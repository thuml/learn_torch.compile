
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


# kernel path: /tmp/torchinductor_youkaichao/oy/coyzvass4ltortxejaujchiyrba63fexx7nmterx3aqwbscixgp6.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_div_native_batch_norm_backward_threshold_backward_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_native_batch_norm_backward_threshold_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1536
    x1 = (xindex // 1536)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1536*r2) + (150528*x1)), rmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (x0 + (1536*(r2 // 49)) + (3072*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr2 + (x0 + (1536*r2) + (150528*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 49.0
        tmp3 = tmp1 / tmp2
        tmp4 = 0.0
        tmp5 = tl.where(tmp0, tmp4, tmp3)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp5 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask, tmp15, _tmp14)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, None)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ho/chonjhioh6zo5kjdyygwzlxtoi2crlmnud3emn2pl3emcpwxifyi.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_div_native_batch_norm_backward_threshold_backward_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_batch_norm_backward_threshold_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1536*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t7/ct727ekfdlrcvhlsf7ffyllzkfonl62oj65ekrzf4grp7vu47v5c.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_div_native_batch_norm_backward_threshold_backward_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_batch_norm_backward_threshold_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1536*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pn/cpnn5kiw44ixxzzyxrbl5zewap7cxo55zrtga42sisxd2uc2zge3.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1536
    x2 = (xindex // 75264)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + (1536*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x3), None)
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/56/c56ldyk6vabawm2mwllqavy6fdhunai2ud2zg3jkk4zpnqdqzpjx.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 264
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (12936*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/me/cmekogxedx2nmk6npgit2lfh22aojx4fxxfsyayvcrflkrfb5bgu.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1056
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 264
    x1 = (xindex // 264)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((49*x0) + (12936*(r2 // 49)) + (25872*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (264*r2) + (25872*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/56/c56v44b3bsybrwh2dvird4h365xyvhf35amxntjn5yztuowdg2gp.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 264
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (264*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iu/ciu2odcnvh26xyq4ileboaitkkdya4wybcbpzltkrdanukrqdbwt.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2112
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 264
    y1 = (yindex // 264)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (264*x2) + (12936*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bs/cbsgnpggwwyixvbgbwpnbmkyetdfugtrnrqnyd6slnd2jip5mj3d.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate, x_350], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___5_____3___se_gate => sigmoid_63
# x_350 => mul_453, sigmoid_61
triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_9', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12672
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x0 = xindex % 1584
    r2 = rindex
    x1 = (xindex // 1584)
    x3 = xindex
    tmp15 = tl.load(in_ptr2 + (x0 + (1584*r2) + (77616*x1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 792, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (49*x0) + (38808*x1)), rmask & tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 1584, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-38808) + r2 + (49*x0) + (38808*x1)), rmask & tmp8 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp14 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp24 = tl.sigmoid(tmp23)
    tmp25 = 1.0
    tmp26 = tmp25 - tmp24
    tmp27 = tmp24 * tmp26
    tmp28 = tmp22 * tmp27
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ct/cctbovgdem4cyxrb2lultrcep67ouqwrglnm6l5iinwzu2h7vww6.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1584
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1584*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sb/csb4jfg75fu7tbsjaewxubyk253lzvhlxw65ufbnzasb6gbga4mi.py
# Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]

triton_poi_fused_add_fill_mul_sigmoid_sub_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = 1.0
    tmp4 = tmp3 - tmp2
    tmp5 = tmp1 * tmp4
    tmp6 = tmp5 + tmp3
    tmp7 = tmp2 * tmp6
    tmp8 = tmp0 * tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ef/cefytxc67li57jvqdtol4zyaetsrnx6x4qqaam6kx4tnwuixgqwd.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 132
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (132*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7l/c7l53fkimaz3o5fextugloyvbv5u226ovhxq5wftjzhnrmp6bhp7.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate], Original ATen: [aten.add, aten.cat, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___5_____3___se_gate => sigmoid_63
triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12672
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 1584
    x2 = xindex
    y1 = (yindex // 1584)
    y3 = yindex
    tmp15 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (y0 + (1584*x2) + (77616*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 792, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (49*y0) + (38808*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 1584, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-38808) + x2 + (49*y0) + (38808*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp14 * tmp16
    tmp19 = 49.0
    tmp20 = tmp18 / tmp19
    tmp21 = tmp17 + tmp20
    tmp23 = tl.sigmoid(tmp22)
    tmp24 = 1.0
    tmp25 = tmp24 - tmp23
    tmp26 = tmp22 * tmp25
    tmp27 = tmp26 + tmp24
    tmp28 = tmp23 * tmp27
    tmp29 = tmp21 * tmp28
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ju/cjuttkjqxbjwewly7zqskx6vfjs35plcnj45fbny2vefohghouwj.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 1584
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (77616*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iv/civo5tuzyfytpg7le2mlzvc34caozvo3ieoqk4mnjebutbrapmir.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6336
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1584
    x1 = (xindex // 1584)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((49*x0) + (77616*(r2 // 49)) + (155232*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (1584*r2) + (155232*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jo/cjoce5d67sfu2acnh7o32wsntvxxinulwbarfn3pn5govcllg3fl.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1584
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1584*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ir/cirpgjphox7nnhpt7l4uthqi7u4ymxtmyz6svfasals5oigikufk.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_17', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12672
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1584
    y1 = (yindex // 1584)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (1584*x2) + (77616*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (49*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pa/cpafj2lyevkdsornecr7pqiuusg3zbqdrfwtfzzalkskk4gvx3sl.py
# Source Nodes: [], Original ATen: [aten.cat, aten.mul]

triton_poi_fused_cat_mul_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_mul_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12672
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 1584
    x2 = xindex
    y1 = (yindex // 1584)
    y3 = yindex
    tmp31 = tl.load(in_ptr4 + (y0 + (1584*x2) + (77616*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 396, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (49*y0) + (19404*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 792, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-19404) + x2 + (49*y0) + (19404*y1)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 1188, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + ((-38808) + x2 + (49*y0) + (19404*y1)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp0 >= tmp16
    tmp23 = tl.full([1, 1], 1584, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr3 + ((-58212) + x2 + (49*y0) + (19404*y1)), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tl.where(tmp18, tmp21, tmp27)
    tmp29 = tl.where(tmp11, tmp14, tmp28)
    tmp30 = tl.where(tmp4, tmp7, tmp29)
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp32, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fo/cfom4cs7swqhti6cs53w4itfculjj4ib4huzhn3aq6d2vxcmitw6.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 264
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
    r3 = rindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (12936*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (49*x0) + (12936*r2)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (x0 + (264*r3)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/gd/cgdi5cwfzgqdtn7slcuszfhiawb55v3r7mikkpfwdcxmqeu6lxin.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2112
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 264
    y1 = (yindex // 264)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (264*x2) + (12936*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.002551020408163265
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tv/ctv5ddfg3naaggfjgvvst2gaz6tgzvc57zn3bnghe4ntxbleulcb.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 264
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
    r3 = rindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (12936*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (49*x0) + (12936*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (49*x0) + (12936*r2)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr3 + (x0 + (264*r3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
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


# kernel path: /tmp/torchinductor_youkaichao/nu/cnueiq24yheqrdbzxwd7t7eluyfcrxgsygwfua6l356gky2sttzv.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2112
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 264
    y1 = (yindex // 264)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0 + (264*x2) + (12936*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
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
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/li/clipwbjbdvk3q2ybe6aqpoann7trnh25qodryv4h6z3rzexbowdn.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 264
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
    r3 = rindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (12936*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (49*x0) + (12936*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (49*x0) + (12936*r2)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (49*x0) + (12936*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr4 + (x0 + (264*r3)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp13 = tmp11 - tmp12
    tmp14 = tmp6 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = tmp18 * tmp19
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ll/cllrtub6lwu536mcvk4xuu6e3xdydsmmt2krk3lh2lp5pjyvmsuf.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2112
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 264
    y1 = (yindex // 264)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0 + (264*x2) + (12936*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp9 = tmp7 - tmp8
    tmp11 = 0.002551020408163265
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (49*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oa/coatm7sebiw62ukdfexzwn57rtazqsrw3qc46miwqmicbutouoeb.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_295], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___5_____0___se_gate => sigmoid_51
# x_295 => mul_378, sigmoid_49
triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_25', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 7680
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 960
    x1 = (xindex // 960)
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (960*r2) + (47040*x1)), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = 1.0
    tmp12 = tmp11 - tmp10
    tmp13 = tmp10 * tmp12
    tmp14 = tmp8 * tmp13
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ca/cca6hydldjeytufha2ecfpafken2bhtjspkcpc7h4ufiu6wq67kf.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (960*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fk/cfky5tohder3etpmo4xerqiu3ueqbpij6ytgwy5putub26b3fvt5.py
# Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]

triton_poi_fused_add_fill_mul_sigmoid_sub_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_27', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = 1.0
    tmp4 = tmp3 - tmp2
    tmp5 = tmp1 * tmp4
    tmp6 = tmp5 + tmp3
    tmp7 = tmp2 * tmp6
    tmp8 = tmp0 * tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mp/cmpxhilln2xsfzn7uu2vvqzn3bw2xvpd5q6cqv3cgs4p65fzmwhb.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 80
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (80*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5z/c5zauca3wqrdggdogpaspx65adnet6kkg5pgt2z6ota2m42q6l65.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___5_____0___se_gate => sigmoid_51
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 960
    x1 = (xindex // 960)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp20 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((49*x0) + (47040*(r2 // 49)) + (94080*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (960*(r2 // 49)) + (1920*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (960*(r2 // 49)) + (1920*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (960*r2) + (94080*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr4 + (x0 + (960*r2) + (94080*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 49.0
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


# kernel path: /tmp/torchinductor_youkaichao/gr/cgr3xex7puiut66ci3jhlsta7wlm5w2omdoht26va4zl6rndipa4.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___5_____0___se_gate => sigmoid_51
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (960*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ge/cgeyaoiepvoezbfobopywpmjktcay3l557pb4cr2b7lxqi3x7irj.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___5_____0___se_gate => sigmoid_51
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (960*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ae/caeg4vlui46musnvle2ektpv5mt5y7sye7xlt4ntphmaubaa7gex.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___5_____0___se_gate => sigmoid_51
triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 960
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 49
    y1 = (yindex // 49)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (47040*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (960*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (960*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (960*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (960*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp5 = 49.0
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
    tmp20 = 0.002551020408163265
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tl.store(out_ptr0 + (x2 + (960*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qq/cqqphrornnrebne43nni6n7d73vhvjr7bjq42qd5vkvcfeybkz5r.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 240
    y1 = (yindex // 240)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (720 + y0 + (960*x2) + (47040*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (720 + y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (720 + y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yo/cyowxgkruly6uvwhdgd34criei2ajvgema7zqb3un42v2mdhd7mv.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 240
    y1 = (yindex // 240)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (480 + y0 + (960*x2) + (47040*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (480 + y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (480 + y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2k/c2kepbfnq3yhtmq47mqs7vqjt5ob24w6xinmpf2fqtgybwyjcpl5.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 240
    y1 = (yindex // 240)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (240 + y0 + (960*x2) + (47040*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (240 + y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (240 + y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oi/coiac7v37agarzxtaylfkwzkci6aoc3dbx3ibvpj7ssnco64g3cp.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 240
    y1 = (yindex // 240)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (960*x2) + (47040*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ap/caponmmnyuyy7kez6seahn77pjdy6n5zy52hnm5rtvkbo625i47x.py
# Source Nodes: [], Original ATen: [aten.cat, aten.mul]

triton_poi_fused_cat_mul_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_mul_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7680
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 960
    x2 = xindex
    y1 = (yindex // 960)
    y3 = yindex
    tmp31 = tl.load(in_ptr4 + (y0 + (960*x2) + (188160*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 240, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (196*y0) + (47040*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 480, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-47040) + x2 + (196*y0) + (47040*y1)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 720, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + ((-94080) + x2 + (196*y0) + (47040*y1)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp0 >= tmp16
    tmp23 = tl.full([1, 1], 960, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr3 + ((-141120) + x2 + (196*y0) + (47040*y1)), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tl.where(tmp18, tmp21, tmp27)
    tmp29 = tl.where(tmp11, tmp14, tmp28)
    tmp30 = tl.where(tmp4, tmp7, tmp29)
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp32, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zq/czqeq2nvpedrcu3q2cocd2ig3ylps5okc4plajn5zfk5crfvu6fi.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_38 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 960
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (188160*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d5/cd5sjeazdshr5oki4rjwt6v57yud5hwae7u6unpc3lvhkuttywdt.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12480
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (188160*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (960*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/qx/cqxzvyd3a2dbilasp3xiwp5jaxcagu55aai4et5idscdx3sc72ts.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 960
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


# kernel path: /tmp/torchinductor_youkaichao/gq/cgqtcfhakm2pbp6xtxkt36h2l5cbmbgctjqac5ntr37wmsa52rbl.py
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
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_41', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7680
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 960
    y1 = (yindex // 960)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (960*x2) + (188160*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cu/ccuvsnu3iza5cfskrh6kepux4w2dm4ljkyyqujgsh7mjymq4jlef.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (31360*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kg/ckggd6ltbptshudhdpueghpmx2hahsmhv6oiu5ozdwanzwlhajs5.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2080
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (31360*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (160*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/tu/ctufyaff2ygq7vubwpzdqacesqn542ykwfgdyymbiyy7o5wfbwp6.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 160
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


# kernel path: /tmp/torchinductor_youkaichao/6p/c6prsw7h5zntvgtxhorulpvb2j73xf2ioagczp3b2y6iyjrrwsxq.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1280
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 160
    y1 = (yindex // 160)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (160*x2) + (31360*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4x/c4xprdoklpp33knoydamlqznq5svlckhupzierzpiwrgyo34rprj.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate, x_276], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___4_____3___se_gate => sigmoid_47
# x_276 => mul_353, sigmoid_45
triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_46', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x0 = xindex % 480
    r2 = rindex
    x1 = (xindex // 480)
    x3 = xindex
    tmp15 = tl.load(in_ptr2 + (x0 + (480*r2) + (94080*x1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 240, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (196*x0) + (47040*x1)), rmask & tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 480, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-47040) + r2 + (196*x0) + (47040*x1)), rmask & tmp8 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp14 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp24 = tl.sigmoid(tmp23)
    tmp25 = 1.0
    tmp26 = tmp25 - tmp24
    tmp27 = tmp24 * tmp26
    tmp28 = tmp22 * tmp27
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2d/c2d3pejhfb5upviok7ldmbkzkqyx3kud5q24mnwu7k6rwksrkhnz.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 480
    rnumel = 8
    RBLOCK: tl.constexpr = 8
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


# kernel path: /tmp/torchinductor_youkaichao/zz/czzfmccfrrq4cwc5npkbnduuc5rkurbf2gp4wgc4vz6euq2ln6kb.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate], Original ATen: [aten.add, aten.cat, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___4_____3___se_gate => sigmoid_47
triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3840
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 480
    x2 = xindex
    y1 = (yindex // 480)
    y3 = yindex
    tmp15 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (y0 + (480*x2) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 240, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (196*y0) + (47040*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 480, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-47040) + x2 + (196*y0) + (47040*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp14 * tmp16
    tmp19 = 196.0
    tmp20 = tmp18 / tmp19
    tmp21 = tmp17 + tmp20
    tmp23 = tl.sigmoid(tmp22)
    tmp24 = 1.0
    tmp25 = tmp24 - tmp23
    tmp26 = tmp22 * tmp25
    tmp27 = tmp26 + tmp24
    tmp28 = tmp23 * tmp27
    tmp29 = tmp21 * tmp28
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bx/cbxvnoi73gmrlf53a6spjbenm5vrwtdiubgjtml4q6zxgww5dhgz.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 480
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (94080*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4e/c4e23fx74kpypxhm6foce6rdmalnewhihis3jk23j5bdqhf7pnw7.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_50 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6240
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (94080*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (480*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ob/cobsqazmkntwba6dbhtbzqnxuleqd3od3ri2nrwqw4gb734glq5l.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 480
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


# kernel path: /tmp/torchinductor_youkaichao/bc/cbccahktdfzgy6adlbs5c7bjsw32suhsdhipfmksrmfjkwnwpf3y.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_52 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_52', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3840
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 480
    y1 = (yindex // 480)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (480*x2) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lp/clpggxui5mjytxqyyivigypxxslnkehpmerp3mqdnwgiland5vvz.py
# Source Nodes: [], Original ATen: [aten.cat, aten.mul]

triton_poi_fused_cat_mul_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_mul_53', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3840
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 480
    x2 = xindex
    y1 = (yindex // 480)
    y3 = yindex
    tmp31 = tl.load(in_ptr4 + (y0 + (480*x2) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 120, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (196*y0) + (23520*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 240, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-23520) + x2 + (196*y0) + (23520*y1)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 360, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + ((-47040) + x2 + (196*y0) + (23520*y1)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp0 >= tmp16
    tmp23 = tl.full([1, 1], 480, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr3 + ((-70560) + x2 + (196*y0) + (23520*y1)), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tl.where(tmp18, tmp21, tmp27)
    tmp29 = tl.where(tmp11, tmp14, tmp28)
    tmp30 = tl.where(tmp4, tmp7, tmp29)
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp32, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nm/cnmoaqfuxgejckmhjtwebbs3avjtc2c2p3o2cszctnxvdipqhnvp.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]

triton_red_fused_add_cat_native_batch_norm_backward_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_batch_norm_backward_54', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp21 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp25 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (31360*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr3 + (x0 + (160*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = x0
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tmp1 >= tmp2
        tmp4 = tl.full([1, 1], 80, tl.int64)
        tmp5 = tmp1 < tmp4
        tmp6 = tl.load(in_ptr1 + (r1 + (196*x0) + (15680*r2)), rmask & tmp5 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = tmp1 >= tmp4
        tmp10 = tl.full([1, 1], 160, tl.int64)
        tmp11 = tmp1 < tmp10
        tmp12 = tl.load(in_ptr2 + ((-15680) + r1 + (196*x0) + (15680*r2)), rmask & tmp9 & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
        tmp14 = tl.where(tmp9, tmp12, tmp13)
        tmp15 = tl.where(tmp5, tmp8, tmp14)
        tmp16 = tmp0 + tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
        tmp22 = tmp20 - tmp21
        tmp23 = tmp16 * tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(rmask & xmask, tmp26, _tmp25)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp18, xmask)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp25, xmask)
    tmp27 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tmp25 * tmp27
    tl.store(out_ptr2 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ld/cldvwwnlisvs25jucazsspbazteadogvilfzxeamr22jj7grosgg.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]

triton_poi_fused_add_cat_native_batch_norm_backward_55 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_native_batch_norm_backward_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1280
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 160
    y1 = (yindex // 160)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (y0 + (160*x2) + (31360*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = y0
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 80, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x2 + (196*y0) + (15680*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1, 1], 160, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tl.load(in_ptr2 + ((-15680) + x2 + (196*y0) + (15680*y1)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.where(tmp5, tmp8, tmp14)
    tmp16 = tmp0 + tmp15
    tmp19 = tmp17 - tmp18
    tmp21 = 0.0006377551020408163
    tmp22 = tmp20 * tmp21
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp19 * tmp25
    tmp27 = tmp16 - tmp26
    tmp29 = tmp28 * tmp21
    tmp30 = tmp27 - tmp29
    tmp32 = tmp23 * tmp31
    tmp33 = tmp30 * tmp32
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp33, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rg/crgpsderojrpoczbqugmxkp7jcvuz2v6mftuv2sseuwgryobqwsr.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat]

triton_poi_fused_add_cat_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_56', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 160
    x2 = (xindex // 31360)
    x4 = xindex % 31360
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = x1
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 80, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + (15680*x2)), tmp5 & xmask, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1], 160, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tl.load(in_ptr1 + ((-15680) + x4 + (15680*x2)), tmp9 & xmask, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.where(tmp5, tmp8, tmp14)
    tmp16 = tmp0 + tmp15
    tmp17 = tl.load(in_ptr2 + (x4 + (15680*x2)), tmp5 & xmask, other=0.0)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp5, tmp17, tmp18)
    tmp20 = tl.load(in_ptr3 + ((-15680) + x4 + (15680*x2)), tmp9 & xmask, other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp9, tmp20, tmp21)
    tmp23 = tl.where(tmp5, tmp19, tmp22)
    tmp24 = tmp16 + tmp23
    tl.store(in_out_ptr0 + (x3), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qr/cqrzyrbgsz2idj2hjopofoyccuuqixtk7y6gft3jbd2xwqpfav7z.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]

triton_poi_fused_add_cat_native_batch_norm_backward_57 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_native_batch_norm_backward_57', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1280
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 160
    y1 = (yindex // 160)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr2 + (y0 + (160*x2) + (31360*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = y0
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 80, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr0 + (x2 + (196*y0) + (15680*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1, 1], 160, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tl.load(in_ptr1 + ((-15680) + x2 + (196*y0) + (15680*y1)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.where(tmp5, tmp8, tmp14)
    tmp16 = tmp0 + tmp15
    tmp19 = tmp17 - tmp18
    tmp21 = 0.0006377551020408163
    tmp22 = tmp20 * tmp21
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp19 * tmp25
    tmp27 = tmp16 - tmp26
    tmp29 = tmp28 * tmp21
    tmp30 = tmp27 - tmp29
    tmp32 = tmp23 * tmp31
    tmp33 = tmp30 * tmp32
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp33, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bh/cbhducg5vmls4xw5txfutdewwo2veqg2wvryspxiqnp4wawsgy3b.py
# Source Nodes: [x_218], Original ATen: [aten.mul, aten.silu, aten.sum]
# x_218 => mul_278, sigmoid_33
triton_red_fused_mul_silu_sum_58 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_silu_sum_58', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9984
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 2
    x1 = (xindex // 2) % 624
    x2 = (xindex // 1248)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (98*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (624*r3) + (61152*x0) + (122304*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3j/c3jtneph5l7elhgxbymhpzmbb6awmsiqh2vrbvcgydijjik73jsr.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_218], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => sigmoid_35
# x_218 => mul_278, sigmoid_33
triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_59 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_59', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4992
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = 1.0
    tmp8 = tmp7 - tmp6
    tmp9 = tmp6 * tmp8
    tmp10 = tmp4 * tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tr/ctrwlfonu2bk45lqv3enf5ctygud4c7jsvyn57bq74e2uu6r46zx.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_60', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 624
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (624*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e6/ce6vvcokz5wniyqoo6dexyr7blxeetwnwgvz2cdpfo3gnn3dce6d.py
# Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]

triton_poi_fused_add_fill_mul_sigmoid_sub_61 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_61', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = 1.0
    tmp4 = tmp3 - tmp2
    tmp5 = tmp1 * tmp4
    tmp6 = tmp5 + tmp3
    tmp7 = tmp2 * tmp6
    tmp8 = tmp0 * tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eh/ceh5s2347xxmxkdetko2v5c6yo6cgozq765sigju7cdasfjyw7wc.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_62 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 52
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (52*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mu/cmuuwzihkw6ds37ql42ynrplpva74ptprgoknfwsriu4yzj2goaj.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => sigmoid_35
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_63 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_63', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8112
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (122304*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (624*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = tl.load(in_ptr2 + (x1 + (624*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = 196.0
        tmp9 = tmp7 / tmp8
        tmp10 = tmp6 + tmp9
        tmp11 = tl.load(in_ptr3 + (x1 + (624*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.sigmoid(tmp11)
        tmp13 = 1.0
        tmp14 = tmp13 - tmp12
        tmp15 = tmp11 * tmp14
        tmp16 = tmp15 + tmp13
        tmp17 = tmp12 * tmp16
        tmp18 = tmp10 * tmp17
        tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
        tmp20 = tl.where(tmp2, tmp18, tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ef/cefxhxkacqhl4qnz534ansmlc6givqbmuw2dijgfjqr6k42heyag.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => sigmoid_35
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_64 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_64', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 624
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


# kernel path: /tmp/torchinductor_youkaichao/il/cilohyp4vvqqo4jaejuenbewcsgqt7nu3g4v7zhnthqi6hijmmka.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => sigmoid_35
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_65 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_65', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8112
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 624)
    x0 = xindex % 624
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x0) + (122304*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (624*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = tl.load(in_ptr2 + (x0 + (624*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = 196.0
        tmp9 = tmp7 / tmp8
        tmp10 = tmp6 + tmp9
        tmp11 = tl.load(in_ptr3 + (x0 + (624*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.sigmoid(tmp11)
        tmp13 = 1.0
        tmp14 = tmp13 - tmp12
        tmp15 = tmp11 * tmp14
        tmp16 = tmp15 + tmp13
        tmp17 = tmp12 * tmp16
        tmp18 = tmp10 * tmp17
        tmp19 = tl.load(in_ptr4 + (x0 + (624*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tmp19 - tmp20
        tmp22 = tmp18 * tmp21
        tmp23 = tl.full(tmp22.shape, 0, tmp22.dtype)
        tmp24 = tl.where(tmp2, tmp22, tmp23)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/aw/cawdq3d57hxgz2shtiw4t5i7jrmgjunsxcmbe2mwf3svycz6n3ph.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => sigmoid_35
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_66 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_66', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 624
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (624*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/43/c43ysvyex4qal2jafaqubtspi2vius6l3cdjfydzw4h3ye4qvnfk.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => sigmoid_35
triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_67 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_67', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 624
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (122304*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (624*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (624*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (624*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (624*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp5 = 196.0
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
    tmp20 = 0.0006377551020408163
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tl.store(out_ptr0 + (x2 + (624*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eq/ceqosznflp6zehru3dcr5xlajvctbbv3nwm4pzuhn6kxkilbbinq.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_68 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_68', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4992
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 624
    y1 = (yindex // 624)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (624*x2) + (122304*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ht/chtqg2wkimc4nczjlmnx7sx2coqjth62lgkzfvwdey6ajdjij3nk.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_69 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_69', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8112
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (122304*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (624*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t2/ct2cplnouorll4siwskvbb245yvgrifldofrdy4iasfhtowkhoqp.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_70 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_70', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8112
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 624)
    x0 = xindex % 624
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x0) + (122304*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (624*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.load(in_ptr2 + (x0 + (624*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cy/ccyx2ysbwi2lngcqhpppoerr4oayzi3b7b6npvhyy6y42ulvodyb.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_71 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_71', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4992
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 624
    y1 = (yindex // 624)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (624*x2) + (122304*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (624*x2) + (122304*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.0006377551020408163
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zm/czmascuszom2unvvackikupua5agceb7ea37tm22zkclvopv5yvp.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_72 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_72', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 104
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (20384*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k2/ck2xv334hznhlylczqairy66h3zdwxwwktb2ybbgvx3wlaypsbrt.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_73 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_73', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1352
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (20384*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (104*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/2y/c2ylnljfjituyyao7wga3apxdmhxbg7lekteo54guuwm3q65dtli.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_74 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_74', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 104
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


# kernel path: /tmp/torchinductor_youkaichao/4o/c4on7kj3lc5zl3lacuiuyysdueyc7o4xbnbtdkql64a7q7f44vba.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_75 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_75', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 832
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 104
    y1 = (yindex // 104)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (104*x2) + (20384*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ac/cacv4oeettdnp7okulytbac3v24r6rpwrgxh4oc23nskws2c4nue.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___se_gate, x_200], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___3_____3___se_gate => sigmoid_31
# x_200 => mul_253, sigmoid_29
triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_76 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_76', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x0 = xindex % 624
    r2 = rindex
    x1 = (xindex // 624)
    x3 = xindex
    tmp15 = tl.load(in_ptr2 + (x0 + (624*r2) + (122304*x1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 312, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (196*x0) + (61152*x1)), rmask & tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 624, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-61152) + r2 + (196*x0) + (61152*x1)), rmask & tmp8 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp14 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp24 = tl.sigmoid(tmp23)
    tmp25 = 1.0
    tmp26 = tmp25 - tmp24
    tmp27 = tmp24 * tmp26
    tmp28 = tmp22 * tmp27
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6z/c6z3flawaopfjlztmnscekefyglmu7sopcpz53vwknomedydn56e.py
# Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]

triton_poi_fused_add_fill_mul_sigmoid_sub_77 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_77', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = 1.0
    tmp4 = tmp3 - tmp2
    tmp5 = tmp1 * tmp4
    tmp6 = tmp5 + tmp3
    tmp7 = tmp2 * tmp6
    tmp8 = tmp0 * tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sn/csn3d43xeg6m73zd6pjnd55jbdojgxjhdscwtnexiirsod3uhrbt.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_78 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_78', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 26
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (26*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iz/cizrlzy7llbnosdpps4nuiqlgoyllrghzv5tl3fpeo4ltrzhrr65.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___se_gate], Original ATen: [aten.add, aten.cat, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___3_____3___se_gate => sigmoid_31
triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_79 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_79', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4992
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 624
    x2 = xindex
    y1 = (yindex // 624)
    y3 = yindex
    tmp15 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (y0 + (624*x2) + (122304*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 312, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (196*y0) + (61152*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 624, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-61152) + x2 + (196*y0) + (61152*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp14 * tmp16
    tmp19 = 196.0
    tmp20 = tmp18 / tmp19
    tmp21 = tmp17 + tmp20
    tmp23 = tl.sigmoid(tmp22)
    tmp24 = 1.0
    tmp25 = tmp24 - tmp23
    tmp26 = tmp22 * tmp25
    tmp27 = tmp26 + tmp24
    tmp28 = tmp23 * tmp27
    tmp29 = tmp21 * tmp28
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s3/cs3ybpavwshrorz3anyjwjq45kjarq6i3io3ejk6cvqfk3cclnhq.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_80 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_80', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 624
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (122304*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6v/c6vjw5kaohzvnqjyf5cuqaexrwltqbvwvyotyli6gjce5largovn.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_81 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_81', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8112
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (122304*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (624*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ud/cudae4x5k4gc3uqotocrmhb7nipmo53ghvqu23volb3nayo3wacd.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_82 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_82', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 624
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


# kernel path: /tmp/torchinductor_youkaichao/ig/cig7nmt67hr22aalw4dqslcnahe7irhlolyykcmk7invhm764vcp.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_83 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_83', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4992
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 624
    y1 = (yindex // 624)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (624*x2) + (122304*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ym/cymm3pfavnijsjqoghlqi7odlrkrqkbcwexwcsrhfuz4nxoev4xx.py
# Source Nodes: [], Original ATen: [aten.cat, aten.mul]

triton_poi_fused_cat_mul_84 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_mul_84', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4992
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 624
    x2 = xindex
    y1 = (yindex // 624)
    y3 = yindex
    tmp31 = tl.load(in_ptr4 + (y0 + (624*x2) + (122304*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 156, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (196*y0) + (30576*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 312, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-30576) + x2 + (196*y0) + (30576*y1)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 468, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + ((-61152) + x2 + (196*y0) + (30576*y1)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp0 >= tmp16
    tmp23 = tl.full([1, 1], 624, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr3 + ((-91728) + x2 + (196*y0) + (30576*y1)), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tl.where(tmp18, tmp21, tmp27)
    tmp29 = tl.where(tmp11, tmp14, tmp28)
    tmp30 = tl.where(tmp4, tmp7, tmp29)
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp32, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qm/cqmxshcidlq7pzkdy7wtr3hqb426nxgaoakkabige6xlbrshlw4u.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]

triton_red_fused_add_cat_native_batch_norm_backward_85 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_batch_norm_backward_85', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 104
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp21 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp25 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (20384*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr3 + (x0 + (104*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = x0
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tmp1 >= tmp2
        tmp4 = tl.full([1, 1], 52, tl.int64)
        tmp5 = tmp1 < tmp4
        tmp6 = tl.load(in_ptr1 + (r1 + (196*x0) + (10192*r2)), rmask & tmp5 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = tmp1 >= tmp4
        tmp10 = tl.full([1, 1], 104, tl.int64)
        tmp11 = tmp1 < tmp10
        tmp12 = tl.load(in_ptr2 + ((-10192) + r1 + (196*x0) + (10192*r2)), rmask & tmp9 & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
        tmp14 = tl.where(tmp9, tmp12, tmp13)
        tmp15 = tl.where(tmp5, tmp8, tmp14)
        tmp16 = tmp0 + tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
        tmp22 = tmp20 - tmp21
        tmp23 = tmp16 * tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(rmask & xmask, tmp26, _tmp25)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp18, xmask)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp25, xmask)
    tmp27 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tmp25 * tmp27
    tl.store(out_ptr2 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3y/c3y6bllzhxxkuwa66c5wvwacecfo5gp7nlb2yv2q5475piuzs5jr.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]

triton_poi_fused_add_cat_native_batch_norm_backward_86 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_native_batch_norm_backward_86', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 832
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 104
    y1 = (yindex // 104)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (y0 + (104*x2) + (20384*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = y0
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 52, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x2 + (196*y0) + (10192*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1, 1], 104, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tl.load(in_ptr2 + ((-10192) + x2 + (196*y0) + (10192*y1)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.where(tmp5, tmp8, tmp14)
    tmp16 = tmp0 + tmp15
    tmp19 = tmp17 - tmp18
    tmp21 = 0.0006377551020408163
    tmp22 = tmp20 * tmp21
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp19 * tmp25
    tmp27 = tmp16 - tmp26
    tmp29 = tmp28 * tmp21
    tmp30 = tmp27 - tmp29
    tmp32 = tmp23 * tmp31
    tmp33 = tmp30 * tmp32
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp33, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s4/cs4junkfnztpca4l4ax5r2yb3yj2lyi2dww4syg7khz7gewyasza.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat]

triton_poi_fused_add_cat_87 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_87', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 104
    x2 = (xindex // 20384)
    x4 = xindex % 20384
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = x1
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 52, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + (10192*x2)), tmp5 & xmask, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1], 104, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tl.load(in_ptr1 + ((-10192) + x4 + (10192*x2)), tmp9 & xmask, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.where(tmp5, tmp8, tmp14)
    tmp16 = tmp0 + tmp15
    tmp17 = tl.load(in_ptr2 + (x4 + (10192*x2)), tmp5 & xmask, other=0.0)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp5, tmp17, tmp18)
    tmp20 = tl.load(in_ptr3 + ((-10192) + x4 + (10192*x2)), tmp9 & xmask, other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp9, tmp20, tmp21)
    tmp23 = tl.where(tmp5, tmp19, tmp22)
    tmp24 = tmp16 + tmp23
    tl.store(in_out_ptr0 + (x3), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dj/cdj5mdq6mdycyncnagzmjnw4yeaqdggw63alfyngr42gmtu6g2ov.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]

triton_poi_fused_add_cat_native_batch_norm_backward_88 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_native_batch_norm_backward_88', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 832
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 104
    y1 = (yindex // 104)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr2 + (y0 + (104*x2) + (20384*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = y0
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 52, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr0 + (x2 + (196*y0) + (10192*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1, 1], 104, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tl.load(in_ptr1 + ((-10192) + x2 + (196*y0) + (10192*y1)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.where(tmp5, tmp8, tmp14)
    tmp16 = tmp0 + tmp15
    tmp19 = tmp17 - tmp18
    tmp21 = 0.0006377551020408163
    tmp22 = tmp20 * tmp21
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp19 * tmp25
    tmp27 = tmp16 - tmp26
    tmp29 = tmp28 * tmp21
    tmp30 = tmp27 - tmp29
    tmp32 = tmp23 * tmp31
    tmp33 = tmp30 * tmp32
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp33, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j5/cj5finpeazbuszslizutkbhbyfxfml6ocry2ks6eolr2slukekgs.py
# Source Nodes: [x_142], Original ATen: [aten.mul, aten.silu, aten.sum]
# x_142 => mul_178, sigmoid_17
triton_red_fused_mul_silu_sum_89 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_silu_sum_89', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5376
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 2
    x1 = (xindex // 2) % 336
    x2 = (xindex // 672)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (98*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (336*r3) + (32928*x0) + (65856*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w6/cw6roldwkbdh7gsfrolevmecuspq62vvc3qhujut2et2ta753vbf.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_142], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_19
# x_142 => mul_178, sigmoid_17
triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_90 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_90', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2688
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = 1.0
    tmp8 = tmp7 - tmp6
    tmp9 = tmp6 * tmp8
    tmp10 = tmp4 * tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u5/cu5a2fcx5bx6zki47t3hoki26txycnmrnxeyjimkgbu5jwqihm2p.py
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
    size_hints=[512, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_91', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 336
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (336*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fc/cfcudpeu4tlobsbmcsb6nqufyiqyvaghnbnxj3yjvh5afkeecsd7.py
# Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]

triton_poi_fused_add_fill_mul_sigmoid_sub_92 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_92', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = 1.0
    tmp4 = tmp3 - tmp2
    tmp5 = tmp1 * tmp4
    tmp6 = tmp5 + tmp3
    tmp7 = tmp2 * tmp6
    tmp8 = tmp0 * tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ye/cyewl2aw45imnzhl7jvefos34664jp7phmviotayhf46hrs7m533.py
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
    size_hints=[16, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_93', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 14
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (14*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lt/cltalkxlpfeushmcmytfazia3ffsemnf6kuwhzzkjwwljiqn5tc3.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_19
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_94 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_94', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4368
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (65856*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (336*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = tl.load(in_ptr2 + (x1 + (336*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = 196.0
        tmp9 = tmp7 / tmp8
        tmp10 = tmp6 + tmp9
        tmp11 = tl.load(in_ptr3 + (x1 + (336*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.sigmoid(tmp11)
        tmp13 = 1.0
        tmp14 = tmp13 - tmp12
        tmp15 = tmp11 * tmp14
        tmp16 = tmp15 + tmp13
        tmp17 = tmp12 * tmp16
        tmp18 = tmp10 * tmp17
        tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
        tmp20 = tl.where(tmp2, tmp18, tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wz/cwzbvdy2piby2aqhwz6a7i7ez7u4xe5yrr4u4wt4tzv4fyupnuhe.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_19
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_95 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_95', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 336
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


# kernel path: /tmp/torchinductor_youkaichao/vh/cvhvagifjpf4vbkbywojpvu5qcwvxf76cm44fhiokuyflgm4zrr6.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_19
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_96 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_96', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4368
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 336)
    x0 = xindex % 336
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x0) + (65856*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (336*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = tl.load(in_ptr2 + (x0 + (336*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = 196.0
        tmp9 = tmp7 / tmp8
        tmp10 = tmp6 + tmp9
        tmp11 = tl.load(in_ptr3 + (x0 + (336*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.sigmoid(tmp11)
        tmp13 = 1.0
        tmp14 = tmp13 - tmp12
        tmp15 = tmp11 * tmp14
        tmp16 = tmp15 + tmp13
        tmp17 = tmp12 * tmp16
        tmp18 = tmp10 * tmp17
        tmp19 = tl.load(in_ptr4 + (x0 + (336*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tmp19 - tmp20
        tmp22 = tmp18 * tmp21
        tmp23 = tl.full(tmp22.shape, 0, tmp22.dtype)
        tmp24 = tl.where(tmp2, tmp22, tmp23)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4l/c4ltglqicfctusku4qpiokrihvwiypr3ps6mocfmblfvbevrhmdj.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_19
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_97 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_97', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 336
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (336*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mx/cmxjwta4bhhjsyovihrxgjk4ikymyyymlqf76frfk4hvwxsbptlg.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_19
triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_98 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_98', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 336
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (65856*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (336*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (336*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (336*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (336*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp5 = 196.0
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
    tmp20 = 0.0006377551020408163
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tl.store(out_ptr0 + (x2 + (336*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5p/c5pyafj7egymnn6d4h5iag3l7vdoxoa2fhpfg63x5jb75qvemp74.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_99 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_99', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (224 + y0 + (336*x2) + (65856*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (224 + y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (224 + y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/36/c36bteqjvlzyeijoufoecw2doebnagujlt7scp3fabizwak2hx7u.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_100 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_100', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (112 + y0 + (336*x2) + (65856*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (112 + y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (112 + y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d5/cd5vl3epqvrg4unkw75aupgunjpkyn6zdybteqyhyrgrbwynourh.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_101 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_101', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (336*x2) + (65856*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oc/cochvzibsu6fvs5x45ke2l3wmn5axn437su2qecirm65ugr6tzsh.py
# Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_cat_mul_native_batch_norm_backward_102 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_mul_native_batch_norm_backward_102', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 336
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp29 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp33 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        r3 = rindex
        tmp23 = tl.load(in_ptr3 + (x0 + (336*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_ptr4 + (x0 + (336*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 112, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (r1 + (784*x0) + (87808*r2)), rmask & tmp4 & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
        tmp7 = tl.where(tmp4, tmp5, tmp6)
        tmp8 = tmp0 >= tmp3
        tmp9 = tl.full([1, 1], 224, tl.int64)
        tmp10 = tmp0 < tmp9
        tmp11 = tmp8 & tmp10
        tmp12 = tl.load(in_ptr1 + ((-87808) + r1 + (784*x0) + (87808*r2)), rmask & tmp11 & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
        tmp14 = tl.where(tmp11, tmp12, tmp13)
        tmp15 = tmp0 >= tmp9
        tmp16 = tl.full([1, 1], 336, tl.int64)
        tmp17 = tmp0 < tmp16
        tmp18 = tl.load(in_ptr2 + ((-175616) + r1 + (784*x0) + (87808*r2)), rmask & tmp15 & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
        tmp20 = tl.where(tmp15, tmp18, tmp19)
        tmp21 = tl.where(tmp11, tmp14, tmp20)
        tmp22 = tl.where(tmp4, tmp7, tmp21)
        tmp24 = tmp22 * tmp23
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
        tmp30 = tmp28 - tmp29
        tmp31 = tmp24 * tmp30
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
        tmp34 = _tmp33 + tmp32
        _tmp33 = tl.where(rmask & xmask, tmp34, _tmp33)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp26, xmask)
    tmp33 = tl.sum(_tmp33, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp33, xmask)
    tmp35 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp36 = tmp33 * tmp35
    tl.store(out_ptr2 + (x0), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h7/ch7xd4u2pm4mq5qyearjgjiwbkdhx6a5baow2kslqw3f5rg3scke.py
# Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_cat_convolution_backward_mul_native_batch_norm_backward_103 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_backward_mul_native_batch_norm_backward_103', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2688
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 336
    x2 = xindex
    y1 = (yindex // 336)
    y3 = yindex
    tmp23 = tl.load(in_ptr3 + (y0 + (336*x2) + (263424*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (y0 + (336*x2) + (263424*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr9 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (784*y0) + (87808*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 224, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-87808) + x2 + (784*y0) + (87808*y1)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 336, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-175616) + x2 + (784*y0) + (87808*y1)), tmp15 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tl.where(tmp11, tmp14, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tmp24 = tmp22 * tmp23
    tmp27 = tmp25 - tmp26
    tmp29 = 0.00015943877551020407
    tmp30 = tmp28 * tmp29
    tmp32 = tmp31 * tmp31
    tmp33 = tmp30 * tmp32
    tmp34 = tmp27 * tmp33
    tmp35 = tmp24 - tmp34
    tmp37 = tmp36 * tmp29
    tmp38 = tmp35 - tmp37
    tmp40 = tmp31 * tmp39
    tmp41 = tmp38 * tmp40
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (784*y3)), tmp41, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cd/ccdk4mmm54bhdz5k5uhz333rybnvvracrtsupbvovqxow2w2ovbs.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_104 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_104', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 56
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (43904*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uv/cuvsabcotuoygtk4kordcuqlgvz3qgmn7jmq2hjpszcdjzlwlkyr.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_105 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_105', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2744
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((784*x1) + (43904*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (56*r2) + (7168*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vv/cvvfdbon3tub2tgvaydryzcim74jr3h7w73c36ezfhi4ccdaptor.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_106 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_106', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 56
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rl/crltpjglyf3w4ksocefkk7wvvablx47gvcdlrgkrv7ugtkf3mpm4.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_107 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_107', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 56
    y1 = (yindex // 56)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (56*x2) + (43904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qr/cqrpr7o4ekrvpjmp6kn4eldpzvowrcpk4iq5th5tphf3hilzxgsh.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___se_gate, x_123], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___2_____3___se_gate => sigmoid_15
# x_123 => mul_153, sigmoid_13
triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_108 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_108', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 2688
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 336
    r2 = rindex
    x1 = (xindex // 336)
    x3 = xindex
    tmp15 = tl.load(in_ptr2 + (x0 + (336*r2) + (263424*x1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 168, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (784*x0) + (131712*x1)), rmask & tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 336, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-131712) + r2 + (784*x0) + (131712*x1)), rmask & tmp8 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp14 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask & xmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp24 = tl.sigmoid(tmp23)
    tmp25 = 1.0
    tmp26 = tmp25 - tmp24
    tmp27 = tmp24 * tmp26
    tmp28 = tmp22 * tmp27
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/aa/caavyhpfokyuwrzcgvpoqis7hw5g36rfxurjkq2sn7yp4b6xtmcp.py
# Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]

triton_poi_fused_add_fill_mul_sigmoid_sub_109 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_109', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = 1.0
    tmp4 = tmp3 - tmp2
    tmp5 = tmp1 * tmp4
    tmp6 = tmp5 + tmp3
    tmp7 = tmp2 * tmp6
    tmp8 = tmp0 * tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5g/c5gjbsviqtcpanoysol2eoeudtwppnfwdv66o337uzp2naogkkwl.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_110 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_110', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 28
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (28*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n6/cn65kj7r5e6tbslntvzn74nchbsmzo3nb4iepyaq3mkoylorftqd.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___se_gate], Original ATen: [aten.add, aten.cat, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___2_____3___se_gate => sigmoid_15
triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_111 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_111', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2688
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 336
    x2 = xindex
    y1 = (yindex // 336)
    y3 = yindex
    tmp15 = tl.load(in_ptr2 + (y3), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (y0 + (336*x2) + (263424*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 168, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (784*y0) + (131712*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 336, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-131712) + x2 + (784*y0) + (131712*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp14 * tmp16
    tmp19 = 784.0
    tmp20 = tmp18 / tmp19
    tmp21 = tmp17 + tmp20
    tmp23 = tl.sigmoid(tmp22)
    tmp24 = 1.0
    tmp25 = tmp24 - tmp23
    tmp26 = tmp22 * tmp25
    tmp27 = tmp26 + tmp24
    tmp28 = tmp23 * tmp27
    tmp29 = tmp21 * tmp28
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7m/c7meb5iyuidfzqoa7pwcgpv22pnb6aqcv3gs5wrbhabdakxirol4.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_112 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_112', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 336
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (263424*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jk/cjknseuox2drr6ibk2dvlh4mceciekng33cokwvzidenzpj5c5wa.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_113 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_113', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16464
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((784*x1) + (263424*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (336*r2) + (43008*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ib/cib56732dxgmooi3fmqfa2y4dqm2rz2kymfsrko5nvwzuv2zbhdz.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_114 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_114', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 336
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sb/csbfg3iwv5ea6bgvackycym7untzqob2ru7mxyzlog27lkbc6fsq.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_115 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_115', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2688
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 336
    y1 = (yindex // 336)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (336*x2) + (263424*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (784*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/av/cav6llps4uqcwzdurcfxtnenogejqiq62daftk54rjpnremsder3.py
# Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_cat_mul_native_batch_norm_backward_116 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_mul_native_batch_norm_backward_116', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 336
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        r3 = rindex
        tmp15 = tl.load(in_ptr2 + (x0 + (336*r3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 168, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (r1 + (784*x0) + (131712*r2)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
        tmp7 = tl.where(tmp4, tmp5, tmp6)
        tmp8 = tmp0 >= tmp3
        tmp9 = tl.full([1, 1], 336, tl.int64)
        tmp10 = tmp0 < tmp9
        tmp11 = tl.load(in_ptr1 + ((-131712) + r1 + (784*x0) + (131712*r2)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
        tmp13 = tl.where(tmp8, tmp11, tmp12)
        tmp14 = tl.where(tmp4, tmp7, tmp13)
        tmp16 = tmp14 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vt/cvt2mnygff44hjghsuc7blir66zu5m7u2zjk2hsbhezcz2ph3evm.py
# Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_cat_mul_native_batch_norm_backward_117 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_mul_native_batch_norm_backward_117', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16464
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 49)
    x0 = xindex % 49
    tmp18 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp15 = tl.load(in_ptr2 + (x1 + (336*r2) + (43008*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr3 + (x1 + (336*r2) + (43008*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x1
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 168, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + ((784*x1) + (131712*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
        tmp7 = tl.where(tmp4, tmp5, tmp6)
        tmp8 = tmp0 >= tmp3
        tmp9 = tl.full([1, 1], 336, tl.int64)
        tmp10 = tmp0 < tmp9
        tmp11 = tl.load(in_ptr1 + ((-131712) + (784*x1) + (131712*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
        tmp13 = tl.where(tmp8, tmp11, tmp12)
        tmp14 = tl.where(tmp4, tmp7, tmp13)
        tmp16 = tmp14 * tmp15
        tmp19 = tmp17 - tmp18
        tmp20 = tmp16 * tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h6/ch6qtu37zaa5ghxpccxpjqbx3td5oiq4bofzldt7kvv7tiihmyja.py
# Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_cat_mul_native_batch_norm_backward_118 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_mul_native_batch_norm_backward_118', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2688
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 336
    x2 = xindex
    y1 = (yindex // 336)
    y3 = yindex
    tmp15 = tl.load(in_ptr2 + (y0 + (336*x2) + (263424*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (y0 + (336*x2) + (263424*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 168, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (784*y0) + (131712*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 336, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-131712) + x2 + (784*y0) + (131712*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp16 = tmp14 * tmp15
    tmp19 = tmp17 - tmp18
    tmp21 = 0.00015943877551020407
    tmp22 = tmp20 * tmp21
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp19 * tmp25
    tmp27 = tmp16 - tmp26
    tmp29 = tmp28 * tmp21
    tmp30 = tmp27 - tmp29
    tmp32 = tmp23 * tmp31
    tmp33 = tmp30 * tmp32
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp33, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4j/c4jbny5h7mlfbjj7xhm6ajyrzm7jf7fiobls5ra4u3ahrlqtrxgr.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]

triton_red_fused_add_cat_native_batch_norm_backward_119 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_batch_norm_backward_119', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 56
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp21 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp25 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (43904*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr3 + (x0 + (56*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = x0
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tmp1 >= tmp2
        tmp4 = tl.full([1, 1], 28, tl.int64)
        tmp5 = tmp1 < tmp4
        tmp6 = tl.load(in_ptr1 + (r1 + (784*x0) + (21952*r2)), rmask & tmp5 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = tmp1 >= tmp4
        tmp10 = tl.full([1, 1], 56, tl.int64)
        tmp11 = tmp1 < tmp10
        tmp12 = tl.load(in_ptr2 + ((-21952) + r1 + (784*x0) + (21952*r2)), rmask & tmp9 & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
        tmp14 = tl.where(tmp9, tmp12, tmp13)
        tmp15 = tl.where(tmp5, tmp8, tmp14)
        tmp16 = tmp0 + tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
        tmp22 = tmp20 - tmp21
        tmp23 = tmp16 * tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(rmask & xmask, tmp26, _tmp25)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp18, xmask)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp25, xmask)
    tmp27 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tmp25 * tmp27
    tl.store(out_ptr2 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ce/cceia7bgrkndngvs5sato2f7db3s6aaydt7qch4hciq24kq6rkj7.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]

triton_poi_fused_add_cat_native_batch_norm_backward_120 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_native_batch_norm_backward_120', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 56
    y1 = (yindex // 56)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (y0 + (56*x2) + (43904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = y0
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 28, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x2 + (784*y0) + (21952*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1, 1], 56, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tl.load(in_ptr2 + ((-21952) + x2 + (784*y0) + (21952*y1)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.where(tmp5, tmp8, tmp14)
    tmp16 = tmp0 + tmp15
    tmp19 = tmp17 - tmp18
    tmp21 = 0.00015943877551020407
    tmp22 = tmp20 * tmp21
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp19 * tmp25
    tmp27 = tmp16 - tmp26
    tmp29 = tmp28 * tmp21
    tmp30 = tmp27 - tmp29
    tmp32 = tmp23 * tmp31
    tmp33 = tmp30 * tmp32
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp33, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/er/certalkpuhqtf6kdjbhaihrpzcoossyg7qrntdoqk6mag5xpy7h7.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat]

triton_poi_fused_add_cat_121 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_121', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 351232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 56
    x2 = (xindex // 43904)
    x4 = xindex % 43904
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = x1
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 28, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + (21952*x2)), tmp5 & xmask, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1], 56, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tl.load(in_ptr1 + ((-21952) + x4 + (21952*x2)), tmp9 & xmask, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.where(tmp5, tmp8, tmp14)
    tmp16 = tmp0 + tmp15
    tmp17 = tl.load(in_ptr2 + (x4 + (21952*x2)), tmp5 & xmask, other=0.0)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp5, tmp17, tmp18)
    tmp20 = tl.load(in_ptr3 + ((-21952) + x4 + (21952*x2)), tmp9 & xmask, other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp9, tmp20, tmp21)
    tmp23 = tl.where(tmp5, tmp19, tmp22)
    tmp24 = tmp16 + tmp23
    tl.store(in_out_ptr0 + (x3), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/64/c64rptxcktoo3por6gjv6an32s4osknnordcd3rnol7nh2xindvi.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]

triton_poi_fused_add_cat_native_batch_norm_backward_122 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_native_batch_norm_backward_122', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 56
    y1 = (yindex // 56)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr2 + (y0 + (56*x2) + (43904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = y0
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 28, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr0 + (x2 + (784*y0) + (21952*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1, 1], 56, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tl.load(in_ptr1 + ((-21952) + x2 + (784*y0) + (21952*y1)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.where(tmp5, tmp8, tmp14)
    tmp16 = tmp0 + tmp15
    tmp19 = tmp17 - tmp18
    tmp21 = 0.00015943877551020407
    tmp22 = tmp20 * tmp21
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp19 * tmp25
    tmp27 = tmp16 - tmp26
    tmp29 = tmp28 * tmp21
    tmp30 = tmp27 - tmp29
    tmp32 = tmp23 * tmp31
    tmp33 = tmp30 * tmp32
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (784*y3)), tmp33, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s7/cs76d4m7gxdvhsy5gxtcn3y7xpt5fbzxzfihk7my7y744cfkforr.py
# Source Nodes: [x_65], Original ATen: [aten.mul, aten.silu, aten.sum]
# x_65 => mul_78, sigmoid_1
triton_red_fused_mul_silu_sum_123 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_silu_sum_123', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 13440
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 7
    x1 = (xindex // 7) % 240
    x2 = (xindex // 1680)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (112*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (240*r3) + (26880*x0) + (188160*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2y/c2yday32sn65hu3ov6odsjjnw6irjubrouxlsizwtnbczzs475t7.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_65], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => sigmoid_3
# x_65 => mul_78, sigmoid_1
triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_124 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_124', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (7*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = 1.0
    tmp8 = tmp7 - tmp6
    tmp9 = tmp6 * tmp8
    tmp10 = tmp4 * tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/im/cimsucu4myjdic63g263ef5gpg2ygnnsqen42kx2hjzvnedmc7eo.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_125 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_125', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    rnumel = 8
    RBLOCK: tl.constexpr = 8
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


# kernel path: /tmp/torchinductor_youkaichao/4c/c4cn5qpqm6ekqqv4o4sh7fg7hxilk24hrjo6glj4chqrtgykf3iy.py
# Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]

triton_poi_fused_add_fill_mul_sigmoid_sub_126 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_126', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = 1.0
    tmp4 = tmp3 - tmp2
    tmp5 = tmp1 * tmp4
    tmp6 = tmp5 + tmp3
    tmp7 = tmp2 * tmp6
    tmp8 = tmp0 * tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wn/cwndmjrceokxvw23autnwbrgkfcgdcmngee5cgormsualmph25on.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_127 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_127', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 20
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (20*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e5/ce5mz4qxwoimwd65icosa7d72xh3asfikg7ppdnapjqfjdh27mju.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => sigmoid_3
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_128 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_128', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 11760
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((784*x1) + (188160*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (240*((r2 + (128*x0)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x1 + (240*((r2 + (128*x0)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x1 + (240*r2) + (30720*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 784.0
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
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cf/ccfw6bokhrpqrncghzqbnuqy3yoll4yk2gbbcrlbiodxrddeyhpo.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => sigmoid_3
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_129 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_129', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v2/cv2g6ufydyxj6tbbdfa5pvs3di6nhco2luuuqgja4egboq4psids.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => sigmoid_3
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_130 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_130', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 11760
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 240
    x1 = (xindex // 240)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((784*x0) + (188160*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (240*((r2 + (128*x1)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (240*((r2 + (128*x1)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (240*r2) + (30720*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr4 + (x0 + (240*r2) + (30720*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 784.0
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
        tmp19 = tmp15 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask & xmask, tmp22, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4n/c4nomgnb2ptopr2b4gro22y3hwuqd7kdkfeoppuxc7esgeqgi65q.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => sigmoid_3
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_131 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_131', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (240*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vv/cvva43ma2rdtl3dqsaajqnyagab3de2uoe4cpyfwd6rrz2extmxv.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => sigmoid_3
triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_132 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_132', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 240
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (188160*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (240*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (240*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (240*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (240*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp5 = 784.0
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
    tmp20 = 0.00015943877551020407
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tl.store(out_ptr0 + (x2 + (240*y3)), tmp29, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/63/c63s4rmplxc7vks7iqxbwcslql3hik477a665iaceyarqeayjzap.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_133 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_133', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 480
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 60
    y1 = (yindex // 60)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (180 + y0 + (240*x2) + (188160*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (180 + y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (180 + y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s6/cs6p45v63lcjvjfgkhv7wisaew7rpf26nvepmktntylyvgubwv4d.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_134 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_134', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 480
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 60
    y1 = (yindex // 60)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (120 + y0 + (240*x2) + (188160*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (120 + y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (120 + y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bt/cbttsmm7tjigpvatoa363x57ykwvlc3fuj5jwy2hpntavcq3twog.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_135 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_135', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 480
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 60
    y1 = (yindex // 60)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (60 + y0 + (240*x2) + (188160*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (60 + y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (60 + y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yg/cygleujg3wjwg5o7tgvvv3untzmafirc57aaywdtxcmwufcxbkkr.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_136 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_136', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 480
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 60
    y1 = (yindex // 60)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (240*x2) + (188160*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ly/cly2b4p2wcqqbpzlt2inrkufecvjyjnwwtjx3cfjqo4g54vsrch2.py
# Source Nodes: [], Original ATen: [aten.cat, aten.mul]

triton_poi_fused_cat_mul_137 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_mul_137', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 240
    x2 = xindex
    y1 = (yindex // 240)
    y3 = yindex
    tmp31 = tl.load(in_ptr4 + (y0 + (240*x2) + (752640*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 60, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (3136*y0) + (188160*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 120, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-188160) + x2 + (3136*y0) + (188160*y1)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 180, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + ((-376320) + x2 + (3136*y0) + (188160*y1)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp0 >= tmp16
    tmp23 = tl.full([1, 1], 240, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr3 + ((-564480) + x2 + (3136*y0) + (188160*y1)), tmp22 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tl.where(tmp18, tmp21, tmp27)
    tmp29 = tl.where(tmp11, tmp14, tmp28)
    tmp30 = tl.where(tmp4, tmp7, tmp29)
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr0 + (x2 + (3136*y3)), tmp32, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fk/cfk6wv2okrm3l2tf4k3njmw26zq33h5cmy7is6xsskm5zoxa7sv6.py
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
    size_hints=[1024, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_138', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 960
    rnumel = 6272
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
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (752640*(r2 // 3136)) + (1505280*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sy/csyhwyld2zvr4hfuwj7dxr3rzb4ljoggcggismtcu36sedh5hwhh.py
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
    size_hints=[256, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_139', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/v5/cv5phfwyafdcvoxitnfnwytpcnfkebgejlhzwod6t7e362v62c64.py
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
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_140', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 47040
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x1) + (752640*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (240*r2) + (30720*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t7/ct754yjpb7gbdjjeurzbyi43rdnmkqof6rlk32t4uvzhhlkfpjmz.py
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
    size_hints=[256, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_141', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ww/cwwapfwisv32sei2p7sgiyeafxoeoklus2fq6me2fo34b3k466x5.py
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
    size_hints=[2048, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_142', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 240
    y1 = (yindex // 240)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (240*x2) + (752640*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (3136*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4i/c4isjli6it77rsocvrwokp2zadtuaeh7faspqg3uxyfevan3nqod.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_143 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_143', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 40
    x1 = (xindex // 40)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (125440*(r2 // 3136)) + (250880*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kr/ckrmeh4qshr2cawvebk3cjr7y47xdpvvbvr6dtzamii6f4ktmckh.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_144 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_144', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 40
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (40*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i7/ci77fha6n7t7ruxvyqxc5dc5makkiksmkyflz3cvepwqjnjirjsf.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_145 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_145', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7840
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x1) + (125440*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (40*r2) + (5120*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/do/cdobba5fns2ycqbokpzu7outqlv45357zayb7b2dhlgbazmimw4e.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_146 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_146', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 40
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oy/coyyevmtqlzjhknm5trlkxizqm4gwww2pztpldtgylhec4awzng7.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_147 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_147', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 320
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 40
    y1 = (yindex // 40)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (40*x2) + (125440*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (3136*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nt/cntoykvelw7orejx3ajzse5uodtb5ugm3hti2czahqslgqh2423a.py
# Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_cat_native_batch_norm_backward_threshold_backward_148 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_batch_norm_backward_threshold_backward_148', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 480
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 120
    x1 = (xindex // 120)
    x3 = xindex
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (120*r2) + (752640*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = x0
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tmp1 >= tmp2
        tmp4 = tl.full([1, 1], 60, tl.int64)
        tmp5 = tmp1 < tmp4
        tmp6 = tl.load(in_ptr1 + ((3136*x3) + (188160*(r2 // 3136)) + (r2 % 3136)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = tmp1 >= tmp4
        tmp10 = tl.full([1, 1], 120, tl.int64)
        tmp11 = tmp1 < tmp10
        tmp12 = tl.load(in_ptr2 + ((-188160) + (3136*x3) + (188160*(r2 // 3136)) + (r2 % 3136)), rmask & tmp9 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
        tmp14 = tl.where(tmp9, tmp12, tmp13)
        tmp15 = tl.where(tmp5, tmp8, tmp14)
        tmp16 = 0.0
        tmp17 = tl.where(tmp0, tmp16, tmp15)
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/de/cdezfihpylyz4xqfilgf7sutguvwdfxfwkg6wdnpcxbgjy6tl46o.py
# Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_cat_native_batch_norm_backward_threshold_backward_149 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_batch_norm_backward_threshold_backward_149', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 120
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (120*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lz/clziinfb6z7jybbaeyfumnrez5sz4x5xigj7yvyj4vgvprcqxdvt.py
# Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_cat_native_batch_norm_backward_threshold_backward_150 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_batch_norm_backward_threshold_backward_150', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 23520
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp19 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (120*r2) + (15360*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp18 = tl.load(in_ptr3 + (x1 + (120*r2) + (15360*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = x1
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tmp1 >= tmp2
        tmp4 = tl.full([1, 1], 60, tl.int64)
        tmp5 = tmp1 < tmp4
        tmp6 = tl.load(in_ptr1 + ((3136*x1) + (188160*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = tmp1 >= tmp4
        tmp10 = tl.full([1, 1], 120, tl.int64)
        tmp11 = tmp1 < tmp10
        tmp12 = tl.load(in_ptr2 + ((-188160) + (3136*x1) + (188160*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & tmp9 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
        tmp14 = tl.where(tmp9, tmp12, tmp13)
        tmp15 = tl.where(tmp5, tmp8, tmp14)
        tmp16 = 0.0
        tmp17 = tl.where(tmp0, tmp16, tmp15)
        tmp20 = tmp18 - tmp19
        tmp21 = tmp17 * tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vx/cvxbaxvvktznz4rijhfsov2ubvflc26mk2fgyx6tvrqghg3zcvek.py
# Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_cat_native_batch_norm_backward_threshold_backward_151 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_batch_norm_backward_threshold_backward_151', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 120
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mt/cmtiwxtbtozx3snj75ab2nccaapkerkohcixfg65euel2s3feovd.py
# Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_cat_native_batch_norm_backward_threshold_backward_152 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_native_batch_norm_backward_threshold_backward_152', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 120
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3136
    y1 = (yindex // 3136)
    tmp0 = tl.load(in_ptr0 + (x2 + (120*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp18 = tl.load(in_ptr3 + (x2 + (120*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = x2
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 60, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (y0 + (3136*x2) + (188160*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1, 1], 120, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tl.load(in_ptr2 + ((-188160) + y0 + (3136*x2) + (188160*y1)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.where(tmp5, tmp8, tmp14)
    tmp16 = 0.0
    tmp17 = tl.where(tmp0, tmp16, tmp15)
    tmp20 = tmp18 - tmp19
    tmp22 = 3.985969387755102e-05
    tmp23 = tmp21 * tmp22
    tmp25 = tmp24 * tmp24
    tmp26 = tmp23 * tmp25
    tmp27 = tmp20 * tmp26
    tmp28 = tmp17 - tmp27
    tmp30 = tmp29 * tmp22
    tmp31 = tmp28 - tmp30
    tmp33 = tmp24 * tmp32
    tmp34 = tmp31 * tmp33
    tl.store(out_ptr0 + (x2 + (120*y3)), tmp34, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lb/clbscd75xhmv2z6x4rvl6sejh562lu7y4ws5omlmxixhshqtqcdl.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_153', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 23520
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (120*r2) + (15360*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x1) + (376320*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n7/cn7gu6k5sfshcpe2toz2zx6ipmecnk5cgfv3cu55z5h6ddcblv2x.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_154 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_154', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 120
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fi/cfijf4tj6ax5ent5ora7esnummgtjyk44a6mr7rg3iioqysvidgv.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_155 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_155', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 23520
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 120
    x1 = (xindex // 120)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (120*r2) + (15360*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (376320*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (120*r2) + (15360*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/on/con7p4gl6kbxqeakn3gj7mhphsfma2gkkoln5ibjzkhdscml3hq5.py
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
    size_hints=[128, 256],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_156', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 120
    rnumel = 196
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
        tmp0 = tl.load(in_ptr0 + (x0 + (120*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7h/c7hdurnoxnn44pydz4e6jjv554msd6k4tikhfb4b4pxdqyv3ovca.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_157 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_157', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 120
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3136
    y1 = (yindex // 3136)
    tmp0 = tl.load(in_ptr0 + (x2 + (120*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (3136*x2) + (376320*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (120*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (120*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ju/cjuwmpp6noisawq27nwkmxbbawho4bqzpob53kt4jliakht3d6fd.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]

triton_red_fused_add_cat_native_batch_norm_backward_158 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_batch_norm_backward_158', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 40
    x1 = (xindex // 40)
    x3 = xindex
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp21 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp25 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (125440*(r2 // 3136)) + (250880*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr3 + (x0 + (40*r2) + (250880*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = x0
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tmp1 >= tmp2
        tmp4 = tl.full([1, 1], 20, tl.int64)
        tmp5 = tmp1 < tmp4
        tmp6 = tl.load(in_ptr1 + ((3136*x3) + (62720*(r2 // 3136)) + (r2 % 3136)), rmask & tmp5 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = tmp1 >= tmp4
        tmp10 = tl.full([1, 1], 40, tl.int64)
        tmp11 = tmp1 < tmp10
        tmp12 = tl.load(in_ptr2 + ((-62720) + (3136*x3) + (62720*(r2 // 3136)) + (r2 % 3136)), rmask & tmp9 & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
        tmp14 = tl.where(tmp9, tmp12, tmp13)
        tmp15 = tl.where(tmp5, tmp8, tmp14)
        tmp16 = tmp0 + tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
        tmp22 = tmp20 - tmp21
        tmp23 = tmp16 * tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(rmask & xmask, tmp26, _tmp25)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, xmask)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2x/c2x45azn4vno6mm6hmbyahnwk6bw5vjhebscwt7jniyfgd6iueud.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]

triton_per_fused_add_cat_native_batch_norm_backward_159 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_batch_norm_backward_159', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 40
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (40*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x3/cx3sks46kgl2pkvfoxkrnrr7hx46mqdz4gs7b2fiicvpuf5jkhmh.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]

triton_poi_fused_add_cat_native_batch_norm_backward_160 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_native_batch_norm_backward_160', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 320
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 40
    y1 = (yindex // 40)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr2 + (y0 + (40*x2) + (125440*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = y0
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 20, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr0 + (x2 + (3136*y0) + (62720*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1, 1], 40, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tl.load(in_ptr1 + ((-62720) + x2 + (3136*y0) + (62720*y1)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.where(tmp5, tmp8, tmp14)
    tmp16 = tmp0 + tmp15
    tmp19 = tmp17 - tmp18
    tmp21 = 3.985969387755102e-05
    tmp22 = tmp20 * tmp21
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp19 * tmp25
    tmp27 = tmp16 - tmp26
    tmp29 = tmp28 * tmp21
    tmp30 = tmp27 - tmp29
    tmp32 = tmp23 * tmp31
    tmp33 = tmp30 * tmp32
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (3136*y3)), tmp33, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/de/cdeh5rhyz22xynthcpyoihlvi4hof7us65e2vu66hin6dz363c66.py
# Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_cat_native_batch_norm_backward_threshold_backward_161 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_batch_norm_backward_threshold_backward_161', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 192
    x1 = (xindex // 192)
    x3 = xindex
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r2) + (1204224*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = x0
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tmp1 >= tmp2
        tmp4 = tl.full([1, 1], 96, tl.int64)
        tmp5 = tmp1 < tmp4
        tmp6 = tl.load(in_ptr1 + ((3136*x3) + (301056*(r2 // 3136)) + (r2 % 3136)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = tmp1 >= tmp4
        tmp10 = tl.full([1, 1], 192, tl.int64)
        tmp11 = tmp1 < tmp10
        tmp12 = tl.load(in_ptr2 + ((-301056) + (3136*x3) + (301056*(r2 // 3136)) + (r2 % 3136)), rmask & tmp9 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
        tmp14 = tl.where(tmp9, tmp12, tmp13)
        tmp15 = tl.where(tmp5, tmp8, tmp14)
        tmp16 = 0.0
        tmp17 = tl.where(tmp0, tmp16, tmp15)
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4g/c4gqekqu2f73fgpyauoobuw2pesr5k4cu3v7gvbozpi35chmpilw.py
# Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_cat_native_batch_norm_backward_threshold_backward_162 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_batch_norm_backward_threshold_backward_162', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/oe/coewcq5ah6wuf3utlmdihtikadf3tb4vcxye6pxbmqplmvqzazvv.py
# Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_cat_native_batch_norm_backward_threshold_backward_163 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_batch_norm_backward_threshold_backward_163', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 37632
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp19 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (192*r2) + (24576*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp18 = tl.load(in_ptr3 + (x1 + (192*r2) + (24576*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = x1
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tmp1 >= tmp2
        tmp4 = tl.full([1, 1], 96, tl.int64)
        tmp5 = tmp1 < tmp4
        tmp6 = tl.load(in_ptr1 + ((3136*x1) + (301056*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = tmp1 >= tmp4
        tmp10 = tl.full([1, 1], 192, tl.int64)
        tmp11 = tmp1 < tmp10
        tmp12 = tl.load(in_ptr2 + ((-301056) + (3136*x1) + (301056*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & tmp9 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
        tmp14 = tl.where(tmp9, tmp12, tmp13)
        tmp15 = tl.where(tmp5, tmp8, tmp14)
        tmp16 = 0.0
        tmp17 = tl.where(tmp0, tmp16, tmp15)
        tmp20 = tmp18 - tmp19
        tmp21 = tmp17 * tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gc/cgc3jcl6wsb6slu4saurqegmzuhxhgajjgu2lxcvqbum5mjnoa5g.py
# Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_cat_native_batch_norm_backward_threshold_backward_164 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_batch_norm_backward_threshold_backward_164', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zx/czxnifp7rqryb5etp7hydnej6i4rgpru5uqbkyc4fiqc6l6eelq5.py
# Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_cat_native_batch_norm_backward_threshold_backward_165 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_native_batch_norm_backward_threshold_backward_165', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 3136
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
    tmp0 = tl.load(in_ptr0 + (y0 + (192*x2) + (602112*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp18 = tl.load(in_ptr3 + (y0 + (192*x2) + (602112*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = y0
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 96, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x2 + (3136*y0) + (301056*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1, 1], 192, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tl.load(in_ptr2 + ((-301056) + x2 + (3136*y0) + (301056*y1)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.where(tmp5, tmp8, tmp14)
    tmp16 = 0.0
    tmp17 = tl.where(tmp0, tmp16, tmp15)
    tmp20 = tmp18 - tmp19
    tmp22 = 3.985969387755102e-05
    tmp23 = tmp21 * tmp22
    tmp25 = tmp24 * tmp24
    tmp26 = tmp23 * tmp25
    tmp27 = tmp20 * tmp26
    tmp28 = tmp17 - tmp27
    tmp30 = tmp29 * tmp22
    tmp31 = tmp28 - tmp30
    tmp33 = tmp24 * tmp32
    tmp34 = tmp31 * tmp33
    tl.store(out_ptr0 + (x2 + (3136*y3)), tmp34, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c3/cc3ctoijvkg5rqx2cvvjmqq2snszo3w5xekhdndv7eqhh4w3zf2g.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_166 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_166', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (401408 + x2 + (3136*y0) + (602112*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (200704*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5x/c5xae2iqien4vt4jkahsjkqb7zxm4ewobagrggxxn4ktuurixhl5.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_167 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_167', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (200704 + x2 + (3136*y0) + (602112*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (200704*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ek/cek3toychvlq632c65ng7qtdrxqvnvtxoqiem7isgdbq7cd6zgka.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_168 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_168', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y0) + (602112*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (200704*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ar/cartrsrkt4dixmqi2ypk77b6u7n2yrjg7htrzcem47l4okbgkkq2.py
# Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_cat_native_batch_norm_backward_threshold_backward_169 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_batch_norm_backward_threshold_backward_169', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 192
    x1 = (xindex // 192)
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp30 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp34 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r2) + (4816896*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp29 = tl.load(in_ptr4 + (x0 + (192*r2) + (4816896*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = x0
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tmp1 >= tmp2
        tmp4 = tl.full([1, 1], 64, tl.int64)
        tmp5 = tmp1 < tmp4
        tmp6 = tl.load(in_ptr1 + ((12544*x0) + (802816*(r2 // 12544)) + (1605632*x1) + (r2 % 12544)), rmask & tmp5 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = tmp1 >= tmp4
        tmp10 = tl.full([1, 1], 128, tl.int64)
        tmp11 = tmp1 < tmp10
        tmp12 = tmp9 & tmp11
        tmp13 = tl.load(in_ptr2 + ((-802816) + (12544*x0) + (802816*(r2 // 12544)) + (1605632*x1) + (r2 % 12544)), rmask & tmp12 & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
        tmp15 = tl.where(tmp12, tmp13, tmp14)
        tmp16 = tmp1 >= tmp10
        tmp17 = tl.full([1, 1], 192, tl.int64)
        tmp18 = tmp1 < tmp17
        tmp19 = tl.load(in_ptr3 + ((-1605632) + (12544*x0) + (802816*(r2 // 12544)) + (1605632*x1) + (r2 % 12544)), rmask & tmp16 & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
        tmp21 = tl.where(tmp16, tmp19, tmp20)
        tmp22 = tl.where(tmp12, tmp15, tmp21)
        tmp23 = tl.where(tmp5, tmp8, tmp22)
        tmp24 = 0.0
        tmp25 = tl.where(tmp0, tmp24, tmp23)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(rmask & xmask, tmp28, _tmp27)
        tmp31 = tmp29 - tmp30
        tmp32 = tmp25 * tmp31
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
        tmp35 = _tmp34 + tmp33
        _tmp34 = tl.where(rmask & xmask, tmp35, _tmp34)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp27, xmask)
    tmp34 = tl.sum(_tmp34, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp34, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/37/c37dgwyrminxjwududbtxyzzao47em2hqcwghuzxvmpslls4u7zc.py
# Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_cat_native_batch_norm_backward_threshold_backward_170 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_batch_norm_backward_threshold_backward_170', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/qy/cqy5gj2wf2d3ym4ohoogl46sdklft4dhgzqtbur2rhxn7tfshcvr.py
# Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_cat_native_batch_norm_backward_threshold_backward_171 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_native_batch_norm_backward_threshold_backward_171', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 12544
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
    tmp0 = tl.load(in_ptr0 + (y0 + (192*x2) + (2408448*y1)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp26 = tl.load(in_ptr4 + (y0 + (192*x2) + (2408448*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = y0
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 64, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.load(in_ptr1 + (x2 + (12544*y0) + (802816*y1)), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp1 >= tmp4
    tmp10 = tl.full([1, 1], 128, tl.int64)
    tmp11 = tmp1 < tmp10
    tmp12 = tmp9 & tmp11
    tmp13 = tl.load(in_ptr2 + ((-802816) + x2 + (12544*y0) + (802816*y1)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tmp1 >= tmp10
    tmp17 = tl.full([1, 1], 192, tl.int64)
    tmp18 = tmp1 < tmp17
    tmp19 = tl.load(in_ptr3 + ((-1605632) + x2 + (12544*y0) + (802816*y1)), tmp16 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp16, tmp19, tmp20)
    tmp22 = tl.where(tmp12, tmp15, tmp21)
    tmp23 = tl.where(tmp5, tmp8, tmp22)
    tmp24 = 0.0
    tmp25 = tl.where(tmp0, tmp24, tmp23)
    tmp28 = tmp26 - tmp27
    tmp30 = 9.964923469387754e-06
    tmp31 = tmp29 * tmp30
    tmp33 = tmp32 * tmp32
    tmp34 = tmp31 * tmp33
    tmp35 = tmp28 * tmp34
    tmp36 = tmp25 - tmp35
    tmp38 = tmp37 * tmp30
    tmp39 = tmp36 - tmp38
    tl.store(out_ptr0 + (x2 + (12544*y3)), tmp39, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wq/cwqpuqokf4ixcn7jxd6vkiq4o52jbm3clgehkmjry3m77y5bqmne.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_172 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_172', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_ptr0 + (1204224 + x2 + (12544*y0) + (2408448*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (96 + y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (96 + y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (y0 + (96*x2) + (1204224*y1)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ha/cha46vjnvblqi3egq7l2xyf5i3qjepzix5mkaumsg2jd32e6pzfg.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_173 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_173', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y0) + (2408448*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (y0 + (96*x2) + (1204224*y1)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rj/crjuup6iupf7y7dlwlmtvhocbihnfyolecsojyyqvlrvgxl3m55e.py
# Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward]

triton_red_fused_cat_native_batch_norm_backward_174 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_batch_norm_backward_174', 'mutated_arg_names': []}
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
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp32 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7720*x0)
        tmp1 = tl.full([1, 1], 100352, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.broadcast_to(x1, [XBLOCK, RBLOCK])
        tmp4 = tl.full([1, 1], 0, tl.int64)
        tmp5 = tmp3 >= tmp4
        tmp6 = tl.full([1, 1], 16, tl.int64)
        tmp7 = tmp3 < tmp6
        tmp8 = tmp7 & tmp2
        tmp9 = tl.load(in_ptr0 + ((12544*x1) + (200704*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
        tmp11 = tl.where(tmp8, tmp9, tmp10)
        tmp12 = tmp3 >= tmp6
        tmp13 = tl.full([1, 1], 32, tl.int64)
        tmp14 = tmp3 < tmp13
        tmp15 = tmp12 & tmp2
        tmp16 = tl.load(in_ptr1 + ((-200704) + (12544*x1) + (200704*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp15 & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
        tmp18 = tl.where(tmp15, tmp16, tmp17)
        tmp19 = tl.where(tmp7, tmp11, tmp18)
        tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
        tmp21 = tl.where(tmp2, tmp19, tmp20)
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
        tmp25 = tl.load(in_ptr2 + (x1 + (32*((r2 + (7720*x0)) % 100352))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp26 = tl.load(in_ptr3 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp27 = tmp25 - tmp26
        tmp28 = tmp19 * tmp27
        tmp29 = tl.full(tmp28.shape, 0, tmp28.dtype)
        tmp30 = tl.where(tmp2, tmp28, tmp29)
        tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
        tmp33 = _tmp32 + tmp31
        _tmp32 = tl.where(rmask & xmask, tmp33, _tmp32)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp23, xmask)
    tmp32 = tl.sum(_tmp32, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bj/cbjpyscllhwvoagjemvybscyexcah66dty2bcg4ow6npdzsjd3si.py
# Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward]

triton_per_fused_cat_native_batch_norm_backward_175 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_batch_norm_backward_175', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/pz/cpzp4wcbdrr235zp35k54lxtjq7ixt7ajwfvikn7bf3fwtylui3z.py
# Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward]

triton_per_fused_cat_native_batch_norm_backward_176 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_batch_norm_backward_176', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/rh/crht6m6r5aeiysfkqdfxgffifp6lhd5jt6wqr3ficifh45r5q4qs.py
# Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_cat_convolution_backward_native_batch_norm_backward_177 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_backward_native_batch_norm_backward_177', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 32
    x2 = xindex
    y1 = (yindex // 32)
    y3 = yindex
    tmp15 = tl.load(in_ptr2 + (y0 + (32*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (12544*y0) + (200704*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 32, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-200704) + x2 + (12544*y0) + (200704*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp17 = tmp15 - tmp16
    tmp19 = 9.964923469387754e-06
    tmp20 = tmp18 * tmp19
    tmp22 = tmp21 * tmp21
    tmp23 = tmp20 * tmp22
    tmp24 = tmp17 * tmp23
    tmp25 = tmp14 - tmp24
    tmp27 = tmp26 * tmp19
    tmp28 = tmp25 - tmp27
    tmp30 = tmp21 * tmp29
    tmp31 = tmp28 * tmp30
    tl.store(out_ptr0 + (x2 + (12544*y3)), tmp31, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5r/c5rgdwxbximrc3r2irycw7jrqchgwnu5oimy3adpkxgcwaacvk4w.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_178 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_178', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (32*r2) + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((12544*x1) + (401408*((r2 + (128*x0)) // 12544)) + ((r2 + (128*x0)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pd/cpdqpbsuvoxjmdkwjdswudlt57wxwnpbl6bd2i7ewwtkxqeofijb.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_179 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_179', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 32
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (784*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mk/cmko4e4gg3v2chkzxdhyf3liptrtzked52rjimdvv4nzjxdejeu2.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_180 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_180', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r2) + (4096*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((12544*x0) + (401408*((r2 + (128*x1)) // 12544)) + ((r2 + (128*x1)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (32*r2) + (4096*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pq/cpqdam5j6ksojyvypst74x3zxd2o3k6jgistgzogpjctqk6okeyq.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_181 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32, 1024],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_181', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 784
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


# kernel path: /tmp/torchinductor_youkaichao/5r/c5rj42q4iryzolmjwocoppk4c6o64mbwbwudd3vw7o5byqarfrfh.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_182 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_182', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 100352
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 12544
    y1 = (yindex // 12544)
    tmp0 = tl.load(in_ptr0 + (x2 + (32*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (12544*x2) + (401408*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (32*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (32*y3)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7u/c7ujspxb375fcoj2wsqojzqnqgks3gsokcp2n7cgy224lwha3s67.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_cat_native_batch_norm_backward_threshold_backward_183 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_batch_norm_backward_threshold_backward_183', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 416
    rnumel = 7720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp29 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp38 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7720*x0)
        tmp1 = tl.full([1, 1], 100352, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (32*((r2 + (7720*x0)) % 100352))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.broadcast_to(x1, [XBLOCK, RBLOCK])
        tmp7 = tl.full([1, 1], 0, tl.int64)
        tmp8 = tmp6 >= tmp7
        tmp9 = tl.full([1, 1], 16, tl.int64)
        tmp10 = tmp6 < tmp9
        tmp11 = tmp10 & tmp2
        tmp12 = tl.load(in_ptr1 + ((12544*x1) + (200704*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp11 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
        tmp14 = tl.where(tmp11, tmp12, tmp13)
        tmp15 = tmp6 >= tmp9
        tmp16 = tl.full([1, 1], 32, tl.int64)
        tmp17 = tmp6 < tmp16
        tmp18 = tmp15 & tmp2
        tmp19 = tl.load(in_ptr2 + ((-200704) + (12544*x1) + (200704*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp18 & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
        tmp21 = tl.where(tmp18, tmp19, tmp20)
        tmp22 = tl.where(tmp10, tmp14, tmp21)
        tmp23 = tl.load(in_ptr3 + ((12544*x1) + (401408*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tmp22 + tmp23
        tmp25 = tl.where(tmp5, tmp4, tmp24)
        tmp26 = tl.full(tmp25.shape, 0, tmp25.dtype)
        tmp27 = tl.where(tmp2, tmp25, tmp26)
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
        tmp30 = _tmp29 + tmp28
        _tmp29 = tl.where(rmask & xmask, tmp30, _tmp29)
        tmp31 = tl.load(in_ptr4 + (x1 + (32*((r2 + (7720*x0)) % 100352))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp32 = tl.load(in_ptr5 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp33 = tmp31 - tmp32
        tmp34 = tmp25 * tmp33
        tmp35 = tl.full(tmp34.shape, 0, tmp34.dtype)
        tmp36 = tl.where(tmp2, tmp34, tmp35)
        tmp37 = tl.broadcast_to(tmp36, [XBLOCK, RBLOCK])
        tmp39 = _tmp38 + tmp37
        _tmp38 = tl.where(rmask & xmask, tmp39, _tmp38)
    tmp29 = tl.sum(_tmp29, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp29, xmask)
    tmp38 = tl.sum(_tmp38, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp38, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gc/cgcgscwuzwzbgunieefc6hmd4j76c7qplru267kwqeoy4v5g767s.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_cat_convolution_backward_native_batch_norm_backward_threshold_backward_184 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_convolution_backward_native_batch_norm_backward_threshold_backward_184', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 12544
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
    tmp0 = tl.load(in_ptr0 + (y0 + (32*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_out_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr3 + (y0 + (32*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp3 = y0
    tmp4 = tl.full([1, 1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.full([1, 1], 16, tl.int64)
    tmp7 = tmp3 < tmp6
    tmp8 = tl.load(in_ptr1 + (x2 + (12544*y0) + (200704*y1)), tmp7 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 >= tmp6
    tmp12 = tl.full([1, 1], 32, tl.int64)
    tmp13 = tmp3 < tmp12
    tmp14 = tl.load(in_ptr2 + ((-200704) + x2 + (12544*y0) + (200704*y1)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp11, tmp14, tmp15)
    tmp17 = tl.where(tmp7, tmp10, tmp16)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.where(tmp2, tmp1, tmp19)
    tmp23 = tmp21 - tmp22
    tmp25 = 9.964923469387754e-06
    tmp26 = tmp24 * tmp25
    tmp28 = tmp27 * tmp27
    tmp29 = tmp26 * tmp28
    tmp30 = tmp23 * tmp29
    tmp31 = tmp20 - tmp30
    tmp33 = tmp32 * tmp25
    tmp34 = tmp31 - tmp33
    tmp36 = tmp27 * tmp35
    tmp37 = tmp34 * tmp36
    tl.store(out_ptr0 + (y0 + (32*x2) + (401408*y1)), tmp37, xmask & ymask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_139, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_148, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_158, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_168, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_178, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_189, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_201, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_213, primals_215, primals_216, primals_217, primals_218, primals_219, primals_221, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_232, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_244, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_256, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_267, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_277, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_288, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_299, primals_301, primals_302, primals_303, primals_480, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, getitem_6, getitem_7, cat, squeeze_10, getitem_13, getitem_17, getitem_21, cat_1, squeeze_13, getitem_26, getitem_29, cat_2, squeeze_16, getitem_32, getitem_33, cat_3, squeeze_19, relu_4, convolution_12, squeeze_22, getitem_40, getitem_43, cat_4, squeeze_25, add_46, convolution_15, squeeze_28, getitem_52, getitem_57, getitem_62, getitem_67, cat_5, squeeze_31, add_56, mean, convolution_20, mul_79, convolution_21, mul_80, convolution_22, squeeze_34, getitem_72, getitem_73, cat_6, squeeze_37, getitem_78, getitem_81, cat_7, squeeze_40, add_71, mean_1, convolution_27, mul_104, convolution_28, getitem_84, getitem_85, cat_8, squeeze_43, getitem_88, getitem_89, cat_9, squeeze_46, getitem_94, getitem_97, cat_10, squeeze_49, add_87, mean_2, convolution_35, mul_129, convolution_36, getitem_100, getitem_101, cat_11, squeeze_52, getitem_104, getitem_105, cat_12, squeeze_55, getitem_110, getitem_113, cat_13, squeeze_58, add_103, mean_3, convolution_43, mul_154, convolution_44, getitem_116, getitem_117, cat_14, squeeze_61, add_109, convolution_47, squeeze_64, getitem_125, getitem_129, getitem_133, cat_15, squeeze_67, add_119, mean_4, convolution_51, mul_179, convolution_52, mul_180, convolution_53, squeeze_70, getitem_138, getitem_139, cat_16, squeeze_73, getitem_146, getitem_151, getitem_156, getitem_161, cat_17, squeeze_76, add_134, mean_5, convolution_60, mul_204, convolution_61, getitem_164, getitem_165, cat_18, squeeze_79, getitem_168, getitem_169, cat_19, squeeze_82, getitem_176, getitem_181, getitem_186, getitem_191, cat_20, squeeze_85, add_150, mean_6, convolution_70, mul_229, convolution_71, getitem_194, getitem_195, cat_21, squeeze_88, getitem_198, getitem_199, cat_22, squeeze_91, getitem_206, getitem_211, getitem_216, getitem_221, cat_23, squeeze_94, add_166, mean_7, convolution_80, mul_254, convolution_81, getitem_224, getitem_225, cat_24, squeeze_97, add_172, convolution_84, squeeze_100, mul_270, convolution_85, squeeze_103, add_182, mean_8, convolution_86, mul_279, convolution_87, mul_280, convolution_88, squeeze_106, getitem_234, getitem_235, cat_25, squeeze_109, getitem_242, getitem_247, getitem_252, getitem_257, cat_26, squeeze_112, add_197, mean_9, convolution_95, mul_304, convolution_96, getitem_260, getitem_261, cat_27, squeeze_115, getitem_264, getitem_265, cat_28, squeeze_118, getitem_272, getitem_277, getitem_282, getitem_287, cat_29, squeeze_121, add_213, mean_10, convolution_105, mul_329, convolution_106, getitem_290, getitem_291, cat_30, squeeze_124, getitem_294, getitem_295, cat_31, squeeze_127, getitem_302, getitem_307, getitem_312, getitem_317, cat_32, squeeze_130, add_229, mean_11, convolution_115, mul_354, convolution_116, getitem_320, getitem_321, cat_33, squeeze_133, add_235, convolution_119, squeeze_136, getitem_330, getitem_335, getitem_340, getitem_345, cat_34, squeeze_139, add_245, mean_12, convolution_124, mul_379, convolution_125, mul_380, convolution_126, squeeze_142, add_250, convolution_127, squeeze_145, getitem_356, getitem_361, getitem_366, getitem_371, cat_35, squeeze_148, add_260, mean_13, convolution_132, mul_404, convolution_133, getitem_374, getitem_375, cat_36, squeeze_151, add_266, convolution_136, squeeze_154, getitem_384, getitem_389, getitem_394, getitem_399, cat_37, squeeze_157, add_276, mean_14, convolution_141, mul_429, convolution_142, getitem_402, getitem_403, cat_38, squeeze_160, add_282, convolution_145, squeeze_163, getitem_412, getitem_417, getitem_422, getitem_427, cat_39, squeeze_166, add_292, mean_15, convolution_150, mul_454, convolution_151, getitem_430, getitem_431, cat_40, squeeze_169, add_298, convolution_154, squeeze_172, view, permute_1, le, unsqueeze_234, unsqueeze_246, unsqueeze_258, mul_508, unsqueeze_270, unsqueeze_282, unsqueeze_294, mul_548, unsqueeze_306, unsqueeze_318, unsqueeze_330, mul_588, unsqueeze_342, unsqueeze_354, unsqueeze_366, mul_628, unsqueeze_378, unsqueeze_390, unsqueeze_402, mul_668, unsqueeze_414, unsqueeze_426, unsqueeze_438, mul_708, unsqueeze_450, unsqueeze_462, unsqueeze_474, mul_748, unsqueeze_486, unsqueeze_498, unsqueeze_510, mul_788, unsqueeze_522, unsqueeze_534, unsqueeze_546, mul_828, unsqueeze_558, unsqueeze_570, unsqueeze_582, mul_868, unsqueeze_594, unsqueeze_606, unsqueeze_618, mul_908, unsqueeze_630, unsqueeze_642, unsqueeze_654, mul_948, unsqueeze_666, unsqueeze_678, unsqueeze_690, mul_988, unsqueeze_702, unsqueeze_714, unsqueeze_726, mul_1028, unsqueeze_738, unsqueeze_750, unsqueeze_762, mul_1068, unsqueeze_774, unsqueeze_786, unsqueeze_798, mul_1108, unsqueeze_810, unsqueeze_822, le_1, unsqueeze_834, unsqueeze_846, unsqueeze_858, le_3, unsqueeze_870, le_4, unsqueeze_882, unsqueeze_894, unsqueeze_906, unsqueeze_918, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, ), (1, ))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_7, (192, ), (1, ))
    assert_size_stride(primals_9, (192, ), (1, ))
    assert_size_stride(primals_11, (40, ), (1, ))
    assert_size_stride(primals_13, (120, ), (1, ))
    assert_size_stride(primals_15, (120, ), (1, ))
    assert_size_stride(primals_17, (40, ), (1, ))
    assert_size_stride(primals_19, (240, ), (1, ))
    assert_size_stride(primals_21, (240, ), (1, ))
    assert_size_stride(primals_23, (56, ), (1, ))
    assert_size_stride(primals_25, (336, ), (1, ))
    assert_size_stride(primals_27, (336, ), (1, ))
    assert_size_stride(primals_29, (56, ), (1, ))
    assert_size_stride(primals_31, (336, ), (1, ))
    assert_size_stride(primals_33, (336, ), (1, ))
    assert_size_stride(primals_35, (56, ), (1, ))
    assert_size_stride(primals_37, (336, ), (1, ))
    assert_size_stride(primals_39, (336, ), (1, ))
    assert_size_stride(primals_41, (56, ), (1, ))
    assert_size_stride(primals_43, (336, ), (1, ))
    assert_size_stride(primals_45, (336, ), (1, ))
    assert_size_stride(primals_47, (104, ), (1, ))
    assert_size_stride(primals_49, (624, ), (1, ))
    assert_size_stride(primals_51, (624, ), (1, ))
    assert_size_stride(primals_53, (104, ), (1, ))
    assert_size_stride(primals_55, (624, ), (1, ))
    assert_size_stride(primals_57, (624, ), (1, ))
    assert_size_stride(primals_59, (104, ), (1, ))
    assert_size_stride(primals_61, (624, ), (1, ))
    assert_size_stride(primals_63, (624, ), (1, ))
    assert_size_stride(primals_65, (104, ), (1, ))
    assert_size_stride(primals_67, (624, ), (1, ))
    assert_size_stride(primals_69, (624, ), (1, ))
    assert_size_stride(primals_71, (160, ), (1, ))
    assert_size_stride(primals_73, (480, ), (1, ))
    assert_size_stride(primals_75, (480, ), (1, ))
    assert_size_stride(primals_77, (160, ), (1, ))
    assert_size_stride(primals_79, (480, ), (1, ))
    assert_size_stride(primals_81, (480, ), (1, ))
    assert_size_stride(primals_83, (160, ), (1, ))
    assert_size_stride(primals_85, (480, ), (1, ))
    assert_size_stride(primals_87, (480, ), (1, ))
    assert_size_stride(primals_89, (160, ), (1, ))
    assert_size_stride(primals_91, (960, ), (1, ))
    assert_size_stride(primals_93, (960, ), (1, ))
    assert_size_stride(primals_95, (264, ), (1, ))
    assert_size_stride(primals_97, (1584, ), (1, ))
    assert_size_stride(primals_99, (1584, ), (1, ))
    assert_size_stride(primals_101, (264, ), (1, ))
    assert_size_stride(primals_103, (1584, ), (1, ))
    assert_size_stride(primals_105, (1584, ), (1, ))
    assert_size_stride(primals_107, (264, ), (1, ))
    assert_size_stride(primals_109, (1584, ), (1, ))
    assert_size_stride(primals_111, (1584, ), (1, ))
    assert_size_stride(primals_113, (264, ), (1, ))
    assert_size_stride(primals_115, (1536, ), (1, ))
    assert_size_stride(primals_117, (32, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_118, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_119, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_120, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_121, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_122, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_123, (64, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_124, (64, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_125, (20, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_126, (20, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_127, (60, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_128, (60, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_129, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_130, (20, 60, 1, 1), (60, 1, 1, 1))
    assert_size_stride(primals_131, (20, 60, 1, 1), (60, 1, 1, 1))
    assert_size_stride(primals_132, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_133, (60, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_134, (60, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_135, (60, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_136, (60, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_137, (20, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_139, (240, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_141, (56, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_142, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_143, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_144, (168, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_145, (168, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_146, (28, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_148, (336, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_150, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_151, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_152, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_153, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_154, (168, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_155, (168, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_156, (28, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_158, (336, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_160, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_161, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_162, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_163, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_164, (168, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_165, (168, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_166, (28, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_168, (336, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_170, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_171, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_172, (336, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_173, (112, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_174, (112, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_175, (112, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_176, (14, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_178, (336, 14, 1, 1), (14, 1, 1, 1))
    assert_size_stride(primals_180, (104, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_181, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_182, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_183, (156, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_184, (156, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_185, (156, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_186, (156, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_187, (26, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(primals_189, (624, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(primals_191, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_192, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_193, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_194, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_195, (156, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_196, (156, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_197, (156, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_198, (156, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_199, (26, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(primals_201, (624, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(primals_203, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_204, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_205, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_206, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_207, (156, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_208, (156, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_209, (156, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_210, (156, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_211, (26, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(primals_213, (624, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(primals_215, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_216, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_217, (624, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(primals_218, (624, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_219, (52, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(primals_221, (624, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_223, (160, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(primals_224, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_225, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_226, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_227, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_228, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_229, (120, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_230, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_232, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_234, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_235, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_236, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_237, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_238, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_239, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_240, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_241, (120, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_242, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_244, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_246, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_247, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_248, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_249, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_250, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_251, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_252, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_253, (120, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_254, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_256, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_258, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_259, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_260, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_261, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_262, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_263, (240, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_264, (240, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_265, (80, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_267, (960, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_269, (264, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_270, (1584, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(primals_271, (396, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_272, (396, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_273, (396, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_274, (396, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_275, (132, 1584, 1, 1), (1584, 1, 1, 1))
    assert_size_stride(primals_277, (1584, 132, 1, 1), (132, 1, 1, 1))
    assert_size_stride(primals_279, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(primals_280, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(primals_281, (1584, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(primals_282, (396, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_283, (396, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_284, (396, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_285, (396, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_286, (132, 1584, 1, 1), (1584, 1, 1, 1))
    assert_size_stride(primals_288, (1584, 132, 1, 1), (132, 1, 1, 1))
    assert_size_stride(primals_290, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(primals_291, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(primals_292, (1584, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(primals_293, (396, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_294, (396, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_295, (396, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_296, (396, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_297, (132, 1584, 1, 1), (1584, 1, 1, 1))
    assert_size_stride(primals_299, (1584, 132, 1, 1), (132, 1, 1, 1))
    assert_size_stride(primals_301, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(primals_302, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(primals_303, (1536, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(primals_480, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(squeeze_1, (32, ), (1, ))
    assert_size_stride(relu, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(convolution_1, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(squeeze_4, (32, ), (1, ))
    assert_size_stride(relu_1, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(convolution_2, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(squeeze_7, (32, ), (1, ))
    assert_size_stride(getitem_6, (8, 16, 112, 112), (401408, 12544, 112, 1))
    assert_size_stride(getitem_7, (8, 16, 112, 112), (401408, 12544, 112, 1))
    assert_size_stride(cat, (8, 192, 112, 112), (2408448, 1, 21504, 192))
    assert_size_stride(squeeze_10, (192, ), (1, ))
    assert_size_stride(getitem_13, (8, 64, 112, 112), (2408448, 1, 21504, 192))
    assert_size_stride(getitem_17, (8, 64, 112, 112), (2408448, 1, 21504, 192))
    assert_size_stride(getitem_21, (8, 64, 112, 112), (2408448, 1, 21504, 192))
    assert_size_stride(cat_1, (8, 192, 56, 56), (602112, 1, 10752, 192))
    assert_size_stride(squeeze_13, (192, ), (1, ))
    assert_size_stride(getitem_26, (8, 96, 56, 56), (602112, 1, 10752, 192))
    assert_size_stride(getitem_29, (8, 96, 56, 56), (602112, 1, 10752, 192))
    assert_size_stride(cat_2, (8, 40, 56, 56), (125440, 1, 2240, 40))
    assert_size_stride(squeeze_16, (40, ), (1, ))
    assert_size_stride(getitem_32, (8, 20, 56, 56), (125440, 1, 2240, 40))
    assert_size_stride(getitem_33, (8, 20, 56, 56), (125440, 1, 2240, 40))
    assert_size_stride(cat_3, (8, 120, 56, 56), (376320, 1, 6720, 120))
    assert_size_stride(squeeze_19, (120, ), (1, ))
    assert_size_stride(relu_4, (8, 120, 56, 56), (376320, 1, 6720, 120))
    assert_size_stride(convolution_12, (8, 120, 56, 56), (376320, 1, 6720, 120))
    assert_size_stride(squeeze_22, (120, ), (1, ))
    assert_size_stride(getitem_40, (8, 60, 56, 56), (376320, 1, 6720, 120))
    assert_size_stride(getitem_43, (8, 60, 56, 56), (376320, 1, 6720, 120))
    assert_size_stride(cat_4, (8, 40, 56, 56), (125440, 1, 2240, 40))
    assert_size_stride(squeeze_25, (40, ), (1, ))
    assert_size_stride(add_46, (8, 40, 56, 56), (125440, 1, 2240, 40))
    assert_size_stride(convolution_15, (8, 240, 56, 56), (752640, 1, 13440, 240))
    assert_size_stride(squeeze_28, (240, ), (1, ))
    assert_size_stride(getitem_52, (8, 60, 56, 56), (752640, 3136, 56, 1))
    assert_size_stride(getitem_57, (8, 60, 56, 56), (752640, 3136, 56, 1))
    assert_size_stride(getitem_62, (8, 60, 56, 56), (752640, 3136, 56, 1))
    assert_size_stride(getitem_67, (8, 60, 56, 56), (752640, 3136, 56, 1))
    assert_size_stride(cat_5, (8, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(squeeze_31, (240, ), (1, ))
    assert_size_stride(add_56, (8, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(mean, (8, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(convolution_20, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(mul_79, (8, 20, 1, 1), (20, 1, 20, 20))
    assert_size_stride(convolution_21, (8, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(mul_80, (8, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(convolution_22, (8, 56, 28, 28), (43904, 1, 1568, 56))
    assert_size_stride(squeeze_34, (56, ), (1, ))
    assert_size_stride(getitem_72, (8, 28, 28, 28), (43904, 1, 1568, 56))
    assert_size_stride(getitem_73, (8, 28, 28, 28), (43904, 1, 1568, 56))
    assert_size_stride(cat_6, (8, 336, 28, 28), (263424, 1, 9408, 336))
    assert_size_stride(squeeze_37, (336, ), (1, ))
    assert_size_stride(getitem_78, (8, 168, 28, 28), (263424, 784, 28, 1))
    assert_size_stride(getitem_81, (8, 168, 28, 28), (263424, 784, 28, 1))
    assert_size_stride(cat_7, (8, 336, 28, 28), (263424, 1, 9408, 336))
    assert_size_stride(squeeze_40, (336, ), (1, ))
    assert_size_stride(add_71, (8, 336, 28, 28), (263424, 1, 9408, 336))
    assert_size_stride(mean_1, (8, 336, 1, 1), (336, 1, 336, 336))
    assert_size_stride(convolution_27, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(mul_104, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(convolution_28, (8, 336, 1, 1), (336, 1, 336, 336))
    assert_size_stride(getitem_84, (8, 168, 28, 28), (263424, 784, 28, 1))
    assert_size_stride(getitem_85, (8, 168, 28, 28), (263424, 784, 28, 1))
    assert_size_stride(cat_8, (8, 56, 28, 28), (43904, 1, 1568, 56))
    assert_size_stride(squeeze_43, (56, ), (1, ))
    assert_size_stride(getitem_88, (8, 28, 28, 28), (43904, 1, 1568, 56))
    assert_size_stride(getitem_89, (8, 28, 28, 28), (43904, 1, 1568, 56))
    assert_size_stride(cat_9, (8, 336, 28, 28), (263424, 1, 9408, 336))
    assert_size_stride(squeeze_46, (336, ), (1, ))
    assert_size_stride(getitem_94, (8, 168, 28, 28), (263424, 784, 28, 1))
    assert_size_stride(getitem_97, (8, 168, 28, 28), (263424, 784, 28, 1))
    assert_size_stride(cat_10, (8, 336, 28, 28), (263424, 1, 9408, 336))
    assert_size_stride(squeeze_49, (336, ), (1, ))
    assert_size_stride(add_87, (8, 336, 28, 28), (263424, 1, 9408, 336))
    assert_size_stride(mean_2, (8, 336, 1, 1), (336, 1, 336, 336))
    assert_size_stride(convolution_35, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(mul_129, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(convolution_36, (8, 336, 1, 1), (336, 1, 336, 336))
    assert_size_stride(getitem_100, (8, 168, 28, 28), (263424, 784, 28, 1))
    assert_size_stride(getitem_101, (8, 168, 28, 28), (263424, 784, 28, 1))
    assert_size_stride(cat_11, (8, 56, 28, 28), (43904, 1, 1568, 56))
    assert_size_stride(squeeze_52, (56, ), (1, ))
    assert_size_stride(getitem_104, (8, 28, 28, 28), (43904, 1, 1568, 56))
    assert_size_stride(getitem_105, (8, 28, 28, 28), (43904, 1, 1568, 56))
    assert_size_stride(cat_12, (8, 336, 28, 28), (263424, 1, 9408, 336))
    assert_size_stride(squeeze_55, (336, ), (1, ))
    assert_size_stride(getitem_110, (8, 168, 28, 28), (263424, 784, 28, 1))
    assert_size_stride(getitem_113, (8, 168, 28, 28), (263424, 784, 28, 1))
    assert_size_stride(cat_13, (8, 336, 28, 28), (263424, 1, 9408, 336))
    assert_size_stride(squeeze_58, (336, ), (1, ))
    assert_size_stride(add_103, (8, 336, 28, 28), (263424, 1, 9408, 336))
    assert_size_stride(mean_3, (8, 336, 1, 1), (336, 1, 336, 336))
    assert_size_stride(convolution_43, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(mul_154, (8, 28, 1, 1), (28, 1, 28, 28))
    assert_size_stride(convolution_44, (8, 336, 1, 1), (336, 1, 336, 336))
    assert_size_stride(getitem_116, (8, 168, 28, 28), (263424, 784, 28, 1))
    assert_size_stride(getitem_117, (8, 168, 28, 28), (263424, 784, 28, 1))
    assert_size_stride(cat_14, (8, 56, 28, 28), (43904, 1, 1568, 56))
    assert_size_stride(squeeze_61, (56, ), (1, ))
    assert_size_stride(add_109, (8, 56, 28, 28), (43904, 1, 1568, 56))
    assert_size_stride(convolution_47, (8, 336, 28, 28), (263424, 1, 9408, 336))
    assert_size_stride(squeeze_64, (336, ), (1, ))
    assert_size_stride(getitem_125, (8, 112, 28, 28), (263424, 784, 28, 1))
    assert_size_stride(getitem_129, (8, 112, 28, 28), (263424, 784, 28, 1))
    assert_size_stride(getitem_133, (8, 112, 28, 28), (263424, 784, 28, 1))
    assert_size_stride(cat_15, (8, 336, 14, 14), (65856, 1, 4704, 336))
    assert_size_stride(squeeze_67, (336, ), (1, ))
    assert_size_stride(add_119, (8, 336, 14, 14), (65856, 1, 4704, 336))
    assert_size_stride(mean_4, (8, 336, 1, 1), (336, 1, 336, 336))
    assert_size_stride(convolution_51, (8, 14, 1, 1), (14, 1, 14, 14))
    assert_size_stride(mul_179, (8, 14, 1, 1), (14, 1, 14, 14))
    assert_size_stride(convolution_52, (8, 336, 1, 1), (336, 1, 336, 336))
    assert_size_stride(mul_180, (8, 336, 14, 14), (65856, 1, 4704, 336))
    assert_size_stride(convolution_53, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_70, (104, ), (1, ))
    assert_size_stride(getitem_138, (8, 52, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(getitem_139, (8, 52, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(cat_16, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(squeeze_73, (624, ), (1, ))
    assert_size_stride(getitem_146, (8, 156, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(getitem_151, (8, 156, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(getitem_156, (8, 156, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(getitem_161, (8, 156, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(cat_17, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(squeeze_76, (624, ), (1, ))
    assert_size_stride(add_134, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(mean_5, (8, 624, 1, 1), (624, 1, 624, 624))
    assert_size_stride(convolution_60, (8, 26, 1, 1), (26, 1, 26, 26))
    assert_size_stride(mul_204, (8, 26, 1, 1), (26, 1, 26, 26))
    assert_size_stride(convolution_61, (8, 624, 1, 1), (624, 1, 624, 624))
    assert_size_stride(getitem_164, (8, 312, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(getitem_165, (8, 312, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(cat_18, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_79, (104, ), (1, ))
    assert_size_stride(getitem_168, (8, 52, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(getitem_169, (8, 52, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(cat_19, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(squeeze_82, (624, ), (1, ))
    assert_size_stride(getitem_176, (8, 156, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(getitem_181, (8, 156, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(getitem_186, (8, 156, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(getitem_191, (8, 156, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(cat_20, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(squeeze_85, (624, ), (1, ))
    assert_size_stride(add_150, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(mean_6, (8, 624, 1, 1), (624, 1, 624, 624))
    assert_size_stride(convolution_70, (8, 26, 1, 1), (26, 1, 26, 26))
    assert_size_stride(mul_229, (8, 26, 1, 1), (26, 1, 26, 26))
    assert_size_stride(convolution_71, (8, 624, 1, 1), (624, 1, 624, 624))
    assert_size_stride(getitem_194, (8, 312, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(getitem_195, (8, 312, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(cat_21, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_88, (104, ), (1, ))
    assert_size_stride(getitem_198, (8, 52, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(getitem_199, (8, 52, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(cat_22, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(squeeze_91, (624, ), (1, ))
    assert_size_stride(getitem_206, (8, 156, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(getitem_211, (8, 156, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(getitem_216, (8, 156, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(getitem_221, (8, 156, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(cat_23, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(squeeze_94, (624, ), (1, ))
    assert_size_stride(add_166, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(mean_7, (8, 624, 1, 1), (624, 1, 624, 624))
    assert_size_stride(convolution_80, (8, 26, 1, 1), (26, 1, 26, 26))
    assert_size_stride(mul_254, (8, 26, 1, 1), (26, 1, 26, 26))
    assert_size_stride(convolution_81, (8, 624, 1, 1), (624, 1, 624, 624))
    assert_size_stride(getitem_224, (8, 312, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(getitem_225, (8, 312, 14, 14), (122304, 196, 14, 1))
    assert_size_stride(cat_24, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(squeeze_97, (104, ), (1, ))
    assert_size_stride(add_172, (8, 104, 14, 14), (20384, 1, 1456, 104))
    assert_size_stride(convolution_84, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(squeeze_100, (624, ), (1, ))
    assert_size_stride(mul_270, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(convolution_85, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(squeeze_103, (624, ), (1, ))
    assert_size_stride(add_182, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(mean_8, (8, 624, 1, 1), (624, 1, 624, 624))
    assert_size_stride(convolution_86, (8, 52, 1, 1), (52, 1, 52, 52))
    assert_size_stride(mul_279, (8, 52, 1, 1), (52, 1, 52, 52))
    assert_size_stride(convolution_87, (8, 624, 1, 1), (624, 1, 624, 624))
    assert_size_stride(mul_280, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(convolution_88, (8, 160, 14, 14), (31360, 1, 2240, 160))
    assert_size_stride(squeeze_106, (160, ), (1, ))
    assert_size_stride(getitem_234, (8, 80, 14, 14), (31360, 1, 2240, 160))
    assert_size_stride(getitem_235, (8, 80, 14, 14), (31360, 1, 2240, 160))
    assert_size_stride(cat_25, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(squeeze_109, (480, ), (1, ))
    assert_size_stride(getitem_242, (8, 120, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(getitem_247, (8, 120, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(getitem_252, (8, 120, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(getitem_257, (8, 120, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(cat_26, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(squeeze_112, (480, ), (1, ))
    assert_size_stride(add_197, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(mean_9, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(convolution_95, (8, 80, 1, 1), (80, 1, 80, 80))
    assert_size_stride(mul_304, (8, 80, 1, 1), (80, 1, 80, 80))
    assert_size_stride(convolution_96, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(getitem_260, (8, 240, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(getitem_261, (8, 240, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(cat_27, (8, 160, 14, 14), (31360, 1, 2240, 160))
    assert_size_stride(squeeze_115, (160, ), (1, ))
    assert_size_stride(getitem_264, (8, 80, 14, 14), (31360, 1, 2240, 160))
    assert_size_stride(getitem_265, (8, 80, 14, 14), (31360, 1, 2240, 160))
    assert_size_stride(cat_28, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(squeeze_118, (480, ), (1, ))
    assert_size_stride(getitem_272, (8, 120, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(getitem_277, (8, 120, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(getitem_282, (8, 120, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(getitem_287, (8, 120, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(cat_29, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(squeeze_121, (480, ), (1, ))
    assert_size_stride(add_213, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(mean_10, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(convolution_105, (8, 80, 1, 1), (80, 1, 80, 80))
    assert_size_stride(mul_329, (8, 80, 1, 1), (80, 1, 80, 80))
    assert_size_stride(convolution_106, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(getitem_290, (8, 240, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(getitem_291, (8, 240, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(cat_30, (8, 160, 14, 14), (31360, 1, 2240, 160))
    assert_size_stride(squeeze_124, (160, ), (1, ))
    assert_size_stride(getitem_294, (8, 80, 14, 14), (31360, 1, 2240, 160))
    assert_size_stride(getitem_295, (8, 80, 14, 14), (31360, 1, 2240, 160))
    assert_size_stride(cat_31, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(squeeze_127, (480, ), (1, ))
    assert_size_stride(getitem_302, (8, 120, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(getitem_307, (8, 120, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(getitem_312, (8, 120, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(getitem_317, (8, 120, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(cat_32, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(squeeze_130, (480, ), (1, ))
    assert_size_stride(add_229, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(mean_11, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(convolution_115, (8, 80, 1, 1), (80, 1, 80, 80))
    assert_size_stride(mul_354, (8, 80, 1, 1), (80, 1, 80, 80))
    assert_size_stride(convolution_116, (8, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(getitem_320, (8, 240, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(getitem_321, (8, 240, 14, 14), (94080, 196, 14, 1))
    assert_size_stride(cat_33, (8, 160, 14, 14), (31360, 1, 2240, 160))
    assert_size_stride(squeeze_133, (160, ), (1, ))
    assert_size_stride(add_235, (8, 160, 14, 14), (31360, 1, 2240, 160))
    assert_size_stride(convolution_119, (8, 960, 14, 14), (188160, 1, 13440, 960))
    assert_size_stride(squeeze_136, (960, ), (1, ))
    assert_size_stride(getitem_330, (8, 240, 14, 14), (188160, 196, 14, 1))
    assert_size_stride(getitem_335, (8, 240, 14, 14), (188160, 196, 14, 1))
    assert_size_stride(getitem_340, (8, 240, 14, 14), (188160, 196, 14, 1))
    assert_size_stride(getitem_345, (8, 240, 14, 14), (188160, 196, 14, 1))
    assert_size_stride(cat_34, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(squeeze_139, (960, ), (1, ))
    assert_size_stride(add_245, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(mean_12, (8, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(convolution_124, (8, 80, 1, 1), (80, 1, 80, 80))
    assert_size_stride(mul_379, (8, 80, 1, 1), (80, 1, 80, 80))
    assert_size_stride(convolution_125, (8, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(mul_380, (8, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_126, (8, 264, 7, 7), (12936, 1, 1848, 264))
    assert_size_stride(squeeze_142, (264, ), (1, ))
    assert_size_stride(add_250, (8, 264, 7, 7), (12936, 1, 1848, 264))
    assert_size_stride(convolution_127, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
    assert_size_stride(squeeze_145, (1584, ), (1, ))
    assert_size_stride(getitem_356, (8, 396, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(getitem_361, (8, 396, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(getitem_366, (8, 396, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(getitem_371, (8, 396, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(cat_35, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
    assert_size_stride(squeeze_148, (1584, ), (1, ))
    assert_size_stride(add_260, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
    assert_size_stride(mean_13, (8, 1584, 1, 1), (1584, 1, 1584, 1584))
    assert_size_stride(convolution_132, (8, 132, 1, 1), (132, 1, 132, 132))
    assert_size_stride(mul_404, (8, 132, 1, 1), (132, 1, 132, 132))
    assert_size_stride(convolution_133, (8, 1584, 1, 1), (1584, 1, 1584, 1584))
    assert_size_stride(getitem_374, (8, 792, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(getitem_375, (8, 792, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(cat_36, (8, 264, 7, 7), (12936, 1, 1848, 264))
    assert_size_stride(squeeze_151, (264, ), (1, ))
    assert_size_stride(add_266, (8, 264, 7, 7), (12936, 1, 1848, 264))
    assert_size_stride(convolution_136, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
    assert_size_stride(squeeze_154, (1584, ), (1, ))
    assert_size_stride(getitem_384, (8, 396, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(getitem_389, (8, 396, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(getitem_394, (8, 396, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(getitem_399, (8, 396, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(cat_37, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
    assert_size_stride(squeeze_157, (1584, ), (1, ))
    assert_size_stride(add_276, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
    assert_size_stride(mean_14, (8, 1584, 1, 1), (1584, 1, 1584, 1584))
    assert_size_stride(convolution_141, (8, 132, 1, 1), (132, 1, 132, 132))
    assert_size_stride(mul_429, (8, 132, 1, 1), (132, 1, 132, 132))
    assert_size_stride(convolution_142, (8, 1584, 1, 1), (1584, 1, 1584, 1584))
    assert_size_stride(getitem_402, (8, 792, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(getitem_403, (8, 792, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(cat_38, (8, 264, 7, 7), (12936, 1, 1848, 264))
    assert_size_stride(squeeze_160, (264, ), (1, ))
    assert_size_stride(add_282, (8, 264, 7, 7), (12936, 1, 1848, 264))
    assert_size_stride(convolution_145, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
    assert_size_stride(squeeze_163, (1584, ), (1, ))
    assert_size_stride(getitem_412, (8, 396, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(getitem_417, (8, 396, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(getitem_422, (8, 396, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(getitem_427, (8, 396, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(cat_39, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
    assert_size_stride(squeeze_166, (1584, ), (1, ))
    assert_size_stride(add_292, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
    assert_size_stride(mean_15, (8, 1584, 1, 1), (1584, 1, 1584, 1584))
    assert_size_stride(convolution_150, (8, 132, 1, 1), (132, 1, 132, 132))
    assert_size_stride(mul_454, (8, 132, 1, 1), (132, 1, 132, 132))
    assert_size_stride(convolution_151, (8, 1584, 1, 1), (1584, 1, 1584, 1584))
    assert_size_stride(getitem_430, (8, 792, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(getitem_431, (8, 792, 7, 7), (77616, 49, 7, 1))
    assert_size_stride(cat_40, (8, 264, 7, 7), (12936, 1, 1848, 264))
    assert_size_stride(squeeze_169, (264, ), (1, ))
    assert_size_stride(add_298, (8, 264, 7, 7), (12936, 1, 1848, 264))
    assert_size_stride(convolution_154, (8, 1536, 7, 7), (75264, 1, 10752, 1536))
    assert_size_stride(squeeze_172, (1536, ), (1, ))
    assert_size_stride(view, (8, 1536), (1536, 1))
    assert_size_stride(permute_1, (1000, 1536), (1536, 1))
    assert_size_stride(le, (8, 1536, 7, 7), (75264, 1, 10752, 1536))
    assert_size_stride(unsqueeze_234, (1, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(unsqueeze_246, (1, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(unsqueeze_258, (1, 1584, 1, 1), (1584, 1, 1, 1))
    assert_size_stride(mul_508, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
    assert_size_stride(unsqueeze_270, (1, 1584, 1, 1), (1584, 1, 1, 1))
    assert_size_stride(unsqueeze_282, (1, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(unsqueeze_294, (1, 1584, 1, 1), (1584, 1, 1, 1))
    assert_size_stride(mul_548, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
    assert_size_stride(unsqueeze_306, (1, 1584, 1, 1), (1584, 1, 1, 1))
    assert_size_stride(unsqueeze_318, (1, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(unsqueeze_330, (1, 1584, 1, 1), (1584, 1, 1, 1))
    assert_size_stride(mul_588, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
    assert_size_stride(unsqueeze_342, (1, 1584, 1, 1), (1584, 1, 1, 1))
    assert_size_stride(unsqueeze_354, (1, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(unsqueeze_366, (1, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(mul_628, (8, 960, 14, 14), (188160, 1, 13440, 960))
    assert_size_stride(unsqueeze_378, (1, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(unsqueeze_390, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_402, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(mul_668, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(unsqueeze_414, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_426, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_438, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(mul_708, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(unsqueeze_450, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_462, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_474, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(mul_748, (8, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(unsqueeze_486, (1, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(unsqueeze_498, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_510, (1, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(mul_788, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(unsqueeze_522, (1, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(unsqueeze_534, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(unsqueeze_546, (1, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(mul_828, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(unsqueeze_558, (1, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(unsqueeze_570, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(unsqueeze_582, (1, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(mul_868, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(unsqueeze_594, (1, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(unsqueeze_606, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(unsqueeze_618, (1, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(mul_908, (8, 624, 14, 14), (122304, 1, 8736, 624))
    assert_size_stride(unsqueeze_630, (1, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(unsqueeze_642, (1, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(unsqueeze_654, (1, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(mul_948, (8, 336, 28, 28), (263424, 1, 9408, 336))
    assert_size_stride(unsqueeze_666, (1, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(unsqueeze_678, (1, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(unsqueeze_690, (1, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(mul_988, (8, 336, 28, 28), (263424, 1, 9408, 336))
    assert_size_stride(unsqueeze_702, (1, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(unsqueeze_714, (1, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(unsqueeze_726, (1, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(mul_1028, (8, 336, 28, 28), (263424, 1, 9408, 336))
    assert_size_stride(unsqueeze_738, (1, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(unsqueeze_750, (1, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(unsqueeze_762, (1, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(mul_1068, (8, 336, 28, 28), (263424, 1, 9408, 336))
    assert_size_stride(unsqueeze_774, (1, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(unsqueeze_786, (1, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(unsqueeze_798, (1, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(mul_1108, (8, 240, 56, 56), (752640, 1, 13440, 240))
    assert_size_stride(unsqueeze_810, (1, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(unsqueeze_822, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(le_1, (8, 120, 56, 56), (376320, 1, 6720, 120))
    assert_size_stride(unsqueeze_834, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_846, (1, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(unsqueeze_858, (1, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(le_3, (8, 192, 56, 56), (602112, 1, 10752, 192))
    assert_size_stride(unsqueeze_870, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(le_4, (8, 192, 112, 112), (2408448, 1, 21504, 192))
    assert_size_stride(unsqueeze_882, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_894, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(unsqueeze_906, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(unsqueeze_918, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_1, out=buf0)
        del permute_1
        buf1 = empty((1000, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), view, out=buf1)
        del view
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf3 = empty_strided((1536, 4), (1, 1536), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((1536, 4), (1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_div_native_batch_norm_backward_threshold_backward_1.run(le, buf0, convolution_154, unsqueeze_234, buf3, buf5, 6144, 98, grid=grid(6144), stream=stream0)
        buf4 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_2.run(buf3, buf4, 1536, 4, grid=grid(1536), stream=stream0)
        del buf3
        buf6 = empty((1536, ), device='cuda', dtype=torch.float32)
        buf7 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_3.run(buf5, squeeze_172, buf6, buf7, 1536, 4, grid=grid(1536), stream=stream0)
        del buf5
        buf8 = empty_strided((8, 1536, 7, 7), (75264, 1, 10752, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_4.run(le, buf0, convolution_154, unsqueeze_234, buf6, squeeze_172, buf4, primals_115, buf8, 602112, grid=grid(602112), stream=stream0)
        del buf0
        del buf6
        del convolution_154
        del le
        del primals_115
        del squeeze_172
        del unsqueeze_234
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        buf9 = aten.convolution_backward(buf8, add_298, primals_303, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_298
        del buf8
        del primals_303
        buf10 = buf9[0]
        buf11 = buf9[1]
        del buf9
        buf12 = empty((264, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_5.run(buf10, buf12, 264, 392, grid=grid(264), stream=stream0)
        buf13 = empty_strided((264, 4), (1, 264), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_6.run(buf10, cat_40, unsqueeze_246, buf13, 1056, 98, grid=grid(1056), stream=stream0)
        buf14 = empty((264, ), device='cuda', dtype=torch.float32)
        buf16 = empty((264, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_7.run(buf13, squeeze_169, buf14, buf16, 264, 4, grid=grid(264), stream=stream0)
        del buf13
        buf15 = empty((8, 264, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_8.run(buf10, cat_40, unsqueeze_246, buf14, squeeze_169, buf12, primals_113, buf15, 2112, 49, grid=grid(2112, 49), stream=stream0)
        del cat_40
        del primals_113
        del squeeze_169
        del unsqueeze_246
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf17 = aten.convolution_backward(reinterpret_tensor(buf15, (8, 132, 7, 7), (12936, 49, 7, 1), 6468), getitem_431, primals_302, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_431
        del primals_302
        buf18 = buf17[0]
        buf19 = buf17[1]
        del buf17
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf20 = aten.convolution_backward(reinterpret_tensor(buf15, (8, 132, 7, 7), (12936, 49, 7, 1), 0), getitem_430, primals_301, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_430
        del primals_301
        buf21 = buf20[0]
        buf22 = buf20[1]
        del buf20
        buf23 = empty_strided((8, 1584, 1, 1), (1584, 1, 12672, 12672), device='cuda', dtype=torch.float32)
        buf24 = reinterpret_tensor(buf23, (8, 1584, 1, 1), (1584, 1, 1, 1), 0); del buf23  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate, x_350], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_9.run(buf24, buf21, buf18, add_292, convolution_151, 12672, 49, grid=grid(12672), stream=stream0)
        buf25 = empty((1584, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_10.run(buf24, buf25, 1584, 8, grid=grid(1584), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf26 = aten.convolution_backward(buf24, mul_454, primals_299, [1584], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf24
        del mul_454
        del primals_299
        buf27 = buf26[0]
        buf28 = buf26[1]
        del buf26
        buf29 = buf27; del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_11.run(buf29, convolution_150, 1056, grid=grid(1056), stream=stream0)
        del convolution_150
        buf30 = empty((132, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf29, buf30, 132, 8, grid=grid(132), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf31 = aten.convolution_backward(buf29, mean_15, primals_297, [132], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf29
        del mean_15
        del primals_297
        buf32 = buf31[0]
        buf33 = buf31[1]
        del buf31
        buf34 = empty((8, 1584, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate], Original ATen: [aten.add, aten.cat, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_13.run(buf21, buf18, convolution_151, buf32, add_292, buf34, 12672, 49, grid=grid(12672, 49), stream=stream0)
        del add_292
        del buf18
        del buf21
        del convolution_151
        buf35 = empty((1584, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_14.run(buf34, buf35, 1584, 392, grid=grid(1584), stream=stream0)
        buf36 = empty_strided((1584, 4), (1, 1584), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_15.run(buf34, cat_39, unsqueeze_258, buf36, 6336, 98, grid=grid(6336), stream=stream0)
        buf37 = empty((1584, ), device='cuda', dtype=torch.float32)
        buf39 = empty((1584, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_16.run(buf36, squeeze_166, buf37, buf39, 1584, 4, grid=grid(1584), stream=stream0)
        buf38 = buf34; del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_17.run(buf38, cat_39, unsqueeze_258, buf37, squeeze_166, buf35, primals_111, 12672, 49, grid=grid(12672, 49), stream=stream0)
        del cat_39
        del primals_111
        del squeeze_166
        del unsqueeze_258
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf40 = aten.convolution_backward(reinterpret_tensor(buf38, (8, 396, 7, 7), (77616, 49, 7, 1), 58212), getitem_427, primals_296, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 396, [True, True, False])
        del getitem_427
        del primals_296
        buf41 = buf40[0]
        buf42 = buf40[1]
        del buf40
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf43 = aten.convolution_backward(reinterpret_tensor(buf38, (8, 396, 7, 7), (77616, 49, 7, 1), 38808), getitem_422, primals_295, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 396, [True, True, False])
        del getitem_422
        del primals_295
        buf44 = buf43[0]
        buf45 = buf43[1]
        del buf43
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf46 = aten.convolution_backward(reinterpret_tensor(buf38, (8, 396, 7, 7), (77616, 49, 7, 1), 19404), getitem_417, primals_294, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 396, [True, True, False])
        del getitem_417
        del primals_294
        buf47 = buf46[0]
        buf48 = buf46[1]
        del buf46
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf49 = aten.convolution_backward(reinterpret_tensor(buf38, (8, 396, 7, 7), (77616, 49, 7, 1), 0), getitem_412, primals_293, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 396, [True, True, False])
        del getitem_412
        del primals_293
        buf50 = buf49[0]
        buf51 = buf49[1]
        del buf49
        buf52 = buf38; del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul]
        triton_poi_fused_cat_mul_18.run(buf50, buf47, buf44, buf41, mul_508, buf52, 12672, 49, grid=grid(12672, 49), stream=stream0)
        del buf41
        del buf44
        del buf47
        del buf50
        del mul_508
        buf53 = buf37; del buf37  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_14.run(buf52, buf53, 1584, 392, grid=grid(1584), stream=stream0)
        buf54 = buf36; del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_15.run(buf52, convolution_145, unsqueeze_270, buf54, 6336, 98, grid=grid(6336), stream=stream0)
        buf55 = empty((1584, ), device='cuda', dtype=torch.float32)
        buf56 = empty((1584, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_16.run(buf54, squeeze_163, buf55, buf56, 1584, 4, grid=grid(1584), stream=stream0)
        buf57 = buf52; del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_17.run(buf57, convolution_145, unsqueeze_270, buf55, squeeze_163, buf53, primals_109, 12672, 49, grid=grid(12672, 49), stream=stream0)
        del convolution_145
        del primals_109
        del squeeze_163
        del unsqueeze_270
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf58 = aten.convolution_backward(buf57, add_282, primals_292, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_282
        del primals_292
        buf59 = buf58[0]
        buf60 = buf58[1]
        del buf58
        buf61 = buf14; del buf14  # reuse
        buf62 = empty((264, ), device='cuda', dtype=torch.float32)
        buf64 = empty((264, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_19.run(buf10, buf59, cat_38, unsqueeze_282, squeeze_160, buf61, buf62, buf64, 264, 392, grid=grid(264), stream=stream0)
        buf63 = buf15; del buf15  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_20.run(buf10, buf59, cat_38, unsqueeze_282, buf62, squeeze_160, buf61, primals_107, buf63, 2112, 49, grid=grid(2112, 49), stream=stream0)
        del cat_38
        del primals_107
        del squeeze_160
        del unsqueeze_282
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf65 = aten.convolution_backward(reinterpret_tensor(buf63, (8, 132, 7, 7), (12936, 49, 7, 1), 6468), getitem_403, primals_291, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_403
        del primals_291
        buf66 = buf65[0]
        buf67 = buf65[1]
        del buf65
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf68 = aten.convolution_backward(reinterpret_tensor(buf63, (8, 132, 7, 7), (12936, 49, 7, 1), 0), getitem_402, primals_290, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_402
        del primals_290
        buf69 = buf68[0]
        buf70 = buf68[1]
        del buf68
        buf71 = reinterpret_tensor(buf32, (8, 1584, 1, 1), (1584, 1, 12672, 12672), 0); del buf32  # reuse
        buf72 = reinterpret_tensor(buf71, (8, 1584, 1, 1), (1584, 1, 1, 1), 0); del buf71  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate, x_331], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_9.run(buf72, buf69, buf66, add_276, convolution_142, 12672, 49, grid=grid(12672), stream=stream0)
        buf73 = buf55; del buf55  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_10.run(buf72, buf73, 1584, 8, grid=grid(1584), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf74 = aten.convolution_backward(buf72, mul_429, primals_288, [1584], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf72
        del mul_429
        del primals_288
        buf75 = buf74[0]
        buf76 = buf74[1]
        del buf74
        buf77 = buf75; del buf75  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_11.run(buf77, convolution_141, 1056, grid=grid(1056), stream=stream0)
        del convolution_141
        buf78 = empty((132, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf77, buf78, 132, 8, grid=grid(132), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf79 = aten.convolution_backward(buf77, mean_14, primals_286, [132], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf77
        del mean_14
        del primals_286
        buf80 = buf79[0]
        buf81 = buf79[1]
        del buf79
        buf82 = buf57; del buf57  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate], Original ATen: [aten.add, aten.cat, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_13.run(buf69, buf66, convolution_142, buf80, add_276, buf82, 12672, 49, grid=grid(12672, 49), stream=stream0)
        del add_276
        del buf66
        del buf69
        del convolution_142
        buf83 = empty((1584, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_14.run(buf82, buf83, 1584, 392, grid=grid(1584), stream=stream0)
        buf84 = buf54; del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_15.run(buf82, cat_37, unsqueeze_294, buf84, 6336, 98, grid=grid(6336), stream=stream0)
        buf85 = empty((1584, ), device='cuda', dtype=torch.float32)
        buf87 = empty((1584, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_16.run(buf84, squeeze_157, buf85, buf87, 1584, 4, grid=grid(1584), stream=stream0)
        buf86 = buf82; del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_17.run(buf86, cat_37, unsqueeze_294, buf85, squeeze_157, buf83, primals_105, 12672, 49, grid=grid(12672, 49), stream=stream0)
        del cat_37
        del primals_105
        del squeeze_157
        del unsqueeze_294
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf88 = aten.convolution_backward(reinterpret_tensor(buf86, (8, 396, 7, 7), (77616, 49, 7, 1), 58212), getitem_399, primals_285, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 396, [True, True, False])
        del getitem_399
        del primals_285
        buf89 = buf88[0]
        buf90 = buf88[1]
        del buf88
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf91 = aten.convolution_backward(reinterpret_tensor(buf86, (8, 396, 7, 7), (77616, 49, 7, 1), 38808), getitem_394, primals_284, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 396, [True, True, False])
        del getitem_394
        del primals_284
        buf92 = buf91[0]
        buf93 = buf91[1]
        del buf91
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf94 = aten.convolution_backward(reinterpret_tensor(buf86, (8, 396, 7, 7), (77616, 49, 7, 1), 19404), getitem_389, primals_283, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 396, [True, True, False])
        del getitem_389
        del primals_283
        buf95 = buf94[0]
        buf96 = buf94[1]
        del buf94
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf97 = aten.convolution_backward(reinterpret_tensor(buf86, (8, 396, 7, 7), (77616, 49, 7, 1), 0), getitem_384, primals_282, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 396, [True, True, False])
        del getitem_384
        del primals_282
        buf98 = buf97[0]
        buf99 = buf97[1]
        del buf97
        buf100 = buf86; del buf86  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul]
        triton_poi_fused_cat_mul_18.run(buf98, buf95, buf92, buf89, mul_548, buf100, 12672, 49, grid=grid(12672, 49), stream=stream0)
        del buf89
        del buf92
        del buf95
        del buf98
        del mul_548
        buf101 = buf85; del buf85  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_14.run(buf100, buf101, 1584, 392, grid=grid(1584), stream=stream0)
        buf102 = buf84; del buf84  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_15.run(buf100, convolution_136, unsqueeze_306, buf102, 6336, 98, grid=grid(6336), stream=stream0)
        buf103 = empty((1584, ), device='cuda', dtype=torch.float32)
        buf104 = empty((1584, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_16.run(buf102, squeeze_154, buf103, buf104, 1584, 4, grid=grid(1584), stream=stream0)
        buf105 = buf100; del buf100  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_17.run(buf105, convolution_136, unsqueeze_306, buf103, squeeze_154, buf101, primals_103, 12672, 49, grid=grid(12672, 49), stream=stream0)
        del convolution_136
        del primals_103
        del squeeze_154
        del unsqueeze_306
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf106 = aten.convolution_backward(buf105, add_266, primals_281, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_266
        del primals_281
        buf107 = buf106[0]
        buf108 = buf106[1]
        del buf106
        buf109 = buf62; del buf62  # reuse
        buf110 = empty((264, ), device='cuda', dtype=torch.float32)
        buf112 = empty((264, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_21.run(buf10, buf59, buf107, cat_36, unsqueeze_318, squeeze_151, buf109, buf110, buf112, 264, 392, grid=grid(264), stream=stream0)
        buf111 = buf63; del buf63  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_22.run(buf10, buf59, buf107, cat_36, unsqueeze_318, buf110, squeeze_151, buf109, primals_101, buf111, 2112, 49, grid=grid(2112, 49), stream=stream0)
        del cat_36
        del primals_101
        del squeeze_151
        del unsqueeze_318
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf113 = aten.convolution_backward(reinterpret_tensor(buf111, (8, 132, 7, 7), (12936, 49, 7, 1), 6468), getitem_375, primals_280, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_375
        del primals_280
        buf114 = buf113[0]
        buf115 = buf113[1]
        del buf113
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf116 = aten.convolution_backward(reinterpret_tensor(buf111, (8, 132, 7, 7), (12936, 49, 7, 1), 0), getitem_374, primals_279, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf111
        del getitem_374
        del primals_279
        buf117 = buf116[0]
        buf118 = buf116[1]
        del buf116
        buf119 = reinterpret_tensor(buf80, (8, 1584, 1, 1), (1584, 1, 12672, 12672), 0); del buf80  # reuse
        buf120 = reinterpret_tensor(buf119, (8, 1584, 1, 1), (1584, 1, 1, 1), 0); del buf119  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_312], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_9.run(buf120, buf117, buf114, add_260, convolution_133, 12672, 49, grid=grid(12672), stream=stream0)
        buf121 = buf103; del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_10.run(buf120, buf121, 1584, 8, grid=grid(1584), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf122 = aten.convolution_backward(buf120, mul_404, primals_277, [1584], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf120
        del mul_404
        del primals_277
        buf123 = buf122[0]
        buf124 = buf122[1]
        del buf122
        buf125 = buf123; del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_11.run(buf125, convolution_132, 1056, grid=grid(1056), stream=stream0)
        del convolution_132
        buf126 = empty((132, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf125, buf126, 132, 8, grid=grid(132), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf127 = aten.convolution_backward(buf125, mean_13, primals_275, [132], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf125
        del mean_13
        del primals_275
        buf128 = buf127[0]
        buf129 = buf127[1]
        del buf127
        buf130 = buf105; del buf105  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate], Original ATen: [aten.add, aten.cat, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_13.run(buf117, buf114, convolution_133, buf128, add_260, buf130, 12672, 49, grid=grid(12672, 49), stream=stream0)
        del add_260
        del buf114
        del buf117
        del buf128
        del convolution_133
        buf131 = empty((1584, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_14.run(buf130, buf131, 1584, 392, grid=grid(1584), stream=stream0)
        buf132 = buf102; del buf102  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_15.run(buf130, cat_35, unsqueeze_330, buf132, 6336, 98, grid=grid(6336), stream=stream0)
        buf133 = empty((1584, ), device='cuda', dtype=torch.float32)
        buf135 = empty((1584, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_16.run(buf132, squeeze_148, buf133, buf135, 1584, 4, grid=grid(1584), stream=stream0)
        buf134 = buf130; del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_17.run(buf134, cat_35, unsqueeze_330, buf133, squeeze_148, buf131, primals_99, 12672, 49, grid=grid(12672, 49), stream=stream0)
        del cat_35
        del primals_99
        del squeeze_148
        del unsqueeze_330
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf136 = aten.convolution_backward(reinterpret_tensor(buf134, (8, 396, 7, 7), (77616, 49, 7, 1), 58212), getitem_371, primals_274, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 396, [True, True, False])
        del getitem_371
        del primals_274
        buf137 = buf136[0]
        buf138 = buf136[1]
        del buf136
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf139 = aten.convolution_backward(reinterpret_tensor(buf134, (8, 396, 7, 7), (77616, 49, 7, 1), 38808), getitem_366, primals_273, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 396, [True, True, False])
        del getitem_366
        del primals_273
        buf140 = buf139[0]
        buf141 = buf139[1]
        del buf139
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf142 = aten.convolution_backward(reinterpret_tensor(buf134, (8, 396, 7, 7), (77616, 49, 7, 1), 19404), getitem_361, primals_272, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 396, [True, True, False])
        del getitem_361
        del primals_272
        buf143 = buf142[0]
        buf144 = buf142[1]
        del buf142
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf145 = aten.convolution_backward(reinterpret_tensor(buf134, (8, 396, 7, 7), (77616, 49, 7, 1), 0), getitem_356, primals_271, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 396, [True, True, False])
        del getitem_356
        del primals_271
        buf146 = buf145[0]
        buf147 = buf145[1]
        del buf145
        buf148 = buf134; del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul]
        triton_poi_fused_cat_mul_18.run(buf146, buf143, buf140, buf137, mul_588, buf148, 12672, 49, grid=grid(12672, 49), stream=stream0)
        del buf137
        del buf140
        del buf143
        del buf146
        del mul_588
        buf149 = buf133; del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_14.run(buf148, buf149, 1584, 392, grid=grid(1584), stream=stream0)
        buf150 = buf132; del buf132  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_15.run(buf148, convolution_127, unsqueeze_342, buf150, 6336, 98, grid=grid(6336), stream=stream0)
        buf151 = empty((1584, ), device='cuda', dtype=torch.float32)
        buf152 = empty((1584, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_16.run(buf150, squeeze_145, buf151, buf152, 1584, 4, grid=grid(1584), stream=stream0)
        del buf150
        buf153 = buf148; del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_17.run(buf153, convolution_127, unsqueeze_342, buf151, squeeze_145, buf149, primals_97, 12672, 49, grid=grid(12672, 49), stream=stream0)
        del buf151
        del convolution_127
        del primals_97
        del squeeze_145
        del unsqueeze_342
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf154 = aten.convolution_backward(buf153, add_250, primals_270, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_250
        del buf153
        del primals_270
        buf155 = buf154[0]
        buf156 = buf154[1]
        del buf154
        buf157 = buf110; del buf110  # reuse
        buf158 = empty((264, ), device='cuda', dtype=torch.float32)
        buf160 = empty((264, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_23.run(buf10, buf59, buf107, buf155, convolution_126, unsqueeze_354, squeeze_142, buf157, buf158, buf160, 264, 392, grid=grid(264), stream=stream0)
        buf159 = buf10; del buf10  # reuse
        buf161 = buf159; del buf159  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_24.run(buf161, buf59, buf107, buf155, convolution_126, unsqueeze_354, buf158, squeeze_142, buf157, primals_95, 2112, 49, grid=grid(2112, 49), stream=stream0)
        del buf107
        del buf155
        del buf158
        del buf59
        del convolution_126
        del primals_95
        del squeeze_142
        del unsqueeze_354
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf162 = aten.convolution_backward(buf161, mul_380, primals_269, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf161
        del mul_380
        del primals_269
        buf163 = buf162[0]
        buf164 = buf162[1]
        del buf162
        buf165 = empty_strided((8, 960, 1, 1), (960, 1, 7680, 7680), device='cuda', dtype=torch.float32)
        buf166 = reinterpret_tensor(buf165, (8, 960, 1, 1), (960, 1, 1, 1), 0); del buf165  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_295], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_25.run(buf166, buf163, add_245, convolution_125, 7680, 49, grid=grid(7680), stream=stream0)
        buf167 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_26.run(buf166, buf167, 960, 8, grid=grid(960), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf168 = aten.convolution_backward(buf166, mul_379, primals_267, [960], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf166
        del mul_379
        del primals_267
        buf169 = buf168[0]
        buf170 = buf168[1]
        del buf168
        buf171 = buf169; del buf169  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_27.run(buf171, convolution_124, 640, grid=grid(640), stream=stream0)
        del convolution_124
        buf172 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_28.run(buf171, buf172, 80, 8, grid=grid(80), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf173 = aten.convolution_backward(buf171, mean_12, primals_265, [80], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf171
        del mean_12
        del primals_265
        buf174 = buf173[0]
        buf175 = buf173[1]
        del buf173
        buf176 = empty_strided((960, 4), (1, 960), device='cuda', dtype=torch.float32)
        buf178 = empty_strided((960, 4), (1, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_29.run(buf163, convolution_125, buf174, add_245, cat_34, unsqueeze_366, buf176, buf178, 3840, 98, grid=grid(3840), stream=stream0)
        buf177 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_30.run(buf176, buf177, 960, 4, grid=grid(960), stream=stream0)
        del buf176
        buf179 = empty((960, ), device='cuda', dtype=torch.float32)
        buf181 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_31.run(buf178, squeeze_139, buf179, buf181, 960, 4, grid=grid(960), stream=stream0)
        buf180 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_32.run(buf163, convolution_125, buf174, add_245, cat_34, unsqueeze_366, buf179, squeeze_139, buf177, buf180, 392, 960, grid=grid(392, 960), stream=stream0)
        del add_245
        del buf163
        del buf174
        del cat_34
        del convolution_125
        del unsqueeze_366
        buf182 = empty((8, 240, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_33.run(buf180, squeeze_139, primals_93, buf182, 1920, 49, grid=grid(1920, 49), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf183 = aten.convolution_backward(buf182, getitem_345, primals_264, [0], [2, 2], [4, 4], [1, 1], False, [0, 0], 240, [True, True, False])
        del getitem_345
        del primals_264
        buf184 = buf183[0]
        buf185 = buf183[1]
        del buf183
        buf186 = buf182; del buf182  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_34.run(buf180, squeeze_139, primals_93, buf186, 1920, 49, grid=grid(1920, 49), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf187 = aten.convolution_backward(buf186, getitem_340, primals_263, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 240, [True, True, False])
        del getitem_340
        del primals_263
        buf188 = buf187[0]
        buf189 = buf187[1]
        del buf187
        buf190 = buf186; del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_35.run(buf180, squeeze_139, primals_93, buf190, 1920, 49, grid=grid(1920, 49), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf191 = aten.convolution_backward(buf190, getitem_335, primals_262, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 240, [True, True, False])
        del getitem_335
        del primals_262
        buf192 = buf191[0]
        buf193 = buf191[1]
        del buf191
        buf194 = buf190; del buf190  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_36.run(buf180, squeeze_139, primals_93, buf194, 1920, 49, grid=grid(1920, 49), stream=stream0)
        del buf180
        del primals_93
        del squeeze_139
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf195 = aten.convolution_backward(buf194, getitem_330, primals_261, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 240, [True, True, False])
        del buf194
        del getitem_330
        del primals_261
        buf196 = buf195[0]
        buf197 = buf195[1]
        del buf195
        buf198 = empty((8, 960, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul]
        triton_poi_fused_cat_mul_37.run(buf196, buf192, buf188, buf184, mul_628, buf198, 7680, 196, grid=grid(7680, 196), stream=stream0)
        del buf184
        del buf188
        del buf192
        del buf196
        del mul_628
        buf199 = buf179; del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_38.run(buf198, buf199, 960, 1568, grid=grid(960), stream=stream0)
        buf200 = empty((960, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_39.run(buf198, convolution_119, unsqueeze_378, buf200, 12480, 121, grid=grid(12480), stream=stream0)
        buf201 = empty((960, ), device='cuda', dtype=torch.float32)
        buf202 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_40.run(buf200, squeeze_136, buf201, buf202, 960, 13, grid=grid(960), stream=stream0)
        del buf200
        buf203 = buf198; del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_41.run(buf203, convolution_119, unsqueeze_378, buf201, squeeze_136, buf199, primals_91, 7680, 196, grid=grid(7680, 196), stream=stream0)
        del convolution_119
        del primals_91
        del squeeze_136
        del unsqueeze_378
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf204 = aten.convolution_backward(buf203, add_235, primals_260, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_235
        del primals_260
        buf205 = buf204[0]
        buf206 = buf204[1]
        del buf204
        buf207 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_42.run(buf205, buf207, 160, 1568, grid=grid(160), stream=stream0)
        buf208 = empty((160, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_43.run(buf205, cat_33, unsqueeze_390, buf208, 2080, 121, grid=grid(2080), stream=stream0)
        buf209 = empty((160, ), device='cuda', dtype=torch.float32)
        buf211 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_44.run(buf208, squeeze_133, buf209, buf211, 160, 13, grid=grid(160), stream=stream0)
        buf210 = empty((8, 160, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_45.run(buf205, cat_33, unsqueeze_390, buf209, squeeze_133, buf207, primals_89, buf210, 1280, 196, grid=grid(1280, 196), stream=stream0)
        del cat_33
        del primals_89
        del squeeze_133
        del unsqueeze_390
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf212 = aten.convolution_backward(reinterpret_tensor(buf210, (8, 80, 14, 14), (31360, 196, 14, 1), 15680), getitem_321, primals_259, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_321
        del primals_259
        buf213 = buf212[0]
        buf214 = buf212[1]
        del buf212
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf215 = aten.convolution_backward(reinterpret_tensor(buf210, (8, 80, 14, 14), (31360, 196, 14, 1), 0), getitem_320, primals_258, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_320
        del primals_258
        buf216 = buf215[0]
        buf217 = buf215[1]
        del buf215
        buf218 = reinterpret_tensor(buf178, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf178  # reuse
        buf219 = reinterpret_tensor(buf218, (8, 480, 1, 1), (480, 1, 1, 1), 0); del buf218  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate, x_276], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_46.run(buf219, buf216, buf213, add_229, convolution_116, 3840, 196, grid=grid(3840), stream=stream0)
        buf220 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_47.run(buf219, buf220, 480, 8, grid=grid(480), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf221 = aten.convolution_backward(buf219, mul_354, primals_256, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf219
        del mul_354
        del primals_256
        buf222 = buf221[0]
        buf223 = buf221[1]
        del buf221
        buf224 = buf222; del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_27.run(buf224, convolution_115, 640, grid=grid(640), stream=stream0)
        del convolution_115
        buf225 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_28.run(buf224, buf225, 80, 8, grid=grid(80), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf226 = aten.convolution_backward(buf224, mean_11, primals_254, [80], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf224
        del mean_11
        del primals_254
        buf227 = buf226[0]
        buf228 = buf226[1]
        del buf226
        buf229 = empty((8, 480, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate], Original ATen: [aten.add, aten.cat, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_48.run(buf216, buf213, convolution_116, buf227, add_229, buf229, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del add_229
        del buf213
        del buf216
        del convolution_116
        buf230 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_49.run(buf229, buf230, 480, 1568, grid=grid(480), stream=stream0)
        buf231 = empty((480, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_50.run(buf229, cat_32, unsqueeze_402, buf231, 6240, 121, grid=grid(6240), stream=stream0)
        buf232 = empty((480, ), device='cuda', dtype=torch.float32)
        buf234 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_51.run(buf231, squeeze_130, buf232, buf234, 480, 13, grid=grid(480), stream=stream0)
        buf233 = buf229; del buf229  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_52.run(buf233, cat_32, unsqueeze_402, buf232, squeeze_130, buf230, primals_87, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del cat_32
        del primals_87
        del squeeze_130
        del unsqueeze_402
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf235 = aten.convolution_backward(reinterpret_tensor(buf233, (8, 120, 14, 14), (94080, 196, 14, 1), 70560), getitem_317, primals_253, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 120, [True, True, False])
        del getitem_317
        del primals_253
        buf236 = buf235[0]
        buf237 = buf235[1]
        del buf235
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf238 = aten.convolution_backward(reinterpret_tensor(buf233, (8, 120, 14, 14), (94080, 196, 14, 1), 47040), getitem_312, primals_252, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 120, [True, True, False])
        del getitem_312
        del primals_252
        buf239 = buf238[0]
        buf240 = buf238[1]
        del buf238
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf241 = aten.convolution_backward(reinterpret_tensor(buf233, (8, 120, 14, 14), (94080, 196, 14, 1), 23520), getitem_307, primals_251, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
        del getitem_307
        del primals_251
        buf242 = buf241[0]
        buf243 = buf241[1]
        del buf241
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf244 = aten.convolution_backward(reinterpret_tensor(buf233, (8, 120, 14, 14), (94080, 196, 14, 1), 0), getitem_302, primals_250, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 120, [True, True, False])
        del getitem_302
        del primals_250
        buf245 = buf244[0]
        buf246 = buf244[1]
        del buf244
        buf247 = buf233; del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul]
        triton_poi_fused_cat_mul_53.run(buf245, buf242, buf239, buf236, mul_668, buf247, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del buf236
        del buf239
        del buf242
        del buf245
        del mul_668
        buf248 = buf232; del buf232  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_49.run(buf247, buf248, 480, 1568, grid=grid(480), stream=stream0)
        buf249 = buf231; del buf231  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_50.run(buf247, cat_31, unsqueeze_414, buf249, 6240, 121, grid=grid(6240), stream=stream0)
        buf250 = empty((480, ), device='cuda', dtype=torch.float32)
        buf252 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_51.run(buf249, squeeze_127, buf250, buf252, 480, 13, grid=grid(480), stream=stream0)
        buf251 = buf247; del buf247  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_52.run(buf251, cat_31, unsqueeze_414, buf250, squeeze_127, buf248, primals_85, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del cat_31
        del primals_85
        del squeeze_127
        del unsqueeze_414
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf253 = aten.convolution_backward(reinterpret_tensor(buf251, (8, 240, 14, 14), (94080, 196, 14, 1), 47040), getitem_295, primals_249, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_295
        del primals_249
        buf254 = buf253[0]
        buf255 = buf253[1]
        del buf253
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf256 = aten.convolution_backward(reinterpret_tensor(buf251, (8, 240, 14, 14), (94080, 196, 14, 1), 0), getitem_294, primals_248, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_294
        del primals_248
        buf257 = buf256[0]
        buf258 = buf256[1]
        del buf256
        buf259 = buf209; del buf209  # reuse
        buf260 = empty((160, ), device='cuda', dtype=torch.float32)
        buf262 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_red_fused_add_cat_native_batch_norm_backward_54.run(buf205, buf257, buf254, cat_30, unsqueeze_426, squeeze_124, buf259, buf260, buf262, 160, 1568, grid=grid(160), stream=stream0)
        buf261 = buf210; del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_poi_fused_add_cat_native_batch_norm_backward_55.run(buf205, buf257, buf254, cat_30, unsqueeze_426, buf260, squeeze_124, buf259, primals_83, buf261, 1280, 196, grid=grid(1280, 196), stream=stream0)
        del cat_30
        del primals_83
        del squeeze_124
        del unsqueeze_426
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf263 = aten.convolution_backward(reinterpret_tensor(buf261, (8, 80, 14, 14), (31360, 196, 14, 1), 15680), getitem_291, primals_247, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_291
        del primals_247
        buf264 = buf263[0]
        buf265 = buf263[1]
        del buf263
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf266 = aten.convolution_backward(reinterpret_tensor(buf261, (8, 80, 14, 14), (31360, 196, 14, 1), 0), getitem_290, primals_246, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_290
        del primals_246
        buf267 = buf266[0]
        buf268 = buf266[1]
        del buf266
        buf269 = reinterpret_tensor(buf227, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf227  # reuse
        buf270 = reinterpret_tensor(buf269, (8, 480, 1, 1), (480, 1, 1, 1), 0); del buf269  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate, x_256], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_46.run(buf270, buf267, buf264, add_213, convolution_106, 3840, 196, grid=grid(3840), stream=stream0)
        buf271 = buf250; del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_47.run(buf270, buf271, 480, 8, grid=grid(480), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf272 = aten.convolution_backward(buf270, mul_329, primals_244, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf270
        del mul_329
        del primals_244
        buf273 = buf272[0]
        buf274 = buf272[1]
        del buf272
        buf275 = buf273; del buf273  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_27.run(buf275, convolution_105, 640, grid=grid(640), stream=stream0)
        del convolution_105
        buf276 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_28.run(buf275, buf276, 80, 8, grid=grid(80), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf277 = aten.convolution_backward(buf275, mean_10, primals_242, [80], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf275
        del mean_10
        del primals_242
        buf278 = buf277[0]
        buf279 = buf277[1]
        del buf277
        buf280 = buf251; del buf251  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate], Original ATen: [aten.add, aten.cat, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_48.run(buf267, buf264, convolution_106, buf278, add_213, buf280, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del add_213
        del buf264
        del buf267
        del convolution_106
        buf281 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_49.run(buf280, buf281, 480, 1568, grid=grid(480), stream=stream0)
        buf282 = buf249; del buf249  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_50.run(buf280, cat_29, unsqueeze_438, buf282, 6240, 121, grid=grid(6240), stream=stream0)
        buf283 = empty((480, ), device='cuda', dtype=torch.float32)
        buf285 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_51.run(buf282, squeeze_121, buf283, buf285, 480, 13, grid=grid(480), stream=stream0)
        buf284 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_52.run(buf284, cat_29, unsqueeze_438, buf283, squeeze_121, buf281, primals_81, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del cat_29
        del primals_81
        del squeeze_121
        del unsqueeze_438
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf286 = aten.convolution_backward(reinterpret_tensor(buf284, (8, 120, 14, 14), (94080, 196, 14, 1), 70560), getitem_287, primals_241, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 120, [True, True, False])
        del getitem_287
        del primals_241
        buf287 = buf286[0]
        buf288 = buf286[1]
        del buf286
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf289 = aten.convolution_backward(reinterpret_tensor(buf284, (8, 120, 14, 14), (94080, 196, 14, 1), 47040), getitem_282, primals_240, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 120, [True, True, False])
        del getitem_282
        del primals_240
        buf290 = buf289[0]
        buf291 = buf289[1]
        del buf289
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf292 = aten.convolution_backward(reinterpret_tensor(buf284, (8, 120, 14, 14), (94080, 196, 14, 1), 23520), getitem_277, primals_239, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
        del getitem_277
        del primals_239
        buf293 = buf292[0]
        buf294 = buf292[1]
        del buf292
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf295 = aten.convolution_backward(reinterpret_tensor(buf284, (8, 120, 14, 14), (94080, 196, 14, 1), 0), getitem_272, primals_238, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 120, [True, True, False])
        del getitem_272
        del primals_238
        buf296 = buf295[0]
        buf297 = buf295[1]
        del buf295
        buf298 = buf284; del buf284  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul]
        triton_poi_fused_cat_mul_53.run(buf296, buf293, buf290, buf287, mul_708, buf298, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del buf287
        del buf290
        del buf293
        del buf296
        del mul_708
        buf299 = buf283; del buf283  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_49.run(buf298, buf299, 480, 1568, grid=grid(480), stream=stream0)
        buf300 = buf282; del buf282  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_50.run(buf298, cat_28, unsqueeze_450, buf300, 6240, 121, grid=grid(6240), stream=stream0)
        buf301 = empty((480, ), device='cuda', dtype=torch.float32)
        buf303 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_51.run(buf300, squeeze_118, buf301, buf303, 480, 13, grid=grid(480), stream=stream0)
        buf302 = buf298; del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_52.run(buf302, cat_28, unsqueeze_450, buf301, squeeze_118, buf299, primals_79, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del cat_28
        del primals_79
        del squeeze_118
        del unsqueeze_450
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf304 = aten.convolution_backward(reinterpret_tensor(buf302, (8, 240, 14, 14), (94080, 196, 14, 1), 47040), getitem_265, primals_237, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_265
        del primals_237
        buf305 = buf304[0]
        buf306 = buf304[1]
        del buf304
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf307 = aten.convolution_backward(reinterpret_tensor(buf302, (8, 240, 14, 14), (94080, 196, 14, 1), 0), getitem_264, primals_236, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_264
        del primals_236
        buf308 = buf307[0]
        buf309 = buf307[1]
        del buf307
        buf310 = buf205; del buf205  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_56.run(buf310, buf257, buf254, buf308, buf305, 250880, grid=grid(250880), stream=stream0)
        del buf254
        del buf257
        del buf305
        del buf308
        buf311 = buf260; del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_42.run(buf310, buf311, 160, 1568, grid=grid(160), stream=stream0)
        buf312 = buf208; del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_43.run(buf310, cat_27, unsqueeze_462, buf312, 2080, 121, grid=grid(2080), stream=stream0)
        buf313 = empty((160, ), device='cuda', dtype=torch.float32)
        buf315 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_44.run(buf312, squeeze_115, buf313, buf315, 160, 13, grid=grid(160), stream=stream0)
        del buf312
        buf314 = buf261; del buf261  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_45.run(buf310, cat_27, unsqueeze_462, buf313, squeeze_115, buf311, primals_77, buf314, 1280, 196, grid=grid(1280, 196), stream=stream0)
        del cat_27
        del primals_77
        del squeeze_115
        del unsqueeze_462
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf316 = aten.convolution_backward(reinterpret_tensor(buf314, (8, 80, 14, 14), (31360, 196, 14, 1), 15680), getitem_261, primals_235, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_261
        del primals_235
        buf317 = buf316[0]
        buf318 = buf316[1]
        del buf316
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf319 = aten.convolution_backward(reinterpret_tensor(buf314, (8, 80, 14, 14), (31360, 196, 14, 1), 0), getitem_260, primals_234, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf314
        del getitem_260
        del primals_234
        buf320 = buf319[0]
        buf321 = buf319[1]
        del buf319
        buf322 = reinterpret_tensor(buf278, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf278  # reuse
        buf323 = reinterpret_tensor(buf322, (8, 480, 1, 1), (480, 1, 1, 1), 0); del buf322  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_236], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_46.run(buf323, buf320, buf317, add_197, convolution_96, 3840, 196, grid=grid(3840), stream=stream0)
        buf324 = buf301; del buf301  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_47.run(buf323, buf324, 480, 8, grid=grid(480), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf325 = aten.convolution_backward(buf323, mul_304, primals_232, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf323
        del mul_304
        del primals_232
        buf326 = buf325[0]
        buf327 = buf325[1]
        del buf325
        buf328 = buf326; del buf326  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_27.run(buf328, convolution_95, 640, grid=grid(640), stream=stream0)
        del convolution_95
        buf329 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_28.run(buf328, buf329, 80, 8, grid=grid(80), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf330 = aten.convolution_backward(buf328, mean_9, primals_230, [80], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf328
        del mean_9
        del primals_230
        buf331 = buf330[0]
        buf332 = buf330[1]
        del buf330
        buf333 = buf302; del buf302  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate], Original ATen: [aten.add, aten.cat, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_48.run(buf320, buf317, convolution_96, buf331, add_197, buf333, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del add_197
        del buf317
        del buf331
        del convolution_96
        buf334 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_49.run(buf333, buf334, 480, 1568, grid=grid(480), stream=stream0)
        buf335 = buf300; del buf300  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_50.run(buf333, cat_26, unsqueeze_474, buf335, 6240, 121, grid=grid(6240), stream=stream0)
        buf336 = empty((480, ), device='cuda', dtype=torch.float32)
        buf338 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_51.run(buf335, squeeze_112, buf336, buf338, 480, 13, grid=grid(480), stream=stream0)
        buf337 = buf333; del buf333  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_52.run(buf337, cat_26, unsqueeze_474, buf336, squeeze_112, buf334, primals_75, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del cat_26
        del primals_75
        del squeeze_112
        del unsqueeze_474
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf339 = aten.convolution_backward(reinterpret_tensor(buf337, (8, 120, 14, 14), (94080, 196, 14, 1), 70560), getitem_257, primals_229, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 120, [True, True, False])
        del getitem_257
        del primals_229
        buf340 = buf339[0]
        buf341 = buf339[1]
        del buf339
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf342 = aten.convolution_backward(reinterpret_tensor(buf337, (8, 120, 14, 14), (94080, 196, 14, 1), 47040), getitem_252, primals_228, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 120, [True, True, False])
        del getitem_252
        del primals_228
        buf343 = buf342[0]
        buf344 = buf342[1]
        del buf342
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf345 = aten.convolution_backward(reinterpret_tensor(buf337, (8, 120, 14, 14), (94080, 196, 14, 1), 23520), getitem_247, primals_227, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
        del getitem_247
        del primals_227
        buf346 = buf345[0]
        buf347 = buf345[1]
        del buf345
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf348 = aten.convolution_backward(reinterpret_tensor(buf337, (8, 120, 14, 14), (94080, 196, 14, 1), 0), getitem_242, primals_226, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 120, [True, True, False])
        del getitem_242
        del primals_226
        buf349 = buf348[0]
        buf350 = buf348[1]
        del buf348
        buf351 = buf337; del buf337  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul]
        triton_poi_fused_cat_mul_53.run(buf349, buf346, buf343, buf340, mul_748, buf351, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del buf340
        del buf343
        del buf346
        del buf349
        del mul_748
        buf352 = buf336; del buf336  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_49.run(buf351, buf352, 480, 1568, grid=grid(480), stream=stream0)
        buf353 = buf335; del buf335  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_50.run(buf351, cat_25, unsqueeze_486, buf353, 6240, 121, grid=grid(6240), stream=stream0)
        buf354 = empty((480, ), device='cuda', dtype=torch.float32)
        buf356 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_51.run(buf353, squeeze_109, buf354, buf356, 480, 13, grid=grid(480), stream=stream0)
        del buf353
        buf355 = buf351; del buf351  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_52.run(buf355, cat_25, unsqueeze_486, buf354, squeeze_109, buf352, primals_73, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del cat_25
        del primals_73
        del squeeze_109
        del unsqueeze_486
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf357 = aten.convolution_backward(reinterpret_tensor(buf355, (8, 240, 14, 14), (94080, 196, 14, 1), 47040), getitem_235, primals_225, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_235
        del primals_225
        buf358 = buf357[0]
        buf359 = buf357[1]
        del buf357
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf360 = aten.convolution_backward(reinterpret_tensor(buf355, (8, 240, 14, 14), (94080, 196, 14, 1), 0), getitem_234, primals_224, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf355
        del getitem_234
        del primals_224
        buf361 = buf360[0]
        buf362 = buf360[1]
        del buf360
        buf363 = buf313; del buf313  # reuse
        buf364 = empty((160, ), device='cuda', dtype=torch.float32)
        buf366 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_red_fused_add_cat_native_batch_norm_backward_54.run(buf310, buf361, buf358, convolution_88, unsqueeze_498, squeeze_106, buf363, buf364, buf366, 160, 1568, grid=grid(160), stream=stream0)
        buf365 = buf310; del buf310  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_poi_fused_add_cat_native_batch_norm_backward_57.run(buf365, buf361, buf358, convolution_88, unsqueeze_498, buf364, squeeze_106, buf363, primals_71, 1280, 196, grid=grid(1280, 196), stream=stream0)
        del buf358
        del buf361
        del convolution_88
        del primals_71
        del squeeze_106
        del unsqueeze_498
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf367 = aten.convolution_backward(buf365, mul_280, primals_223, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf365
        del mul_280
        del primals_223
        buf368 = buf367[0]
        buf369 = buf367[1]
        del buf367
        buf370 = empty_strided((8, 624, 1, 1, 2), (1248, 2, 9984, 9984, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_218], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_58.run(buf368, add_182, buf370, 9984, 98, grid=grid(9984), stream=stream0)
        buf371 = empty_strided((8, 624, 1, 1), (624, 1, 4992, 4992), device='cuda', dtype=torch.float32)
        buf372 = reinterpret_tensor(buf371, (8, 624, 1, 1), (624, 1, 1, 1), 0); del buf371  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_218], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_59.run(buf372, buf370, convolution_87, 4992, 2, grid=grid(4992), stream=stream0)
        del buf370
        buf373 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_60.run(buf372, buf373, 624, 8, grid=grid(624), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf374 = aten.convolution_backward(buf372, mul_279, primals_221, [624], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf372
        del mul_279
        del primals_221
        buf375 = buf374[0]
        buf376 = buf374[1]
        del buf374
        buf377 = buf375; del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_61.run(buf377, convolution_86, 416, grid=grid(416), stream=stream0)
        del convolution_86
        buf378 = empty((52, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_62.run(buf377, buf378, 52, 8, grid=grid(52), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf379 = aten.convolution_backward(buf377, mean_8, primals_219, [52], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_8
        del primals_219
        buf380 = buf379[0]
        buf381 = buf379[1]
        del buf379
        buf382 = empty((624, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_63.run(buf368, convolution_87, buf380, add_182, buf382, 8112, 121, grid=grid(8112), stream=stream0)
        buf383 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_64.run(buf382, buf383, 624, 13, grid=grid(624), stream=stream0)
        buf384 = reinterpret_tensor(buf382, (624, 13), (1, 624), 0); del buf382  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_65.run(buf368, convolution_87, buf380, add_182, convolution_85, unsqueeze_510, buf384, 8112, 121, grid=grid(8112), stream=stream0)
        buf385 = empty((624, ), device='cuda', dtype=torch.float32)
        buf387 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_66.run(buf384, squeeze_103, buf385, buf387, 624, 13, grid=grid(624), stream=stream0)
        buf386 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_67.run(buf368, convolution_87, buf380, add_182, convolution_85, unsqueeze_510, buf385, squeeze_103, buf383, buf386, 1568, 624, grid=grid(1568, 624), stream=stream0)
        del add_182
        del convolution_85
        del convolution_87
        del unsqueeze_510
        buf388 = buf368; del buf368  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_68.run(buf386, squeeze_103, primals_69, buf388, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del buf386
        del primals_69
        del squeeze_103
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf389 = aten.convolution_backward(buf388, mul_270, primals_218, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 624, [True, True, False])
        del buf388
        del mul_270
        del primals_218
        buf390 = buf389[0]
        buf391 = buf389[1]
        del buf389
        buf392 = reinterpret_tensor(buf384, (624, 13), (13, 1), 0); del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_69.run(buf390, mul_788, buf392, 8112, 121, grid=grid(8112), stream=stream0)
        buf393 = buf385; del buf385  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_64.run(buf392, buf393, 624, 13, grid=grid(624), stream=stream0)
        buf394 = reinterpret_tensor(buf392, (624, 13), (1, 624), 0); del buf392  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_70.run(buf390, mul_788, convolution_84, unsqueeze_522, buf394, 8112, 121, grid=grid(8112), stream=stream0)
        buf395 = empty((624, ), device='cuda', dtype=torch.float32)
        buf396 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_66.run(buf394, squeeze_100, buf395, buf396, 624, 13, grid=grid(624), stream=stream0)
        buf397 = buf390; del buf390  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_71.run(buf397, mul_788, convolution_84, unsqueeze_522, buf395, squeeze_100, buf393, primals_67, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del convolution_84
        del mul_788
        del primals_67
        del squeeze_100
        del unsqueeze_522
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf398 = aten.convolution_backward(buf397, add_172, primals_217, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_172
        del primals_217
        buf399 = buf398[0]
        buf400 = buf398[1]
        del buf398
        buf401 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_72.run(buf399, buf401, 104, 1568, grid=grid(104), stream=stream0)
        buf402 = empty((104, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_73.run(buf399, cat_24, unsqueeze_534, buf402, 1352, 121, grid=grid(1352), stream=stream0)
        buf403 = empty((104, ), device='cuda', dtype=torch.float32)
        buf405 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_74.run(buf402, squeeze_97, buf403, buf405, 104, 13, grid=grid(104), stream=stream0)
        buf404 = empty((8, 104, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_75.run(buf399, cat_24, unsqueeze_534, buf403, squeeze_97, buf401, primals_65, buf404, 832, 196, grid=grid(832, 196), stream=stream0)
        del cat_24
        del primals_65
        del squeeze_97
        del unsqueeze_534
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf406 = aten.convolution_backward(reinterpret_tensor(buf404, (8, 52, 14, 14), (20384, 196, 14, 1), 10192), getitem_225, primals_216, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_225
        del primals_216
        buf407 = buf406[0]
        buf408 = buf406[1]
        del buf406
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf409 = aten.convolution_backward(reinterpret_tensor(buf404, (8, 52, 14, 14), (20384, 196, 14, 1), 0), getitem_224, primals_215, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_224
        del primals_215
        buf410 = buf409[0]
        buf411 = buf409[1]
        del buf409
        buf412 = reinterpret_tensor(buf380, (8, 624, 1, 1), (624, 1, 4992, 4992), 0); del buf380  # reuse
        buf413 = reinterpret_tensor(buf412, (8, 624, 1, 1), (624, 1, 1, 1), 0); del buf412  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___se_gate, x_200], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_76.run(buf413, buf410, buf407, add_166, convolution_81, 4992, 196, grid=grid(4992), stream=stream0)
        buf414 = buf395; del buf395  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_60.run(buf413, buf414, 624, 8, grid=grid(624), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf415 = aten.convolution_backward(buf413, mul_254, primals_213, [624], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf413
        del mul_254
        del primals_213
        buf416 = buf415[0]
        buf417 = buf415[1]
        del buf415
        buf418 = buf416; del buf416  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_77.run(buf418, convolution_80, 208, grid=grid(208), stream=stream0)
        del convolution_80
        buf419 = empty((26, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_78.run(buf418, buf419, 26, 8, grid=grid(26), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf420 = aten.convolution_backward(buf418, mean_7, primals_211, [26], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf418
        del mean_7
        del primals_211
        buf421 = buf420[0]
        buf422 = buf420[1]
        del buf420
        buf423 = buf397; del buf397  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___se_gate], Original ATen: [aten.add, aten.cat, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_79.run(buf410, buf407, convolution_81, buf421, add_166, buf423, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del add_166
        del buf407
        del buf410
        del convolution_81
        buf424 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_80.run(buf423, buf424, 624, 1568, grid=grid(624), stream=stream0)
        buf425 = reinterpret_tensor(buf394, (624, 13), (13, 1), 0); del buf394  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_81.run(buf423, cat_23, unsqueeze_546, buf425, 8112, 121, grid=grid(8112), stream=stream0)
        buf426 = empty((624, ), device='cuda', dtype=torch.float32)
        buf428 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_82.run(buf425, squeeze_94, buf426, buf428, 624, 13, grid=grid(624), stream=stream0)
        buf427 = buf423; del buf423  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_83.run(buf427, cat_23, unsqueeze_546, buf426, squeeze_94, buf424, primals_63, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del cat_23
        del primals_63
        del squeeze_94
        del unsqueeze_546
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf429 = aten.convolution_backward(reinterpret_tensor(buf427, (8, 156, 14, 14), (122304, 196, 14, 1), 91728), getitem_221, primals_210, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 156, [True, True, False])
        del getitem_221
        del primals_210
        buf430 = buf429[0]
        buf431 = buf429[1]
        del buf429
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf432 = aten.convolution_backward(reinterpret_tensor(buf427, (8, 156, 14, 14), (122304, 196, 14, 1), 61152), getitem_216, primals_209, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 156, [True, True, False])
        del getitem_216
        del primals_209
        buf433 = buf432[0]
        buf434 = buf432[1]
        del buf432
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf435 = aten.convolution_backward(reinterpret_tensor(buf427, (8, 156, 14, 14), (122304, 196, 14, 1), 30576), getitem_211, primals_208, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 156, [True, True, False])
        del getitem_211
        del primals_208
        buf436 = buf435[0]
        buf437 = buf435[1]
        del buf435
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf438 = aten.convolution_backward(reinterpret_tensor(buf427, (8, 156, 14, 14), (122304, 196, 14, 1), 0), getitem_206, primals_207, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 156, [True, True, False])
        del getitem_206
        del primals_207
        buf439 = buf438[0]
        buf440 = buf438[1]
        del buf438
        buf441 = buf427; del buf427  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul]
        triton_poi_fused_cat_mul_84.run(buf439, buf436, buf433, buf430, mul_828, buf441, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del buf430
        del buf433
        del buf436
        del buf439
        del mul_828
        buf442 = buf426; del buf426  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_80.run(buf441, buf442, 624, 1568, grid=grid(624), stream=stream0)
        buf443 = buf425; del buf425  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_81.run(buf441, cat_22, unsqueeze_558, buf443, 8112, 121, grid=grid(8112), stream=stream0)
        buf444 = empty((624, ), device='cuda', dtype=torch.float32)
        buf446 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_82.run(buf443, squeeze_91, buf444, buf446, 624, 13, grid=grid(624), stream=stream0)
        buf445 = buf441; del buf441  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_83.run(buf445, cat_22, unsqueeze_558, buf444, squeeze_91, buf442, primals_61, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del cat_22
        del primals_61
        del squeeze_91
        del unsqueeze_558
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf447 = aten.convolution_backward(reinterpret_tensor(buf445, (8, 312, 14, 14), (122304, 196, 14, 1), 61152), getitem_199, primals_206, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_199
        del primals_206
        buf448 = buf447[0]
        buf449 = buf447[1]
        del buf447
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf450 = aten.convolution_backward(reinterpret_tensor(buf445, (8, 312, 14, 14), (122304, 196, 14, 1), 0), getitem_198, primals_205, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_198
        del primals_205
        buf451 = buf450[0]
        buf452 = buf450[1]
        del buf450
        buf453 = buf403; del buf403  # reuse
        buf454 = empty((104, ), device='cuda', dtype=torch.float32)
        buf456 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_red_fused_add_cat_native_batch_norm_backward_85.run(buf399, buf451, buf448, cat_21, unsqueeze_570, squeeze_88, buf453, buf454, buf456, 104, 1568, grid=grid(104), stream=stream0)
        buf455 = buf404; del buf404  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_poi_fused_add_cat_native_batch_norm_backward_86.run(buf399, buf451, buf448, cat_21, unsqueeze_570, buf454, squeeze_88, buf453, primals_59, buf455, 832, 196, grid=grid(832, 196), stream=stream0)
        del cat_21
        del primals_59
        del squeeze_88
        del unsqueeze_570
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf457 = aten.convolution_backward(reinterpret_tensor(buf455, (8, 52, 14, 14), (20384, 196, 14, 1), 10192), getitem_195, primals_204, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_195
        del primals_204
        buf458 = buf457[0]
        buf459 = buf457[1]
        del buf457
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf460 = aten.convolution_backward(reinterpret_tensor(buf455, (8, 52, 14, 14), (20384, 196, 14, 1), 0), getitem_194, primals_203, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_194
        del primals_203
        buf461 = buf460[0]
        buf462 = buf460[1]
        del buf460
        buf463 = reinterpret_tensor(buf421, (8, 624, 1, 1), (624, 1, 4992, 4992), 0); del buf421  # reuse
        buf464 = reinterpret_tensor(buf463, (8, 624, 1, 1), (624, 1, 1, 1), 0); del buf463  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate, x_180], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_76.run(buf464, buf461, buf458, add_150, convolution_71, 4992, 196, grid=grid(4992), stream=stream0)
        buf465 = buf444; del buf444  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_60.run(buf464, buf465, 624, 8, grid=grid(624), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf466 = aten.convolution_backward(buf464, mul_229, primals_201, [624], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf464
        del mul_229
        del primals_201
        buf467 = buf466[0]
        buf468 = buf466[1]
        del buf466
        buf469 = buf467; del buf467  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_77.run(buf469, convolution_70, 208, grid=grid(208), stream=stream0)
        del convolution_70
        buf470 = empty((26, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_78.run(buf469, buf470, 26, 8, grid=grid(26), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf471 = aten.convolution_backward(buf469, mean_6, primals_199, [26], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf469
        del mean_6
        del primals_199
        buf472 = buf471[0]
        buf473 = buf471[1]
        del buf471
        buf474 = buf445; del buf445  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate], Original ATen: [aten.add, aten.cat, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_79.run(buf461, buf458, convolution_71, buf472, add_150, buf474, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del add_150
        del buf458
        del buf461
        del convolution_71
        buf475 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_80.run(buf474, buf475, 624, 1568, grid=grid(624), stream=stream0)
        buf476 = buf443; del buf443  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_81.run(buf474, cat_20, unsqueeze_582, buf476, 8112, 121, grid=grid(8112), stream=stream0)
        buf477 = empty((624, ), device='cuda', dtype=torch.float32)
        buf479 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_82.run(buf476, squeeze_85, buf477, buf479, 624, 13, grid=grid(624), stream=stream0)
        buf478 = buf474; del buf474  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_83.run(buf478, cat_20, unsqueeze_582, buf477, squeeze_85, buf475, primals_57, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del cat_20
        del primals_57
        del squeeze_85
        del unsqueeze_582
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf480 = aten.convolution_backward(reinterpret_tensor(buf478, (8, 156, 14, 14), (122304, 196, 14, 1), 91728), getitem_191, primals_198, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 156, [True, True, False])
        del getitem_191
        del primals_198
        buf481 = buf480[0]
        buf482 = buf480[1]
        del buf480
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf483 = aten.convolution_backward(reinterpret_tensor(buf478, (8, 156, 14, 14), (122304, 196, 14, 1), 61152), getitem_186, primals_197, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 156, [True, True, False])
        del getitem_186
        del primals_197
        buf484 = buf483[0]
        buf485 = buf483[1]
        del buf483
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf486 = aten.convolution_backward(reinterpret_tensor(buf478, (8, 156, 14, 14), (122304, 196, 14, 1), 30576), getitem_181, primals_196, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 156, [True, True, False])
        del getitem_181
        del primals_196
        buf487 = buf486[0]
        buf488 = buf486[1]
        del buf486
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf489 = aten.convolution_backward(reinterpret_tensor(buf478, (8, 156, 14, 14), (122304, 196, 14, 1), 0), getitem_176, primals_195, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 156, [True, True, False])
        del getitem_176
        del primals_195
        buf490 = buf489[0]
        buf491 = buf489[1]
        del buf489
        buf492 = buf478; del buf478  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul]
        triton_poi_fused_cat_mul_84.run(buf490, buf487, buf484, buf481, mul_868, buf492, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del buf481
        del buf484
        del buf487
        del buf490
        del mul_868
        buf493 = buf477; del buf477  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_80.run(buf492, buf493, 624, 1568, grid=grid(624), stream=stream0)
        buf494 = buf476; del buf476  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_81.run(buf492, cat_19, unsqueeze_594, buf494, 8112, 121, grid=grid(8112), stream=stream0)
        buf495 = empty((624, ), device='cuda', dtype=torch.float32)
        buf497 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_82.run(buf494, squeeze_82, buf495, buf497, 624, 13, grid=grid(624), stream=stream0)
        buf496 = buf492; del buf492  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_83.run(buf496, cat_19, unsqueeze_594, buf495, squeeze_82, buf493, primals_55, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del cat_19
        del primals_55
        del squeeze_82
        del unsqueeze_594
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf498 = aten.convolution_backward(reinterpret_tensor(buf496, (8, 312, 14, 14), (122304, 196, 14, 1), 61152), getitem_169, primals_194, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_169
        del primals_194
        buf499 = buf498[0]
        buf500 = buf498[1]
        del buf498
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf501 = aten.convolution_backward(reinterpret_tensor(buf496, (8, 312, 14, 14), (122304, 196, 14, 1), 0), getitem_168, primals_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_168
        del primals_193
        buf502 = buf501[0]
        buf503 = buf501[1]
        del buf501
        buf504 = buf399; del buf399  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_87.run(buf504, buf451, buf448, buf502, buf499, 163072, grid=grid(163072), stream=stream0)
        del buf448
        del buf451
        del buf499
        del buf502
        buf505 = buf454; del buf454  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_72.run(buf504, buf505, 104, 1568, grid=grid(104), stream=stream0)
        buf506 = buf402; del buf402  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_73.run(buf504, cat_18, unsqueeze_606, buf506, 1352, 121, grid=grid(1352), stream=stream0)
        buf507 = empty((104, ), device='cuda', dtype=torch.float32)
        buf509 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_74.run(buf506, squeeze_79, buf507, buf509, 104, 13, grid=grid(104), stream=stream0)
        del buf506
        buf508 = buf455; del buf455  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_75.run(buf504, cat_18, unsqueeze_606, buf507, squeeze_79, buf505, primals_53, buf508, 832, 196, grid=grid(832, 196), stream=stream0)
        del cat_18
        del primals_53
        del squeeze_79
        del unsqueeze_606
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf510 = aten.convolution_backward(reinterpret_tensor(buf508, (8, 52, 14, 14), (20384, 196, 14, 1), 10192), getitem_165, primals_192, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_165
        del primals_192
        buf511 = buf510[0]
        buf512 = buf510[1]
        del buf510
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf513 = aten.convolution_backward(reinterpret_tensor(buf508, (8, 52, 14, 14), (20384, 196, 14, 1), 0), getitem_164, primals_191, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf508
        del getitem_164
        del primals_191
        buf514 = buf513[0]
        buf515 = buf513[1]
        del buf513
        buf516 = reinterpret_tensor(buf472, (8, 624, 1, 1), (624, 1, 4992, 4992), 0); del buf472  # reuse
        buf517 = reinterpret_tensor(buf516, (8, 624, 1, 1), (624, 1, 1, 1), 0); del buf516  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate, x_160], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_76.run(buf517, buf514, buf511, add_134, convolution_61, 4992, 196, grid=grid(4992), stream=stream0)
        buf518 = buf495; del buf495  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_60.run(buf517, buf518, 624, 8, grid=grid(624), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf519 = aten.convolution_backward(buf517, mul_204, primals_189, [624], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf517
        del mul_204
        del primals_189
        buf520 = buf519[0]
        buf521 = buf519[1]
        del buf519
        buf522 = buf520; del buf520  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_77.run(buf522, convolution_60, 208, grid=grid(208), stream=stream0)
        del convolution_60
        buf523 = empty((26, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_78.run(buf522, buf523, 26, 8, grid=grid(26), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf524 = aten.convolution_backward(buf522, mean_5, primals_187, [26], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf522
        del mean_5
        del primals_187
        buf525 = buf524[0]
        buf526 = buf524[1]
        del buf524
        buf527 = buf496; del buf496  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate], Original ATen: [aten.add, aten.cat, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_79.run(buf514, buf511, convolution_61, buf525, add_134, buf527, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del add_134
        del buf511
        del buf514
        del buf525
        del convolution_61
        buf528 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_80.run(buf527, buf528, 624, 1568, grid=grid(624), stream=stream0)
        buf529 = buf494; del buf494  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_81.run(buf527, cat_17, unsqueeze_618, buf529, 8112, 121, grid=grid(8112), stream=stream0)
        buf530 = empty((624, ), device='cuda', dtype=torch.float32)
        buf532 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_82.run(buf529, squeeze_76, buf530, buf532, 624, 13, grid=grid(624), stream=stream0)
        buf531 = buf527; del buf527  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_83.run(buf531, cat_17, unsqueeze_618, buf530, squeeze_76, buf528, primals_51, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del cat_17
        del primals_51
        del squeeze_76
        del unsqueeze_618
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf533 = aten.convolution_backward(reinterpret_tensor(buf531, (8, 156, 14, 14), (122304, 196, 14, 1), 91728), getitem_161, primals_186, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 156, [True, True, False])
        del getitem_161
        del primals_186
        buf534 = buf533[0]
        buf535 = buf533[1]
        del buf533
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf536 = aten.convolution_backward(reinterpret_tensor(buf531, (8, 156, 14, 14), (122304, 196, 14, 1), 61152), getitem_156, primals_185, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 156, [True, True, False])
        del getitem_156
        del primals_185
        buf537 = buf536[0]
        buf538 = buf536[1]
        del buf536
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf539 = aten.convolution_backward(reinterpret_tensor(buf531, (8, 156, 14, 14), (122304, 196, 14, 1), 30576), getitem_151, primals_184, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 156, [True, True, False])
        del getitem_151
        del primals_184
        buf540 = buf539[0]
        buf541 = buf539[1]
        del buf539
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf542 = aten.convolution_backward(reinterpret_tensor(buf531, (8, 156, 14, 14), (122304, 196, 14, 1), 0), getitem_146, primals_183, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 156, [True, True, False])
        del getitem_146
        del primals_183
        buf543 = buf542[0]
        buf544 = buf542[1]
        del buf542
        buf545 = buf531; del buf531  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul]
        triton_poi_fused_cat_mul_84.run(buf543, buf540, buf537, buf534, mul_908, buf545, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del buf534
        del buf537
        del buf540
        del buf543
        del mul_908
        buf546 = buf530; del buf530  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_80.run(buf545, buf546, 624, 1568, grid=grid(624), stream=stream0)
        buf547 = buf529; del buf529  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_81.run(buf545, cat_16, unsqueeze_630, buf547, 8112, 121, grid=grid(8112), stream=stream0)
        buf548 = empty((624, ), device='cuda', dtype=torch.float32)
        buf550 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_82.run(buf547, squeeze_73, buf548, buf550, 624, 13, grid=grid(624), stream=stream0)
        del buf547
        buf549 = buf545; del buf545  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_83.run(buf549, cat_16, unsqueeze_630, buf548, squeeze_73, buf546, primals_49, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del buf548
        del cat_16
        del primals_49
        del squeeze_73
        del unsqueeze_630
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf551 = aten.convolution_backward(reinterpret_tensor(buf549, (8, 312, 14, 14), (122304, 196, 14, 1), 61152), getitem_139, primals_182, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_139
        del primals_182
        buf552 = buf551[0]
        buf553 = buf551[1]
        del buf551
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf554 = aten.convolution_backward(reinterpret_tensor(buf549, (8, 312, 14, 14), (122304, 196, 14, 1), 0), getitem_138, primals_181, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf549
        del getitem_138
        del primals_181
        buf555 = buf554[0]
        buf556 = buf554[1]
        del buf554
        buf557 = buf507; del buf507  # reuse
        buf558 = empty((104, ), device='cuda', dtype=torch.float32)
        buf560 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_red_fused_add_cat_native_batch_norm_backward_85.run(buf504, buf555, buf552, convolution_53, unsqueeze_642, squeeze_70, buf557, buf558, buf560, 104, 1568, grid=grid(104), stream=stream0)
        buf559 = buf504; del buf504  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_poi_fused_add_cat_native_batch_norm_backward_88.run(buf559, buf555, buf552, convolution_53, unsqueeze_642, buf558, squeeze_70, buf557, primals_47, 832, 196, grid=grid(832, 196), stream=stream0)
        del buf552
        del buf555
        del buf558
        del convolution_53
        del primals_47
        del squeeze_70
        del unsqueeze_642
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf561 = aten.convolution_backward(buf559, mul_180, primals_180, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf559
        del mul_180
        del primals_180
        buf562 = buf561[0]
        buf563 = buf561[1]
        del buf561
        buf564 = empty_strided((8, 336, 1, 1, 2), (672, 2, 5376, 5376, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_142], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_89.run(buf562, add_119, buf564, 5376, 98, grid=grid(5376), stream=stream0)
        buf565 = empty_strided((8, 336, 1, 1), (336, 1, 2688, 2688), device='cuda', dtype=torch.float32)
        buf566 = reinterpret_tensor(buf565, (8, 336, 1, 1), (336, 1, 1, 1), 0); del buf565  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_142], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_90.run(buf566, buf564, convolution_52, 2688, 2, grid=grid(2688), stream=stream0)
        del buf564
        buf567 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_91.run(buf566, buf567, 336, 8, grid=grid(336), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf568 = aten.convolution_backward(buf566, mul_179, primals_178, [336], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf566
        del mul_179
        del primals_178
        buf569 = buf568[0]
        buf570 = buf568[1]
        del buf568
        buf571 = buf569; del buf569  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_92.run(buf571, convolution_51, 112, grid=grid(112), stream=stream0)
        del convolution_51
        buf572 = empty((14, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_93.run(buf571, buf572, 14, 8, grid=grid(14), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf573 = aten.convolution_backward(buf571, mean_4, primals_176, [14], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf571
        del mean_4
        del primals_176
        buf574 = buf573[0]
        buf575 = buf573[1]
        del buf573
        buf576 = empty((336, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_94.run(buf562, convolution_52, buf574, add_119, buf576, 4368, 121, grid=grid(4368), stream=stream0)
        buf577 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_95.run(buf576, buf577, 336, 13, grid=grid(336), stream=stream0)
        buf578 = reinterpret_tensor(buf576, (336, 13), (1, 336), 0); del buf576  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_96.run(buf562, convolution_52, buf574, add_119, cat_15, unsqueeze_654, buf578, 4368, 121, grid=grid(4368), stream=stream0)
        buf579 = empty((336, ), device='cuda', dtype=torch.float32)
        buf581 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_97.run(buf578, squeeze_67, buf579, buf581, 336, 13, grid=grid(336), stream=stream0)
        del buf578
        buf580 = empty_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_98.run(buf562, convolution_52, buf574, add_119, cat_15, unsqueeze_654, buf579, squeeze_67, buf577, buf580, 1568, 336, grid=grid(1568, 336), stream=stream0)
        del add_119
        del buf562
        del cat_15
        del convolution_52
        del unsqueeze_654
        buf582 = empty((8, 112, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_99.run(buf580, squeeze_67, primals_45, buf582, 896, 196, grid=grid(896, 196), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf583 = aten.convolution_backward(buf582, getitem_133, primals_175, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 112, [True, True, False])
        del getitem_133
        del primals_175
        buf584 = buf583[0]
        buf585 = buf583[1]
        del buf583
        buf586 = buf582; del buf582  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_100.run(buf580, squeeze_67, primals_45, buf586, 896, 196, grid=grid(896, 196), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf587 = aten.convolution_backward(buf586, getitem_129, primals_174, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 112, [True, True, False])
        del getitem_129
        del primals_174
        buf588 = buf587[0]
        buf589 = buf587[1]
        del buf587
        buf590 = buf586; del buf586  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_101.run(buf580, squeeze_67, primals_45, buf590, 896, 196, grid=grid(896, 196), stream=stream0)
        del buf580
        del primals_45
        del squeeze_67
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf591 = aten.convolution_backward(buf590, getitem_125, primals_173, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 112, [True, True, False])
        del buf590
        del getitem_125
        del primals_173
        buf592 = buf591[0]
        buf593 = buf591[1]
        del buf591
        buf594 = buf579; del buf579  # reuse
        buf595 = empty((336, ), device='cuda', dtype=torch.float32)
        buf597 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_cat_mul_native_batch_norm_backward_102.run(buf592, buf588, buf584, mul_948, convolution_47, unsqueeze_666, squeeze_64, buf594, buf595, buf597, 336, 6272, grid=grid(336), stream=stream0)
        buf596 = empty((8, 336, 28, 28), device='cuda', dtype=torch.float32)
        buf598 = buf596; del buf596  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_cat_convolution_backward_mul_native_batch_norm_backward_103.run(buf598, buf592, buf588, buf584, mul_948, convolution_47, unsqueeze_666, buf595, squeeze_64, buf594, primals_43, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del buf584
        del buf588
        del buf592
        del convolution_47
        del mul_948
        del primals_43
        del squeeze_64
        del unsqueeze_666
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf599 = aten.convolution_backward(buf598, add_109, primals_172, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_109
        del primals_172
        buf600 = buf599[0]
        buf601 = buf599[1]
        del buf599
        buf602 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_104.run(buf600, buf602, 56, 6272, grid=grid(56), stream=stream0)
        buf603 = empty((56, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_105.run(buf600, cat_14, unsqueeze_678, buf603, 2744, 128, grid=grid(2744), stream=stream0)
        buf604 = empty((56, ), device='cuda', dtype=torch.float32)
        buf606 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_106.run(buf603, squeeze_61, buf604, buf606, 56, 49, grid=grid(56), stream=stream0)
        buf605 = empty((8, 56, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_107.run(buf600, cat_14, unsqueeze_678, buf604, squeeze_61, buf602, primals_41, buf605, 448, 784, grid=grid(448, 784), stream=stream0)
        del cat_14
        del primals_41
        del squeeze_61
        del unsqueeze_678
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf607 = aten.convolution_backward(reinterpret_tensor(buf605, (8, 28, 28, 28), (43904, 784, 28, 1), 21952), getitem_117, primals_171, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_117
        del primals_171
        buf608 = buf607[0]
        buf609 = buf607[1]
        del buf607
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf610 = aten.convolution_backward(reinterpret_tensor(buf605, (8, 28, 28, 28), (43904, 784, 28, 1), 0), getitem_116, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_116
        del primals_170
        buf611 = buf610[0]
        buf612 = buf610[1]
        del buf610
        buf613 = reinterpret_tensor(buf574, (8, 336, 1, 1), (336, 1, 2688, 2688), 0); del buf574  # reuse
        buf614 = reinterpret_tensor(buf613, (8, 336, 1, 1), (336, 1, 1, 1), 0); del buf613  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___se_gate, x_123], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_108.run(buf614, buf611, buf608, add_103, convolution_44, 2688, 784, grid=grid(2688), stream=stream0)
        buf615 = buf595; del buf595  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_91.run(buf614, buf615, 336, 8, grid=grid(336), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf616 = aten.convolution_backward(buf614, mul_154, primals_168, [336], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf614
        del mul_154
        del primals_168
        buf617 = buf616[0]
        buf618 = buf616[1]
        del buf616
        buf619 = buf617; del buf617  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_109.run(buf619, convolution_43, 224, grid=grid(224), stream=stream0)
        del convolution_43
        buf620 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_110.run(buf619, buf620, 28, 8, grid=grid(28), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf621 = aten.convolution_backward(buf619, mean_3, primals_166, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf619
        del mean_3
        del primals_166
        buf622 = buf621[0]
        buf623 = buf621[1]
        del buf621
        buf624 = buf598; del buf598  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___se_gate], Original ATen: [aten.add, aten.cat, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_111.run(buf611, buf608, convolution_44, buf622, add_103, buf624, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del add_103
        del buf608
        del buf611
        del convolution_44
        buf625 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_112.run(buf624, buf625, 336, 6272, grid=grid(336), stream=stream0)
        buf626 = empty((336, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_113.run(buf624, cat_13, unsqueeze_690, buf626, 16464, 128, grid=grid(16464), stream=stream0)
        buf627 = empty((336, ), device='cuda', dtype=torch.float32)
        buf629 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_114.run(buf626, squeeze_58, buf627, buf629, 336, 49, grid=grid(336), stream=stream0)
        buf628 = buf624; del buf624  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_115.run(buf628, cat_13, unsqueeze_690, buf627, squeeze_58, buf625, primals_39, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del cat_13
        del primals_39
        del squeeze_58
        del unsqueeze_690
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf630 = aten.convolution_backward(reinterpret_tensor(buf628, (8, 168, 28, 28), (263424, 784, 28, 1), 131712), getitem_113, primals_165, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 168, [True, True, False])
        del getitem_113
        del primals_165
        buf631 = buf630[0]
        buf632 = buf630[1]
        del buf630
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf633 = aten.convolution_backward(reinterpret_tensor(buf628, (8, 168, 28, 28), (263424, 784, 28, 1), 0), getitem_110, primals_164, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 168, [True, True, False])
        del getitem_110
        del primals_164
        buf634 = buf633[0]
        buf635 = buf633[1]
        del buf633
        buf636 = buf627; del buf627  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_cat_mul_native_batch_norm_backward_116.run(buf634, buf631, mul_988, buf636, 336, 6272, grid=grid(336), stream=stream0)
        buf637 = buf626; del buf626  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_cat_mul_native_batch_norm_backward_117.run(buf634, buf631, mul_988, cat_12, unsqueeze_702, buf637, 16464, 128, grid=grid(16464), stream=stream0)
        buf638 = empty((336, ), device='cuda', dtype=torch.float32)
        buf640 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_114.run(buf637, squeeze_55, buf638, buf640, 336, 49, grid=grid(336), stream=stream0)
        buf639 = buf628; del buf628  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_cat_mul_native_batch_norm_backward_118.run(buf634, buf631, mul_988, cat_12, unsqueeze_702, buf638, squeeze_55, buf636, primals_37, buf639, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del buf631
        del buf634
        del cat_12
        del mul_988
        del primals_37
        del squeeze_55
        del unsqueeze_702
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf641 = aten.convolution_backward(reinterpret_tensor(buf639, (8, 168, 28, 28), (263424, 784, 28, 1), 131712), getitem_105, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_105
        del primals_163
        buf642 = buf641[0]
        buf643 = buf641[1]
        del buf641
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf644 = aten.convolution_backward(reinterpret_tensor(buf639, (8, 168, 28, 28), (263424, 784, 28, 1), 0), getitem_104, primals_162, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_104
        del primals_162
        buf645 = buf644[0]
        buf646 = buf644[1]
        del buf644
        buf647 = buf604; del buf604  # reuse
        buf648 = empty((56, ), device='cuda', dtype=torch.float32)
        buf650 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_red_fused_add_cat_native_batch_norm_backward_119.run(buf600, buf645, buf642, cat_11, unsqueeze_714, squeeze_52, buf647, buf648, buf650, 56, 6272, grid=grid(56), stream=stream0)
        buf649 = buf605; del buf605  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_poi_fused_add_cat_native_batch_norm_backward_120.run(buf600, buf645, buf642, cat_11, unsqueeze_714, buf648, squeeze_52, buf647, primals_35, buf649, 448, 784, grid=grid(448, 784), stream=stream0)
        del cat_11
        del primals_35
        del squeeze_52
        del unsqueeze_714
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf651 = aten.convolution_backward(reinterpret_tensor(buf649, (8, 28, 28, 28), (43904, 784, 28, 1), 21952), getitem_101, primals_161, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_101
        del primals_161
        buf652 = buf651[0]
        buf653 = buf651[1]
        del buf651
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf654 = aten.convolution_backward(reinterpret_tensor(buf649, (8, 28, 28, 28), (43904, 784, 28, 1), 0), getitem_100, primals_160, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_100
        del primals_160
        buf655 = buf654[0]
        buf656 = buf654[1]
        del buf654
        buf657 = reinterpret_tensor(buf622, (8, 336, 1, 1), (336, 1, 2688, 2688), 0); del buf622  # reuse
        buf658 = reinterpret_tensor(buf657, (8, 336, 1, 1), (336, 1, 1, 1), 0); del buf657  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___se_gate, x_103], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_108.run(buf658, buf655, buf652, add_87, convolution_36, 2688, 784, grid=grid(2688), stream=stream0)
        buf659 = buf638; del buf638  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_91.run(buf658, buf659, 336, 8, grid=grid(336), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf660 = aten.convolution_backward(buf658, mul_129, primals_158, [336], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf658
        del mul_129
        del primals_158
        buf661 = buf660[0]
        buf662 = buf660[1]
        del buf660
        buf663 = buf661; del buf661  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_109.run(buf663, convolution_35, 224, grid=grid(224), stream=stream0)
        del convolution_35
        buf664 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_110.run(buf663, buf664, 28, 8, grid=grid(28), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf665 = aten.convolution_backward(buf663, mean_2, primals_156, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf663
        del mean_2
        del primals_156
        buf666 = buf665[0]
        buf667 = buf665[1]
        del buf665
        buf668 = buf639; del buf639  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___se_gate], Original ATen: [aten.add, aten.cat, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_111.run(buf655, buf652, convolution_36, buf666, add_87, buf668, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del add_87
        del buf652
        del buf655
        del convolution_36
        buf669 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_112.run(buf668, buf669, 336, 6272, grid=grid(336), stream=stream0)
        buf670 = buf637; del buf637  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_113.run(buf668, cat_10, unsqueeze_726, buf670, 16464, 128, grid=grid(16464), stream=stream0)
        buf671 = empty((336, ), device='cuda', dtype=torch.float32)
        buf673 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_114.run(buf670, squeeze_49, buf671, buf673, 336, 49, grid=grid(336), stream=stream0)
        buf672 = buf668; del buf668  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_115.run(buf672, cat_10, unsqueeze_726, buf671, squeeze_49, buf669, primals_33, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del cat_10
        del primals_33
        del squeeze_49
        del unsqueeze_726
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf674 = aten.convolution_backward(reinterpret_tensor(buf672, (8, 168, 28, 28), (263424, 784, 28, 1), 131712), getitem_97, primals_155, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 168, [True, True, False])
        del getitem_97
        del primals_155
        buf675 = buf674[0]
        buf676 = buf674[1]
        del buf674
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf677 = aten.convolution_backward(reinterpret_tensor(buf672, (8, 168, 28, 28), (263424, 784, 28, 1), 0), getitem_94, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 168, [True, True, False])
        del getitem_94
        del primals_154
        buf678 = buf677[0]
        buf679 = buf677[1]
        del buf677
        buf680 = buf671; del buf671  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_cat_mul_native_batch_norm_backward_116.run(buf678, buf675, mul_1028, buf680, 336, 6272, grid=grid(336), stream=stream0)
        buf681 = buf670; del buf670  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_cat_mul_native_batch_norm_backward_117.run(buf678, buf675, mul_1028, cat_9, unsqueeze_738, buf681, 16464, 128, grid=grid(16464), stream=stream0)
        buf682 = empty((336, ), device='cuda', dtype=torch.float32)
        buf684 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_114.run(buf681, squeeze_46, buf682, buf684, 336, 49, grid=grid(336), stream=stream0)
        buf683 = buf672; del buf672  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_cat_mul_native_batch_norm_backward_118.run(buf678, buf675, mul_1028, cat_9, unsqueeze_738, buf682, squeeze_46, buf680, primals_31, buf683, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del buf675
        del buf678
        del cat_9
        del mul_1028
        del primals_31
        del squeeze_46
        del unsqueeze_738
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf685 = aten.convolution_backward(reinterpret_tensor(buf683, (8, 168, 28, 28), (263424, 784, 28, 1), 131712), getitem_89, primals_153, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_89
        del primals_153
        buf686 = buf685[0]
        buf687 = buf685[1]
        del buf685
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf688 = aten.convolution_backward(reinterpret_tensor(buf683, (8, 168, 28, 28), (263424, 784, 28, 1), 0), getitem_88, primals_152, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_88
        del primals_152
        buf689 = buf688[0]
        buf690 = buf688[1]
        del buf688
        buf691 = buf600; del buf600  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_121.run(buf691, buf645, buf642, buf689, buf686, 351232, grid=grid(351232), stream=stream0)
        del buf642
        del buf645
        del buf686
        del buf689
        buf692 = buf648; del buf648  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_104.run(buf691, buf692, 56, 6272, grid=grid(56), stream=stream0)
        buf693 = buf603; del buf603  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_105.run(buf691, cat_8, unsqueeze_750, buf693, 2744, 128, grid=grid(2744), stream=stream0)
        buf694 = empty((56, ), device='cuda', dtype=torch.float32)
        buf696 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_106.run(buf693, squeeze_43, buf694, buf696, 56, 49, grid=grid(56), stream=stream0)
        del buf693
        buf695 = buf649; del buf649  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_107.run(buf691, cat_8, unsqueeze_750, buf694, squeeze_43, buf692, primals_29, buf695, 448, 784, grid=grid(448, 784), stream=stream0)
        del cat_8
        del primals_29
        del squeeze_43
        del unsqueeze_750
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf697 = aten.convolution_backward(reinterpret_tensor(buf695, (8, 28, 28, 28), (43904, 784, 28, 1), 21952), getitem_85, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_85
        del primals_151
        buf698 = buf697[0]
        buf699 = buf697[1]
        del buf697
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf700 = aten.convolution_backward(reinterpret_tensor(buf695, (8, 28, 28, 28), (43904, 784, 28, 1), 0), getitem_84, primals_150, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf695
        del getitem_84
        del primals_150
        buf701 = buf700[0]
        buf702 = buf700[1]
        del buf700
        buf703 = reinterpret_tensor(buf666, (8, 336, 1, 1), (336, 1, 2688, 2688), 0); del buf666  # reuse
        buf704 = reinterpret_tensor(buf703, (8, 336, 1, 1), (336, 1, 1, 1), 0); del buf703  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_83], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_108.run(buf704, buf701, buf698, add_71, convolution_28, 2688, 784, grid=grid(2688), stream=stream0)
        buf705 = buf682; del buf682  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_91.run(buf704, buf705, 336, 8, grid=grid(336), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf706 = aten.convolution_backward(buf704, mul_104, primals_148, [336], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf704
        del mul_104
        del primals_148
        buf707 = buf706[0]
        buf708 = buf706[1]
        del buf706
        buf709 = buf707; del buf707  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_109.run(buf709, convolution_27, 224, grid=grid(224), stream=stream0)
        del convolution_27
        buf710 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_110.run(buf709, buf710, 28, 8, grid=grid(28), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf711 = aten.convolution_backward(buf709, mean_1, primals_146, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf709
        del mean_1
        del primals_146
        buf712 = buf711[0]
        buf713 = buf711[1]
        del buf711
        buf714 = buf683; del buf683  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate], Original ATen: [aten.add, aten.cat, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_111.run(buf701, buf698, convolution_28, buf712, add_71, buf714, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del add_71
        del buf698
        del buf701
        del buf712
        del convolution_28
        buf715 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_112.run(buf714, buf715, 336, 6272, grid=grid(336), stream=stream0)
        buf716 = buf681; del buf681  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_113.run(buf714, cat_7, unsqueeze_762, buf716, 16464, 128, grid=grid(16464), stream=stream0)
        buf717 = empty((336, ), device='cuda', dtype=torch.float32)
        buf719 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_114.run(buf716, squeeze_40, buf717, buf719, 336, 49, grid=grid(336), stream=stream0)
        buf718 = buf714; del buf714  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_115.run(buf718, cat_7, unsqueeze_762, buf717, squeeze_40, buf715, primals_27, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del cat_7
        del primals_27
        del squeeze_40
        del unsqueeze_762
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf720 = aten.convolution_backward(reinterpret_tensor(buf718, (8, 168, 28, 28), (263424, 784, 28, 1), 131712), getitem_81, primals_145, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 168, [True, True, False])
        del getitem_81
        del primals_145
        buf721 = buf720[0]
        buf722 = buf720[1]
        del buf720
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf723 = aten.convolution_backward(reinterpret_tensor(buf718, (8, 168, 28, 28), (263424, 784, 28, 1), 0), getitem_78, primals_144, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 168, [True, True, False])
        del getitem_78
        del primals_144
        buf724 = buf723[0]
        buf725 = buf723[1]
        del buf723
        buf726 = buf717; del buf717  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_cat_mul_native_batch_norm_backward_116.run(buf724, buf721, mul_1068, buf726, 336, 6272, grid=grid(336), stream=stream0)
        buf727 = buf716; del buf716  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_cat_mul_native_batch_norm_backward_117.run(buf724, buf721, mul_1068, cat_6, unsqueeze_774, buf727, 16464, 128, grid=grid(16464), stream=stream0)
        buf728 = empty((336, ), device='cuda', dtype=torch.float32)
        buf730 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_114.run(buf727, squeeze_37, buf728, buf730, 336, 49, grid=grid(336), stream=stream0)
        del buf727
        buf729 = buf718; del buf718  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_cat_mul_native_batch_norm_backward_118.run(buf724, buf721, mul_1068, cat_6, unsqueeze_774, buf728, squeeze_37, buf726, primals_25, buf729, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del buf721
        del buf724
        del buf728
        del cat_6
        del mul_1068
        del primals_25
        del squeeze_37
        del unsqueeze_774
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf731 = aten.convolution_backward(reinterpret_tensor(buf729, (8, 168, 28, 28), (263424, 784, 28, 1), 131712), getitem_73, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_73
        del primals_143
        buf732 = buf731[0]
        buf733 = buf731[1]
        del buf731
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf734 = aten.convolution_backward(reinterpret_tensor(buf729, (8, 168, 28, 28), (263424, 784, 28, 1), 0), getitem_72, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf729
        del getitem_72
        del primals_142
        buf735 = buf734[0]
        buf736 = buf734[1]
        del buf734
        buf737 = buf694; del buf694  # reuse
        buf738 = empty((56, ), device='cuda', dtype=torch.float32)
        buf740 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_red_fused_add_cat_native_batch_norm_backward_119.run(buf691, buf735, buf732, convolution_22, unsqueeze_786, squeeze_34, buf737, buf738, buf740, 56, 6272, grid=grid(56), stream=stream0)
        buf739 = buf691; del buf691  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_poi_fused_add_cat_native_batch_norm_backward_122.run(buf739, buf735, buf732, convolution_22, unsqueeze_786, buf738, squeeze_34, buf737, primals_23, 448, 784, grid=grid(448, 784), stream=stream0)
        del buf732
        del buf735
        del buf738
        del convolution_22
        del primals_23
        del squeeze_34
        del unsqueeze_786
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf741 = aten.convolution_backward(buf739, mul_80, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf739
        del mul_80
        del primals_141
        buf742 = buf741[0]
        buf743 = buf741[1]
        del buf741
        buf744 = empty_strided((8, 240, 1, 1, 7), (1680, 7, 13440, 13440, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_65], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_123.run(buf742, add_56, buf744, 13440, 112, grid=grid(13440), stream=stream0)
        buf745 = empty_strided((8, 240, 1, 1), (240, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf746 = reinterpret_tensor(buf745, (8, 240, 1, 1), (240, 1, 1, 1), 0); del buf745  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_65], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_124.run(buf746, buf744, convolution_21, 1920, 7, grid=grid(1920), stream=stream0)
        del buf744
        buf747 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_125.run(buf746, buf747, 240, 8, grid=grid(240), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf748 = aten.convolution_backward(buf746, mul_79, primals_139, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf746
        del mul_79
        del primals_139
        buf749 = buf748[0]
        buf750 = buf748[1]
        del buf748
        buf751 = buf749; del buf749  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_126.run(buf751, convolution_20, 160, grid=grid(160), stream=stream0)
        del convolution_20
        buf752 = empty((20, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_127.run(buf751, buf752, 20, 8, grid=grid(20), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf753 = aten.convolution_backward(buf751, mean, primals_137, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean
        del primals_137
        buf754 = buf753[0]
        buf755 = buf753[1]
        del buf753
        buf756 = empty((240, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_128.run(buf742, convolution_21, buf754, add_56, buf756, 11760, 128, grid=grid(11760), stream=stream0)
        buf757 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_129.run(buf756, buf757, 240, 49, grid=grid(240), stream=stream0)
        buf758 = reinterpret_tensor(buf756, (240, 49), (1, 240), 0); del buf756  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_130.run(buf742, convolution_21, buf754, add_56, cat_5, unsqueeze_798, buf758, 11760, 128, grid=grid(11760), stream=stream0)
        buf759 = empty((240, ), device='cuda', dtype=torch.float32)
        buf761 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_131.run(buf758, squeeze_31, buf759, buf761, 240, 49, grid=grid(240), stream=stream0)
        del buf758
        buf760 = reinterpret_tensor(buf203, (8, 240, 28, 28), (188160, 1, 6720, 240), 0); del buf203  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_132.run(buf742, convolution_21, buf754, add_56, cat_5, unsqueeze_798, buf759, squeeze_31, buf757, buf760, 6272, 240, grid=grid(6272, 240), stream=stream0)
        del add_56
        del buf742
        del buf754
        del cat_5
        del convolution_21
        del unsqueeze_798
        buf762 = reinterpret_tensor(buf320, (8, 60, 28, 28), (47040, 784, 28, 1), 0); del buf320  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_133.run(buf760, squeeze_31, primals_21, buf762, 480, 784, grid=grid(480, 784), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf763 = aten.convolution_backward(buf762, getitem_67, primals_136, [0], [2, 2], [4, 4], [1, 1], False, [0, 0], 60, [True, True, False])
        del getitem_67
        del primals_136
        buf764 = buf763[0]
        buf765 = buf763[1]
        del buf763
        buf766 = buf762; del buf762  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_134.run(buf760, squeeze_31, primals_21, buf766, 480, 784, grid=grid(480, 784), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf767 = aten.convolution_backward(buf766, getitem_62, primals_135, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 60, [True, True, False])
        del getitem_62
        del primals_135
        buf768 = buf767[0]
        buf769 = buf767[1]
        del buf767
        buf770 = buf766; del buf766  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_135.run(buf760, squeeze_31, primals_21, buf770, 480, 784, grid=grid(480, 784), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf771 = aten.convolution_backward(buf770, getitem_57, primals_134, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 60, [True, True, False])
        del getitem_57
        del primals_134
        buf772 = buf771[0]
        buf773 = buf771[1]
        del buf771
        buf774 = buf770; del buf770  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_136.run(buf760, squeeze_31, primals_21, buf774, 480, 784, grid=grid(480, 784), stream=stream0)
        del buf760
        del primals_21
        del squeeze_31
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf775 = aten.convolution_backward(buf774, getitem_52, primals_133, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 60, [True, True, False])
        del buf774
        del getitem_52
        del primals_133
        buf776 = buf775[0]
        buf777 = buf775[1]
        del buf775
        buf778 = empty((8, 240, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul]
        triton_poi_fused_cat_mul_137.run(buf776, buf772, buf768, buf764, mul_1108, buf778, 1920, 3136, grid=grid(1920, 3136), stream=stream0)
        del buf764
        del buf768
        del buf772
        del buf776
        del mul_1108
        buf779 = reinterpret_tensor(buf201, (240, 4), (1, 240), 0); del buf201  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_138.run(buf778, buf779, 960, 6272, grid=grid(960), stream=stream0)
        buf780 = buf759; del buf759  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_139.run(buf779, buf780, 240, 4, grid=grid(240), stream=stream0)
        del buf779
        buf781 = empty((240, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_140.run(buf778, convolution_15, unsqueeze_810, buf781, 47040, 128, grid=grid(47040), stream=stream0)
        buf782 = empty((240, ), device='cuda', dtype=torch.float32)
        buf783 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_141.run(buf781, squeeze_28, buf782, buf783, 240, 196, grid=grid(240), stream=stream0)
        del buf781
        buf784 = buf778; del buf778  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_142.run(buf784, convolution_15, unsqueeze_810, buf782, squeeze_28, buf780, primals_19, 1920, 3136, grid=grid(1920, 3136), stream=stream0)
        del buf782
        del convolution_15
        del primals_19
        del squeeze_28
        del unsqueeze_810
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf785 = aten.convolution_backward(buf784, add_46, primals_132, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_46
        del buf784
        del primals_132
        buf786 = buf785[0]
        buf787 = buf785[1]
        del buf785
        buf788 = reinterpret_tensor(buf751, (40, 4), (1, 40), 0); del buf751  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_143.run(buf786, buf788, 160, 6272, grid=grid(160), stream=stream0)
        buf789 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_144.run(buf788, buf789, 40, 4, grid=grid(40), stream=stream0)
        buf790 = empty((40, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_145.run(buf786, cat_4, unsqueeze_822, buf790, 7840, 128, grid=grid(7840), stream=stream0)
        buf791 = empty((40, ), device='cuda', dtype=torch.float32)
        buf793 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_146.run(buf790, squeeze_25, buf791, buf793, 40, 196, grid=grid(40), stream=stream0)
        del buf790
        buf792 = empty((8, 40, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_147.run(buf786, cat_4, unsqueeze_822, buf791, squeeze_25, buf789, primals_17, buf792, 320, 3136, grid=grid(320, 3136), stream=stream0)
        del cat_4
        del primals_17
        del squeeze_25
        del unsqueeze_822
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf794 = aten.convolution_backward(reinterpret_tensor(buf792, (8, 20, 56, 56), (125440, 3136, 56, 1), 62720), getitem_43, primals_131, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_43
        del primals_131
        buf795 = buf794[0]
        buf796 = buf794[1]
        del buf794
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf797 = aten.convolution_backward(reinterpret_tensor(buf792, (8, 20, 56, 56), (125440, 3136, 56, 1), 0), getitem_40, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf792
        del getitem_40
        del primals_130
        buf798 = buf797[0]
        buf799 = buf797[1]
        del buf797
        buf800 = reinterpret_tensor(buf354, (120, 4), (1, 120), 0); del buf354  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_cat_native_batch_norm_backward_threshold_backward_148.run(le_1, buf798, buf795, buf800, 480, 6272, grid=grid(480), stream=stream0)
        buf801 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_cat_native_batch_norm_backward_threshold_backward_149.run(buf800, buf801, 120, 4, grid=grid(120), stream=stream0)
        del buf800
        buf802 = empty((120, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_cat_native_batch_norm_backward_threshold_backward_150.run(le_1, buf798, buf795, convolution_12, unsqueeze_834, buf802, 23520, 128, grid=grid(23520), stream=stream0)
        buf803 = empty((120, ), device='cuda', dtype=torch.float32)
        buf805 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_cat_native_batch_norm_backward_threshold_backward_151.run(buf802, squeeze_22, buf803, buf805, 120, 196, grid=grid(120), stream=stream0)
        buf804 = empty_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_cat_native_batch_norm_backward_threshold_backward_152.run(le_1, buf798, buf795, convolution_12, unsqueeze_834, buf803, squeeze_22, buf801, primals_15, buf804, 25088, 120, grid=grid(25088, 120), stream=stream0)
        del buf795
        del buf798
        del convolution_12
        del le_1
        del primals_15
        del squeeze_22
        del unsqueeze_834
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf806 = aten.convolution_backward(buf804, relu_4, primals_129, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 120, [True, True, False])
        del primals_129
        buf807 = buf806[0]
        buf808 = buf806[1]
        del buf806
        buf809 = buf802; del buf802  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_153.run(relu_4, buf807, buf809, 23520, 128, grid=grid(23520), stream=stream0)
        buf810 = buf803; del buf803  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_154.run(buf809, buf810, 120, 196, grid=grid(120), stream=stream0)
        buf811 = reinterpret_tensor(buf809, (120, 196), (1, 120), 0); del buf809  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_155.run(relu_4, buf807, cat_3, unsqueeze_846, buf811, 23520, 128, grid=grid(23520), stream=stream0)
        buf812 = empty((120, ), device='cuda', dtype=torch.float32)
        buf814 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_156.run(buf811, squeeze_19, buf812, buf814, 120, 196, grid=grid(120), stream=stream0)
        del buf811
        buf813 = buf804; del buf804  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_157.run(relu_4, buf807, cat_3, unsqueeze_846, buf812, squeeze_19, buf810, primals_13, buf813, 25088, 120, grid=grid(25088, 120), stream=stream0)
        del buf807
        del buf812
        del cat_3
        del primals_13
        del relu_4
        del squeeze_19
        del unsqueeze_846
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf815 = aten.convolution_backward(reinterpret_tensor(buf813, (8, 60, 56, 56), (376320, 1, 6720, 120), 60), getitem_33, primals_128, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_33
        del primals_128
        buf816 = buf815[0]
        buf817 = buf815[1]
        del buf815
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf818 = aten.convolution_backward(reinterpret_tensor(buf813, (8, 60, 56, 56), (376320, 1, 6720, 120), 0), getitem_32, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf813
        del getitem_32
        del primals_127
        buf819 = buf818[0]
        buf820 = buf818[1]
        del buf818
        buf821 = buf788; del buf788  # reuse
        buf823 = reinterpret_tensor(buf364, (40, 4), (1, 40), 0); del buf364  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_red_fused_add_cat_native_batch_norm_backward_158.run(buf786, buf819, buf816, cat_2, unsqueeze_858, buf821, buf823, 160, 6272, grid=grid(160), stream=stream0)
        buf822 = buf791; del buf791  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_144.run(buf821, buf822, 40, 4, grid=grid(40), stream=stream0)
        del buf821
        buf824 = empty((40, ), device='cuda', dtype=torch.float32)
        buf826 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_per_fused_add_cat_native_batch_norm_backward_159.run(buf823, squeeze_16, buf824, buf826, 40, 4, grid=grid(40), stream=stream0)
        del buf823
        buf825 = buf786; del buf786  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_poi_fused_add_cat_native_batch_norm_backward_160.run(buf825, buf819, buf816, cat_2, unsqueeze_858, buf824, squeeze_16, buf822, primals_11, 320, 3136, grid=grid(320, 3136), stream=stream0)
        del buf816
        del buf819
        del buf824
        del cat_2
        del primals_11
        del squeeze_16
        del unsqueeze_858
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf827 = aten.convolution_backward(reinterpret_tensor(buf825, (8, 20, 56, 56), (125440, 3136, 56, 1), 62720), getitem_29, primals_126, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_29
        del primals_126
        buf828 = buf827[0]
        buf829 = buf827[1]
        del buf827
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf830 = aten.convolution_backward(reinterpret_tensor(buf825, (8, 20, 56, 56), (125440, 3136, 56, 1), 0), getitem_26, primals_125, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf825
        del getitem_26
        del primals_125
        buf831 = buf830[0]
        buf832 = buf830[1]
        del buf830
        buf833 = empty_strided((192, 4), (1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_cat_native_batch_norm_backward_threshold_backward_161.run(le_3, buf831, buf828, buf833, 768, 6272, grid=grid(768), stream=stream0)
        buf834 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_cat_native_batch_norm_backward_threshold_backward_162.run(buf833, buf834, 192, 4, grid=grid(192), stream=stream0)
        buf835 = empty((192, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_cat_native_batch_norm_backward_threshold_backward_163.run(le_3, buf831, buf828, cat_1, unsqueeze_870, buf835, 37632, 128, grid=grid(37632), stream=stream0)
        buf836 = empty((192, ), device='cuda', dtype=torch.float32)
        buf838 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_cat_native_batch_norm_backward_threshold_backward_164.run(buf835, squeeze_13, buf836, buf838, 192, 196, grid=grid(192), stream=stream0)
        del buf835
        buf837 = empty((8, 192, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_cat_native_batch_norm_backward_threshold_backward_165.run(le_3, buf831, buf828, cat_1, unsqueeze_870, buf836, squeeze_13, buf834, primals_9, buf837, 1536, 3136, grid=grid(1536, 3136), stream=stream0)
        del buf828
        del buf831
        del cat_1
        del le_3
        del primals_9
        del squeeze_13
        del unsqueeze_870
        buf839 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_166.run(buf837, buf839, 512, 3136, grid=grid(512, 3136), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf840 = aten.convolution_backward(buf839, getitem_21, primals_124, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 64, [True, True, False])
        del getitem_21
        del primals_124
        buf841 = buf840[0]
        buf842 = buf840[1]
        del buf840
        buf843 = buf839; del buf839  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_167.run(buf837, buf843, 512, 3136, grid=grid(512, 3136), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf844 = aten.convolution_backward(buf843, getitem_17, primals_123, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 64, [True, True, False])
        del getitem_17
        del primals_123
        buf845 = buf844[0]
        buf846 = buf844[1]
        del buf844
        buf847 = buf843; del buf843  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_168.run(buf837, buf847, 512, 3136, grid=grid(512, 3136), stream=stream0)
        del buf837
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf848 = aten.convolution_backward(buf847, getitem_13, primals_122, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False])
        del buf847
        del getitem_13
        del primals_122
        buf849 = buf848[0]
        buf850 = buf848[1]
        del buf848
        buf851 = buf833; del buf833  # reuse
        buf853 = empty_strided((192, 4), (1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_cat_native_batch_norm_backward_threshold_backward_169.run(le_4, buf849, buf845, buf841, cat, unsqueeze_882, buf851, buf853, 768, 25088, grid=grid(768), stream=stream0)
        buf852 = buf836; del buf836  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_cat_native_batch_norm_backward_threshold_backward_162.run(buf851, buf852, 192, 4, grid=grid(192), stream=stream0)
        del buf851
        buf854 = empty((192, ), device='cuda', dtype=torch.float32)
        buf856 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_cat_native_batch_norm_backward_threshold_backward_170.run(buf853, squeeze_10, buf854, buf856, 192, 4, grid=grid(192), stream=stream0)
        del buf853
        buf855 = empty((8, 192, 112, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_cat_native_batch_norm_backward_threshold_backward_171.run(le_4, buf849, buf845, buf841, cat, unsqueeze_882, buf854, squeeze_10, buf852, buf855, 1536, 12544, grid=grid(1536, 12544), stream=stream0)
        del buf841
        del buf845
        del buf849
        del buf854
        del cat
        del le_4
        del unsqueeze_882
        buf857 = empty_strided((8, 96, 112, 112), (1204224, 1, 10752, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_172.run(buf855, squeeze_10, primals_7, buf857, 768, 12544, grid=grid(768, 12544), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf858 = aten.convolution_backward(buf857, getitem_7, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_7
        del primals_121
        buf859 = buf858[0]
        buf860 = buf858[1]
        del buf858
        buf861 = buf857; del buf857  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_173.run(buf855, squeeze_10, primals_7, buf861, 768, 12544, grid=grid(768, 12544), stream=stream0)
        del buf855
        del primals_7
        del squeeze_10
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf862 = aten.convolution_backward(buf861, getitem_6, primals_120, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf861
        del getitem_6
        del primals_120
        buf863 = buf862[0]
        buf864 = buf862[1]
        del buf862
        buf865 = reinterpret_tensor(buf377, (32, 13), (13, 1), 0); del buf377  # reuse
        buf867 = empty((32, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward]
        triton_red_fused_cat_native_batch_norm_backward_174.run(buf863, buf859, convolution_2, unsqueeze_894, buf865, buf867, 416, 7720, grid=grid(416), stream=stream0)
        buf866 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward]
        triton_per_fused_cat_native_batch_norm_backward_175.run(buf865, buf866, 32, 13, grid=grid(32), stream=stream0)
        buf868 = empty((32, ), device='cuda', dtype=torch.float32)
        buf869 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward]
        triton_per_fused_cat_native_batch_norm_backward_176.run(buf867, squeeze_7, buf868, buf869, 32, 13, grid=grid(32), stream=stream0)
        buf870 = empty((8, 32, 112, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_cat_convolution_backward_native_batch_norm_backward_177.run(buf863, buf859, convolution_2, unsqueeze_894, buf868, squeeze_7, buf866, primals_5, buf870, 256, 12544, grid=grid(256, 12544), stream=stream0)
        del convolution_2
        del primals_5
        del squeeze_7
        del unsqueeze_894
        # Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward, aten.native_batch_norm_backward]
        buf871 = aten.convolution_backward(buf870, relu_1, primals_119, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_119
        buf872 = buf871[0]
        buf873 = buf871[1]
        del buf871
        buf874 = empty((32, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_178.run(relu_1, buf872, buf874, 25088, 128, grid=grid(25088), stream=stream0)
        buf875 = buf868; del buf868  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_179.run(buf874, buf875, 32, 784, grid=grid(32), stream=stream0)
        buf876 = reinterpret_tensor(buf874, (32, 784), (1, 32), 0); del buf874  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_180.run(relu_1, buf872, convolution_1, unsqueeze_906, buf876, 25088, 128, grid=grid(25088), stream=stream0)
        buf877 = empty((32, ), device='cuda', dtype=torch.float32)
        buf878 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_181.run(buf876, squeeze_4, buf877, buf878, 32, 784, grid=grid(32), stream=stream0)
        del buf876
        buf879 = reinterpret_tensor(buf870, (8, 32, 112, 112), (401408, 1, 3584, 32), 0); del buf870  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_182.run(relu_1, buf872, convolution_1, unsqueeze_906, buf877, squeeze_4, buf875, primals_3, buf879, 100352, 32, grid=grid(100352, 32), stream=stream0)
        del buf872
        del convolution_1
        del primals_3
        del relu_1
        del squeeze_4
        del unsqueeze_906
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf880 = aten.convolution_backward(buf879, relu, primals_118, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del primals_118
        buf881 = buf880[0]
        buf882 = buf880[1]
        del buf880
        buf883 = buf867; del buf867  # reuse
        buf885 = buf865; del buf865  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_cat_native_batch_norm_backward_threshold_backward_183.run(relu, buf863, buf859, buf881, convolution, unsqueeze_918, buf883, buf885, 416, 7720, grid=grid(416), stream=stream0)
        buf884 = buf877; del buf877  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_cat_native_batch_norm_backward_175.run(buf883, buf884, 32, 13, grid=grid(32), stream=stream0)
        del buf883
        buf886 = empty((32, ), device='cuda', dtype=torch.float32)
        buf888 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_cat_native_batch_norm_backward_176.run(buf885, squeeze_1, buf886, buf888, 32, 13, grid=grid(32), stream=stream0)
        del buf885
        buf887 = buf881; del buf881  # reuse
        buf889 = buf879; del buf879  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_cat_convolution_backward_native_batch_norm_backward_threshold_backward_184.run(buf887, relu, buf863, buf859, convolution, unsqueeze_918, buf886, squeeze_1, buf884, primals_1, buf889, 256, 12544, grid=grid(256, 12544), stream=stream0)
        del buf859
        del buf863
        del buf886
        del buf887
        del convolution
        del primals_1
        del relu
        del squeeze_1
        del unsqueeze_918
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf890 = aten.convolution_backward(buf889, primals_480, primals_117, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf889
        del primals_117
        del primals_480
        buf891 = buf890[1]
        return (buf888, buf884, buf878, buf875, buf869, buf866, buf856, buf852, buf838, buf834, buf826, buf822, buf814, buf810, buf805, buf801, buf793, buf789, buf783, buf780, buf761, buf757, buf740, buf737, buf730, buf726, buf719, buf715, buf696, buf692, buf684, buf680, buf673, buf669, buf650, buf647, buf640, buf636, buf629, buf625, buf606, buf602, buf597, buf594, buf581, buf577, buf560, buf557, buf550, buf546, buf532, buf528, buf509, buf505, buf497, buf493, buf479, buf475, buf456, buf453, buf446, buf442, buf428, buf424, buf405, buf401, buf396, buf393, buf387, buf383, buf366, buf363, buf356, buf352, buf338, buf334, buf315, buf311, buf303, buf299, buf285, buf281, buf262, buf259, buf252, buf248, buf234, buf230, buf211, buf207, buf202, buf199, buf181, buf177, buf160, buf157, buf152, buf149, buf135, buf131, buf112, buf109, buf104, buf101, buf87, buf83, buf64, buf61, buf56, buf53, buf39, buf35, buf16, buf12, buf7, buf4, buf891, buf882, buf873, buf864, buf860, buf850, buf846, buf842, buf832, buf829, buf820, buf817, buf808, buf799, buf796, buf787, buf777, buf773, buf769, buf765, buf755, buf752, buf750, buf747, buf743, buf736, buf733, buf725, buf722, buf713, buf710, buf708, buf705, buf702, buf699, buf690, buf687, buf679, buf676, buf667, buf664, buf662, buf659, buf656, buf653, buf646, buf643, buf635, buf632, buf623, buf620, buf618, buf615, buf612, buf609, buf601, buf593, buf589, buf585, buf575, buf572, buf570, buf567, buf563, buf556, buf553, buf544, buf541, buf538, buf535, buf526, buf523, buf521, buf518, buf515, buf512, buf503, buf500, buf491, buf488, buf485, buf482, buf473, buf470, buf468, buf465, buf462, buf459, buf452, buf449, buf440, buf437, buf434, buf431, buf422, buf419, buf417, buf414, buf411, buf408, buf400, buf391, buf381, buf378, buf376, buf373, buf369, buf362, buf359, buf350, buf347, buf344, buf341, buf332, buf329, buf327, buf324, buf321, buf318, buf309, buf306, buf297, buf294, buf291, buf288, buf279, buf276, buf274, buf271, buf268, buf265, buf258, buf255, buf246, buf243, buf240, buf237, buf228, buf225, buf223, buf220, buf217, buf214, buf206, buf197, buf193, buf189, buf185, buf175, buf172, buf170, buf167, buf164, buf156, buf147, buf144, buf141, buf138, buf129, buf126, buf124, buf121, buf118, buf115, buf108, buf99, buf96, buf93, buf90, buf81, buf78, buf76, buf73, buf70, buf67, buf60, buf51, buf48, buf45, buf42, buf33, buf30, buf28, buf25, buf22, buf19, buf11, reinterpret_tensor(buf1, (1000, 1536), (1536, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((64, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((64, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((20, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((20, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((60, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((60, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((20, 60, 1, 1), (60, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((20, 60, 1, 1), (60, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((60, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((60, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((60, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((60, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((20, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((240, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((56, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((168, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((168, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((28, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((336, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((168, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((168, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((28, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((336, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((168, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((168, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((28, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((336, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((336, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((112, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((112, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((112, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((14, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((336, 14, 1, 1), (14, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((104, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((156, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((156, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((156, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((156, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((26, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((624, 26, 1, 1), (26, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((156, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((156, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((156, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((156, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((26, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((624, 26, 1, 1), (26, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((156, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((156, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((156, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((156, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((26, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((624, 26, 1, 1), (26, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((624, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((624, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((52, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((624, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((160, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((120, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((120, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((120, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((240, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((240, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((80, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((960, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((264, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((1584, 264, 1, 1), (264, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((396, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((396, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((396, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((396, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((132, 1584, 1, 1), (1584, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((1584, 132, 1, 1), (132, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((1584, 264, 1, 1), (264, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((396, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((396, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((396, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((396, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((132, 1584, 1, 1), (1584, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((1584, 132, 1, 1), (132, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((1584, 264, 1, 1), (264, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((396, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((396, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((396, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((396, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((132, 1584, 1, 1), (1584, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((1584, 132, 1, 1), (132, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((1536, 264, 1, 1), (264, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_6 = rand_strided((8, 16, 112, 112), (401408, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((8, 16, 112, 112), (401408, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    cat = rand_strided((8, 192, 112, 112), (2408448, 1, 21504, 192), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_13 = rand_strided((8, 64, 112, 112), (2408448, 1, 21504, 192), device='cuda:0', dtype=torch.float32)
    getitem_17 = rand_strided((8, 64, 112, 112), (2408448, 1, 21504, 192), device='cuda:0', dtype=torch.float32)
    getitem_21 = rand_strided((8, 64, 112, 112), (2408448, 1, 21504, 192), device='cuda:0', dtype=torch.float32)
    cat_1 = rand_strided((8, 192, 56, 56), (602112, 1, 10752, 192), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_26 = rand_strided((8, 96, 56, 56), (602112, 1, 10752, 192), device='cuda:0', dtype=torch.float32)
    getitem_29 = rand_strided((8, 96, 56, 56), (602112, 1, 10752, 192), device='cuda:0', dtype=torch.float32)
    cat_2 = rand_strided((8, 40, 56, 56), (125440, 1, 2240, 40), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_32 = rand_strided((8, 20, 56, 56), (125440, 1, 2240, 40), device='cuda:0', dtype=torch.float32)
    getitem_33 = rand_strided((8, 20, 56, 56), (125440, 1, 2240, 40), device='cuda:0', dtype=torch.float32)
    cat_3 = rand_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_40 = rand_strided((8, 60, 56, 56), (376320, 1, 6720, 120), device='cuda:0', dtype=torch.float32)
    getitem_43 = rand_strided((8, 60, 56, 56), (376320, 1, 6720, 120), device='cuda:0', dtype=torch.float32)
    cat_4 = rand_strided((8, 40, 56, 56), (125440, 1, 2240, 40), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_46 = rand_strided((8, 40, 56, 56), (125440, 1, 2240, 40), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 240, 56, 56), (752640, 1, 13440, 240), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_52 = rand_strided((8, 60, 56, 56), (752640, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    getitem_57 = rand_strided((8, 60, 56, 56), (752640, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    getitem_62 = rand_strided((8, 60, 56, 56), (752640, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    getitem_67 = rand_strided((8, 60, 56, 56), (752640, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    cat_5 = rand_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_56 = rand_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda:0', dtype=torch.float32)
    mean = rand_strided((8, 240, 1, 1), (240, 1, 240, 240), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cuda:0', dtype=torch.float32)
    mul_79 = rand_strided((8, 20, 1, 1), (20, 1, 20, 20), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 240, 1, 1), (240, 1, 240, 240), device='cuda:0', dtype=torch.float32)
    mul_80 = rand_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_72 = rand_strided((8, 28, 28, 28), (43904, 1, 1568, 56), device='cuda:0', dtype=torch.float32)
    getitem_73 = rand_strided((8, 28, 28, 28), (43904, 1, 1568, 56), device='cuda:0', dtype=torch.float32)
    cat_6 = rand_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_78 = rand_strided((8, 168, 28, 28), (263424, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    getitem_81 = rand_strided((8, 168, 28, 28), (263424, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    cat_7 = rand_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_71 = rand_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda:0', dtype=torch.float32)
    mean_1 = rand_strided((8, 336, 1, 1), (336, 1, 336, 336), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cuda:0', dtype=torch.float32)
    mul_104 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((8, 336, 1, 1), (336, 1, 336, 336), device='cuda:0', dtype=torch.float32)
    getitem_84 = rand_strided((8, 168, 28, 28), (263424, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    getitem_85 = rand_strided((8, 168, 28, 28), (263424, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    cat_8 = rand_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_88 = rand_strided((8, 28, 28, 28), (43904, 1, 1568, 56), device='cuda:0', dtype=torch.float32)
    getitem_89 = rand_strided((8, 28, 28, 28), (43904, 1, 1568, 56), device='cuda:0', dtype=torch.float32)
    cat_9 = rand_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_94 = rand_strided((8, 168, 28, 28), (263424, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    getitem_97 = rand_strided((8, 168, 28, 28), (263424, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    cat_10 = rand_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_87 = rand_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda:0', dtype=torch.float32)
    mean_2 = rand_strided((8, 336, 1, 1), (336, 1, 336, 336), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cuda:0', dtype=torch.float32)
    mul_129 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((8, 336, 1, 1), (336, 1, 336, 336), device='cuda:0', dtype=torch.float32)
    getitem_100 = rand_strided((8, 168, 28, 28), (263424, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    getitem_101 = rand_strided((8, 168, 28, 28), (263424, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    cat_11 = rand_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_104 = rand_strided((8, 28, 28, 28), (43904, 1, 1568, 56), device='cuda:0', dtype=torch.float32)
    getitem_105 = rand_strided((8, 28, 28, 28), (43904, 1, 1568, 56), device='cuda:0', dtype=torch.float32)
    cat_12 = rand_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_110 = rand_strided((8, 168, 28, 28), (263424, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    getitem_113 = rand_strided((8, 168, 28, 28), (263424, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    cat_13 = rand_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_103 = rand_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda:0', dtype=torch.float32)
    mean_3 = rand_strided((8, 336, 1, 1), (336, 1, 336, 336), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cuda:0', dtype=torch.float32)
    mul_154 = rand_strided((8, 28, 1, 1), (28, 1, 28, 28), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((8, 336, 1, 1), (336, 1, 336, 336), device='cuda:0', dtype=torch.float32)
    getitem_116 = rand_strided((8, 168, 28, 28), (263424, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    getitem_117 = rand_strided((8, 168, 28, 28), (263424, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    cat_14 = rand_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_109 = rand_strided((8, 56, 28, 28), (43904, 1, 1568, 56), device='cuda:0', dtype=torch.float32)
    convolution_47 = rand_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_125 = rand_strided((8, 112, 28, 28), (263424, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    getitem_129 = rand_strided((8, 112, 28, 28), (263424, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    getitem_133 = rand_strided((8, 112, 28, 28), (263424, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    cat_15 = rand_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_119 = rand_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cuda:0', dtype=torch.float32)
    mean_4 = rand_strided((8, 336, 1, 1), (336, 1, 336, 336), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((8, 14, 1, 1), (14, 1, 14, 14), device='cuda:0', dtype=torch.float32)
    mul_179 = rand_strided((8, 14, 1, 1), (14, 1, 14, 14), device='cuda:0', dtype=torch.float32)
    convolution_52 = rand_strided((8, 336, 1, 1), (336, 1, 336, 336), device='cuda:0', dtype=torch.float32)
    mul_180 = rand_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cuda:0', dtype=torch.float32)
    convolution_53 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_70 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_138 = rand_strided((8, 52, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    getitem_139 = rand_strided((8, 52, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    cat_16 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_146 = rand_strided((8, 156, 14, 14), (122304, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_151 = rand_strided((8, 156, 14, 14), (122304, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_156 = rand_strided((8, 156, 14, 14), (122304, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_161 = rand_strided((8, 156, 14, 14), (122304, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_17 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_134 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda:0', dtype=torch.float32)
    mean_5 = rand_strided((8, 624, 1, 1), (624, 1, 624, 624), device='cuda:0', dtype=torch.float32)
    convolution_60 = rand_strided((8, 26, 1, 1), (26, 1, 26, 26), device='cuda:0', dtype=torch.float32)
    mul_204 = rand_strided((8, 26, 1, 1), (26, 1, 26, 26), device='cuda:0', dtype=torch.float32)
    convolution_61 = rand_strided((8, 624, 1, 1), (624, 1, 624, 624), device='cuda:0', dtype=torch.float32)
    getitem_164 = rand_strided((8, 312, 14, 14), (122304, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_165 = rand_strided((8, 312, 14, 14), (122304, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_18 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_168 = rand_strided((8, 52, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    getitem_169 = rand_strided((8, 52, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    cat_19 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda:0', dtype=torch.float32)
    squeeze_82 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_176 = rand_strided((8, 156, 14, 14), (122304, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_181 = rand_strided((8, 156, 14, 14), (122304, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_186 = rand_strided((8, 156, 14, 14), (122304, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_191 = rand_strided((8, 156, 14, 14), (122304, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_20 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_150 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda:0', dtype=torch.float32)
    mean_6 = rand_strided((8, 624, 1, 1), (624, 1, 624, 624), device='cuda:0', dtype=torch.float32)
    convolution_70 = rand_strided((8, 26, 1, 1), (26, 1, 26, 26), device='cuda:0', dtype=torch.float32)
    mul_229 = rand_strided((8, 26, 1, 1), (26, 1, 26, 26), device='cuda:0', dtype=torch.float32)
    convolution_71 = rand_strided((8, 624, 1, 1), (624, 1, 624, 624), device='cuda:0', dtype=torch.float32)
    getitem_194 = rand_strided((8, 312, 14, 14), (122304, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_195 = rand_strided((8, 312, 14, 14), (122304, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_21 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_88 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_198 = rand_strided((8, 52, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    getitem_199 = rand_strided((8, 52, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    cat_22 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_206 = rand_strided((8, 156, 14, 14), (122304, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_211 = rand_strided((8, 156, 14, 14), (122304, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_216 = rand_strided((8, 156, 14, 14), (122304, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_221 = rand_strided((8, 156, 14, 14), (122304, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_23 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda:0', dtype=torch.float32)
    squeeze_94 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_166 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda:0', dtype=torch.float32)
    mean_7 = rand_strided((8, 624, 1, 1), (624, 1, 624, 624), device='cuda:0', dtype=torch.float32)
    convolution_80 = rand_strided((8, 26, 1, 1), (26, 1, 26, 26), device='cuda:0', dtype=torch.float32)
    mul_254 = rand_strided((8, 26, 1, 1), (26, 1, 26, 26), device='cuda:0', dtype=torch.float32)
    convolution_81 = rand_strided((8, 624, 1, 1), (624, 1, 624, 624), device='cuda:0', dtype=torch.float32)
    getitem_224 = rand_strided((8, 312, 14, 14), (122304, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_225 = rand_strided((8, 312, 14, 14), (122304, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_24 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    squeeze_97 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_172 = rand_strided((8, 104, 14, 14), (20384, 1, 1456, 104), device='cuda:0', dtype=torch.float32)
    convolution_84 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda:0', dtype=torch.float32)
    squeeze_100 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_270 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda:0', dtype=torch.float32)
    convolution_85 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda:0', dtype=torch.float32)
    squeeze_103 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_182 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda:0', dtype=torch.float32)
    mean_8 = rand_strided((8, 624, 1, 1), (624, 1, 624, 624), device='cuda:0', dtype=torch.float32)
    convolution_86 = rand_strided((8, 52, 1, 1), (52, 1, 52, 52), device='cuda:0', dtype=torch.float32)
    mul_279 = rand_strided((8, 52, 1, 1), (52, 1, 52, 52), device='cuda:0', dtype=torch.float32)
    convolution_87 = rand_strided((8, 624, 1, 1), (624, 1, 624, 624), device='cuda:0', dtype=torch.float32)
    mul_280 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda:0', dtype=torch.float32)
    convolution_88 = rand_strided((8, 160, 14, 14), (31360, 1, 2240, 160), device='cuda:0', dtype=torch.float32)
    squeeze_106 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_234 = rand_strided((8, 80, 14, 14), (31360, 1, 2240, 160), device='cuda:0', dtype=torch.float32)
    getitem_235 = rand_strided((8, 80, 14, 14), (31360, 1, 2240, 160), device='cuda:0', dtype=torch.float32)
    cat_25 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    squeeze_109 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_242 = rand_strided((8, 120, 14, 14), (94080, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_247 = rand_strided((8, 120, 14, 14), (94080, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_252 = rand_strided((8, 120, 14, 14), (94080, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_257 = rand_strided((8, 120, 14, 14), (94080, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_26 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    squeeze_112 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_197 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    mean_9 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cuda:0', dtype=torch.float32)
    convolution_95 = rand_strided((8, 80, 1, 1), (80, 1, 80, 80), device='cuda:0', dtype=torch.float32)
    mul_304 = rand_strided((8, 80, 1, 1), (80, 1, 80, 80), device='cuda:0', dtype=torch.float32)
    convolution_96 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cuda:0', dtype=torch.float32)
    getitem_260 = rand_strided((8, 240, 14, 14), (94080, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_261 = rand_strided((8, 240, 14, 14), (94080, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_27 = rand_strided((8, 160, 14, 14), (31360, 1, 2240, 160), device='cuda:0', dtype=torch.float32)
    squeeze_115 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_264 = rand_strided((8, 80, 14, 14), (31360, 1, 2240, 160), device='cuda:0', dtype=torch.float32)
    getitem_265 = rand_strided((8, 80, 14, 14), (31360, 1, 2240, 160), device='cuda:0', dtype=torch.float32)
    cat_28 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    squeeze_118 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_272 = rand_strided((8, 120, 14, 14), (94080, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_277 = rand_strided((8, 120, 14, 14), (94080, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_282 = rand_strided((8, 120, 14, 14), (94080, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_287 = rand_strided((8, 120, 14, 14), (94080, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_29 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    squeeze_121 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_213 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    mean_10 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cuda:0', dtype=torch.float32)
    convolution_105 = rand_strided((8, 80, 1, 1), (80, 1, 80, 80), device='cuda:0', dtype=torch.float32)
    mul_329 = rand_strided((8, 80, 1, 1), (80, 1, 80, 80), device='cuda:0', dtype=torch.float32)
    convolution_106 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cuda:0', dtype=torch.float32)
    getitem_290 = rand_strided((8, 240, 14, 14), (94080, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_291 = rand_strided((8, 240, 14, 14), (94080, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_30 = rand_strided((8, 160, 14, 14), (31360, 1, 2240, 160), device='cuda:0', dtype=torch.float32)
    squeeze_124 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_294 = rand_strided((8, 80, 14, 14), (31360, 1, 2240, 160), device='cuda:0', dtype=torch.float32)
    getitem_295 = rand_strided((8, 80, 14, 14), (31360, 1, 2240, 160), device='cuda:0', dtype=torch.float32)
    cat_31 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    squeeze_127 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_302 = rand_strided((8, 120, 14, 14), (94080, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_307 = rand_strided((8, 120, 14, 14), (94080, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_312 = rand_strided((8, 120, 14, 14), (94080, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_317 = rand_strided((8, 120, 14, 14), (94080, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_32 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    squeeze_130 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_229 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    mean_11 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cuda:0', dtype=torch.float32)
    convolution_115 = rand_strided((8, 80, 1, 1), (80, 1, 80, 80), device='cuda:0', dtype=torch.float32)
    mul_354 = rand_strided((8, 80, 1, 1), (80, 1, 80, 80), device='cuda:0', dtype=torch.float32)
    convolution_116 = rand_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cuda:0', dtype=torch.float32)
    getitem_320 = rand_strided((8, 240, 14, 14), (94080, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_321 = rand_strided((8, 240, 14, 14), (94080, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_33 = rand_strided((8, 160, 14, 14), (31360, 1, 2240, 160), device='cuda:0', dtype=torch.float32)
    squeeze_133 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_235 = rand_strided((8, 160, 14, 14), (31360, 1, 2240, 160), device='cuda:0', dtype=torch.float32)
    convolution_119 = rand_strided((8, 960, 14, 14), (188160, 1, 13440, 960), device='cuda:0', dtype=torch.float32)
    squeeze_136 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_330 = rand_strided((8, 240, 14, 14), (188160, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_335 = rand_strided((8, 240, 14, 14), (188160, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_340 = rand_strided((8, 240, 14, 14), (188160, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    getitem_345 = rand_strided((8, 240, 14, 14), (188160, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    cat_34 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    squeeze_139 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_245 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    mean_12 = rand_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cuda:0', dtype=torch.float32)
    convolution_124 = rand_strided((8, 80, 1, 1), (80, 1, 80, 80), device='cuda:0', dtype=torch.float32)
    mul_379 = rand_strided((8, 80, 1, 1), (80, 1, 80, 80), device='cuda:0', dtype=torch.float32)
    convolution_125 = rand_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cuda:0', dtype=torch.float32)
    mul_380 = rand_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    convolution_126 = rand_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cuda:0', dtype=torch.float32)
    squeeze_142 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_250 = rand_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cuda:0', dtype=torch.float32)
    convolution_127 = rand_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cuda:0', dtype=torch.float32)
    squeeze_145 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_356 = rand_strided((8, 396, 7, 7), (77616, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    getitem_361 = rand_strided((8, 396, 7, 7), (77616, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    getitem_366 = rand_strided((8, 396, 7, 7), (77616, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    getitem_371 = rand_strided((8, 396, 7, 7), (77616, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    cat_35 = rand_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cuda:0', dtype=torch.float32)
    squeeze_148 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_260 = rand_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cuda:0', dtype=torch.float32)
    mean_13 = rand_strided((8, 1584, 1, 1), (1584, 1, 1584, 1584), device='cuda:0', dtype=torch.float32)
    convolution_132 = rand_strided((8, 132, 1, 1), (132, 1, 132, 132), device='cuda:0', dtype=torch.float32)
    mul_404 = rand_strided((8, 132, 1, 1), (132, 1, 132, 132), device='cuda:0', dtype=torch.float32)
    convolution_133 = rand_strided((8, 1584, 1, 1), (1584, 1, 1584, 1584), device='cuda:0', dtype=torch.float32)
    getitem_374 = rand_strided((8, 792, 7, 7), (77616, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    getitem_375 = rand_strided((8, 792, 7, 7), (77616, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    cat_36 = rand_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cuda:0', dtype=torch.float32)
    squeeze_151 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_266 = rand_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cuda:0', dtype=torch.float32)
    convolution_136 = rand_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cuda:0', dtype=torch.float32)
    squeeze_154 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_384 = rand_strided((8, 396, 7, 7), (77616, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    getitem_389 = rand_strided((8, 396, 7, 7), (77616, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    getitem_394 = rand_strided((8, 396, 7, 7), (77616, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    getitem_399 = rand_strided((8, 396, 7, 7), (77616, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    cat_37 = rand_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cuda:0', dtype=torch.float32)
    squeeze_157 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_276 = rand_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cuda:0', dtype=torch.float32)
    mean_14 = rand_strided((8, 1584, 1, 1), (1584, 1, 1584, 1584), device='cuda:0', dtype=torch.float32)
    convolution_141 = rand_strided((8, 132, 1, 1), (132, 1, 132, 132), device='cuda:0', dtype=torch.float32)
    mul_429 = rand_strided((8, 132, 1, 1), (132, 1, 132, 132), device='cuda:0', dtype=torch.float32)
    convolution_142 = rand_strided((8, 1584, 1, 1), (1584, 1, 1584, 1584), device='cuda:0', dtype=torch.float32)
    getitem_402 = rand_strided((8, 792, 7, 7), (77616, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    getitem_403 = rand_strided((8, 792, 7, 7), (77616, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    cat_38 = rand_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cuda:0', dtype=torch.float32)
    squeeze_160 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_282 = rand_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cuda:0', dtype=torch.float32)
    convolution_145 = rand_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cuda:0', dtype=torch.float32)
    squeeze_163 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    getitem_412 = rand_strided((8, 396, 7, 7), (77616, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    getitem_417 = rand_strided((8, 396, 7, 7), (77616, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    getitem_422 = rand_strided((8, 396, 7, 7), (77616, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    getitem_427 = rand_strided((8, 396, 7, 7), (77616, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    cat_39 = rand_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cuda:0', dtype=torch.float32)
    squeeze_166 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_292 = rand_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cuda:0', dtype=torch.float32)
    mean_15 = rand_strided((8, 1584, 1, 1), (1584, 1, 1584, 1584), device='cuda:0', dtype=torch.float32)
    convolution_150 = rand_strided((8, 132, 1, 1), (132, 1, 132, 132), device='cuda:0', dtype=torch.float32)
    mul_454 = rand_strided((8, 132, 1, 1), (132, 1, 132, 132), device='cuda:0', dtype=torch.float32)
    convolution_151 = rand_strided((8, 1584, 1, 1), (1584, 1, 1584, 1584), device='cuda:0', dtype=torch.float32)
    getitem_430 = rand_strided((8, 792, 7, 7), (77616, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    getitem_431 = rand_strided((8, 792, 7, 7), (77616, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    cat_40 = rand_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cuda:0', dtype=torch.float32)
    squeeze_169 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_298 = rand_strided((8, 264, 7, 7), (12936, 1, 1848, 264), device='cuda:0', dtype=torch.float32)
    convolution_154 = rand_strided((8, 1536, 7, 7), (75264, 1, 10752, 1536), device='cuda:0', dtype=torch.float32)
    squeeze_172 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    view = rand_strided((8, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    le = rand_strided((8, 1536, 7, 7), (75264, 1, 10752, 1536), device='cuda:0', dtype=torch.bool)
    unsqueeze_234 = rand_strided((1, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_246 = rand_strided((1, 264, 1, 1), (264, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_258 = rand_strided((1, 1584, 1, 1), (1584, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_508 = rand_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cuda:0', dtype=torch.float32)
    unsqueeze_270 = rand_strided((1, 1584, 1, 1), (1584, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_282 = rand_strided((1, 264, 1, 1), (264, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_294 = rand_strided((1, 1584, 1, 1), (1584, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_548 = rand_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cuda:0', dtype=torch.float32)
    unsqueeze_306 = rand_strided((1, 1584, 1, 1), (1584, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_318 = rand_strided((1, 264, 1, 1), (264, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_330 = rand_strided((1, 1584, 1, 1), (1584, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_588 = rand_strided((8, 1584, 7, 7), (77616, 1, 11088, 1584), device='cuda:0', dtype=torch.float32)
    unsqueeze_342 = rand_strided((1, 1584, 1, 1), (1584, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_354 = rand_strided((1, 264, 1, 1), (264, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_366 = rand_strided((1, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_628 = rand_strided((8, 960, 14, 14), (188160, 1, 13440, 960), device='cuda:0', dtype=torch.float32)
    unsqueeze_378 = rand_strided((1, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_390 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_402 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_668 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    unsqueeze_414 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_426 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_438 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_708 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    unsqueeze_450 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_462 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_474 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_748 = rand_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    unsqueeze_486 = rand_strided((1, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_498 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_510 = rand_strided((1, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_788 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda:0', dtype=torch.float32)
    unsqueeze_522 = rand_strided((1, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_534 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_546 = rand_strided((1, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_828 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda:0', dtype=torch.float32)
    unsqueeze_558 = rand_strided((1, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_570 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_582 = rand_strided((1, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_868 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda:0', dtype=torch.float32)
    unsqueeze_594 = rand_strided((1, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_606 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_618 = rand_strided((1, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_908 = rand_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda:0', dtype=torch.float32)
    unsqueeze_630 = rand_strided((1, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_642 = rand_strided((1, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_654 = rand_strided((1, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_948 = rand_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda:0', dtype=torch.float32)
    unsqueeze_666 = rand_strided((1, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_678 = rand_strided((1, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_690 = rand_strided((1, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_988 = rand_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda:0', dtype=torch.float32)
    unsqueeze_702 = rand_strided((1, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_714 = rand_strided((1, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_726 = rand_strided((1, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_1028 = rand_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda:0', dtype=torch.float32)
    unsqueeze_738 = rand_strided((1, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_750 = rand_strided((1, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_762 = rand_strided((1, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_1068 = rand_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cuda:0', dtype=torch.float32)
    unsqueeze_774 = rand_strided((1, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_786 = rand_strided((1, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_798 = rand_strided((1, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_1108 = rand_strided((8, 240, 56, 56), (752640, 1, 13440, 240), device='cuda:0', dtype=torch.float32)
    unsqueeze_810 = rand_strided((1, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_822 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_1 = rand_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cuda:0', dtype=torch.bool)
    unsqueeze_834 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_846 = rand_strided((1, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_858 = rand_strided((1, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_3 = rand_strided((8, 192, 56, 56), (602112, 1, 10752, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_870 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    le_4 = rand_strided((8, 192, 112, 112), (2408448, 1, 21504, 192), device='cuda:0', dtype=torch.bool)
    unsqueeze_882 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_894 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_906 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_918 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_139, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_148, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_158, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_168, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_178, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_189, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_201, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_213, primals_215, primals_216, primals_217, primals_218, primals_219, primals_221, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_232, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_244, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_256, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_267, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_277, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_288, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_299, primals_301, primals_302, primals_303, primals_480, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, getitem_6, getitem_7, cat, squeeze_10, getitem_13, getitem_17, getitem_21, cat_1, squeeze_13, getitem_26, getitem_29, cat_2, squeeze_16, getitem_32, getitem_33, cat_3, squeeze_19, relu_4, convolution_12, squeeze_22, getitem_40, getitem_43, cat_4, squeeze_25, add_46, convolution_15, squeeze_28, getitem_52, getitem_57, getitem_62, getitem_67, cat_5, squeeze_31, add_56, mean, convolution_20, mul_79, convolution_21, mul_80, convolution_22, squeeze_34, getitem_72, getitem_73, cat_6, squeeze_37, getitem_78, getitem_81, cat_7, squeeze_40, add_71, mean_1, convolution_27, mul_104, convolution_28, getitem_84, getitem_85, cat_8, squeeze_43, getitem_88, getitem_89, cat_9, squeeze_46, getitem_94, getitem_97, cat_10, squeeze_49, add_87, mean_2, convolution_35, mul_129, convolution_36, getitem_100, getitem_101, cat_11, squeeze_52, getitem_104, getitem_105, cat_12, squeeze_55, getitem_110, getitem_113, cat_13, squeeze_58, add_103, mean_3, convolution_43, mul_154, convolution_44, getitem_116, getitem_117, cat_14, squeeze_61, add_109, convolution_47, squeeze_64, getitem_125, getitem_129, getitem_133, cat_15, squeeze_67, add_119, mean_4, convolution_51, mul_179, convolution_52, mul_180, convolution_53, squeeze_70, getitem_138, getitem_139, cat_16, squeeze_73, getitem_146, getitem_151, getitem_156, getitem_161, cat_17, squeeze_76, add_134, mean_5, convolution_60, mul_204, convolution_61, getitem_164, getitem_165, cat_18, squeeze_79, getitem_168, getitem_169, cat_19, squeeze_82, getitem_176, getitem_181, getitem_186, getitem_191, cat_20, squeeze_85, add_150, mean_6, convolution_70, mul_229, convolution_71, getitem_194, getitem_195, cat_21, squeeze_88, getitem_198, getitem_199, cat_22, squeeze_91, getitem_206, getitem_211, getitem_216, getitem_221, cat_23, squeeze_94, add_166, mean_7, convolution_80, mul_254, convolution_81, getitem_224, getitem_225, cat_24, squeeze_97, add_172, convolution_84, squeeze_100, mul_270, convolution_85, squeeze_103, add_182, mean_8, convolution_86, mul_279, convolution_87, mul_280, convolution_88, squeeze_106, getitem_234, getitem_235, cat_25, squeeze_109, getitem_242, getitem_247, getitem_252, getitem_257, cat_26, squeeze_112, add_197, mean_9, convolution_95, mul_304, convolution_96, getitem_260, getitem_261, cat_27, squeeze_115, getitem_264, getitem_265, cat_28, squeeze_118, getitem_272, getitem_277, getitem_282, getitem_287, cat_29, squeeze_121, add_213, mean_10, convolution_105, mul_329, convolution_106, getitem_290, getitem_291, cat_30, squeeze_124, getitem_294, getitem_295, cat_31, squeeze_127, getitem_302, getitem_307, getitem_312, getitem_317, cat_32, squeeze_130, add_229, mean_11, convolution_115, mul_354, convolution_116, getitem_320, getitem_321, cat_33, squeeze_133, add_235, convolution_119, squeeze_136, getitem_330, getitem_335, getitem_340, getitem_345, cat_34, squeeze_139, add_245, mean_12, convolution_124, mul_379, convolution_125, mul_380, convolution_126, squeeze_142, add_250, convolution_127, squeeze_145, getitem_356, getitem_361, getitem_366, getitem_371, cat_35, squeeze_148, add_260, mean_13, convolution_132, mul_404, convolution_133, getitem_374, getitem_375, cat_36, squeeze_151, add_266, convolution_136, squeeze_154, getitem_384, getitem_389, getitem_394, getitem_399, cat_37, squeeze_157, add_276, mean_14, convolution_141, mul_429, convolution_142, getitem_402, getitem_403, cat_38, squeeze_160, add_282, convolution_145, squeeze_163, getitem_412, getitem_417, getitem_422, getitem_427, cat_39, squeeze_166, add_292, mean_15, convolution_150, mul_454, convolution_151, getitem_430, getitem_431, cat_40, squeeze_169, add_298, convolution_154, squeeze_172, view, permute_1, le, unsqueeze_234, unsqueeze_246, unsqueeze_258, mul_508, unsqueeze_270, unsqueeze_282, unsqueeze_294, mul_548, unsqueeze_306, unsqueeze_318, unsqueeze_330, mul_588, unsqueeze_342, unsqueeze_354, unsqueeze_366, mul_628, unsqueeze_378, unsqueeze_390, unsqueeze_402, mul_668, unsqueeze_414, unsqueeze_426, unsqueeze_438, mul_708, unsqueeze_450, unsqueeze_462, unsqueeze_474, mul_748, unsqueeze_486, unsqueeze_498, unsqueeze_510, mul_788, unsqueeze_522, unsqueeze_534, unsqueeze_546, mul_828, unsqueeze_558, unsqueeze_570, unsqueeze_582, mul_868, unsqueeze_594, unsqueeze_606, unsqueeze_618, mul_908, unsqueeze_630, unsqueeze_642, unsqueeze_654, mul_948, unsqueeze_666, unsqueeze_678, unsqueeze_690, mul_988, unsqueeze_702, unsqueeze_714, unsqueeze_726, mul_1028, unsqueeze_738, unsqueeze_750, unsqueeze_762, mul_1068, unsqueeze_774, unsqueeze_786, unsqueeze_798, mul_1108, unsqueeze_810, unsqueeze_822, le_1, unsqueeze_834, unsqueeze_846, unsqueeze_858, le_3, unsqueeze_870, le_4, unsqueeze_882, unsqueeze_894, unsqueeze_906, unsqueeze_918, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mixnet_l', benchmark_compiled_module)
