
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
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate, x_379], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___5_____3___se_gate => sigmoid_63
# x_379 => mul_453, sigmoid_61
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
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_324], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___5_____0___se_gate => sigmoid_51
# x_324 => mul_378, sigmoid_49
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


# kernel path: /tmp/torchinductor_youkaichao/yf/cyfcnrphu2u2rqr3lf4xgscr5oagnmhrtecrpm4k3x2pvodg77dw.py
# Source Nodes: [], Original ATen: [aten.cat]

triton_poi_fused_cat_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 196) % 960
    x1 = (xindex // 14) % 14
    x0 = xindex % 14
    x3 = (xindex // 188160)
    x6 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 240, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tl.full([1], 15, tl.int64)
    tmp7 = tmp5 < tmp6
    tmp8 = x0
    tmp9 = tmp8 < tmp6
    tmp10 = tmp7 & tmp9
    tmp11 = tmp10 & tmp4
    tmp12 = tl.load(in_ptr0 + (x0 + (15*x1) + (225*x2) + (54000*x3)), tmp11, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp4, tmp14, tmp15)
    tmp17 = tmp0 >= tmp3
    tmp18 = tl.full([1], 480, tl.int64)
    tmp19 = tmp0 < tmp18
    tmp20 = tmp17 & tmp19
    tmp21 = 1 + x1
    tmp22 = tmp21 >= tmp1
    tmp23 = tl.full([1], 17, tl.int64)
    tmp24 = tmp21 < tmp23
    tmp25 = 1 + x0
    tmp26 = tmp25 >= tmp1
    tmp27 = tmp25 < tmp23
    tmp28 = tmp22 & tmp24
    tmp29 = tmp28 & tmp26
    tmp30 = tmp29 & tmp27
    tmp31 = tmp30 & tmp20
    tmp32 = tl.load(in_ptr1 + ((-69342) + x0 + (17*x1) + (289*x2) + (69360*x3)), tmp31, other=0.0)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp31, tmp32, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp20, tmp34, tmp35)
    tmp37 = tmp0 >= tmp18
    tmp38 = tl.full([1], 720, tl.int64)
    tmp39 = tmp0 < tmp38
    tmp40 = tmp37 & tmp39
    tmp41 = 2 + x1
    tmp42 = tmp41 >= tmp1
    tmp43 = tl.full([1], 19, tl.int64)
    tmp44 = tmp41 < tmp43
    tmp45 = 2 + x0
    tmp46 = tmp45 >= tmp1
    tmp47 = tmp45 < tmp43
    tmp48 = tmp42 & tmp44
    tmp49 = tmp48 & tmp46
    tmp50 = tmp49 & tmp47
    tmp51 = tmp50 & tmp40
    tmp52 = tl.load(in_ptr2 + ((-173240) + x0 + (19*x1) + (361*x2) + (86640*x3)), tmp51, other=0.0)
    tmp53 = tl.full(tmp52.shape, 0.0, tmp52.dtype)
    tmp54 = tl.where(tmp51, tmp52, tmp53)
    tmp55 = tl.full(tmp54.shape, 0.0, tmp54.dtype)
    tmp56 = tl.where(tmp40, tmp54, tmp55)
    tmp57 = tmp0 >= tmp38
    tmp58 = tl.full([1], 960, tl.int64)
    tmp59 = tmp0 < tmp58
    tmp60 = 3 + x1
    tmp61 = tmp60 >= tmp1
    tmp62 = tl.full([1], 21, tl.int64)
    tmp63 = tmp60 < tmp62
    tmp64 = 3 + x0
    tmp65 = tmp64 >= tmp1
    tmp66 = tmp64 < tmp62
    tmp67 = tmp61 & tmp63
    tmp68 = tmp67 & tmp65
    tmp69 = tmp68 & tmp66
    tmp70 = tmp69 & tmp57
    tmp71 = tl.load(in_ptr3 + ((-317454) + x0 + (21*x1) + (441*x2) + (105840*x3)), tmp70, other=0.0)
    tmp72 = tl.full(tmp71.shape, 0.0, tmp71.dtype)
    tmp73 = tl.where(tmp70, tmp71, tmp72)
    tmp74 = tl.full(tmp73.shape, 0.0, tmp73.dtype)
    tmp75 = tl.where(tmp57, tmp73, tmp74)
    tmp76 = tl.where(tmp40, tmp56, tmp75)
    tmp77 = tl.where(tmp20, tmp36, tmp76)
    tmp78 = tl.where(tmp4, tmp16, tmp77)
    tl.store(out_ptr0 + (x6), tmp78, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vb/cvbsrz4wn6btwndvljnommpre65zn6awwnaq3ivpfkpn4batupue.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12480
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
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (188160*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (960*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uq/cuqhqpqbocrunadsuqwpjd7s24v6s42xkwdupz5krt4t5ddgj234.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/af/caf4435m4v2ardf56bycjinsd6sd3vs46lxqlekbig2f7smouv34.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12480
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 960)
    x0 = xindex % 960
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x0) + (188160*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (960*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.load(in_ptr2 + (x0 + (960*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/65/c65w34zu7xva64supwgdr33n77p7ogtavjxw7pn2itq3slliwmes.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_41', 'mutated_arg_names': []}
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
    tmp0 = tl.load(in_ptr0 + (x0 + (960*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/af/cafy2lc7cqsqf5sulke663a7qrqqng6inycdd7636ozsan3zedte.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_42', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp3 = tl.load(in_ptr1 + (y0 + (960*x2) + (188160*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/3o/c3ouu4wtitb6scnz3jspk7ypwhgquuy4l6vke5waalchmf6ffwlj.py
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
    size_hints=[256, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_43', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/jt/cjtwcnde5b4vznsftwjm4qhzplig6gcuv5k7c5c27deso7u2n2ni.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_44', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/jv/cjv7eibujj6ih3klshtmdna56jyl4y5vshqcelz4vm5viwdkfjfv.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_45', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/gh/cghla7a4cpuzi5q6wkacgdkkkkzvrdjvjzguwccxrb2mwlpce7mr.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_46', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/t5/ct5bybh5mmlozll56pmmhpt5lc5oyx6i63drqromxbuxx5wszqjt.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate, x_297], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___4_____3___se_gate => sigmoid_47
# x_297 => mul_353, sigmoid_45
triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_47', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/p5/cp5xq65r5x7tonrsehhd7tn4yvcuphut5doeq4kikoytkegj5rqr.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_48', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/qz/cqzw2zfdstfwo6ioyhdybunx35v7dgmuzngpdlzupsx6udou5lms.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate], Original ATen: [aten.add, aten.cat, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___4_____3___se_gate => sigmoid_47
triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_49', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ox/coxl44j5glbodpcqrzhrwk4lh3swht7dpcimuhqw64bfct6vzalw.py
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
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_50', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ol/colhzz7n5sgnev6wiw4ljlcz6rkbkqpckcz6xw4xwmib5hmtk4gz.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_51', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/k2/ck2yq4fze2mv2a6pg527a6ognv7nevkq46ov7oxiw2hker5bxlnu.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_52', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/p4/cp4fyd46upspo2g4abbvifrqvkfhghkan4i2ahtk2krzlbzk2ahr.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_53', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/6o/c6oomobgtnjjkzrq7wrthz7c46r62qlmhts22ovr2xyfn52fjf3k.py
# Source Nodes: [], Original ATen: [aten.cat, aten.mul]

triton_poi_fused_cat_mul_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_mul_54', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/m6/cm6oue6v6tvdyveg7tepl2j5p2sgdposb67c4twzfdasebde27cp.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]

triton_red_fused_add_cat_native_batch_norm_backward_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_batch_norm_backward_55', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/it/citxke3klgrdd75po6ebx5j5xg4dcng72lolmydeklzsrv33uqr3.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]

triton_poi_fused_add_cat_native_batch_norm_backward_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_native_batch_norm_backward_56', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/rm/crm25xpypx6b2fosaehrgj7f6jp6z2mw4oco2l4ehhlfwspfvifs.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat]

triton_poi_fused_add_cat_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_57', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/co/ccobf2kffneious6lbc5c3w3kii6cv6c7zutvsniwfv3xutoia4q.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]

triton_poi_fused_add_cat_native_batch_norm_backward_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_native_batch_norm_backward_58', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/5l/c5lpryn5srlwm6ynrffr72jbg75ba74q7vm4oceikdqjyfc3cdgn.py
# Source Nodes: [x_239], Original ATen: [aten.mul, aten.silu, aten.sum]
# x_239 => mul_278, sigmoid_33
triton_red_fused_mul_silu_sum_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_silu_sum_59', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/cm/ccm45xp2c6p6ghw4n3sspinp5lggdqaqlbq2m4v5c55lq5ogqcdp.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_239], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => sigmoid_35
# x_239 => mul_278, sigmoid_33
triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_60', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/ny/cny4pvrhnpq2jlnr6jireuk5drs6rsyyk274tucdhcrngj7aaypw.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_61', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ir/cirluubv64dc6faqzubj4t7kcqnnufiu2vj2sjb3dxjmzzsftaxe.py
# Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]

triton_poi_fused_add_fill_mul_sigmoid_sub_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_62', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/yr/cyrjk24xsccceyyccps6ohoxfvtiinwoulcjxiyzrhrwsh33su5m.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_63', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/wp/cwp2sirx4umgozllxpzn7mtfcggmpxy327p56ibj7rvlxuibphds.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => sigmoid_35
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_64', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/6x/c6xreh6g6itijigt5ysdcsnqxjp3umio2zzc6gd3dah4jhjjinxr.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => sigmoid_35
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_65', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/wp/cwpqyo4lfncdf6iojoqpwirgw6gbwfivwyp7afrz76n5gswdhu4h.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => sigmoid_35
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_66 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_66', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/y6/cy6o7vg6ogovirgqrrhtnmecpyzuy2lobpwnfxikc3jzlj7playx.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => sigmoid_35
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_67', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ha/chajvnh342rgenrq45z67lpqj4bx2a6lbhfue2lx47mjaaeymcoy.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => sigmoid_35
triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_68 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_68', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/4u/c4u6ntwriyhbitqf43te7p3wyvzxznvd2hub6qqqp7efyzdxpt43.py
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
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_69', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/vu/cvuimt3ywtkwbkemdqvkoaukmici4tfrrb7im3j55h5yueovgvhk.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_70', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ji/cjiiayv6f2lkekuusow7db7dpjp2s3ajhypgaolmhqmnsqcy3jpv.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_71 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_71', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/u4/cu4qrv2hykmsrux2a3ddabassct7qxqpde4curvvhpalh7lmy3ax.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_72 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_72', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/qu/cquetjwlm35ggbmb2yc7yxvk773u6t37epml2haghrk5jvu6pgxg.py
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
    size_hints=[128, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_73', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/im/cimzzdlxl5xn57ovmf54uqae3ngtc6mjnu3rt33typ63jc6dwawv.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_74', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/3z/c3zkzblznb5ipsniaoxjbohwliacprj6ofjqbpk5okp4kcw352id.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_75 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_75', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/lu/cluapfcsco6jy4nry5amludvmbtfd3f6sumflmfvv2rnlrg2nnt7.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_76 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_76', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/sh/cshb7z45vu3pivh77lzuoc2yogvp75lndcsiiomqglf2dgerezy2.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___se_gate, x_221], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___3_____3___se_gate => sigmoid_31
# x_221 => mul_253, sigmoid_29
triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_77 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_77', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/7z/c7zdjugdcvgieacduqu2myjarite4cz2quzsetu5tjya6ud5qy6s.py
# Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]

triton_poi_fused_add_fill_mul_sigmoid_sub_78 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_78', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/r7/cr7eilpocjf5scm5f7md7qsvalwmcn4aqu7mdahdf33d6tyl3coo.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_79 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_79', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/as/casln4htafmofebspcdy4whyxj2zz7dx6pdaq5st5jd4vh33bnzi.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___se_gate], Original ATen: [aten.add, aten.cat, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___3_____3___se_gate => sigmoid_31
triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_80 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_80', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/zh/czhiu72zasum3fin4ims7hnpapwrjjmtlfeixt32re2364g7j2kz.py
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
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_81', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/u2/cu2qdq5nrc3m5flnq26f6g3ls3i7dr6kqmkz2i2cb7jgkv4pjuj3.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_82 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_82', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/n7/cn72la52u3664b5jxu5hlqzhpdgdsnvm3kfose46qfotls6qau3a.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_83 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_83', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/xb/cxbsigqflbtldscjj4t4abgqtueeiypt6tap34ipabxlgcrq375l.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_84 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_84', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/k5/ck5l4nzg3kgjyyoa5hektq6o3afexv5ebidunn7k73ypi76abtqg.py
# Source Nodes: [], Original ATen: [aten.cat, aten.mul]

triton_poi_fused_cat_mul_85 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_mul_85', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/od/code775x43jnbtcv5mllg2hkp4jiyykjwwb73uq7cq7p6uzucxwo.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]

triton_red_fused_add_cat_native_batch_norm_backward_86 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_batch_norm_backward_86', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/gg/cggmkd3j7a2x6eroxuuvs2oa2aacffj4f3cobxupkwfeybf6lbe4.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]

triton_poi_fused_add_cat_native_batch_norm_backward_87 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_native_batch_norm_backward_87', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/cw/ccwgsw7mmxfsinxlwdg47yklnxhis6lrz77ftxdpfpfp2seeca2i.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat]

triton_poi_fused_add_cat_88 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_88', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/gl/cglbxlohbmhguvcbiw677dpdjo5zr6grvgamda7ekgqgoqz57hkt.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]

triton_poi_fused_add_cat_native_batch_norm_backward_89 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_native_batch_norm_backward_89', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/5s/c5snr7mbmy2fin2t5rrtgjp2zotcj3h2gzsmx4dqezcjt4kaip7p.py
# Source Nodes: [x_163], Original ATen: [aten.mul, aten.silu, aten.sum]
# x_163 => mul_178, sigmoid_17
triton_red_fused_mul_silu_sum_90 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_silu_sum_90', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/xn/cxn2nuv4snzzw7jwqndhm2gzlgad2ux5zlmh4bsr64m6ugdi3f2j.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_163], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_19
# x_163 => mul_178, sigmoid_17
triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_91 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_91', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/5z/c5zvpgg3mzc5xwibdhzihyus4snakixduhiwu7emgnzqsd7k5vfk.py
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


# kernel path: /tmp/torchinductor_youkaichao/is/cisvdy45yttdffpabw6ql435mapcp42y5rx77beyioulekbcygwg.py
# Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]

triton_poi_fused_add_fill_mul_sigmoid_sub_93 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_93', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/bx/cbxwpvs2o3tqeqhh4lbikf3bqm2biqjp7vs6tgf6q3nihpk6s2ud.py
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
    size_hints=[16, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_94', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/7r/c7r3wxkaxx76mowajoqnbb2wy5ten7dt3tbotbyb6qwrn4z5y6bg.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_19
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_95 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_95', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/6x/c6xq5wyfunhumjmyn7k4c7wptmrw2jacg2drnccmqafzah5gubpl.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_19
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_96 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_96', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/vq/cvqmr6x7wfaeueqjvv2iel7nfhbwrtjegeqq53co5ehdgyujlqpn.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_19
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_97 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_97', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/gu/cgupye74rx5rfb4tkvu5ido55k6xvwtrl54clzpnzu6gjnzwv3cf.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_19
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_98 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_98', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/bd/cbdsw2yabtxulzorvf3axuy7qb5c6w2grxgwmfjlhywr6efc5mcf.py
# Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___3_____0___se_gate => sigmoid_19
triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_99 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_99', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/l2/cl2xlv7gima4i3txncpyzy3ysufsc3rbmmgnasfxibrx7hdc5hu5.py
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
    tmp0 = tl.load(in_ptr0 + (224 + y0 + (336*x2) + (65856*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (224 + y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (224 + y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lp/clph6awghkhc4a2lih23b6bn363mkpjlhn4rwpfpr6hceylo6lm6.py
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
    tmp0 = tl.load(in_ptr0 + (112 + y0 + (336*x2) + (65856*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (112 + y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (112 + y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/te/ctedscjz7bbe3dgkzzdjodqakrtnjursj6lgqjd5ok5u7ooadten.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_102 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_102', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/hv/chv3qszh3otgywu5clqjz6rbcuqw5mmnaoigqmfc3uzivp5czda2.py
# Source Nodes: [], Original ATen: [aten.cat]

triton_poi_fused_cat_103 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_103', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2107392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 784) % 336
    x1 = (xindex // 28) % 28
    x0 = xindex % 28
    x3 = (xindex // 263424)
    x6 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tl.full([1], 29, tl.int64)
    tmp7 = tmp5 < tmp6
    tmp8 = x0
    tmp9 = tmp8 < tmp6
    tmp10 = tmp7 & tmp9
    tmp11 = tmp10 & tmp4
    tmp12 = tl.load(in_ptr0 + (x0 + (29*x1) + (841*x2) + (94192*x3)), tmp11, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp4, tmp14, tmp15)
    tmp17 = tmp0 >= tmp3
    tmp18 = tl.full([1], 224, tl.int64)
    tmp19 = tmp0 < tmp18
    tmp20 = tmp17 & tmp19
    tmp21 = 1 + x1
    tmp22 = tmp21 >= tmp1
    tmp23 = tl.full([1], 31, tl.int64)
    tmp24 = tmp21 < tmp23
    tmp25 = 1 + x0
    tmp26 = tmp25 >= tmp1
    tmp27 = tmp25 < tmp23
    tmp28 = tmp22 & tmp24
    tmp29 = tmp28 & tmp26
    tmp30 = tmp29 & tmp27
    tmp31 = tmp30 & tmp20
    tmp32 = tl.load(in_ptr1 + ((-107600) + x0 + (31*x1) + (961*x2) + (107632*x3)), tmp31, other=0.0)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp31, tmp32, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp20, tmp34, tmp35)
    tmp37 = tmp0 >= tmp18
    tmp38 = tl.full([1], 336, tl.int64)
    tmp39 = tmp0 < tmp38
    tmp40 = 2 + x1
    tmp41 = tmp40 >= tmp1
    tmp42 = tl.full([1], 33, tl.int64)
    tmp43 = tmp40 < tmp42
    tmp44 = 2 + x0
    tmp45 = tmp44 >= tmp1
    tmp46 = tmp44 < tmp42
    tmp47 = tmp41 & tmp43
    tmp48 = tmp47 & tmp45
    tmp49 = tmp48 & tmp46
    tmp50 = tmp49 & tmp37
    tmp51 = tl.load(in_ptr2 + ((-243868) + x0 + (33*x1) + (1089*x2) + (121968*x3)), tmp50, other=0.0)
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp50, tmp51, tmp52)
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp37, tmp53, tmp54)
    tmp56 = tl.where(tmp20, tmp36, tmp55)
    tmp57 = tl.where(tmp4, tmp16, tmp56)
    tl.store(out_ptr0 + (x6), tmp57, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zq/czqe42xgbux6gmbnkmli552rkq65pkzkrhmalaarh3ockby3cqoj.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_104 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_104', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16464
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((784*x1) + (263424*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (336*r2) + (43008*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6i/c6iabpuenduzjda3x5hg7ensxwzzn6ijlrrallm7t5pyah3ge35s.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_105 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_105', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qx/cqx5xdvtkc2l2rbba2a2u7tvvpxw72c3tcnmbpbr3gxqbhdkrsh4.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_106 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_106', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16464
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 336
    x1 = (xindex // 336)
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((784*x0) + (263424*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (336*r2) + (43008*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (336*r2) + (43008*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 - tmp4
        tmp6 = tmp2 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wo/cwomjqgjubspza3lojkkabeadz3ehc5rzy3rrbh2t2ahrhiryhpw.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_107 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_107', 'mutated_arg_names': []}
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
    tmp0 = tl.load(in_ptr0 + (x0 + (336*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uz/cuz5ll2fw34gumh7s6mjw5ddaqn6qcgajgs2ymkwfiymhtz2yo4d.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_108 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_108', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp3 = tl.load(in_ptr1 + (y0 + (336*x2) + (263424*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.00015943877551020407
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
    tl.store(in_out_ptr0 + (x2 + (784*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nk/cnkne4gaoi25w6eulv4g6d74co6dlcu2errzs7sxtclcbucdm4j2.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_109 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_109', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/od/codpml566ixvdzjwq2nnoixdzo6f6mky6vticgxtx4ocm2qn5a6x.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_110 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_110', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/y7/cy7pwk2ey5sznbpla62x66lbjsl2jiycgrrx7bf3ira2eol2cf2p.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_111 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_111', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/2f/c2f7pzmut74bbwa2xwlwnczbfq2espqvivb3tpns2cprhrvpbcqy.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_112 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_112', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/va/cvaeoofgcmgvndmdoctjlqtgkw77maot72yiwl5bgmrd4mcn522x.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___se_gate, x_138], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___2_____3___se_gate => sigmoid_15
# x_138 => mul_153, sigmoid_13
triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_113 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_113', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/l4/cl43aggpgfxx53yiy6flvlispddnwlbmj4ksyxx5mtufv45adclm.py
# Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]

triton_poi_fused_add_fill_mul_sigmoid_sub_114 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_114', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/jf/cjff77nrc3hdh66qojcosqdj4t6cxkjcc2wwd5akdgb5phhafx3u.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_115 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_115', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/vz/cvzfuwxgskrws74b7pvcsjkhkuhzpvvpgeoleh2enb3otafhyf2m.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___se_gate], Original ATen: [aten.add, aten.cat, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___2_____3___se_gate => sigmoid_15
triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_116 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_116', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/vn/cvnjypvyrzd43lbz3qdc6dfbmarmdeftbrr2ffblzx53thmlqdtr.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_117 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_117', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ml/cmlcrbcbl6u3hoz4kzmb7hog44dz7gtkbj4kz2ry3mhmxdjfwdet.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_118 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_118', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/x7/cx76bhjs3nwkxtuklles5ovlhhxzr2vy53xoqbzhnlboollmdwpv.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_119 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_119', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/fl/cfll6x2d6wgynlhsmuu32zy7nizpky3lp2gi6tjezgxkdgvwmtod.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_120 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_120', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/zh/czhdgppqznn7rdpi374zjz3cz3pfpu3jaif2uemzvyc5ktmybwej.py
# Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_cat_mul_native_batch_norm_backward_121 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_mul_native_batch_norm_backward_121', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/hr/chrtnvehwfzvlvioj4cfi6avmom2hyubq7pjnflfugzsopngbdx3.py
# Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_cat_mul_native_batch_norm_backward_122 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_mul_native_batch_norm_backward_122', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ja/cjabjpzveal6g3wevmdoec7fvw7tjdhcrv5i4rqo3vyxwisnbo47.py
# Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_cat_mul_native_batch_norm_backward_123 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_mul_native_batch_norm_backward_123', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/kh/ckhzqlpkyszus25whknnbcbbswamnuwxgg5zf5isinzxc4tbusyf.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]

triton_red_fused_add_cat_native_batch_norm_backward_124 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_batch_norm_backward_124', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/wd/cwdzvgabbah3awvki7m6ccov22ptjrbit6xylazket4bai2oftzm.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]

triton_poi_fused_add_cat_native_batch_norm_backward_125 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_native_batch_norm_backward_125', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/3n/c3nx2eakn4kfq6b5g7ihwgmzvcrmylqu6v27wzf3cacowfan4em7.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat]

triton_poi_fused_add_cat_126 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_126', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/wj/cwjufoun4wsndvbii4zubielqml6heoa2w3ncdk4w662rsp2wtki.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]

triton_poi_fused_add_cat_native_batch_norm_backward_127 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_native_batch_norm_backward_127', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/vl/cvlr3t2baycqst5qkaa3ncpixjczmmykvpgpxpcll2pcxk7lfrob.py
# Source Nodes: [x_80], Original ATen: [aten.mul, aten.silu, aten.sum]
# x_80 => mul_78, sigmoid_1
triton_red_fused_mul_silu_sum_128 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_silu_sum_128', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/wf/cwffeedvuog2rcmt2wi4hs67h7nyxp6agvy37mzhiwpmxvypow5y.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_80], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => sigmoid_3
# x_80 => mul_78, sigmoid_1
triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_129 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_129', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/ib/ciben3b3wjclbq52fo6u7mzmu7ienriwuehn2v7nsy2u4j3wn2rx.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_130 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_130', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/w6/cw6s5hway5zdetkk2fen5wqroqnvudmprmbs53uvwdvkcazu6qlg.py
# Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]

triton_poi_fused_add_fill_mul_sigmoid_sub_131 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_sub_131', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/fm/cfmne6c73vnkx4mi7hdae2oxv4kvse3ufoylgoaufar64rpewrvi.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_132 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_132', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/tc/ctcjxn4zn4nzpr5f2svrwa5w75nr2w575b2y67ki6tghh3tk5wbm.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => sigmoid_3
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_133 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_133', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/in/cinauayr66ju2avmzskc7rryh4nvbftlw6v36vq3wiyfgnptg3k6.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => sigmoid_3
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_134 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_134', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/lx/clx6svrdkiavehlacn2dkxw6a7c36tsvdafjctkmm3jnllelk7ox.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => sigmoid_3
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_135 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_135', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/by/cbyba5fqws2dste4wmflu2ewafommq2nqyvrmdkv3hnk2ewpclnq.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => sigmoid_3
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_136 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_136', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/td/ctdwjvhp6ollnhpklnt7xa2zswq5mnkirqlryoqx6zuv2altcghi.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => sigmoid_3
triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_137 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_137', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/kx/ckxtfusl473bolbzg2bi7bx2fqkume6hs555esbscbuad3tjdzra.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_138 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_138', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ne/cnevbzcpfn43jqw52ag757pi2lswmlvsa2sblspgiwjzfuepmlvb.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_139 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_139', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/av/cavtpbpdpwftoflaisone4fu4unnjillqrcjfh3liddcd7s4iyhv.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_140 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_140', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ha/chadeofvb56h3dl635gqlyu2kn2sxs6lzv7ogv7ieqntpbnyta3z.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_141 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_141', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ct/ccto3w7xjh535mtsehpz5pjd6uyrx3fnoj5zkkfex7wu35hx46wf.py
# Source Nodes: [], Original ATen: [aten.cat]

triton_poi_fused_cat_142 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_142', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6021120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 3136) % 240
    x1 = (xindex // 56) % 56
    x0 = xindex % 56
    x3 = (xindex // 752640)
    x6 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 60, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tl.full([1], 57, tl.int64)
    tmp7 = tmp5 < tmp6
    tmp8 = x0
    tmp9 = tmp8 < tmp6
    tmp10 = tmp7 & tmp9
    tmp11 = tmp10 & tmp4
    tmp12 = tl.load(in_ptr0 + (x0 + (57*x1) + (3249*x2) + (194940*x3)), tmp11, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp4, tmp14, tmp15)
    tmp17 = tmp0 >= tmp3
    tmp18 = tl.full([1], 120, tl.int64)
    tmp19 = tmp0 < tmp18
    tmp20 = tmp17 & tmp19
    tmp21 = 1 + x1
    tmp22 = tmp21 >= tmp1
    tmp23 = tl.full([1], 59, tl.int64)
    tmp24 = tmp21 < tmp23
    tmp25 = 1 + x0
    tmp26 = tmp25 >= tmp1
    tmp27 = tmp25 < tmp23
    tmp28 = tmp22 & tmp24
    tmp29 = tmp28 & tmp26
    tmp30 = tmp29 & tmp27
    tmp31 = tmp30 & tmp20
    tmp32 = tl.load(in_ptr1 + ((-208800) + x0 + (59*x1) + (3481*x2) + (208860*x3)), tmp31, other=0.0)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp31, tmp32, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp20, tmp34, tmp35)
    tmp37 = tmp0 >= tmp18
    tmp38 = tl.full([1], 180, tl.int64)
    tmp39 = tmp0 < tmp38
    tmp40 = tmp37 & tmp39
    tmp41 = 2 + x1
    tmp42 = tmp41 >= tmp1
    tmp43 = tl.full([1], 61, tl.int64)
    tmp44 = tmp41 < tmp43
    tmp45 = 2 + x0
    tmp46 = tmp45 >= tmp1
    tmp47 = tmp45 < tmp43
    tmp48 = tmp42 & tmp44
    tmp49 = tmp48 & tmp46
    tmp50 = tmp49 & tmp47
    tmp51 = tmp50 & tmp40
    tmp52 = tl.load(in_ptr2 + ((-446396) + x0 + (61*x1) + (3721*x2) + (223260*x3)), tmp51, other=0.0)
    tmp53 = tl.full(tmp52.shape, 0.0, tmp52.dtype)
    tmp54 = tl.where(tmp51, tmp52, tmp53)
    tmp55 = tl.full(tmp54.shape, 0.0, tmp54.dtype)
    tmp56 = tl.where(tmp40, tmp54, tmp55)
    tmp57 = tmp0 >= tmp38
    tmp58 = tl.full([1], 240, tl.int64)
    tmp59 = tmp0 < tmp58
    tmp60 = 3 + x1
    tmp61 = tmp60 >= tmp1
    tmp62 = tl.full([1], 63, tl.int64)
    tmp63 = tmp60 < tmp62
    tmp64 = 3 + x0
    tmp65 = tmp64 >= tmp1
    tmp66 = tmp64 < tmp62
    tmp67 = tmp61 & tmp63
    tmp68 = tmp67 & tmp65
    tmp69 = tmp68 & tmp66
    tmp70 = tmp69 & tmp57
    tmp71 = tl.load(in_ptr3 + ((-714228) + x0 + (63*x1) + (3969*x2) + (238140*x3)), tmp70, other=0.0)
    tmp72 = tl.full(tmp71.shape, 0.0, tmp71.dtype)
    tmp73 = tl.where(tmp70, tmp71, tmp72)
    tmp74 = tl.full(tmp73.shape, 0.0, tmp73.dtype)
    tmp75 = tl.where(tmp57, tmp73, tmp74)
    tmp76 = tl.where(tmp40, tmp56, tmp75)
    tmp77 = tl.where(tmp20, tmp36, tmp76)
    tmp78 = tl.where(tmp4, tmp16, tmp77)
    tl.store(out_ptr0 + (x6), tmp78, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/v7/cv75ebzdgu2qbmgmqgducqxje3oyrsume7ve54jergnvtu32cvf4.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_143 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_143', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 47040
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x1) + (752640*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (240*r2) + (30720*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ae/cae2auliuqtj6thrwk2l7uzlydgjrzr3is2motvpbxk6nblb2b7y.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_144 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_144', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nx/cnxzkdqwh2464a2t3n2hf4ey4mozdz7ek2sgb4tvpppuhavffc5e.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_145 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_145', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 47040
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 240
    x1 = (xindex // 240)
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (752640*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (240*r2) + (30720*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (240*r2) + (30720*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 - tmp4
        tmp6 = tmp2 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sm/csmu52c3feod6car27karjfyx2hi5y6k5knlzrevihux6clqjbph.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_146 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 256],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_146', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 240
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
        tmp0 = tl.load(in_ptr0 + (x0 + (240*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mn/cmnhrcybzqonsnstssoy63conpsoqy3au7pl66chhtsiedey4nku.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_147 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_147', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp3 = tl.load(in_ptr1 + (y0 + (240*x2) + (752640*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 3.985969387755102e-05
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
    tl.store(in_out_ptr0 + (x2 + (3136*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ny/cnyhonlqv2jb2wbtieth63l5wahdckxyszdi36odn6dchq235epp.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_148 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_148', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/uk/cuk3b3bfzwrtjw3sodqn5xglapjhunn2y5arvnm5ecjou6cfsdbh.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_149 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_149', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/j3/cj3hse52x2jeq4beorflislzrxmnd5zewkvnx3t5v5la66d6dpzz.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_150 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_150', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/5r/c5r4vmyfiw65ggqspx3kfrjv747rhajd32p3qtdcwql3rt3krldx.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_151 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_151', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/h7/ch7saekfjbmfmf24xlzpy2loxaravuronuftqtognq5stgg5pevq.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_152 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_152', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/wf/cwfvqliaor3us55sevz7rfjrwi4aztl4vrz7j7jzheyfvpbaetxn.py
# Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_cat_native_batch_norm_backward_threshold_backward_153 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_batch_norm_backward_threshold_backward_153', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/gr/cgrjl7tnpxezhtn3ixphaamx4ovu2fa4hgajo7vdyagchk6lrz7m.py
# Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_cat_native_batch_norm_backward_threshold_backward_154 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_batch_norm_backward_threshold_backward_154', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/aa/caalqmz4pfjxc2evref33tqahszgn4folki2zbx52u2xnc4wlent.py
# Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_cat_native_batch_norm_backward_threshold_backward_155 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_batch_norm_backward_threshold_backward_155', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/el/celcftkrul4745u7ou2jv2feylchulauzeequzqfguudptvaas6w.py
# Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_cat_native_batch_norm_backward_threshold_backward_156 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_batch_norm_backward_threshold_backward_156', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/jv/cjv7swchxm2gamsxor4dnm6dduyw3vwthxez7lwki3d7g2zk5mxt.py
# Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_cat_native_batch_norm_backward_threshold_backward_157 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_native_batch_norm_backward_threshold_backward_157', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ja/cjanjwlj545fl4v5evsmrezdznay4jxmmmczievto65avr7ovf3w.py
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_158', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/vc/cvc3nycixz5norimlbqt3xnv5rau4scvokts4ayrgtzc2sdq3x2n.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_159 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_159', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/4k/c4kkwhblmxr4eqn6dvygj3j3z5ccytpulxy7a2jmolyrgsv7v4g2.py
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_160', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/4n/c4n5tcslcjy2nijdbgsg6cdcbz5vk237jstj2uutk7o2wttoh4ct.py
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
    size_hints=[128, 256],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_161', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/3t/c3teoxlpojsrlakyfhflrmni3ph5retgl56ofytbmxczczul2wjl.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_162 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_162', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/b4/cb466v2xr5wndrmhv553y5teualeeya3oceuwalfiu72feztobnt.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]

triton_red_fused_add_cat_native_batch_norm_backward_163 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_batch_norm_backward_163', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/hr/chrgdhs46nzb4kp2zkcngw7lkspxvhdvix3zxdv6yinqmeh6hx6g.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]

triton_per_fused_add_cat_native_batch_norm_backward_164 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cat_native_batch_norm_backward_164', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ft/cftzzslcc4x2fyqmf6lsofcmz44asrbls4gjvelcg2ca7xlerqtr.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]

triton_poi_fused_add_cat_native_batch_norm_backward_165 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_native_batch_norm_backward_165', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/ko/ckorp56cjsy6jruhmywk4ga2jncc6i2xfjd7stycwdpgymmfjtkz.py
# Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_cat_native_batch_norm_backward_threshold_backward_166 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_batch_norm_backward_threshold_backward_166', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/4b/c4bhsp2yf5ijdjp3utumvu535rdx372vjfqvb7u375bg2kbtmgu5.py
# Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_cat_native_batch_norm_backward_threshold_backward_167 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_batch_norm_backward_threshold_backward_167', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/o5/co5cmjt3hy7v4e774jnwuiz55qmz4troq7nduxkvcmzngkv25izf.py
# Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_cat_native_batch_norm_backward_threshold_backward_168 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_batch_norm_backward_threshold_backward_168', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/q4/cq43sh6r6pziknmpxvfmeaf5buhorangcjzwg3svqadnhdmmncqk.py
# Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_cat_native_batch_norm_backward_threshold_backward_169 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_batch_norm_backward_threshold_backward_169', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/w3/cw3dvcbgpbhgisjsx4t3twucnolplf776vex4a5sqyr6cylnsqh7.py
# Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_cat_native_batch_norm_backward_threshold_backward_170 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_native_batch_norm_backward_threshold_backward_170', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/du/cdu4n2h44s3aamy7xzyzekdr6zfv5cecjoqq7rngphkoyd5krppj.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_171 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_171', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/gg/cggsczha5e6t77uffzeuqgifvavlg4sdyj3yul35d6lmgg4x7pn3.py
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
    size_hints=[512, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_172', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/e5/ce5jgjfmbmu4kcplmvblkfb3paqvx3t5e6bk4wnx7jvxrcid5ldr.py
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
    size_hints=[512, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_173', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/oi/coi5josegyowdyettbbc2fhy6r5vp2ikwve3eggzm6zcb46lgypw.py
# Source Nodes: [], Original ATen: [aten.cat]

triton_poi_fused_cat_174 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_174', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19267584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 12544) % 192
    x1 = (xindex // 112) % 112
    x0 = xindex % 112
    x3 = (xindex // 2408448)
    x6 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x1
    tmp6 = tl.full([1], 113, tl.int64)
    tmp7 = tmp5 < tmp6
    tmp8 = x0
    tmp9 = tmp8 < tmp6
    tmp10 = tmp7 & tmp9
    tmp11 = tmp10 & tmp4
    tmp12 = tl.load(in_ptr0 + (x0 + (113*x1) + (12769*x2) + (817216*x3)), tmp11, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp4, tmp14, tmp15)
    tmp17 = tmp0 >= tmp3
    tmp18 = tl.full([1], 128, tl.int64)
    tmp19 = tmp0 < tmp18
    tmp20 = tmp17 & tmp19
    tmp21 = 1 + x1
    tmp22 = tmp21 >= tmp1
    tmp23 = tl.full([1], 115, tl.int64)
    tmp24 = tmp21 < tmp23
    tmp25 = 1 + x0
    tmp26 = tmp25 >= tmp1
    tmp27 = tmp25 < tmp23
    tmp28 = tmp22 & tmp24
    tmp29 = tmp28 & tmp26
    tmp30 = tmp29 & tmp27
    tmp31 = tmp30 & tmp20
    tmp32 = tl.load(in_ptr1 + ((-846284) + x0 + (115*x1) + (13225*x2) + (846400*x3)), tmp31, other=0.0)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp31, tmp32, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp20, tmp34, tmp35)
    tmp37 = tmp0 >= tmp18
    tmp38 = tl.full([1], 192, tl.int64)
    tmp39 = tmp0 < tmp38
    tmp40 = 2 + x1
    tmp41 = tmp40 >= tmp1
    tmp42 = tl.full([1], 117, tl.int64)
    tmp43 = tmp40 < tmp42
    tmp44 = 2 + x0
    tmp45 = tmp44 >= tmp1
    tmp46 = tmp44 < tmp42
    tmp47 = tmp41 & tmp43
    tmp48 = tmp47 & tmp45
    tmp49 = tmp48 & tmp46
    tmp50 = tmp49 & tmp37
    tmp51 = tl.load(in_ptr2 + ((-1751956) + x0 + (117*x1) + (13689*x2) + (876096*x3)), tmp50, other=0.0)
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp50, tmp51, tmp52)
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp37, tmp53, tmp54)
    tmp56 = tl.where(tmp20, tmp36, tmp55)
    tmp57 = tl.where(tmp4, tmp16, tmp56)
    tl.store(out_ptr0 + (x6), tmp57, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pe/cpet2vzquf7sxw5ucu4wsidzsal2qarec5efvp3do65rd2qgh7zt.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_175 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_175', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (192*r2) + (37632*x0)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((12544*x1) + (2408448*((r2 + (196*x0)) // 12544)) + ((r2 + (196*x0)) % 12544)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jb/cjbthcr4u4ce7nzukoztlkuufuqjy64d4x4wtx23jx4tbg7liazg.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_176 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_176', 'mutated_arg_names': []}
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
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xd/cxd2jz4zcqbs2264aj5k3ojnri3jxtc2ceasczagqlanu4td4q2j.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_177 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_177', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 192
    x1 = (xindex // 192)
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r2) + (37632*x1)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((12544*x0) + (2408448*((r2 + (196*x1)) // 12544)) + ((r2 + (196*x1)) % 12544)), rmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (192*r2) + (37632*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp3 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vz/cvzvmmg3jtu5m2oglobgfbwybbpqhpo3xolr2mdffg2cet4ly4ce.py
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
    size_hints=[256, 512],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_178', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 512
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


# kernel path: /tmp/torchinductor_youkaichao/5q/c5qvpvt55mpoom7opbv5nuazs64rigkpflkevmturqu57ljnqksd.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_native_batch_norm_backward_threshold_backward_179 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_threshold_backward_179', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 100352
    xnumel = 192
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
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (12544*x2) + (2408448*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (192*y3)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tmp6 = tmp4 - tmp5
    tmp8 = 9.964923469387754e-06
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tmp6 * tmp12
    tmp14 = tmp3 - tmp13
    tmp16 = tmp15 * tmp8
    tmp17 = tmp14 - tmp16
    tmp19 = tmp10 * tmp18
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2a/c2atn7epuk6bkqdqxtws7qusxwxgxe4qis5j4ewv4teachcrttsf.py
# Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward]

triton_red_fused_cat_native_batch_norm_backward_180 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_batch_norm_backward_180', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/2w/c2w72mw2nxi46wjuv3d2pejqyh3mavb4u677bs2bqjqezh5prubq.py
# Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward]

triton_per_fused_cat_native_batch_norm_backward_181 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_batch_norm_backward_181', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/s5/cs5yxhdyxntqhlfozvfdf24e52pfqz4ejcxtq6rxohjko5g5riy7.py
# Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward]

triton_per_fused_cat_native_batch_norm_backward_182 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_batch_norm_backward_182', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/rj/crjce6knvupz3av2i5eca4vw2t2q3uk637knkpfikoiay2ewvwgn.py
# Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_cat_convolution_backward_native_batch_norm_backward_183 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_backward_native_batch_norm_backward_183', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/bg/cbgwm7clxlbqiomdqatvp25awwvwphd6zarlqwbdx7noujtcvrax.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_184 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_184', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/tx/ctxoehj6kgnztx7kjhedyd77szjnhzvnbfeysiqa6o5ir66tfogf.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_185 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_185', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/45/c45t6csp2x64xqomptjn2bcnjdkus4xmuwiwtm33oo3zrssuyzxo.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_186 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_186', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/k3/ck3iyruyaicjs4ffdgvqpv634gx3iwt3inh3mqyfpqwd6wgyltvl.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_187 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_187', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/zn/cznarahitgpmzfi37oknwr3cjfoulzvcal4scvywm7cwjuu26cq5.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_188 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_188', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/6x/c6x35jf67fcqx2ayxfdabirnfbd3broeqr7u2xx3dapbdexi7ev3.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_cat_native_batch_norm_backward_threshold_backward_189 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_batch_norm_backward_threshold_backward_189', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/qx/cqxnstocs2qiv3v7dy6zmobiilor43vob4jhdsg4xx3pbz2vvrah.py
# Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_cat_convolution_backward_native_batch_norm_backward_threshold_backward_190 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_convolution_backward_native_batch_norm_backward_threshold_backward_190', 'mutated_arg_names': ['in_out_ptr0']},
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
    primals_1, primals_2, primals_4, primals_6, primals_8, primals_10, primals_11, primals_12, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_26, primals_27, primals_28, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_54, primals_55, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_105, primals_106, primals_107, primals_108, primals_110, primals_112, primals_114, primals_116, primals_118, primals_120, primals_122, primals_124, primals_126, primals_128, primals_130, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_146, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_155, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_165, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_175, primals_177, primals_178, primals_179, primals_180, primals_182, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_193, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_205, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_217, primals_219, primals_220, primals_221, primals_222, primals_223, primals_225, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_236, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_248, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_260, primals_262, primals_263, primals_264, primals_265, primals_267, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_277, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_288, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_299, primals_301, primals_302, primals_303, constant_pad_nd, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, getitem_6, getitem_7, cat, squeeze_10, constant_pad_nd_1, constant_pad_nd_2, constant_pad_nd_3, cat_1, squeeze_13, getitem_26, getitem_29, cat_2, squeeze_16, getitem_32, getitem_33, cat_3, squeeze_19, relu_4, convolution_12, squeeze_22, getitem_40, getitem_43, cat_4, squeeze_25, add_46, convolution_15, squeeze_28, constant_pad_nd_4, constant_pad_nd_5, constant_pad_nd_6, constant_pad_nd_7, cat_5, squeeze_31, add_56, mean, convolution_20, mul_79, convolution_21, mul_80, convolution_22, squeeze_34, getitem_72, getitem_73, cat_6, squeeze_37, getitem_78, getitem_81, cat_7, squeeze_40, add_71, mean_1, convolution_27, mul_104, convolution_28, getitem_84, getitem_85, cat_8, squeeze_43, getitem_88, getitem_89, cat_9, squeeze_46, getitem_94, getitem_97, cat_10, squeeze_49, add_87, mean_2, convolution_35, mul_129, convolution_36, getitem_100, getitem_101, cat_11, squeeze_52, getitem_104, getitem_105, cat_12, squeeze_55, getitem_110, getitem_113, cat_13, squeeze_58, add_103, mean_3, convolution_43, mul_154, convolution_44, getitem_116, getitem_117, cat_14, squeeze_61, add_109, convolution_47, squeeze_64, constant_pad_nd_8, constant_pad_nd_9, constant_pad_nd_10, cat_15, squeeze_67, add_119, mean_4, convolution_51, mul_179, convolution_52, mul_180, convolution_53, squeeze_70, getitem_138, getitem_139, cat_16, squeeze_73, getitem_146, getitem_151, getitem_156, getitem_161, cat_17, squeeze_76, add_134, mean_5, convolution_60, mul_204, convolution_61, getitem_164, getitem_165, cat_18, squeeze_79, getitem_168, getitem_169, cat_19, squeeze_82, getitem_176, getitem_181, getitem_186, getitem_191, cat_20, squeeze_85, add_150, mean_6, convolution_70, mul_229, convolution_71, getitem_194, getitem_195, cat_21, squeeze_88, getitem_198, getitem_199, cat_22, squeeze_91, getitem_206, getitem_211, getitem_216, getitem_221, cat_23, squeeze_94, add_166, mean_7, convolution_80, mul_254, convolution_81, getitem_224, getitem_225, cat_24, squeeze_97, add_172, convolution_84, squeeze_100, mul_270, convolution_85, squeeze_103, add_182, mean_8, convolution_86, mul_279, convolution_87, mul_280, convolution_88, squeeze_106, getitem_234, getitem_235, cat_25, squeeze_109, getitem_242, getitem_247, getitem_252, getitem_257, cat_26, squeeze_112, add_197, mean_9, convolution_95, mul_304, convolution_96, getitem_260, getitem_261, cat_27, squeeze_115, getitem_264, getitem_265, cat_28, squeeze_118, getitem_272, getitem_277, getitem_282, getitem_287, cat_29, squeeze_121, add_213, mean_10, convolution_105, mul_329, convolution_106, getitem_290, getitem_291, cat_30, squeeze_124, getitem_294, getitem_295, cat_31, squeeze_127, getitem_302, getitem_307, getitem_312, getitem_317, cat_32, squeeze_130, add_229, mean_11, convolution_115, mul_354, convolution_116, getitem_320, getitem_321, cat_33, squeeze_133, add_235, convolution_119, squeeze_136, constant_pad_nd_11, constant_pad_nd_12, constant_pad_nd_13, constant_pad_nd_14, cat_34, squeeze_139, add_245, mean_12, convolution_124, mul_379, convolution_125, mul_380, convolution_126, squeeze_142, add_250, convolution_127, squeeze_145, getitem_356, getitem_361, getitem_366, getitem_371, cat_35, squeeze_148, add_260, mean_13, convolution_132, mul_404, convolution_133, getitem_374, getitem_375, cat_36, squeeze_151, add_266, convolution_136, squeeze_154, getitem_384, getitem_389, getitem_394, getitem_399, cat_37, squeeze_157, add_276, mean_14, convolution_141, mul_429, convolution_142, getitem_402, getitem_403, cat_38, squeeze_160, add_282, convolution_145, squeeze_163, getitem_412, getitem_417, getitem_422, getitem_427, cat_39, squeeze_166, add_292, mean_15, convolution_150, mul_454, convolution_151, getitem_430, getitem_431, cat_40, squeeze_169, add_298, convolution_154, squeeze_172, view, permute_1, le, unsqueeze_234, unsqueeze_246, unsqueeze_258, mul_508, unsqueeze_270, unsqueeze_282, unsqueeze_294, mul_548, unsqueeze_306, unsqueeze_318, unsqueeze_330, mul_588, unsqueeze_342, unsqueeze_354, unsqueeze_366, mul_628, unsqueeze_378, unsqueeze_390, unsqueeze_402, mul_668, unsqueeze_414, unsqueeze_426, unsqueeze_438, mul_708, unsqueeze_450, unsqueeze_462, unsqueeze_474, mul_748, unsqueeze_486, unsqueeze_498, unsqueeze_510, mul_788, unsqueeze_522, unsqueeze_534, unsqueeze_546, mul_828, unsqueeze_558, unsqueeze_570, unsqueeze_582, mul_868, unsqueeze_594, unsqueeze_606, unsqueeze_618, mul_908, unsqueeze_630, unsqueeze_642, unsqueeze_654, mul_948, unsqueeze_666, unsqueeze_678, unsqueeze_690, mul_988, unsqueeze_702, unsqueeze_714, unsqueeze_726, mul_1028, unsqueeze_738, unsqueeze_750, unsqueeze_762, mul_1068, unsqueeze_774, unsqueeze_786, unsqueeze_798, mul_1108, unsqueeze_810, unsqueeze_822, le_1, unsqueeze_834, unsqueeze_846, unsqueeze_858, le_3, unsqueeze_870, le_4, unsqueeze_882, unsqueeze_894, unsqueeze_906, unsqueeze_918, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_4, (32, ), (1, ))
    assert_size_stride(primals_6, (32, ), (1, ))
    assert_size_stride(primals_8, (192, ), (1, ))
    assert_size_stride(primals_10, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_11, (64, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_12, (64, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_13, (192, ), (1, ))
    assert_size_stride(primals_15, (40, ), (1, ))
    assert_size_stride(primals_17, (120, ), (1, ))
    assert_size_stride(primals_19, (120, ), (1, ))
    assert_size_stride(primals_21, (40, ), (1, ))
    assert_size_stride(primals_23, (240, ), (1, ))
    assert_size_stride(primals_25, (60, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_26, (60, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_27, (60, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_28, (60, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_29, (240, ), (1, ))
    assert_size_stride(primals_31, (56, ), (1, ))
    assert_size_stride(primals_33, (336, ), (1, ))
    assert_size_stride(primals_35, (336, ), (1, ))
    assert_size_stride(primals_37, (56, ), (1, ))
    assert_size_stride(primals_39, (336, ), (1, ))
    assert_size_stride(primals_41, (336, ), (1, ))
    assert_size_stride(primals_43, (56, ), (1, ))
    assert_size_stride(primals_45, (336, ), (1, ))
    assert_size_stride(primals_47, (336, ), (1, ))
    assert_size_stride(primals_49, (56, ), (1, ))
    assert_size_stride(primals_51, (336, ), (1, ))
    assert_size_stride(primals_53, (112, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_54, (112, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_55, (112, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_56, (336, ), (1, ))
    assert_size_stride(primals_58, (104, ), (1, ))
    assert_size_stride(primals_60, (624, ), (1, ))
    assert_size_stride(primals_62, (624, ), (1, ))
    assert_size_stride(primals_64, (104, ), (1, ))
    assert_size_stride(primals_66, (624, ), (1, ))
    assert_size_stride(primals_68, (624, ), (1, ))
    assert_size_stride(primals_70, (104, ), (1, ))
    assert_size_stride(primals_72, (624, ), (1, ))
    assert_size_stride(primals_74, (624, ), (1, ))
    assert_size_stride(primals_76, (104, ), (1, ))
    assert_size_stride(primals_78, (624, ), (1, ))
    assert_size_stride(primals_80, (624, ), (1, ))
    assert_size_stride(primals_82, (160, ), (1, ))
    assert_size_stride(primals_84, (480, ), (1, ))
    assert_size_stride(primals_86, (480, ), (1, ))
    assert_size_stride(primals_88, (160, ), (1, ))
    assert_size_stride(primals_90, (480, ), (1, ))
    assert_size_stride(primals_92, (480, ), (1, ))
    assert_size_stride(primals_94, (160, ), (1, ))
    assert_size_stride(primals_96, (480, ), (1, ))
    assert_size_stride(primals_98, (480, ), (1, ))
    assert_size_stride(primals_100, (160, ), (1, ))
    assert_size_stride(primals_102, (960, ), (1, ))
    assert_size_stride(primals_104, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_105, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_106, (240, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_107, (240, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_108, (960, ), (1, ))
    assert_size_stride(primals_110, (264, ), (1, ))
    assert_size_stride(primals_112, (1584, ), (1, ))
    assert_size_stride(primals_114, (1584, ), (1, ))
    assert_size_stride(primals_116, (264, ), (1, ))
    assert_size_stride(primals_118, (1584, ), (1, ))
    assert_size_stride(primals_120, (1584, ), (1, ))
    assert_size_stride(primals_122, (264, ), (1, ))
    assert_size_stride(primals_124, (1584, ), (1, ))
    assert_size_stride(primals_126, (1584, ), (1, ))
    assert_size_stride(primals_128, (264, ), (1, ))
    assert_size_stride(primals_130, (1536, ), (1, ))
    assert_size_stride(primals_132, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_133, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_134, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_135, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_136, (20, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_137, (20, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_138, (60, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_139, (60, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_140, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_141, (20, 60, 1, 1), (60, 1, 1, 1))
    assert_size_stride(primals_142, (20, 60, 1, 1), (60, 1, 1, 1))
    assert_size_stride(primals_143, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_144, (20, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_146, (240, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_148, (56, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_149, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_150, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_151, (168, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_152, (168, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_153, (28, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_155, (336, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_157, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_158, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_159, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_160, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_161, (168, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_162, (168, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_163, (28, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_165, (336, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_167, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_168, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_169, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_170, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_171, (168, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_172, (168, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_173, (28, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_175, (336, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(primals_177, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_178, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_179, (336, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_180, (14, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_182, (336, 14, 1, 1), (14, 1, 1, 1))
    assert_size_stride(primals_184, (104, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_185, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_186, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_187, (156, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_188, (156, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_189, (156, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_190, (156, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_191, (26, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(primals_193, (624, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(primals_195, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_196, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_197, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_198, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_199, (156, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_200, (156, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_201, (156, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_202, (156, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_203, (26, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(primals_205, (624, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(primals_207, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_208, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_209, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_210, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_211, (156, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_212, (156, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_213, (156, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_214, (156, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_215, (26, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(primals_217, (624, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(primals_219, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_220, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_221, (624, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(primals_222, (624, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_223, (52, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(primals_225, (624, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_227, (160, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(primals_228, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_229, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_230, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_231, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_232, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_233, (120, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_234, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_236, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_238, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_239, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_240, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_241, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_242, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_243, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_244, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_245, (120, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_246, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_248, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_250, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_251, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_252, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_253, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_254, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_255, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_256, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_257, (120, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(primals_258, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_260, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_262, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_263, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_264, (960, 160, 1, 1), (160, 1, 1, 1))
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
    assert_size_stride(constant_pad_nd, (8, 3, 225, 225), (151875, 1, 675, 3))
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
    assert_size_stride(constant_pad_nd_1, (8, 64, 113, 113), (817216, 1, 7232, 64))
    assert_size_stride(constant_pad_nd_2, (8, 64, 115, 115), (846400, 1, 7360, 64))
    assert_size_stride(constant_pad_nd_3, (8, 64, 117, 117), (876096, 1, 7488, 64))
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
    assert_size_stride(constant_pad_nd_4, (8, 60, 57, 57), (194940, 1, 3420, 60))
    assert_size_stride(constant_pad_nd_5, (8, 60, 59, 59), (208860, 1, 3540, 60))
    assert_size_stride(constant_pad_nd_6, (8, 60, 61, 61), (223260, 1, 3660, 60))
    assert_size_stride(constant_pad_nd_7, (8, 60, 63, 63), (238140, 1, 3780, 60))
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
    assert_size_stride(constant_pad_nd_8, (8, 112, 29, 29), (94192, 1, 3248, 112))
    assert_size_stride(constant_pad_nd_9, (8, 112, 31, 31), (107632, 1, 3472, 112))
    assert_size_stride(constant_pad_nd_10, (8, 112, 33, 33), (121968, 1, 3696, 112))
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
    assert_size_stride(constant_pad_nd_11, (8, 240, 15, 15), (54000, 1, 3600, 240))
    assert_size_stride(constant_pad_nd_12, (8, 240, 17, 17), (69360, 1, 4080, 240))
    assert_size_stride(constant_pad_nd_13, (8, 240, 19, 19), (86640, 1, 4560, 240))
    assert_size_stride(constant_pad_nd_14, (8, 240, 21, 21), (105840, 1, 5040, 240))
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
        triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_4.run(le, buf0, convolution_154, unsqueeze_234, buf6, squeeze_172, buf4, primals_130, buf8, 602112, grid=grid(602112), stream=stream0)
        del buf0
        del buf6
        del convolution_154
        del le
        del primals_130
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
        triton_poi_fused_native_batch_norm_backward_8.run(buf10, cat_40, unsqueeze_246, buf14, squeeze_169, buf12, primals_128, buf15, 2112, 49, grid=grid(2112, 49), stream=stream0)
        del cat_40
        del primals_128
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
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___se_gate, x_379], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
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
        triton_poi_fused_native_batch_norm_backward_17.run(buf38, cat_39, unsqueeze_258, buf37, squeeze_166, buf35, primals_126, 12672, 49, grid=grid(12672, 49), stream=stream0)
        del cat_39
        del primals_126
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
        triton_poi_fused_native_batch_norm_backward_17.run(buf57, convolution_145, unsqueeze_270, buf55, squeeze_163, buf53, primals_124, 12672, 49, grid=grid(12672, 49), stream=stream0)
        del convolution_145
        del primals_124
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
        triton_poi_fused_add_native_batch_norm_backward_20.run(buf10, buf59, cat_38, unsqueeze_282, buf62, squeeze_160, buf61, primals_122, buf63, 2112, 49, grid=grid(2112, 49), stream=stream0)
        del cat_38
        del primals_122
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
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate, x_360], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
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
        triton_poi_fused_native_batch_norm_backward_17.run(buf86, cat_37, unsqueeze_294, buf85, squeeze_157, buf83, primals_120, 12672, 49, grid=grid(12672, 49), stream=stream0)
        del cat_37
        del primals_120
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
        triton_poi_fused_native_batch_norm_backward_17.run(buf105, convolution_136, unsqueeze_306, buf103, squeeze_154, buf101, primals_118, 12672, 49, grid=grid(12672, 49), stream=stream0)
        del convolution_136
        del primals_118
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
        triton_poi_fused_add_native_batch_norm_backward_22.run(buf10, buf59, buf107, cat_36, unsqueeze_318, buf110, squeeze_151, buf109, primals_116, buf111, 2112, 49, grid=grid(2112, 49), stream=stream0)
        del cat_36
        del primals_116
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
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_341], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
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
        triton_poi_fused_native_batch_norm_backward_17.run(buf134, cat_35, unsqueeze_330, buf133, squeeze_148, buf131, primals_114, 12672, 49, grid=grid(12672, 49), stream=stream0)
        del cat_35
        del primals_114
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
        triton_poi_fused_native_batch_norm_backward_17.run(buf153, convolution_127, unsqueeze_342, buf151, squeeze_145, buf149, primals_112, 12672, 49, grid=grid(12672, 49), stream=stream0)
        del buf151
        del convolution_127
        del primals_112
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
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_24.run(buf161, buf59, buf107, buf155, convolution_126, unsqueeze_354, buf158, squeeze_142, buf157, primals_110, 2112, 49, grid=grid(2112, 49), stream=stream0)
        del buf107
        del buf155
        del buf158
        del buf59
        del convolution_126
        del primals_110
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
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_324], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
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
        triton_poi_fused_convolution_backward_33.run(buf180, squeeze_139, primals_108, buf182, 1920, 49, grid=grid(1920, 49), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf183 = aten.convolution_backward(buf182, constant_pad_nd_14, primals_107, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 240, [True, True, False])
        del constant_pad_nd_14
        del primals_107
        buf184 = buf183[0]
        buf185 = buf183[1]
        del buf183
        buf186 = buf182; del buf182  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_34.run(buf180, squeeze_139, primals_108, buf186, 1920, 49, grid=grid(1920, 49), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf187 = aten.convolution_backward(buf186, constant_pad_nd_13, primals_106, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 240, [True, True, False])
        del constant_pad_nd_13
        del primals_106
        buf188 = buf187[0]
        buf189 = buf187[1]
        del buf187
        buf190 = buf186; del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_35.run(buf180, squeeze_139, primals_108, buf190, 1920, 49, grid=grid(1920, 49), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf191 = aten.convolution_backward(buf190, constant_pad_nd_12, primals_105, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 240, [True, True, False])
        del constant_pad_nd_12
        del primals_105
        buf192 = buf191[0]
        buf193 = buf191[1]
        del buf191
        buf194 = buf190; del buf190  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_36.run(buf180, squeeze_139, primals_108, buf194, 1920, 49, grid=grid(1920, 49), stream=stream0)
        del buf180
        del primals_108
        del squeeze_139
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf195 = aten.convolution_backward(buf194, constant_pad_nd_11, primals_104, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 240, [True, True, False])
        del buf194
        del constant_pad_nd_11
        del primals_104
        buf196 = buf195[0]
        buf197 = buf195[1]
        del buf195
        buf198 = empty((8, 960, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_37.run(buf196, buf192, buf188, buf184, buf198, 1505280, grid=grid(1505280), stream=stream0)
        del buf184
        del buf188
        del buf192
        del buf196
        buf199 = empty((960, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_38.run(buf198, mul_628, buf199, 12480, 121, grid=grid(12480), stream=stream0)
        buf200 = buf179; del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_39.run(buf199, buf200, 960, 13, grid=grid(960), stream=stream0)
        buf201 = reinterpret_tensor(buf199, (960, 13), (1, 960), 0); del buf199  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_40.run(buf198, mul_628, convolution_119, unsqueeze_378, buf201, 12480, 121, grid=grid(12480), stream=stream0)
        buf202 = empty((960, ), device='cuda', dtype=torch.float32)
        buf203 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_41.run(buf201, squeeze_136, buf202, buf203, 960, 13, grid=grid(960), stream=stream0)
        del buf201
        buf204 = buf198; del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_42.run(buf204, mul_628, convolution_119, unsqueeze_378, buf202, squeeze_136, buf200, primals_102, 7680, 196, grid=grid(7680, 196), stream=stream0)
        del buf202
        del convolution_119
        del mul_628
        del primals_102
        del squeeze_136
        del unsqueeze_378
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf205 = aten.convolution_backward(buf204, add_235, primals_264, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_235
        del primals_264
        buf206 = buf205[0]
        buf207 = buf205[1]
        del buf205
        buf208 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_43.run(buf206, buf208, 160, 1568, grid=grid(160), stream=stream0)
        buf209 = empty((160, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_44.run(buf206, cat_33, unsqueeze_390, buf209, 2080, 121, grid=grid(2080), stream=stream0)
        buf210 = empty((160, ), device='cuda', dtype=torch.float32)
        buf212 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_45.run(buf209, squeeze_133, buf210, buf212, 160, 13, grid=grid(160), stream=stream0)
        buf211 = empty((8, 160, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_46.run(buf206, cat_33, unsqueeze_390, buf210, squeeze_133, buf208, primals_100, buf211, 1280, 196, grid=grid(1280, 196), stream=stream0)
        del cat_33
        del primals_100
        del squeeze_133
        del unsqueeze_390
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf213 = aten.convolution_backward(reinterpret_tensor(buf211, (8, 80, 14, 14), (31360, 196, 14, 1), 15680), getitem_321, primals_263, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_321
        del primals_263
        buf214 = buf213[0]
        buf215 = buf213[1]
        del buf213
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf216 = aten.convolution_backward(reinterpret_tensor(buf211, (8, 80, 14, 14), (31360, 196, 14, 1), 0), getitem_320, primals_262, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_320
        del primals_262
        buf217 = buf216[0]
        buf218 = buf216[1]
        del buf216
        buf219 = reinterpret_tensor(buf178, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf178  # reuse
        buf220 = reinterpret_tensor(buf219, (8, 480, 1, 1), (480, 1, 1, 1), 0); del buf219  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate, x_297], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_47.run(buf220, buf217, buf214, add_229, convolution_116, 3840, 196, grid=grid(3840), stream=stream0)
        buf221 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_48.run(buf220, buf221, 480, 8, grid=grid(480), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf222 = aten.convolution_backward(buf220, mul_354, primals_260, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf220
        del mul_354
        del primals_260
        buf223 = buf222[0]
        buf224 = buf222[1]
        del buf222
        buf225 = buf223; del buf223  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_27.run(buf225, convolution_115, 640, grid=grid(640), stream=stream0)
        del convolution_115
        buf226 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_28.run(buf225, buf226, 80, 8, grid=grid(80), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf227 = aten.convolution_backward(buf225, mean_11, primals_258, [80], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf225
        del mean_11
        del primals_258
        buf228 = buf227[0]
        buf229 = buf227[1]
        del buf227
        buf230 = empty((8, 480, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___se_gate], Original ATen: [aten.add, aten.cat, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_49.run(buf217, buf214, convolution_116, buf228, add_229, buf230, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del add_229
        del buf214
        del buf217
        del convolution_116
        buf231 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_50.run(buf230, buf231, 480, 1568, grid=grid(480), stream=stream0)
        buf232 = empty((480, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_51.run(buf230, cat_32, unsqueeze_402, buf232, 6240, 121, grid=grid(6240), stream=stream0)
        buf233 = empty((480, ), device='cuda', dtype=torch.float32)
        buf235 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_52.run(buf232, squeeze_130, buf233, buf235, 480, 13, grid=grid(480), stream=stream0)
        buf234 = buf230; del buf230  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_53.run(buf234, cat_32, unsqueeze_402, buf233, squeeze_130, buf231, primals_98, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del cat_32
        del primals_98
        del squeeze_130
        del unsqueeze_402
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf236 = aten.convolution_backward(reinterpret_tensor(buf234, (8, 120, 14, 14), (94080, 196, 14, 1), 70560), getitem_317, primals_257, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 120, [True, True, False])
        del getitem_317
        del primals_257
        buf237 = buf236[0]
        buf238 = buf236[1]
        del buf236
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf239 = aten.convolution_backward(reinterpret_tensor(buf234, (8, 120, 14, 14), (94080, 196, 14, 1), 47040), getitem_312, primals_256, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 120, [True, True, False])
        del getitem_312
        del primals_256
        buf240 = buf239[0]
        buf241 = buf239[1]
        del buf239
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf242 = aten.convolution_backward(reinterpret_tensor(buf234, (8, 120, 14, 14), (94080, 196, 14, 1), 23520), getitem_307, primals_255, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
        del getitem_307
        del primals_255
        buf243 = buf242[0]
        buf244 = buf242[1]
        del buf242
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf245 = aten.convolution_backward(reinterpret_tensor(buf234, (8, 120, 14, 14), (94080, 196, 14, 1), 0), getitem_302, primals_254, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 120, [True, True, False])
        del getitem_302
        del primals_254
        buf246 = buf245[0]
        buf247 = buf245[1]
        del buf245
        buf248 = buf234; del buf234  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul]
        triton_poi_fused_cat_mul_54.run(buf246, buf243, buf240, buf237, mul_668, buf248, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del buf237
        del buf240
        del buf243
        del buf246
        del mul_668
        buf249 = buf233; del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_50.run(buf248, buf249, 480, 1568, grid=grid(480), stream=stream0)
        buf250 = buf232; del buf232  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_51.run(buf248, cat_31, unsqueeze_414, buf250, 6240, 121, grid=grid(6240), stream=stream0)
        buf251 = empty((480, ), device='cuda', dtype=torch.float32)
        buf253 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_52.run(buf250, squeeze_127, buf251, buf253, 480, 13, grid=grid(480), stream=stream0)
        buf252 = buf248; del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_53.run(buf252, cat_31, unsqueeze_414, buf251, squeeze_127, buf249, primals_96, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del cat_31
        del primals_96
        del squeeze_127
        del unsqueeze_414
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf254 = aten.convolution_backward(reinterpret_tensor(buf252, (8, 240, 14, 14), (94080, 196, 14, 1), 47040), getitem_295, primals_253, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_295
        del primals_253
        buf255 = buf254[0]
        buf256 = buf254[1]
        del buf254
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf257 = aten.convolution_backward(reinterpret_tensor(buf252, (8, 240, 14, 14), (94080, 196, 14, 1), 0), getitem_294, primals_252, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_294
        del primals_252
        buf258 = buf257[0]
        buf259 = buf257[1]
        del buf257
        buf260 = buf210; del buf210  # reuse
        buf261 = empty((160, ), device='cuda', dtype=torch.float32)
        buf263 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_red_fused_add_cat_native_batch_norm_backward_55.run(buf206, buf258, buf255, cat_30, unsqueeze_426, squeeze_124, buf260, buf261, buf263, 160, 1568, grid=grid(160), stream=stream0)
        buf262 = buf211; del buf211  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_poi_fused_add_cat_native_batch_norm_backward_56.run(buf206, buf258, buf255, cat_30, unsqueeze_426, buf261, squeeze_124, buf260, primals_94, buf262, 1280, 196, grid=grid(1280, 196), stream=stream0)
        del cat_30
        del primals_94
        del squeeze_124
        del unsqueeze_426
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf264 = aten.convolution_backward(reinterpret_tensor(buf262, (8, 80, 14, 14), (31360, 196, 14, 1), 15680), getitem_291, primals_251, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_291
        del primals_251
        buf265 = buf264[0]
        buf266 = buf264[1]
        del buf264
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf267 = aten.convolution_backward(reinterpret_tensor(buf262, (8, 80, 14, 14), (31360, 196, 14, 1), 0), getitem_290, primals_250, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_290
        del primals_250
        buf268 = buf267[0]
        buf269 = buf267[1]
        del buf267
        buf270 = reinterpret_tensor(buf228, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf228  # reuse
        buf271 = reinterpret_tensor(buf270, (8, 480, 1, 1), (480, 1, 1, 1), 0); del buf270  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate, x_277], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_47.run(buf271, buf268, buf265, add_213, convolution_106, 3840, 196, grid=grid(3840), stream=stream0)
        buf272 = buf251; del buf251  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_48.run(buf271, buf272, 480, 8, grid=grid(480), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf273 = aten.convolution_backward(buf271, mul_329, primals_248, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf271
        del mul_329
        del primals_248
        buf274 = buf273[0]
        buf275 = buf273[1]
        del buf273
        buf276 = buf274; del buf274  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_27.run(buf276, convolution_105, 640, grid=grid(640), stream=stream0)
        del convolution_105
        buf277 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_28.run(buf276, buf277, 80, 8, grid=grid(80), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf278 = aten.convolution_backward(buf276, mean_10, primals_246, [80], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf276
        del mean_10
        del primals_246
        buf279 = buf278[0]
        buf280 = buf278[1]
        del buf278
        buf281 = buf252; del buf252  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___se_gate], Original ATen: [aten.add, aten.cat, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_49.run(buf268, buf265, convolution_106, buf279, add_213, buf281, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del add_213
        del buf265
        del buf268
        del convolution_106
        buf282 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_50.run(buf281, buf282, 480, 1568, grid=grid(480), stream=stream0)
        buf283 = buf250; del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_51.run(buf281, cat_29, unsqueeze_438, buf283, 6240, 121, grid=grid(6240), stream=stream0)
        buf284 = empty((480, ), device='cuda', dtype=torch.float32)
        buf286 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_52.run(buf283, squeeze_121, buf284, buf286, 480, 13, grid=grid(480), stream=stream0)
        buf285 = buf281; del buf281  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_53.run(buf285, cat_29, unsqueeze_438, buf284, squeeze_121, buf282, primals_92, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del cat_29
        del primals_92
        del squeeze_121
        del unsqueeze_438
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf287 = aten.convolution_backward(reinterpret_tensor(buf285, (8, 120, 14, 14), (94080, 196, 14, 1), 70560), getitem_287, primals_245, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 120, [True, True, False])
        del getitem_287
        del primals_245
        buf288 = buf287[0]
        buf289 = buf287[1]
        del buf287
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf290 = aten.convolution_backward(reinterpret_tensor(buf285, (8, 120, 14, 14), (94080, 196, 14, 1), 47040), getitem_282, primals_244, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 120, [True, True, False])
        del getitem_282
        del primals_244
        buf291 = buf290[0]
        buf292 = buf290[1]
        del buf290
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf293 = aten.convolution_backward(reinterpret_tensor(buf285, (8, 120, 14, 14), (94080, 196, 14, 1), 23520), getitem_277, primals_243, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
        del getitem_277
        del primals_243
        buf294 = buf293[0]
        buf295 = buf293[1]
        del buf293
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf296 = aten.convolution_backward(reinterpret_tensor(buf285, (8, 120, 14, 14), (94080, 196, 14, 1), 0), getitem_272, primals_242, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 120, [True, True, False])
        del getitem_272
        del primals_242
        buf297 = buf296[0]
        buf298 = buf296[1]
        del buf296
        buf299 = buf285; del buf285  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul]
        triton_poi_fused_cat_mul_54.run(buf297, buf294, buf291, buf288, mul_708, buf299, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del buf288
        del buf291
        del buf294
        del buf297
        del mul_708
        buf300 = buf284; del buf284  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_50.run(buf299, buf300, 480, 1568, grid=grid(480), stream=stream0)
        buf301 = buf283; del buf283  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_51.run(buf299, cat_28, unsqueeze_450, buf301, 6240, 121, grid=grid(6240), stream=stream0)
        buf302 = empty((480, ), device='cuda', dtype=torch.float32)
        buf304 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_52.run(buf301, squeeze_118, buf302, buf304, 480, 13, grid=grid(480), stream=stream0)
        buf303 = buf299; del buf299  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_53.run(buf303, cat_28, unsqueeze_450, buf302, squeeze_118, buf300, primals_90, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del cat_28
        del primals_90
        del squeeze_118
        del unsqueeze_450
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf305 = aten.convolution_backward(reinterpret_tensor(buf303, (8, 240, 14, 14), (94080, 196, 14, 1), 47040), getitem_265, primals_241, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_265
        del primals_241
        buf306 = buf305[0]
        buf307 = buf305[1]
        del buf305
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf308 = aten.convolution_backward(reinterpret_tensor(buf303, (8, 240, 14, 14), (94080, 196, 14, 1), 0), getitem_264, primals_240, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_264
        del primals_240
        buf309 = buf308[0]
        buf310 = buf308[1]
        del buf308
        buf311 = buf206; del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_57.run(buf311, buf258, buf255, buf309, buf306, 250880, grid=grid(250880), stream=stream0)
        del buf255
        del buf258
        del buf306
        del buf309
        buf312 = buf261; del buf261  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_43.run(buf311, buf312, 160, 1568, grid=grid(160), stream=stream0)
        buf313 = buf209; del buf209  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_44.run(buf311, cat_27, unsqueeze_462, buf313, 2080, 121, grid=grid(2080), stream=stream0)
        buf314 = empty((160, ), device='cuda', dtype=torch.float32)
        buf316 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_45.run(buf313, squeeze_115, buf314, buf316, 160, 13, grid=grid(160), stream=stream0)
        del buf313
        buf315 = buf262; del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_46.run(buf311, cat_27, unsqueeze_462, buf314, squeeze_115, buf312, primals_88, buf315, 1280, 196, grid=grid(1280, 196), stream=stream0)
        del cat_27
        del primals_88
        del squeeze_115
        del unsqueeze_462
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf317 = aten.convolution_backward(reinterpret_tensor(buf315, (8, 80, 14, 14), (31360, 196, 14, 1), 15680), getitem_261, primals_239, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_261
        del primals_239
        buf318 = buf317[0]
        buf319 = buf317[1]
        del buf317
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf320 = aten.convolution_backward(reinterpret_tensor(buf315, (8, 80, 14, 14), (31360, 196, 14, 1), 0), getitem_260, primals_238, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf315
        del getitem_260
        del primals_238
        buf321 = buf320[0]
        buf322 = buf320[1]
        del buf320
        buf323 = reinterpret_tensor(buf279, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf279  # reuse
        buf324 = reinterpret_tensor(buf323, (8, 480, 1, 1), (480, 1, 1, 1), 0); del buf323  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_257], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_47.run(buf324, buf321, buf318, add_197, convolution_96, 3840, 196, grid=grid(3840), stream=stream0)
        buf325 = buf302; del buf302  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_48.run(buf324, buf325, 480, 8, grid=grid(480), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf326 = aten.convolution_backward(buf324, mul_304, primals_236, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf324
        del mul_304
        del primals_236
        buf327 = buf326[0]
        buf328 = buf326[1]
        del buf326
        buf329 = buf327; del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_27.run(buf329, convolution_95, 640, grid=grid(640), stream=stream0)
        del convolution_95
        buf330 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_28.run(buf329, buf330, 80, 8, grid=grid(80), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf331 = aten.convolution_backward(buf329, mean_9, primals_234, [80], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf329
        del mean_9
        del primals_234
        buf332 = buf331[0]
        buf333 = buf331[1]
        del buf331
        buf334 = buf303; del buf303  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate], Original ATen: [aten.add, aten.cat, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_49.run(buf321, buf318, convolution_96, buf332, add_197, buf334, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del add_197
        del buf318
        del buf332
        del convolution_96
        buf335 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_50.run(buf334, buf335, 480, 1568, grid=grid(480), stream=stream0)
        buf336 = buf301; del buf301  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_51.run(buf334, cat_26, unsqueeze_474, buf336, 6240, 121, grid=grid(6240), stream=stream0)
        buf337 = empty((480, ), device='cuda', dtype=torch.float32)
        buf339 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_52.run(buf336, squeeze_112, buf337, buf339, 480, 13, grid=grid(480), stream=stream0)
        buf338 = buf334; del buf334  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_53.run(buf338, cat_26, unsqueeze_474, buf337, squeeze_112, buf335, primals_86, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del cat_26
        del primals_86
        del squeeze_112
        del unsqueeze_474
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf340 = aten.convolution_backward(reinterpret_tensor(buf338, (8, 120, 14, 14), (94080, 196, 14, 1), 70560), getitem_257, primals_233, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 120, [True, True, False])
        del getitem_257
        del primals_233
        buf341 = buf340[0]
        buf342 = buf340[1]
        del buf340
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf343 = aten.convolution_backward(reinterpret_tensor(buf338, (8, 120, 14, 14), (94080, 196, 14, 1), 47040), getitem_252, primals_232, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 120, [True, True, False])
        del getitem_252
        del primals_232
        buf344 = buf343[0]
        buf345 = buf343[1]
        del buf343
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf346 = aten.convolution_backward(reinterpret_tensor(buf338, (8, 120, 14, 14), (94080, 196, 14, 1), 23520), getitem_247, primals_231, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
        del getitem_247
        del primals_231
        buf347 = buf346[0]
        buf348 = buf346[1]
        del buf346
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf349 = aten.convolution_backward(reinterpret_tensor(buf338, (8, 120, 14, 14), (94080, 196, 14, 1), 0), getitem_242, primals_230, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 120, [True, True, False])
        del getitem_242
        del primals_230
        buf350 = buf349[0]
        buf351 = buf349[1]
        del buf349
        buf352 = buf338; del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul]
        triton_poi_fused_cat_mul_54.run(buf350, buf347, buf344, buf341, mul_748, buf352, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del buf341
        del buf344
        del buf347
        del buf350
        del mul_748
        buf353 = buf337; del buf337  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_50.run(buf352, buf353, 480, 1568, grid=grid(480), stream=stream0)
        buf354 = buf336; del buf336  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_51.run(buf352, cat_25, unsqueeze_486, buf354, 6240, 121, grid=grid(6240), stream=stream0)
        buf355 = empty((480, ), device='cuda', dtype=torch.float32)
        buf357 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_52.run(buf354, squeeze_109, buf355, buf357, 480, 13, grid=grid(480), stream=stream0)
        del buf354
        buf356 = buf352; del buf352  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_53.run(buf356, cat_25, unsqueeze_486, buf355, squeeze_109, buf353, primals_84, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del cat_25
        del primals_84
        del squeeze_109
        del unsqueeze_486
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf358 = aten.convolution_backward(reinterpret_tensor(buf356, (8, 240, 14, 14), (94080, 196, 14, 1), 47040), getitem_235, primals_229, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_235
        del primals_229
        buf359 = buf358[0]
        buf360 = buf358[1]
        del buf358
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf361 = aten.convolution_backward(reinterpret_tensor(buf356, (8, 240, 14, 14), (94080, 196, 14, 1), 0), getitem_234, primals_228, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf356
        del getitem_234
        del primals_228
        buf362 = buf361[0]
        buf363 = buf361[1]
        del buf361
        buf364 = buf314; del buf314  # reuse
        buf365 = empty((160, ), device='cuda', dtype=torch.float32)
        buf367 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_red_fused_add_cat_native_batch_norm_backward_55.run(buf311, buf362, buf359, convolution_88, unsqueeze_498, squeeze_106, buf364, buf365, buf367, 160, 1568, grid=grid(160), stream=stream0)
        buf366 = buf311; del buf311  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_poi_fused_add_cat_native_batch_norm_backward_58.run(buf366, buf362, buf359, convolution_88, unsqueeze_498, buf365, squeeze_106, buf364, primals_82, 1280, 196, grid=grid(1280, 196), stream=stream0)
        del buf359
        del buf362
        del convolution_88
        del primals_82
        del squeeze_106
        del unsqueeze_498
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf368 = aten.convolution_backward(buf366, mul_280, primals_227, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf366
        del mul_280
        del primals_227
        buf369 = buf368[0]
        buf370 = buf368[1]
        del buf368
        buf371 = empty_strided((8, 624, 1, 1, 2), (1248, 2, 9984, 9984, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_239], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_59.run(buf369, add_182, buf371, 9984, 98, grid=grid(9984), stream=stream0)
        buf372 = empty_strided((8, 624, 1, 1), (624, 1, 4992, 4992), device='cuda', dtype=torch.float32)
        buf373 = reinterpret_tensor(buf372, (8, 624, 1, 1), (624, 1, 1, 1), 0); del buf372  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_239], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_60.run(buf373, buf371, convolution_87, 4992, 2, grid=grid(4992), stream=stream0)
        del buf371
        buf374 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_61.run(buf373, buf374, 624, 8, grid=grid(624), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf375 = aten.convolution_backward(buf373, mul_279, primals_225, [624], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf373
        del mul_279
        del primals_225
        buf376 = buf375[0]
        buf377 = buf375[1]
        del buf375
        buf378 = buf376; del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_62.run(buf378, convolution_86, 416, grid=grid(416), stream=stream0)
        del convolution_86
        buf379 = empty((52, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_63.run(buf378, buf379, 52, 8, grid=grid(52), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf380 = aten.convolution_backward(buf378, mean_8, primals_223, [52], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_8
        del primals_223
        buf381 = buf380[0]
        buf382 = buf380[1]
        del buf380
        buf383 = empty((624, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_64.run(buf369, convolution_87, buf381, add_182, buf383, 8112, 121, grid=grid(8112), stream=stream0)
        buf384 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_65.run(buf383, buf384, 624, 13, grid=grid(624), stream=stream0)
        buf385 = reinterpret_tensor(buf383, (624, 13), (1, 624), 0); del buf383  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_66.run(buf369, convolution_87, buf381, add_182, convolution_85, unsqueeze_510, buf385, 8112, 121, grid=grid(8112), stream=stream0)
        buf386 = empty((624, ), device='cuda', dtype=torch.float32)
        buf388 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_67.run(buf385, squeeze_103, buf386, buf388, 624, 13, grid=grid(624), stream=stream0)
        buf387 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_68.run(buf369, convolution_87, buf381, add_182, convolution_85, unsqueeze_510, buf386, squeeze_103, buf384, buf387, 1568, 624, grid=grid(1568, 624), stream=stream0)
        del add_182
        del convolution_85
        del convolution_87
        del unsqueeze_510
        buf389 = buf369; del buf369  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_69.run(buf387, squeeze_103, primals_80, buf389, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del buf387
        del primals_80
        del squeeze_103
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf390 = aten.convolution_backward(buf389, mul_270, primals_222, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 624, [True, True, False])
        del buf389
        del mul_270
        del primals_222
        buf391 = buf390[0]
        buf392 = buf390[1]
        del buf390
        buf393 = reinterpret_tensor(buf385, (624, 13), (13, 1), 0); del buf385  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_70.run(buf391, mul_788, buf393, 8112, 121, grid=grid(8112), stream=stream0)
        buf394 = buf386; del buf386  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_65.run(buf393, buf394, 624, 13, grid=grid(624), stream=stream0)
        buf395 = reinterpret_tensor(buf393, (624, 13), (1, 624), 0); del buf393  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_71.run(buf391, mul_788, convolution_84, unsqueeze_522, buf395, 8112, 121, grid=grid(8112), stream=stream0)
        buf396 = empty((624, ), device='cuda', dtype=torch.float32)
        buf397 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_67.run(buf395, squeeze_100, buf396, buf397, 624, 13, grid=grid(624), stream=stream0)
        buf398 = buf391; del buf391  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_72.run(buf398, mul_788, convolution_84, unsqueeze_522, buf396, squeeze_100, buf394, primals_78, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del convolution_84
        del mul_788
        del primals_78
        del squeeze_100
        del unsqueeze_522
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf399 = aten.convolution_backward(buf398, add_172, primals_221, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_172
        del primals_221
        buf400 = buf399[0]
        buf401 = buf399[1]
        del buf399
        buf402 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_73.run(buf400, buf402, 104, 1568, grid=grid(104), stream=stream0)
        buf403 = empty((104, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_74.run(buf400, cat_24, unsqueeze_534, buf403, 1352, 121, grid=grid(1352), stream=stream0)
        buf404 = empty((104, ), device='cuda', dtype=torch.float32)
        buf406 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_75.run(buf403, squeeze_97, buf404, buf406, 104, 13, grid=grid(104), stream=stream0)
        buf405 = empty((8, 104, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_76.run(buf400, cat_24, unsqueeze_534, buf404, squeeze_97, buf402, primals_76, buf405, 832, 196, grid=grid(832, 196), stream=stream0)
        del cat_24
        del primals_76
        del squeeze_97
        del unsqueeze_534
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf407 = aten.convolution_backward(reinterpret_tensor(buf405, (8, 52, 14, 14), (20384, 196, 14, 1), 10192), getitem_225, primals_220, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_225
        del primals_220
        buf408 = buf407[0]
        buf409 = buf407[1]
        del buf407
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf410 = aten.convolution_backward(reinterpret_tensor(buf405, (8, 52, 14, 14), (20384, 196, 14, 1), 0), getitem_224, primals_219, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_224
        del primals_219
        buf411 = buf410[0]
        buf412 = buf410[1]
        del buf410
        buf413 = reinterpret_tensor(buf381, (8, 624, 1, 1), (624, 1, 4992, 4992), 0); del buf381  # reuse
        buf414 = reinterpret_tensor(buf413, (8, 624, 1, 1), (624, 1, 1, 1), 0); del buf413  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___se_gate, x_221], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_77.run(buf414, buf411, buf408, add_166, convolution_81, 4992, 196, grid=grid(4992), stream=stream0)
        buf415 = buf396; del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_61.run(buf414, buf415, 624, 8, grid=grid(624), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf416 = aten.convolution_backward(buf414, mul_254, primals_217, [624], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf414
        del mul_254
        del primals_217
        buf417 = buf416[0]
        buf418 = buf416[1]
        del buf416
        buf419 = buf417; del buf417  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_78.run(buf419, convolution_80, 208, grid=grid(208), stream=stream0)
        del convolution_80
        buf420 = empty((26, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_79.run(buf419, buf420, 26, 8, grid=grid(26), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf421 = aten.convolution_backward(buf419, mean_7, primals_215, [26], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf419
        del mean_7
        del primals_215
        buf422 = buf421[0]
        buf423 = buf421[1]
        del buf421
        buf424 = buf398; del buf398  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___se_gate], Original ATen: [aten.add, aten.cat, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_80.run(buf411, buf408, convolution_81, buf422, add_166, buf424, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del add_166
        del buf408
        del buf411
        del convolution_81
        buf425 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_81.run(buf424, buf425, 624, 1568, grid=grid(624), stream=stream0)
        buf426 = reinterpret_tensor(buf395, (624, 13), (13, 1), 0); del buf395  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_82.run(buf424, cat_23, unsqueeze_546, buf426, 8112, 121, grid=grid(8112), stream=stream0)
        buf427 = empty((624, ), device='cuda', dtype=torch.float32)
        buf429 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_83.run(buf426, squeeze_94, buf427, buf429, 624, 13, grid=grid(624), stream=stream0)
        buf428 = buf424; del buf424  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_84.run(buf428, cat_23, unsqueeze_546, buf427, squeeze_94, buf425, primals_74, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del cat_23
        del primals_74
        del squeeze_94
        del unsqueeze_546
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf430 = aten.convolution_backward(reinterpret_tensor(buf428, (8, 156, 14, 14), (122304, 196, 14, 1), 91728), getitem_221, primals_214, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 156, [True, True, False])
        del getitem_221
        del primals_214
        buf431 = buf430[0]
        buf432 = buf430[1]
        del buf430
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf433 = aten.convolution_backward(reinterpret_tensor(buf428, (8, 156, 14, 14), (122304, 196, 14, 1), 61152), getitem_216, primals_213, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 156, [True, True, False])
        del getitem_216
        del primals_213
        buf434 = buf433[0]
        buf435 = buf433[1]
        del buf433
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf436 = aten.convolution_backward(reinterpret_tensor(buf428, (8, 156, 14, 14), (122304, 196, 14, 1), 30576), getitem_211, primals_212, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 156, [True, True, False])
        del getitem_211
        del primals_212
        buf437 = buf436[0]
        buf438 = buf436[1]
        del buf436
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf439 = aten.convolution_backward(reinterpret_tensor(buf428, (8, 156, 14, 14), (122304, 196, 14, 1), 0), getitem_206, primals_211, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 156, [True, True, False])
        del getitem_206
        del primals_211
        buf440 = buf439[0]
        buf441 = buf439[1]
        del buf439
        buf442 = buf428; del buf428  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul]
        triton_poi_fused_cat_mul_85.run(buf440, buf437, buf434, buf431, mul_828, buf442, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del buf431
        del buf434
        del buf437
        del buf440
        del mul_828
        buf443 = buf427; del buf427  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_81.run(buf442, buf443, 624, 1568, grid=grid(624), stream=stream0)
        buf444 = buf426; del buf426  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_82.run(buf442, cat_22, unsqueeze_558, buf444, 8112, 121, grid=grid(8112), stream=stream0)
        buf445 = empty((624, ), device='cuda', dtype=torch.float32)
        buf447 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_83.run(buf444, squeeze_91, buf445, buf447, 624, 13, grid=grid(624), stream=stream0)
        buf446 = buf442; del buf442  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_84.run(buf446, cat_22, unsqueeze_558, buf445, squeeze_91, buf443, primals_72, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del cat_22
        del primals_72
        del squeeze_91
        del unsqueeze_558
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf448 = aten.convolution_backward(reinterpret_tensor(buf446, (8, 312, 14, 14), (122304, 196, 14, 1), 61152), getitem_199, primals_210, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_199
        del primals_210
        buf449 = buf448[0]
        buf450 = buf448[1]
        del buf448
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf451 = aten.convolution_backward(reinterpret_tensor(buf446, (8, 312, 14, 14), (122304, 196, 14, 1), 0), getitem_198, primals_209, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_198
        del primals_209
        buf452 = buf451[0]
        buf453 = buf451[1]
        del buf451
        buf454 = buf404; del buf404  # reuse
        buf455 = empty((104, ), device='cuda', dtype=torch.float32)
        buf457 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_red_fused_add_cat_native_batch_norm_backward_86.run(buf400, buf452, buf449, cat_21, unsqueeze_570, squeeze_88, buf454, buf455, buf457, 104, 1568, grid=grid(104), stream=stream0)
        buf456 = buf405; del buf405  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_poi_fused_add_cat_native_batch_norm_backward_87.run(buf400, buf452, buf449, cat_21, unsqueeze_570, buf455, squeeze_88, buf454, primals_70, buf456, 832, 196, grid=grid(832, 196), stream=stream0)
        del cat_21
        del primals_70
        del squeeze_88
        del unsqueeze_570
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf458 = aten.convolution_backward(reinterpret_tensor(buf456, (8, 52, 14, 14), (20384, 196, 14, 1), 10192), getitem_195, primals_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_195
        del primals_208
        buf459 = buf458[0]
        buf460 = buf458[1]
        del buf458
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf461 = aten.convolution_backward(reinterpret_tensor(buf456, (8, 52, 14, 14), (20384, 196, 14, 1), 0), getitem_194, primals_207, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_194
        del primals_207
        buf462 = buf461[0]
        buf463 = buf461[1]
        del buf461
        buf464 = reinterpret_tensor(buf422, (8, 624, 1, 1), (624, 1, 4992, 4992), 0); del buf422  # reuse
        buf465 = reinterpret_tensor(buf464, (8, 624, 1, 1), (624, 1, 1, 1), 0); del buf464  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate, x_201], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_77.run(buf465, buf462, buf459, add_150, convolution_71, 4992, 196, grid=grid(4992), stream=stream0)
        buf466 = buf445; del buf445  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_61.run(buf465, buf466, 624, 8, grid=grid(624), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf467 = aten.convolution_backward(buf465, mul_229, primals_205, [624], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf465
        del mul_229
        del primals_205
        buf468 = buf467[0]
        buf469 = buf467[1]
        del buf467
        buf470 = buf468; del buf468  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_78.run(buf470, convolution_70, 208, grid=grid(208), stream=stream0)
        del convolution_70
        buf471 = empty((26, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_79.run(buf470, buf471, 26, 8, grid=grid(26), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf472 = aten.convolution_backward(buf470, mean_6, primals_203, [26], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf470
        del mean_6
        del primals_203
        buf473 = buf472[0]
        buf474 = buf472[1]
        del buf472
        buf475 = buf446; del buf446  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___se_gate], Original ATen: [aten.add, aten.cat, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_80.run(buf462, buf459, convolution_71, buf473, add_150, buf475, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del add_150
        del buf459
        del buf462
        del convolution_71
        buf476 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_81.run(buf475, buf476, 624, 1568, grid=grid(624), stream=stream0)
        buf477 = buf444; del buf444  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_82.run(buf475, cat_20, unsqueeze_582, buf477, 8112, 121, grid=grid(8112), stream=stream0)
        buf478 = empty((624, ), device='cuda', dtype=torch.float32)
        buf480 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_83.run(buf477, squeeze_85, buf478, buf480, 624, 13, grid=grid(624), stream=stream0)
        buf479 = buf475; del buf475  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_84.run(buf479, cat_20, unsqueeze_582, buf478, squeeze_85, buf476, primals_68, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del cat_20
        del primals_68
        del squeeze_85
        del unsqueeze_582
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf481 = aten.convolution_backward(reinterpret_tensor(buf479, (8, 156, 14, 14), (122304, 196, 14, 1), 91728), getitem_191, primals_202, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 156, [True, True, False])
        del getitem_191
        del primals_202
        buf482 = buf481[0]
        buf483 = buf481[1]
        del buf481
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf484 = aten.convolution_backward(reinterpret_tensor(buf479, (8, 156, 14, 14), (122304, 196, 14, 1), 61152), getitem_186, primals_201, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 156, [True, True, False])
        del getitem_186
        del primals_201
        buf485 = buf484[0]
        buf486 = buf484[1]
        del buf484
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf487 = aten.convolution_backward(reinterpret_tensor(buf479, (8, 156, 14, 14), (122304, 196, 14, 1), 30576), getitem_181, primals_200, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 156, [True, True, False])
        del getitem_181
        del primals_200
        buf488 = buf487[0]
        buf489 = buf487[1]
        del buf487
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf490 = aten.convolution_backward(reinterpret_tensor(buf479, (8, 156, 14, 14), (122304, 196, 14, 1), 0), getitem_176, primals_199, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 156, [True, True, False])
        del getitem_176
        del primals_199
        buf491 = buf490[0]
        buf492 = buf490[1]
        del buf490
        buf493 = buf479; del buf479  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul]
        triton_poi_fused_cat_mul_85.run(buf491, buf488, buf485, buf482, mul_868, buf493, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del buf482
        del buf485
        del buf488
        del buf491
        del mul_868
        buf494 = buf478; del buf478  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_81.run(buf493, buf494, 624, 1568, grid=grid(624), stream=stream0)
        buf495 = buf477; del buf477  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_82.run(buf493, cat_19, unsqueeze_594, buf495, 8112, 121, grid=grid(8112), stream=stream0)
        buf496 = empty((624, ), device='cuda', dtype=torch.float32)
        buf498 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_83.run(buf495, squeeze_82, buf496, buf498, 624, 13, grid=grid(624), stream=stream0)
        buf497 = buf493; del buf493  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_84.run(buf497, cat_19, unsqueeze_594, buf496, squeeze_82, buf494, primals_66, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del cat_19
        del primals_66
        del squeeze_82
        del unsqueeze_594
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf499 = aten.convolution_backward(reinterpret_tensor(buf497, (8, 312, 14, 14), (122304, 196, 14, 1), 61152), getitem_169, primals_198, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_169
        del primals_198
        buf500 = buf499[0]
        buf501 = buf499[1]
        del buf499
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf502 = aten.convolution_backward(reinterpret_tensor(buf497, (8, 312, 14, 14), (122304, 196, 14, 1), 0), getitem_168, primals_197, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_168
        del primals_197
        buf503 = buf502[0]
        buf504 = buf502[1]
        del buf502
        buf505 = buf400; del buf400  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_88.run(buf505, buf452, buf449, buf503, buf500, 163072, grid=grid(163072), stream=stream0)
        del buf449
        del buf452
        del buf500
        del buf503
        buf506 = buf455; del buf455  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_73.run(buf505, buf506, 104, 1568, grid=grid(104), stream=stream0)
        buf507 = buf403; del buf403  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_74.run(buf505, cat_18, unsqueeze_606, buf507, 1352, 121, grid=grid(1352), stream=stream0)
        buf508 = empty((104, ), device='cuda', dtype=torch.float32)
        buf510 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_75.run(buf507, squeeze_79, buf508, buf510, 104, 13, grid=grid(104), stream=stream0)
        del buf507
        buf509 = buf456; del buf456  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_76.run(buf505, cat_18, unsqueeze_606, buf508, squeeze_79, buf506, primals_64, buf509, 832, 196, grid=grid(832, 196), stream=stream0)
        del cat_18
        del primals_64
        del squeeze_79
        del unsqueeze_606
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf511 = aten.convolution_backward(reinterpret_tensor(buf509, (8, 52, 14, 14), (20384, 196, 14, 1), 10192), getitem_165, primals_196, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_165
        del primals_196
        buf512 = buf511[0]
        buf513 = buf511[1]
        del buf511
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf514 = aten.convolution_backward(reinterpret_tensor(buf509, (8, 52, 14, 14), (20384, 196, 14, 1), 0), getitem_164, primals_195, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf509
        del getitem_164
        del primals_195
        buf515 = buf514[0]
        buf516 = buf514[1]
        del buf514
        buf517 = reinterpret_tensor(buf473, (8, 624, 1, 1), (624, 1, 4992, 4992), 0); del buf473  # reuse
        buf518 = reinterpret_tensor(buf517, (8, 624, 1, 1), (624, 1, 1, 1), 0); del buf517  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate, x_181], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_77.run(buf518, buf515, buf512, add_134, convolution_61, 4992, 196, grid=grid(4992), stream=stream0)
        buf519 = buf496; del buf496  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_61.run(buf518, buf519, 624, 8, grid=grid(624), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf520 = aten.convolution_backward(buf518, mul_204, primals_193, [624], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf518
        del mul_204
        del primals_193
        buf521 = buf520[0]
        buf522 = buf520[1]
        del buf520
        buf523 = buf521; del buf521  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_78.run(buf523, convolution_60, 208, grid=grid(208), stream=stream0)
        del convolution_60
        buf524 = empty((26, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_79.run(buf523, buf524, 26, 8, grid=grid(26), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf525 = aten.convolution_backward(buf523, mean_5, primals_191, [26], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf523
        del mean_5
        del primals_191
        buf526 = buf525[0]
        buf527 = buf525[1]
        del buf525
        buf528 = buf497; del buf497  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___se_gate], Original ATen: [aten.add, aten.cat, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_80.run(buf515, buf512, convolution_61, buf526, add_134, buf528, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del add_134
        del buf512
        del buf515
        del buf526
        del convolution_61
        buf529 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_81.run(buf528, buf529, 624, 1568, grid=grid(624), stream=stream0)
        buf530 = buf495; del buf495  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_82.run(buf528, cat_17, unsqueeze_618, buf530, 8112, 121, grid=grid(8112), stream=stream0)
        buf531 = empty((624, ), device='cuda', dtype=torch.float32)
        buf533 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_83.run(buf530, squeeze_76, buf531, buf533, 624, 13, grid=grid(624), stream=stream0)
        buf532 = buf528; del buf528  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_84.run(buf532, cat_17, unsqueeze_618, buf531, squeeze_76, buf529, primals_62, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del cat_17
        del primals_62
        del squeeze_76
        del unsqueeze_618
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf534 = aten.convolution_backward(reinterpret_tensor(buf532, (8, 156, 14, 14), (122304, 196, 14, 1), 91728), getitem_161, primals_190, [0], [1, 1], [4, 4], [1, 1], False, [0, 0], 156, [True, True, False])
        del getitem_161
        del primals_190
        buf535 = buf534[0]
        buf536 = buf534[1]
        del buf534
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf537 = aten.convolution_backward(reinterpret_tensor(buf532, (8, 156, 14, 14), (122304, 196, 14, 1), 61152), getitem_156, primals_189, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 156, [True, True, False])
        del getitem_156
        del primals_189
        buf538 = buf537[0]
        buf539 = buf537[1]
        del buf537
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf540 = aten.convolution_backward(reinterpret_tensor(buf532, (8, 156, 14, 14), (122304, 196, 14, 1), 30576), getitem_151, primals_188, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 156, [True, True, False])
        del getitem_151
        del primals_188
        buf541 = buf540[0]
        buf542 = buf540[1]
        del buf540
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf543 = aten.convolution_backward(reinterpret_tensor(buf532, (8, 156, 14, 14), (122304, 196, 14, 1), 0), getitem_146, primals_187, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 156, [True, True, False])
        del getitem_146
        del primals_187
        buf544 = buf543[0]
        buf545 = buf543[1]
        del buf543
        buf546 = buf532; del buf532  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul]
        triton_poi_fused_cat_mul_85.run(buf544, buf541, buf538, buf535, mul_908, buf546, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del buf535
        del buf538
        del buf541
        del buf544
        del mul_908
        buf547 = buf531; del buf531  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_81.run(buf546, buf547, 624, 1568, grid=grid(624), stream=stream0)
        buf548 = buf530; del buf530  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_82.run(buf546, cat_16, unsqueeze_630, buf548, 8112, 121, grid=grid(8112), stream=stream0)
        buf549 = empty((624, ), device='cuda', dtype=torch.float32)
        buf551 = empty((624, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_83.run(buf548, squeeze_73, buf549, buf551, 624, 13, grid=grid(624), stream=stream0)
        del buf548
        buf550 = buf546; del buf546  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_84.run(buf550, cat_16, unsqueeze_630, buf549, squeeze_73, buf547, primals_60, 4992, 196, grid=grid(4992, 196), stream=stream0)
        del buf549
        del cat_16
        del primals_60
        del squeeze_73
        del unsqueeze_630
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf552 = aten.convolution_backward(reinterpret_tensor(buf550, (8, 312, 14, 14), (122304, 196, 14, 1), 61152), getitem_139, primals_186, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_139
        del primals_186
        buf553 = buf552[0]
        buf554 = buf552[1]
        del buf552
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf555 = aten.convolution_backward(reinterpret_tensor(buf550, (8, 312, 14, 14), (122304, 196, 14, 1), 0), getitem_138, primals_185, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf550
        del getitem_138
        del primals_185
        buf556 = buf555[0]
        buf557 = buf555[1]
        del buf555
        buf558 = buf508; del buf508  # reuse
        buf559 = empty((104, ), device='cuda', dtype=torch.float32)
        buf561 = empty((104, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_red_fused_add_cat_native_batch_norm_backward_86.run(buf505, buf556, buf553, convolution_53, unsqueeze_642, squeeze_70, buf558, buf559, buf561, 104, 1568, grid=grid(104), stream=stream0)
        buf560 = buf505; del buf505  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_poi_fused_add_cat_native_batch_norm_backward_89.run(buf560, buf556, buf553, convolution_53, unsqueeze_642, buf559, squeeze_70, buf558, primals_58, 832, 196, grid=grid(832, 196), stream=stream0)
        del buf553
        del buf556
        del buf559
        del convolution_53
        del primals_58
        del squeeze_70
        del unsqueeze_642
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf562 = aten.convolution_backward(buf560, mul_180, primals_184, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf560
        del mul_180
        del primals_184
        buf563 = buf562[0]
        buf564 = buf562[1]
        del buf562
        buf565 = empty_strided((8, 336, 1, 1, 2), (672, 2, 5376, 5376, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_163], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_90.run(buf563, add_119, buf565, 5376, 98, grid=grid(5376), stream=stream0)
        buf566 = empty_strided((8, 336, 1, 1), (336, 1, 2688, 2688), device='cuda', dtype=torch.float32)
        buf567 = reinterpret_tensor(buf566, (8, 336, 1, 1), (336, 1, 1, 1), 0); del buf566  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_163], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_91.run(buf567, buf565, convolution_52, 2688, 2, grid=grid(2688), stream=stream0)
        del buf565
        buf568 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_92.run(buf567, buf568, 336, 8, grid=grid(336), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf569 = aten.convolution_backward(buf567, mul_179, primals_182, [336], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf567
        del mul_179
        del primals_182
        buf570 = buf569[0]
        buf571 = buf569[1]
        del buf569
        buf572 = buf570; del buf570  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_93.run(buf572, convolution_51, 112, grid=grid(112), stream=stream0)
        del convolution_51
        buf573 = empty((14, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_94.run(buf572, buf573, 14, 8, grid=grid(14), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf574 = aten.convolution_backward(buf572, mean_4, primals_180, [14], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf572
        del mean_4
        del primals_180
        buf575 = buf574[0]
        buf576 = buf574[1]
        del buf574
        buf577 = empty((336, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_95.run(buf563, convolution_52, buf575, add_119, buf577, 4368, 121, grid=grid(4368), stream=stream0)
        buf578 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_96.run(buf577, buf578, 336, 13, grid=grid(336), stream=stream0)
        buf579 = reinterpret_tensor(buf577, (336, 13), (1, 336), 0); del buf577  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_97.run(buf563, convolution_52, buf575, add_119, cat_15, unsqueeze_654, buf579, 4368, 121, grid=grid(4368), stream=stream0)
        buf580 = empty((336, ), device='cuda', dtype=torch.float32)
        buf582 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_98.run(buf579, squeeze_67, buf580, buf582, 336, 13, grid=grid(336), stream=stream0)
        del buf579
        buf581 = empty_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_99.run(buf563, convolution_52, buf575, add_119, cat_15, unsqueeze_654, buf580, squeeze_67, buf578, buf581, 1568, 336, grid=grid(1568, 336), stream=stream0)
        del add_119
        del buf563
        del cat_15
        del convolution_52
        del unsqueeze_654
        buf583 = empty((8, 112, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_100.run(buf581, squeeze_67, primals_56, buf583, 896, 196, grid=grid(896, 196), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf584 = aten.convolution_backward(buf583, constant_pad_nd_10, primals_55, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 112, [True, True, False])
        del constant_pad_nd_10
        del primals_55
        buf585 = buf584[0]
        buf586 = buf584[1]
        del buf584
        buf587 = buf583; del buf583  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_101.run(buf581, squeeze_67, primals_56, buf587, 896, 196, grid=grid(896, 196), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf588 = aten.convolution_backward(buf587, constant_pad_nd_9, primals_54, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 112, [True, True, False])
        del constant_pad_nd_9
        del primals_54
        buf589 = buf588[0]
        buf590 = buf588[1]
        del buf588
        buf591 = buf587; del buf587  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_102.run(buf581, squeeze_67, primals_56, buf591, 896, 196, grid=grid(896, 196), stream=stream0)
        del buf581
        del primals_56
        del squeeze_67
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf592 = aten.convolution_backward(buf591, constant_pad_nd_8, primals_53, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 112, [True, True, False])
        del buf591
        del constant_pad_nd_8
        del primals_53
        buf593 = buf592[0]
        buf594 = buf592[1]
        del buf592
        buf595 = empty((8, 336, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_103.run(buf593, buf589, buf585, buf595, 2107392, grid=grid(2107392), stream=stream0)
        del buf585
        del buf589
        del buf593
        buf596 = empty((336, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_104.run(buf595, mul_948, buf596, 16464, 128, grid=grid(16464), stream=stream0)
        buf597 = buf580; del buf580  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_105.run(buf596, buf597, 336, 49, grid=grid(336), stream=stream0)
        buf598 = reinterpret_tensor(buf596, (336, 49), (1, 336), 0); del buf596  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_106.run(buf595, mul_948, convolution_47, unsqueeze_666, buf598, 16464, 128, grid=grid(16464), stream=stream0)
        buf599 = empty((336, ), device='cuda', dtype=torch.float32)
        buf600 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_107.run(buf598, squeeze_64, buf599, buf600, 336, 49, grid=grid(336), stream=stream0)
        buf601 = buf595; del buf595  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_108.run(buf601, mul_948, convolution_47, unsqueeze_666, buf599, squeeze_64, buf597, primals_51, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del convolution_47
        del mul_948
        del primals_51
        del squeeze_64
        del unsqueeze_666
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf602 = aten.convolution_backward(buf601, add_109, primals_179, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_109
        del primals_179
        buf603 = buf602[0]
        buf604 = buf602[1]
        del buf602
        buf605 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_109.run(buf603, buf605, 56, 6272, grid=grid(56), stream=stream0)
        buf606 = empty((56, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_110.run(buf603, cat_14, unsqueeze_678, buf606, 2744, 128, grid=grid(2744), stream=stream0)
        buf607 = empty((56, ), device='cuda', dtype=torch.float32)
        buf609 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_111.run(buf606, squeeze_61, buf607, buf609, 56, 49, grid=grid(56), stream=stream0)
        buf608 = empty((8, 56, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_112.run(buf603, cat_14, unsqueeze_678, buf607, squeeze_61, buf605, primals_49, buf608, 448, 784, grid=grid(448, 784), stream=stream0)
        del cat_14
        del primals_49
        del squeeze_61
        del unsqueeze_678
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf610 = aten.convolution_backward(reinterpret_tensor(buf608, (8, 28, 28, 28), (43904, 784, 28, 1), 21952), getitem_117, primals_178, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_117
        del primals_178
        buf611 = buf610[0]
        buf612 = buf610[1]
        del buf610
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf613 = aten.convolution_backward(reinterpret_tensor(buf608, (8, 28, 28, 28), (43904, 784, 28, 1), 0), getitem_116, primals_177, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_116
        del primals_177
        buf614 = buf613[0]
        buf615 = buf613[1]
        del buf613
        buf616 = reinterpret_tensor(buf575, (8, 336, 1, 1), (336, 1, 2688, 2688), 0); del buf575  # reuse
        buf617 = reinterpret_tensor(buf616, (8, 336, 1, 1), (336, 1, 1, 1), 0); del buf616  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___se_gate, x_138], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_113.run(buf617, buf614, buf611, add_103, convolution_44, 2688, 784, grid=grid(2688), stream=stream0)
        buf618 = buf599; del buf599  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_92.run(buf617, buf618, 336, 8, grid=grid(336), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf619 = aten.convolution_backward(buf617, mul_154, primals_175, [336], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf617
        del mul_154
        del primals_175
        buf620 = buf619[0]
        buf621 = buf619[1]
        del buf619
        buf622 = buf620; del buf620  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_114.run(buf622, convolution_43, 224, grid=grid(224), stream=stream0)
        del convolution_43
        buf623 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_115.run(buf622, buf623, 28, 8, grid=grid(28), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf624 = aten.convolution_backward(buf622, mean_3, primals_173, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf622
        del mean_3
        del primals_173
        buf625 = buf624[0]
        buf626 = buf624[1]
        del buf624
        buf627 = buf601; del buf601  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___se_gate], Original ATen: [aten.add, aten.cat, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_116.run(buf614, buf611, convolution_44, buf625, add_103, buf627, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del add_103
        del buf611
        del buf614
        del convolution_44
        buf628 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_117.run(buf627, buf628, 336, 6272, grid=grid(336), stream=stream0)
        buf629 = reinterpret_tensor(buf598, (336, 49), (49, 1), 0); del buf598  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_118.run(buf627, cat_13, unsqueeze_690, buf629, 16464, 128, grid=grid(16464), stream=stream0)
        buf630 = empty((336, ), device='cuda', dtype=torch.float32)
        buf632 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_119.run(buf629, squeeze_58, buf630, buf632, 336, 49, grid=grid(336), stream=stream0)
        buf631 = buf627; del buf627  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_120.run(buf631, cat_13, unsqueeze_690, buf630, squeeze_58, buf628, primals_47, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del cat_13
        del primals_47
        del squeeze_58
        del unsqueeze_690
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf633 = aten.convolution_backward(reinterpret_tensor(buf631, (8, 168, 28, 28), (263424, 784, 28, 1), 131712), getitem_113, primals_172, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 168, [True, True, False])
        del getitem_113
        del primals_172
        buf634 = buf633[0]
        buf635 = buf633[1]
        del buf633
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf636 = aten.convolution_backward(reinterpret_tensor(buf631, (8, 168, 28, 28), (263424, 784, 28, 1), 0), getitem_110, primals_171, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 168, [True, True, False])
        del getitem_110
        del primals_171
        buf637 = buf636[0]
        buf638 = buf636[1]
        del buf636
        buf639 = buf630; del buf630  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_cat_mul_native_batch_norm_backward_121.run(buf637, buf634, mul_988, buf639, 336, 6272, grid=grid(336), stream=stream0)
        buf640 = buf629; del buf629  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_cat_mul_native_batch_norm_backward_122.run(buf637, buf634, mul_988, cat_12, unsqueeze_702, buf640, 16464, 128, grid=grid(16464), stream=stream0)
        buf641 = empty((336, ), device='cuda', dtype=torch.float32)
        buf643 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_119.run(buf640, squeeze_55, buf641, buf643, 336, 49, grid=grid(336), stream=stream0)
        buf642 = buf631; del buf631  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_cat_mul_native_batch_norm_backward_123.run(buf637, buf634, mul_988, cat_12, unsqueeze_702, buf641, squeeze_55, buf639, primals_45, buf642, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del buf634
        del buf637
        del cat_12
        del mul_988
        del primals_45
        del squeeze_55
        del unsqueeze_702
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf644 = aten.convolution_backward(reinterpret_tensor(buf642, (8, 168, 28, 28), (263424, 784, 28, 1), 131712), getitem_105, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_105
        del primals_170
        buf645 = buf644[0]
        buf646 = buf644[1]
        del buf644
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf647 = aten.convolution_backward(reinterpret_tensor(buf642, (8, 168, 28, 28), (263424, 784, 28, 1), 0), getitem_104, primals_169, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_104
        del primals_169
        buf648 = buf647[0]
        buf649 = buf647[1]
        del buf647
        buf650 = buf607; del buf607  # reuse
        buf651 = empty((56, ), device='cuda', dtype=torch.float32)
        buf653 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_red_fused_add_cat_native_batch_norm_backward_124.run(buf603, buf648, buf645, cat_11, unsqueeze_714, squeeze_52, buf650, buf651, buf653, 56, 6272, grid=grid(56), stream=stream0)
        buf652 = buf608; del buf608  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_poi_fused_add_cat_native_batch_norm_backward_125.run(buf603, buf648, buf645, cat_11, unsqueeze_714, buf651, squeeze_52, buf650, primals_43, buf652, 448, 784, grid=grid(448, 784), stream=stream0)
        del cat_11
        del primals_43
        del squeeze_52
        del unsqueeze_714
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf654 = aten.convolution_backward(reinterpret_tensor(buf652, (8, 28, 28, 28), (43904, 784, 28, 1), 21952), getitem_101, primals_168, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_101
        del primals_168
        buf655 = buf654[0]
        buf656 = buf654[1]
        del buf654
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf657 = aten.convolution_backward(reinterpret_tensor(buf652, (8, 28, 28, 28), (43904, 784, 28, 1), 0), getitem_100, primals_167, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_100
        del primals_167
        buf658 = buf657[0]
        buf659 = buf657[1]
        del buf657
        buf660 = reinterpret_tensor(buf625, (8, 336, 1, 1), (336, 1, 2688, 2688), 0); del buf625  # reuse
        buf661 = reinterpret_tensor(buf660, (8, 336, 1, 1), (336, 1, 1, 1), 0); del buf660  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___se_gate, x_118], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_113.run(buf661, buf658, buf655, add_87, convolution_36, 2688, 784, grid=grid(2688), stream=stream0)
        buf662 = buf641; del buf641  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_92.run(buf661, buf662, 336, 8, grid=grid(336), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf663 = aten.convolution_backward(buf661, mul_129, primals_165, [336], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf661
        del mul_129
        del primals_165
        buf664 = buf663[0]
        buf665 = buf663[1]
        del buf663
        buf666 = buf664; del buf664  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_114.run(buf666, convolution_35, 224, grid=grid(224), stream=stream0)
        del convolution_35
        buf667 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_115.run(buf666, buf667, 28, 8, grid=grid(28), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf668 = aten.convolution_backward(buf666, mean_2, primals_163, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf666
        del mean_2
        del primals_163
        buf669 = buf668[0]
        buf670 = buf668[1]
        del buf668
        buf671 = buf642; del buf642  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___se_gate], Original ATen: [aten.add, aten.cat, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_116.run(buf658, buf655, convolution_36, buf669, add_87, buf671, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del add_87
        del buf655
        del buf658
        del convolution_36
        buf672 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_117.run(buf671, buf672, 336, 6272, grid=grid(336), stream=stream0)
        buf673 = buf640; del buf640  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_118.run(buf671, cat_10, unsqueeze_726, buf673, 16464, 128, grid=grid(16464), stream=stream0)
        buf674 = empty((336, ), device='cuda', dtype=torch.float32)
        buf676 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_119.run(buf673, squeeze_49, buf674, buf676, 336, 49, grid=grid(336), stream=stream0)
        buf675 = buf671; del buf671  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_120.run(buf675, cat_10, unsqueeze_726, buf674, squeeze_49, buf672, primals_41, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del cat_10
        del primals_41
        del squeeze_49
        del unsqueeze_726
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf677 = aten.convolution_backward(reinterpret_tensor(buf675, (8, 168, 28, 28), (263424, 784, 28, 1), 131712), getitem_97, primals_162, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 168, [True, True, False])
        del getitem_97
        del primals_162
        buf678 = buf677[0]
        buf679 = buf677[1]
        del buf677
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf680 = aten.convolution_backward(reinterpret_tensor(buf675, (8, 168, 28, 28), (263424, 784, 28, 1), 0), getitem_94, primals_161, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 168, [True, True, False])
        del getitem_94
        del primals_161
        buf681 = buf680[0]
        buf682 = buf680[1]
        del buf680
        buf683 = buf674; del buf674  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_cat_mul_native_batch_norm_backward_121.run(buf681, buf678, mul_1028, buf683, 336, 6272, grid=grid(336), stream=stream0)
        buf684 = buf673; del buf673  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_cat_mul_native_batch_norm_backward_122.run(buf681, buf678, mul_1028, cat_9, unsqueeze_738, buf684, 16464, 128, grid=grid(16464), stream=stream0)
        buf685 = empty((336, ), device='cuda', dtype=torch.float32)
        buf687 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_119.run(buf684, squeeze_46, buf685, buf687, 336, 49, grid=grid(336), stream=stream0)
        buf686 = buf675; del buf675  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_cat_mul_native_batch_norm_backward_123.run(buf681, buf678, mul_1028, cat_9, unsqueeze_738, buf685, squeeze_46, buf683, primals_39, buf686, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del buf678
        del buf681
        del cat_9
        del mul_1028
        del primals_39
        del squeeze_46
        del unsqueeze_738
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf688 = aten.convolution_backward(reinterpret_tensor(buf686, (8, 168, 28, 28), (263424, 784, 28, 1), 131712), getitem_89, primals_160, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_89
        del primals_160
        buf689 = buf688[0]
        buf690 = buf688[1]
        del buf688
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf691 = aten.convolution_backward(reinterpret_tensor(buf686, (8, 168, 28, 28), (263424, 784, 28, 1), 0), getitem_88, primals_159, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_88
        del primals_159
        buf692 = buf691[0]
        buf693 = buf691[1]
        del buf691
        buf694 = buf603; del buf603  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.cat]
        triton_poi_fused_add_cat_126.run(buf694, buf648, buf645, buf692, buf689, 351232, grid=grid(351232), stream=stream0)
        del buf645
        del buf648
        del buf689
        del buf692
        buf695 = buf651; del buf651  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_109.run(buf694, buf695, 56, 6272, grid=grid(56), stream=stream0)
        buf696 = buf606; del buf606  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_110.run(buf694, cat_8, unsqueeze_750, buf696, 2744, 128, grid=grid(2744), stream=stream0)
        buf697 = empty((56, ), device='cuda', dtype=torch.float32)
        buf699 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_111.run(buf696, squeeze_43, buf697, buf699, 56, 49, grid=grid(56), stream=stream0)
        del buf696
        buf698 = buf652; del buf652  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_112.run(buf694, cat_8, unsqueeze_750, buf697, squeeze_43, buf695, primals_37, buf698, 448, 784, grid=grid(448, 784), stream=stream0)
        del cat_8
        del primals_37
        del squeeze_43
        del unsqueeze_750
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf700 = aten.convolution_backward(reinterpret_tensor(buf698, (8, 28, 28, 28), (43904, 784, 28, 1), 21952), getitem_85, primals_158, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_85
        del primals_158
        buf701 = buf700[0]
        buf702 = buf700[1]
        del buf700
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf703 = aten.convolution_backward(reinterpret_tensor(buf698, (8, 28, 28, 28), (43904, 784, 28, 1), 0), getitem_84, primals_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf698
        del getitem_84
        del primals_157
        buf704 = buf703[0]
        buf705 = buf703[1]
        del buf703
        buf706 = reinterpret_tensor(buf669, (8, 336, 1, 1), (336, 1, 2688, 2688), 0); del buf669  # reuse
        buf707 = reinterpret_tensor(buf706, (8, 336, 1, 1), (336, 1, 1, 1), 0); del buf706  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_98], Original ATen: [aten.cat, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_cat_mul_sigmoid_sigmoid_backward_silu_sum_113.run(buf707, buf704, buf701, add_71, convolution_28, 2688, 784, grid=grid(2688), stream=stream0)
        buf708 = buf685; del buf685  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_92.run(buf707, buf708, 336, 8, grid=grid(336), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf709 = aten.convolution_backward(buf707, mul_104, primals_155, [336], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf707
        del mul_104
        del primals_155
        buf710 = buf709[0]
        buf711 = buf709[1]
        del buf709
        buf712 = buf710; del buf710  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_114.run(buf712, convolution_27, 224, grid=grid(224), stream=stream0)
        del convolution_27
        buf713 = empty((28, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_115.run(buf712, buf713, 28, 8, grid=grid(28), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf714 = aten.convolution_backward(buf712, mean_1, primals_153, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf712
        del mean_1
        del primals_153
        buf715 = buf714[0]
        buf716 = buf714[1]
        del buf714
        buf717 = buf686; del buf686  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate], Original ATen: [aten.add, aten.cat, aten.div, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_cat_div_fill_mul_sigmoid_sub_116.run(buf704, buf701, convolution_28, buf715, add_71, buf717, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del add_71
        del buf701
        del buf704
        del buf715
        del convolution_28
        buf718 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_117.run(buf717, buf718, 336, 6272, grid=grid(336), stream=stream0)
        buf719 = buf684; del buf684  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_118.run(buf717, cat_7, unsqueeze_762, buf719, 16464, 128, grid=grid(16464), stream=stream0)
        buf720 = empty((336, ), device='cuda', dtype=torch.float32)
        buf722 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_119.run(buf719, squeeze_40, buf720, buf722, 336, 49, grid=grid(336), stream=stream0)
        buf721 = buf717; del buf717  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_120.run(buf721, cat_7, unsqueeze_762, buf720, squeeze_40, buf718, primals_35, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del cat_7
        del primals_35
        del squeeze_40
        del unsqueeze_762
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf723 = aten.convolution_backward(reinterpret_tensor(buf721, (8, 168, 28, 28), (263424, 784, 28, 1), 131712), getitem_81, primals_152, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 168, [True, True, False])
        del getitem_81
        del primals_152
        buf724 = buf723[0]
        buf725 = buf723[1]
        del buf723
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf726 = aten.convolution_backward(reinterpret_tensor(buf721, (8, 168, 28, 28), (263424, 784, 28, 1), 0), getitem_78, primals_151, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 168, [True, True, False])
        del getitem_78
        del primals_151
        buf727 = buf726[0]
        buf728 = buf726[1]
        del buf726
        buf729 = buf720; del buf720  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_cat_mul_native_batch_norm_backward_121.run(buf727, buf724, mul_1068, buf729, 336, 6272, grid=grid(336), stream=stream0)
        buf730 = buf719; del buf719  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_cat_mul_native_batch_norm_backward_122.run(buf727, buf724, mul_1068, cat_6, unsqueeze_774, buf730, 16464, 128, grid=grid(16464), stream=stream0)
        buf731 = empty((336, ), device='cuda', dtype=torch.float32)
        buf733 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_119.run(buf730, squeeze_37, buf731, buf733, 336, 49, grid=grid(336), stream=stream0)
        del buf730
        buf732 = buf721; del buf721  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_cat_mul_native_batch_norm_backward_123.run(buf727, buf724, mul_1068, cat_6, unsqueeze_774, buf731, squeeze_37, buf729, primals_33, buf732, 2688, 784, grid=grid(2688, 784), stream=stream0)
        del buf724
        del buf727
        del buf731
        del cat_6
        del mul_1068
        del primals_33
        del squeeze_37
        del unsqueeze_774
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf734 = aten.convolution_backward(reinterpret_tensor(buf732, (8, 168, 28, 28), (263424, 784, 28, 1), 131712), getitem_73, primals_150, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_73
        del primals_150
        buf735 = buf734[0]
        buf736 = buf734[1]
        del buf734
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf737 = aten.convolution_backward(reinterpret_tensor(buf732, (8, 168, 28, 28), (263424, 784, 28, 1), 0), getitem_72, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf732
        del getitem_72
        del primals_149
        buf738 = buf737[0]
        buf739 = buf737[1]
        del buf737
        buf740 = buf697; del buf697  # reuse
        buf741 = empty((56, ), device='cuda', dtype=torch.float32)
        buf743 = empty((56, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_red_fused_add_cat_native_batch_norm_backward_124.run(buf694, buf738, buf735, convolution_22, unsqueeze_786, squeeze_34, buf740, buf741, buf743, 56, 6272, grid=grid(56), stream=stream0)
        buf742 = buf694; del buf694  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_poi_fused_add_cat_native_batch_norm_backward_127.run(buf742, buf738, buf735, convolution_22, unsqueeze_786, buf741, squeeze_34, buf740, primals_31, 448, 784, grid=grid(448, 784), stream=stream0)
        del buf735
        del buf738
        del buf741
        del convolution_22
        del primals_31
        del squeeze_34
        del unsqueeze_786
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf744 = aten.convolution_backward(buf742, mul_80, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf742
        del mul_80
        del primals_148
        buf745 = buf744[0]
        buf746 = buf744[1]
        del buf744
        buf747 = empty_strided((8, 240, 1, 1, 7), (1680, 7, 13440, 13440, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_80], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_128.run(buf745, add_56, buf747, 13440, 112, grid=grid(13440), stream=stream0)
        buf748 = empty_strided((8, 240, 1, 1), (240, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf749 = reinterpret_tensor(buf748, (8, 240, 1, 1), (240, 1, 1, 1), 0); del buf748  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_80], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_129.run(buf749, buf747, convolution_21, 1920, 7, grid=grid(1920), stream=stream0)
        del buf747
        buf750 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_130.run(buf749, buf750, 240, 8, grid=grid(240), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf751 = aten.convolution_backward(buf749, mul_79, primals_146, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf749
        del mul_79
        del primals_146
        buf752 = buf751[0]
        buf753 = buf751[1]
        del buf751
        buf754 = buf752; del buf752  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_sub_131.run(buf754, convolution_20, 160, grid=grid(160), stream=stream0)
        del convolution_20
        buf755 = empty((20, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_132.run(buf754, buf755, 20, 8, grid=grid(20), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf756 = aten.convolution_backward(buf754, mean, primals_144, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean
        del primals_144
        buf757 = buf756[0]
        buf758 = buf756[1]
        del buf756
        buf759 = empty((240, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_133.run(buf745, convolution_21, buf757, add_56, buf759, 11760, 128, grid=grid(11760), stream=stream0)
        buf760 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_134.run(buf759, buf760, 240, 49, grid=grid(240), stream=stream0)
        buf761 = reinterpret_tensor(buf759, (240, 49), (1, 240), 0); del buf759  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_135.run(buf745, convolution_21, buf757, add_56, cat_5, unsqueeze_798, buf761, 11760, 128, grid=grid(11760), stream=stream0)
        buf762 = empty((240, ), device='cuda', dtype=torch.float32)
        buf764 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_136.run(buf761, squeeze_31, buf762, buf764, 240, 49, grid=grid(240), stream=stream0)
        del buf761
        buf763 = reinterpret_tensor(buf204, (8, 240, 28, 28), (188160, 1, 6720, 240), 0); del buf204  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_137.run(buf745, convolution_21, buf757, add_56, cat_5, unsqueeze_798, buf762, squeeze_31, buf760, buf763, 6272, 240, grid=grid(6272, 240), stream=stream0)
        del add_56
        del buf745
        del buf757
        del cat_5
        del convolution_21
        del unsqueeze_798
        buf765 = reinterpret_tensor(buf321, (8, 60, 28, 28), (47040, 784, 28, 1), 0); del buf321  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_138.run(buf763, squeeze_31, primals_29, buf765, 480, 784, grid=grid(480, 784), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf766 = aten.convolution_backward(buf765, constant_pad_nd_7, primals_28, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 60, [True, True, False])
        del constant_pad_nd_7
        del primals_28
        buf767 = buf766[0]
        buf768 = buf766[1]
        del buf766
        buf769 = buf765; del buf765  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_139.run(buf763, squeeze_31, primals_29, buf769, 480, 784, grid=grid(480, 784), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf770 = aten.convolution_backward(buf769, constant_pad_nd_6, primals_27, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 60, [True, True, False])
        del constant_pad_nd_6
        del primals_27
        buf771 = buf770[0]
        buf772 = buf770[1]
        del buf770
        buf773 = buf769; del buf769  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_140.run(buf763, squeeze_31, primals_29, buf773, 480, 784, grid=grid(480, 784), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf774 = aten.convolution_backward(buf773, constant_pad_nd_5, primals_26, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 60, [True, True, False])
        del constant_pad_nd_5
        del primals_26
        buf775 = buf774[0]
        buf776 = buf774[1]
        del buf774
        buf777 = buf773; del buf773  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_141.run(buf763, squeeze_31, primals_29, buf777, 480, 784, grid=grid(480, 784), stream=stream0)
        del buf763
        del primals_29
        del squeeze_31
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf778 = aten.convolution_backward(buf777, constant_pad_nd_4, primals_25, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 60, [True, True, False])
        del buf777
        del constant_pad_nd_4
        del primals_25
        buf779 = buf778[0]
        buf780 = buf778[1]
        del buf778
        buf781 = empty((8, 240, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_142.run(buf779, buf775, buf771, buf767, buf781, 6021120, grid=grid(6021120), stream=stream0)
        del buf767
        del buf771
        del buf775
        del buf779
        buf782 = empty((240, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_143.run(buf781, mul_1108, buf782, 47040, 128, grid=grid(47040), stream=stream0)
        buf783 = buf762; del buf762  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_144.run(buf782, buf783, 240, 196, grid=grid(240), stream=stream0)
        buf784 = reinterpret_tensor(buf782, (240, 196), (1, 240), 0); del buf782  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_145.run(buf781, mul_1108, convolution_15, unsqueeze_810, buf784, 47040, 128, grid=grid(47040), stream=stream0)
        buf785 = empty((240, ), device='cuda', dtype=torch.float32)
        buf786 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_146.run(buf784, squeeze_28, buf785, buf786, 240, 196, grid=grid(240), stream=stream0)
        del buf784
        buf787 = buf781; del buf781  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_147.run(buf787, mul_1108, convolution_15, unsqueeze_810, buf785, squeeze_28, buf783, primals_23, 1920, 3136, grid=grid(1920, 3136), stream=stream0)
        del buf785
        del convolution_15
        del mul_1108
        del primals_23
        del squeeze_28
        del unsqueeze_810
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf788 = aten.convolution_backward(buf787, add_46, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_46
        del buf787
        del primals_143
        buf789 = buf788[0]
        buf790 = buf788[1]
        del buf788
        buf791 = reinterpret_tensor(buf754, (40, 4), (1, 40), 0); del buf754  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_148.run(buf789, buf791, 160, 6272, grid=grid(160), stream=stream0)
        buf792 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_149.run(buf791, buf792, 40, 4, grid=grid(40), stream=stream0)
        buf793 = empty((40, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_150.run(buf789, cat_4, unsqueeze_822, buf793, 7840, 128, grid=grid(7840), stream=stream0)
        buf794 = empty((40, ), device='cuda', dtype=torch.float32)
        buf796 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_151.run(buf793, squeeze_25, buf794, buf796, 40, 196, grid=grid(40), stream=stream0)
        del buf793
        buf795 = empty((8, 40, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_152.run(buf789, cat_4, unsqueeze_822, buf794, squeeze_25, buf792, primals_21, buf795, 320, 3136, grid=grid(320, 3136), stream=stream0)
        del cat_4
        del primals_21
        del squeeze_25
        del unsqueeze_822
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf797 = aten.convolution_backward(reinterpret_tensor(buf795, (8, 20, 56, 56), (125440, 3136, 56, 1), 62720), getitem_43, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_43
        del primals_142
        buf798 = buf797[0]
        buf799 = buf797[1]
        del buf797
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf800 = aten.convolution_backward(reinterpret_tensor(buf795, (8, 20, 56, 56), (125440, 3136, 56, 1), 0), getitem_40, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf795
        del getitem_40
        del primals_141
        buf801 = buf800[0]
        buf802 = buf800[1]
        del buf800
        buf803 = reinterpret_tensor(buf355, (120, 4), (1, 120), 0); del buf355  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_cat_native_batch_norm_backward_threshold_backward_153.run(le_1, buf801, buf798, buf803, 480, 6272, grid=grid(480), stream=stream0)
        buf804 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_cat_native_batch_norm_backward_threshold_backward_154.run(buf803, buf804, 120, 4, grid=grid(120), stream=stream0)
        del buf803
        buf805 = empty((120, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_cat_native_batch_norm_backward_threshold_backward_155.run(le_1, buf801, buf798, convolution_12, unsqueeze_834, buf805, 23520, 128, grid=grid(23520), stream=stream0)
        buf806 = empty((120, ), device='cuda', dtype=torch.float32)
        buf808 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_cat_native_batch_norm_backward_threshold_backward_156.run(buf805, squeeze_22, buf806, buf808, 120, 196, grid=grid(120), stream=stream0)
        buf807 = empty_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_cat_native_batch_norm_backward_threshold_backward_157.run(le_1, buf801, buf798, convolution_12, unsqueeze_834, buf806, squeeze_22, buf804, primals_19, buf807, 25088, 120, grid=grid(25088, 120), stream=stream0)
        del buf798
        del buf801
        del convolution_12
        del le_1
        del primals_19
        del squeeze_22
        del unsqueeze_834
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf809 = aten.convolution_backward(buf807, relu_4, primals_140, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 120, [True, True, False])
        del primals_140
        buf810 = buf809[0]
        buf811 = buf809[1]
        del buf809
        buf812 = buf805; del buf805  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_158.run(relu_4, buf810, buf812, 23520, 128, grid=grid(23520), stream=stream0)
        buf813 = buf806; del buf806  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_159.run(buf812, buf813, 120, 196, grid=grid(120), stream=stream0)
        buf814 = reinterpret_tensor(buf812, (120, 196), (1, 120), 0); del buf812  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_160.run(relu_4, buf810, cat_3, unsqueeze_846, buf814, 23520, 128, grid=grid(23520), stream=stream0)
        buf815 = empty((120, ), device='cuda', dtype=torch.float32)
        buf817 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_161.run(buf814, squeeze_19, buf815, buf817, 120, 196, grid=grid(120), stream=stream0)
        del buf814
        buf816 = buf807; del buf807  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_162.run(relu_4, buf810, cat_3, unsqueeze_846, buf815, squeeze_19, buf813, primals_17, buf816, 25088, 120, grid=grid(25088, 120), stream=stream0)
        del buf810
        del buf815
        del cat_3
        del primals_17
        del relu_4
        del squeeze_19
        del unsqueeze_846
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf818 = aten.convolution_backward(reinterpret_tensor(buf816, (8, 60, 56, 56), (376320, 1, 6720, 120), 60), getitem_33, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_33
        del primals_139
        buf819 = buf818[0]
        buf820 = buf818[1]
        del buf818
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf821 = aten.convolution_backward(reinterpret_tensor(buf816, (8, 60, 56, 56), (376320, 1, 6720, 120), 0), getitem_32, primals_138, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf816
        del getitem_32
        del primals_138
        buf822 = buf821[0]
        buf823 = buf821[1]
        del buf821
        buf824 = buf791; del buf791  # reuse
        buf826 = reinterpret_tensor(buf365, (40, 4), (1, 40), 0); del buf365  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_red_fused_add_cat_native_batch_norm_backward_163.run(buf789, buf822, buf819, cat_2, unsqueeze_858, buf824, buf826, 160, 6272, grid=grid(160), stream=stream0)
        buf825 = buf794; del buf794  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_149.run(buf824, buf825, 40, 4, grid=grid(40), stream=stream0)
        del buf824
        buf827 = empty((40, ), device='cuda', dtype=torch.float32)
        buf829 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_per_fused_add_cat_native_batch_norm_backward_164.run(buf826, squeeze_16, buf827, buf829, 40, 4, grid=grid(40), stream=stream0)
        del buf826
        buf828 = buf789; del buf789  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward]
        triton_poi_fused_add_cat_native_batch_norm_backward_165.run(buf828, buf822, buf819, cat_2, unsqueeze_858, buf827, squeeze_16, buf825, primals_15, 320, 3136, grid=grid(320, 3136), stream=stream0)
        del buf819
        del buf822
        del buf827
        del cat_2
        del primals_15
        del squeeze_16
        del unsqueeze_858
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf830 = aten.convolution_backward(reinterpret_tensor(buf828, (8, 20, 56, 56), (125440, 3136, 56, 1), 62720), getitem_29, primals_137, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_29
        del primals_137
        buf831 = buf830[0]
        buf832 = buf830[1]
        del buf830
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf833 = aten.convolution_backward(reinterpret_tensor(buf828, (8, 20, 56, 56), (125440, 3136, 56, 1), 0), getitem_26, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf828
        del getitem_26
        del primals_136
        buf834 = buf833[0]
        buf835 = buf833[1]
        del buf833
        buf836 = empty_strided((192, 4), (1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_cat_native_batch_norm_backward_threshold_backward_166.run(le_3, buf834, buf831, buf836, 768, 6272, grid=grid(768), stream=stream0)
        buf837 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_cat_native_batch_norm_backward_threshold_backward_167.run(buf836, buf837, 192, 4, grid=grid(192), stream=stream0)
        del buf836
        buf838 = empty((192, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_cat_native_batch_norm_backward_threshold_backward_168.run(le_3, buf834, buf831, cat_1, unsqueeze_870, buf838, 37632, 128, grid=grid(37632), stream=stream0)
        buf839 = empty((192, ), device='cuda', dtype=torch.float32)
        buf841 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_cat_native_batch_norm_backward_threshold_backward_169.run(buf838, squeeze_13, buf839, buf841, 192, 196, grid=grid(192), stream=stream0)
        del buf838
        buf840 = empty((8, 192, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_cat_native_batch_norm_backward_threshold_backward_170.run(le_3, buf834, buf831, cat_1, unsqueeze_870, buf839, squeeze_13, buf837, primals_13, buf840, 1536, 3136, grid=grid(1536, 3136), stream=stream0)
        del buf831
        del buf834
        del cat_1
        del le_3
        del primals_13
        del squeeze_13
        del unsqueeze_870
        buf842 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_171.run(buf840, buf842, 512, 3136, grid=grid(512, 3136), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf843 = aten.convolution_backward(buf842, constant_pad_nd_3, primals_12, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 64, [True, True, False])
        del constant_pad_nd_3
        del primals_12
        buf844 = buf843[0]
        buf845 = buf843[1]
        del buf843
        buf846 = buf842; del buf842  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_172.run(buf840, buf846, 512, 3136, grid=grid(512, 3136), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf847 = aten.convolution_backward(buf846, constant_pad_nd_2, primals_11, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 64, [True, True, False])
        del constant_pad_nd_2
        del primals_11
        buf848 = buf847[0]
        buf849 = buf847[1]
        del buf847
        buf850 = buf846; del buf846  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_173.run(buf840, buf850, 512, 3136, grid=grid(512, 3136), stream=stream0)
        del buf840
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf851 = aten.convolution_backward(buf850, constant_pad_nd_1, primals_10, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 64, [True, True, False])
        del buf850
        del constant_pad_nd_1
        del primals_10
        buf852 = buf851[0]
        buf853 = buf851[1]
        del buf851
        buf854 = empty((8, 192, 112, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_174.run(buf852, buf848, buf844, buf854, 19267584, grid=grid(19267584), stream=stream0)
        del buf844
        del buf848
        del buf852
        buf855 = empty((192, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_175.run(le_4, buf854, buf855, 98304, 196, grid=grid(98304), stream=stream0)
        buf856 = buf839; del buf839  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_176.run(buf855, buf856, 192, 512, grid=grid(192), stream=stream0)
        buf857 = reinterpret_tensor(buf855, (192, 512), (1, 192), 0); del buf855  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_177.run(le_4, buf854, cat, unsqueeze_882, buf857, 98304, 196, grid=grid(98304), stream=stream0)
        buf858 = empty((192, ), device='cuda', dtype=torch.float32)
        buf860 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_178.run(buf857, squeeze_10, buf858, buf860, 192, 512, grid=grid(192), stream=stream0)
        del buf857
        buf859 = empty_strided((8, 192, 112, 112), (2408448, 1, 21504, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_native_batch_norm_backward_threshold_backward_179.run(le_4, buf854, cat, unsqueeze_882, buf858, squeeze_10, buf856, primals_8, buf859, 100352, 192, grid=grid(100352, 192), stream=stream0)
        del buf854
        del buf858
        del cat
        del le_4
        del primals_8
        del squeeze_10
        del unsqueeze_882
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf861 = aten.convolution_backward(reinterpret_tensor(buf859, (8, 96, 112, 112), (2408448, 1, 21504, 192), 96), getitem_7, primals_135, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del getitem_7
        del primals_135
        buf862 = buf861[0]
        buf863 = buf861[1]
        del buf861
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf864 = aten.convolution_backward(reinterpret_tensor(buf859, (8, 96, 112, 112), (2408448, 1, 21504, 192), 0), getitem_6, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf859
        del getitem_6
        del primals_134
        buf865 = buf864[0]
        buf866 = buf864[1]
        del buf864
        buf867 = reinterpret_tensor(buf378, (32, 13), (13, 1), 0); del buf378  # reuse
        buf869 = empty((32, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward]
        triton_red_fused_cat_native_batch_norm_backward_180.run(buf865, buf862, convolution_2, unsqueeze_894, buf867, buf869, 416, 7720, grid=grid(416), stream=stream0)
        buf868 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward]
        triton_per_fused_cat_native_batch_norm_backward_181.run(buf867, buf868, 32, 13, grid=grid(32), stream=stream0)
        buf870 = empty((32, ), device='cuda', dtype=torch.float32)
        buf871 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.native_batch_norm_backward]
        triton_per_fused_cat_native_batch_norm_backward_182.run(buf869, squeeze_7, buf870, buf871, 32, 13, grid=grid(32), stream=stream0)
        buf872 = empty((8, 32, 112, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_cat_convolution_backward_native_batch_norm_backward_183.run(buf865, buf862, convolution_2, unsqueeze_894, buf870, squeeze_7, buf868, primals_6, buf872, 256, 12544, grid=grid(256, 12544), stream=stream0)
        del convolution_2
        del primals_6
        del squeeze_7
        del unsqueeze_894
        # Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward, aten.native_batch_norm_backward]
        buf873 = aten.convolution_backward(buf872, relu_1, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_133
        buf874 = buf873[0]
        buf875 = buf873[1]
        del buf873
        buf876 = empty((32, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_184.run(relu_1, buf874, buf876, 25088, 128, grid=grid(25088), stream=stream0)
        buf877 = buf870; del buf870  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_185.run(buf876, buf877, 32, 784, grid=grid(32), stream=stream0)
        buf878 = reinterpret_tensor(buf876, (32, 784), (1, 32), 0); del buf876  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_186.run(relu_1, buf874, convolution_1, unsqueeze_906, buf878, 25088, 128, grid=grid(25088), stream=stream0)
        buf879 = empty((32, ), device='cuda', dtype=torch.float32)
        buf880 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_187.run(buf878, squeeze_4, buf879, buf880, 32, 784, grid=grid(32), stream=stream0)
        del buf878
        buf881 = reinterpret_tensor(buf872, (8, 32, 112, 112), (401408, 1, 3584, 32), 0); del buf872  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_188.run(relu_1, buf874, convolution_1, unsqueeze_906, buf879, squeeze_4, buf877, primals_4, buf881, 100352, 32, grid=grid(100352, 32), stream=stream0)
        del buf874
        del convolution_1
        del primals_4
        del relu_1
        del squeeze_4
        del unsqueeze_906
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf882 = aten.convolution_backward(buf881, relu, primals_132, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del primals_132
        buf883 = buf882[0]
        buf884 = buf882[1]
        del buf882
        buf885 = buf869; del buf869  # reuse
        buf887 = buf867; del buf867  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_cat_native_batch_norm_backward_threshold_backward_189.run(relu, buf865, buf862, buf883, convolution, unsqueeze_918, buf885, buf887, 416, 7720, grid=grid(416), stream=stream0)
        buf886 = buf879; del buf879  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_cat_native_batch_norm_backward_181.run(buf885, buf886, 32, 13, grid=grid(32), stream=stream0)
        del buf885
        buf888 = empty((32, ), device='cuda', dtype=torch.float32)
        buf890 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_cat_native_batch_norm_backward_182.run(buf887, squeeze_1, buf888, buf890, 32, 13, grid=grid(32), stream=stream0)
        del buf887
        buf889 = buf883; del buf883  # reuse
        buf891 = buf881; del buf881  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.cat, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_cat_convolution_backward_native_batch_norm_backward_threshold_backward_190.run(buf889, relu, buf865, buf862, convolution, unsqueeze_918, buf888, squeeze_1, buf886, primals_2, buf891, 256, 12544, grid=grid(256, 12544), stream=stream0)
        del buf862
        del buf865
        del buf888
        del buf889
        del convolution
        del primals_2
        del relu
        del squeeze_1
        del unsqueeze_918
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf892 = aten.convolution_backward(buf891, constant_pad_nd, primals_1, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf891
        del constant_pad_nd
        del primals_1
        buf893 = buf892[1]
        return (buf893, buf890, buf886, buf880, buf877, buf871, buf868, buf860, buf856, buf853, buf849, buf845, buf841, buf837, buf829, buf825, buf817, buf813, buf808, buf804, buf796, buf792, buf786, buf783, buf780, buf776, buf772, buf768, buf764, buf760, buf743, buf740, buf733, buf729, buf722, buf718, buf699, buf695, buf687, buf683, buf676, buf672, buf653, buf650, buf643, buf639, buf632, buf628, buf609, buf605, buf600, buf597, buf594, buf590, buf586, buf582, buf578, buf561, buf558, buf551, buf547, buf533, buf529, buf510, buf506, buf498, buf494, buf480, buf476, buf457, buf454, buf447, buf443, buf429, buf425, buf406, buf402, buf397, buf394, buf388, buf384, buf367, buf364, buf357, buf353, buf339, buf335, buf316, buf312, buf304, buf300, buf286, buf282, buf263, buf260, buf253, buf249, buf235, buf231, buf212, buf208, buf203, buf200, buf197, buf193, buf189, buf185, buf181, buf177, buf160, buf157, buf152, buf149, buf135, buf131, buf112, buf109, buf104, buf101, buf87, buf83, buf64, buf61, buf56, buf53, buf39, buf35, buf16, buf12, buf7, buf4, buf884, buf875, buf866, buf863, buf835, buf832, buf823, buf820, buf811, buf802, buf799, buf790, buf758, buf755, buf753, buf750, buf746, buf739, buf736, buf728, buf725, buf716, buf713, buf711, buf708, buf705, buf702, buf693, buf690, buf682, buf679, buf670, buf667, buf665, buf662, buf659, buf656, buf649, buf646, buf638, buf635, buf626, buf623, buf621, buf618, buf615, buf612, buf604, buf576, buf573, buf571, buf568, buf564, buf557, buf554, buf545, buf542, buf539, buf536, buf527, buf524, buf522, buf519, buf516, buf513, buf504, buf501, buf492, buf489, buf486, buf483, buf474, buf471, buf469, buf466, buf463, buf460, buf453, buf450, buf441, buf438, buf435, buf432, buf423, buf420, buf418, buf415, buf412, buf409, buf401, buf392, buf382, buf379, buf377, buf374, buf370, buf363, buf360, buf351, buf348, buf345, buf342, buf333, buf330, buf328, buf325, buf322, buf319, buf310, buf307, buf298, buf295, buf292, buf289, buf280, buf277, buf275, buf272, buf269, buf266, buf259, buf256, buf247, buf244, buf241, buf238, buf229, buf226, buf224, buf221, buf218, buf215, buf207, buf175, buf172, buf170, buf167, buf164, buf156, buf147, buf144, buf141, buf138, buf129, buf126, buf124, buf121, buf118, buf115, buf108, buf99, buf96, buf93, buf90, buf81, buf78, buf76, buf73, buf70, buf67, buf60, buf51, buf48, buf45, buf42, buf33, buf30, buf28, buf25, buf22, buf19, buf11, reinterpret_tensor(buf1, (1000, 1536), (1536, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((60, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((60, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((60, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((60, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((112, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((112, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((112, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((624, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((240, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((240, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((1584, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((264, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((20, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((20, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((60, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((60, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((20, 60, 1, 1), (60, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((20, 60, 1, 1), (60, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((20, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((240, 20, 1, 1), (20, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((56, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((168, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((168, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((28, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((336, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((168, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((168, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((28, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((336, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((168, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((168, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((28, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((336, 28, 1, 1), (28, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((336, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((14, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((336, 14, 1, 1), (14, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((104, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((156, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((156, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((156, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((156, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((26, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((624, 26, 1, 1), (26, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((156, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((156, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((156, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((156, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((26, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((624, 26, 1, 1), (26, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((156, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((156, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((156, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((156, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((26, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((624, 26, 1, 1), (26, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((624, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((624, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((52, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((624, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((160, 624, 1, 1), (624, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((120, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((120, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((120, 1, 9, 9), (81, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
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
    constant_pad_nd = rand_strided((8, 3, 225, 225), (151875, 1, 675, 3), device='cuda:0', dtype=torch.float32)
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
    constant_pad_nd_1 = rand_strided((8, 64, 113, 113), (817216, 1, 7232, 64), device='cuda:0', dtype=torch.float32)
    constant_pad_nd_2 = rand_strided((8, 64, 115, 115), (846400, 1, 7360, 64), device='cuda:0', dtype=torch.float32)
    constant_pad_nd_3 = rand_strided((8, 64, 117, 117), (876096, 1, 7488, 64), device='cuda:0', dtype=torch.float32)
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
    constant_pad_nd_4 = rand_strided((8, 60, 57, 57), (194940, 1, 3420, 60), device='cuda:0', dtype=torch.float32)
    constant_pad_nd_5 = rand_strided((8, 60, 59, 59), (208860, 1, 3540, 60), device='cuda:0', dtype=torch.float32)
    constant_pad_nd_6 = rand_strided((8, 60, 61, 61), (223260, 1, 3660, 60), device='cuda:0', dtype=torch.float32)
    constant_pad_nd_7 = rand_strided((8, 60, 63, 63), (238140, 1, 3780, 60), device='cuda:0', dtype=torch.float32)
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
    constant_pad_nd_8 = rand_strided((8, 112, 29, 29), (94192, 1, 3248, 112), device='cuda:0', dtype=torch.float32)
    constant_pad_nd_9 = rand_strided((8, 112, 31, 31), (107632, 1, 3472, 112), device='cuda:0', dtype=torch.float32)
    constant_pad_nd_10 = rand_strided((8, 112, 33, 33), (121968, 1, 3696, 112), device='cuda:0', dtype=torch.float32)
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
    constant_pad_nd_11 = rand_strided((8, 240, 15, 15), (54000, 1, 3600, 240), device='cuda:0', dtype=torch.float32)
    constant_pad_nd_12 = rand_strided((8, 240, 17, 17), (69360, 1, 4080, 240), device='cuda:0', dtype=torch.float32)
    constant_pad_nd_13 = rand_strided((8, 240, 19, 19), (86640, 1, 4560, 240), device='cuda:0', dtype=torch.float32)
    constant_pad_nd_14 = rand_strided((8, 240, 21, 21), (105840, 1, 5040, 240), device='cuda:0', dtype=torch.float32)
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
    return print_performance(lambda: call([primals_1, primals_2, primals_4, primals_6, primals_8, primals_10, primals_11, primals_12, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_26, primals_27, primals_28, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_54, primals_55, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_105, primals_106, primals_107, primals_108, primals_110, primals_112, primals_114, primals_116, primals_118, primals_120, primals_122, primals_124, primals_126, primals_128, primals_130, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_146, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_155, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_165, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_175, primals_177, primals_178, primals_179, primals_180, primals_182, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_193, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_205, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_217, primals_219, primals_220, primals_221, primals_222, primals_223, primals_225, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_236, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_248, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_260, primals_262, primals_263, primals_264, primals_265, primals_267, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_277, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_288, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_299, primals_301, primals_302, primals_303, constant_pad_nd, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, getitem_6, getitem_7, cat, squeeze_10, constant_pad_nd_1, constant_pad_nd_2, constant_pad_nd_3, cat_1, squeeze_13, getitem_26, getitem_29, cat_2, squeeze_16, getitem_32, getitem_33, cat_3, squeeze_19, relu_4, convolution_12, squeeze_22, getitem_40, getitem_43, cat_4, squeeze_25, add_46, convolution_15, squeeze_28, constant_pad_nd_4, constant_pad_nd_5, constant_pad_nd_6, constant_pad_nd_7, cat_5, squeeze_31, add_56, mean, convolution_20, mul_79, convolution_21, mul_80, convolution_22, squeeze_34, getitem_72, getitem_73, cat_6, squeeze_37, getitem_78, getitem_81, cat_7, squeeze_40, add_71, mean_1, convolution_27, mul_104, convolution_28, getitem_84, getitem_85, cat_8, squeeze_43, getitem_88, getitem_89, cat_9, squeeze_46, getitem_94, getitem_97, cat_10, squeeze_49, add_87, mean_2, convolution_35, mul_129, convolution_36, getitem_100, getitem_101, cat_11, squeeze_52, getitem_104, getitem_105, cat_12, squeeze_55, getitem_110, getitem_113, cat_13, squeeze_58, add_103, mean_3, convolution_43, mul_154, convolution_44, getitem_116, getitem_117, cat_14, squeeze_61, add_109, convolution_47, squeeze_64, constant_pad_nd_8, constant_pad_nd_9, constant_pad_nd_10, cat_15, squeeze_67, add_119, mean_4, convolution_51, mul_179, convolution_52, mul_180, convolution_53, squeeze_70, getitem_138, getitem_139, cat_16, squeeze_73, getitem_146, getitem_151, getitem_156, getitem_161, cat_17, squeeze_76, add_134, mean_5, convolution_60, mul_204, convolution_61, getitem_164, getitem_165, cat_18, squeeze_79, getitem_168, getitem_169, cat_19, squeeze_82, getitem_176, getitem_181, getitem_186, getitem_191, cat_20, squeeze_85, add_150, mean_6, convolution_70, mul_229, convolution_71, getitem_194, getitem_195, cat_21, squeeze_88, getitem_198, getitem_199, cat_22, squeeze_91, getitem_206, getitem_211, getitem_216, getitem_221, cat_23, squeeze_94, add_166, mean_7, convolution_80, mul_254, convolution_81, getitem_224, getitem_225, cat_24, squeeze_97, add_172, convolution_84, squeeze_100, mul_270, convolution_85, squeeze_103, add_182, mean_8, convolution_86, mul_279, convolution_87, mul_280, convolution_88, squeeze_106, getitem_234, getitem_235, cat_25, squeeze_109, getitem_242, getitem_247, getitem_252, getitem_257, cat_26, squeeze_112, add_197, mean_9, convolution_95, mul_304, convolution_96, getitem_260, getitem_261, cat_27, squeeze_115, getitem_264, getitem_265, cat_28, squeeze_118, getitem_272, getitem_277, getitem_282, getitem_287, cat_29, squeeze_121, add_213, mean_10, convolution_105, mul_329, convolution_106, getitem_290, getitem_291, cat_30, squeeze_124, getitem_294, getitem_295, cat_31, squeeze_127, getitem_302, getitem_307, getitem_312, getitem_317, cat_32, squeeze_130, add_229, mean_11, convolution_115, mul_354, convolution_116, getitem_320, getitem_321, cat_33, squeeze_133, add_235, convolution_119, squeeze_136, constant_pad_nd_11, constant_pad_nd_12, constant_pad_nd_13, constant_pad_nd_14, cat_34, squeeze_139, add_245, mean_12, convolution_124, mul_379, convolution_125, mul_380, convolution_126, squeeze_142, add_250, convolution_127, squeeze_145, getitem_356, getitem_361, getitem_366, getitem_371, cat_35, squeeze_148, add_260, mean_13, convolution_132, mul_404, convolution_133, getitem_374, getitem_375, cat_36, squeeze_151, add_266, convolution_136, squeeze_154, getitem_384, getitem_389, getitem_394, getitem_399, cat_37, squeeze_157, add_276, mean_14, convolution_141, mul_429, convolution_142, getitem_402, getitem_403, cat_38, squeeze_160, add_282, convolution_145, squeeze_163, getitem_412, getitem_417, getitem_422, getitem_427, cat_39, squeeze_166, add_292, mean_15, convolution_150, mul_454, convolution_151, getitem_430, getitem_431, cat_40, squeeze_169, add_298, convolution_154, squeeze_172, view, permute_1, le, unsqueeze_234, unsqueeze_246, unsqueeze_258, mul_508, unsqueeze_270, unsqueeze_282, unsqueeze_294, mul_548, unsqueeze_306, unsqueeze_318, unsqueeze_330, mul_588, unsqueeze_342, unsqueeze_354, unsqueeze_366, mul_628, unsqueeze_378, unsqueeze_390, unsqueeze_402, mul_668, unsqueeze_414, unsqueeze_426, unsqueeze_438, mul_708, unsqueeze_450, unsqueeze_462, unsqueeze_474, mul_748, unsqueeze_486, unsqueeze_498, unsqueeze_510, mul_788, unsqueeze_522, unsqueeze_534, unsqueeze_546, mul_828, unsqueeze_558, unsqueeze_570, unsqueeze_582, mul_868, unsqueeze_594, unsqueeze_606, unsqueeze_618, mul_908, unsqueeze_630, unsqueeze_642, unsqueeze_654, mul_948, unsqueeze_666, unsqueeze_678, unsqueeze_690, mul_988, unsqueeze_702, unsqueeze_714, unsqueeze_726, mul_1028, unsqueeze_738, unsqueeze_750, unsqueeze_762, mul_1068, unsqueeze_774, unsqueeze_786, unsqueeze_798, mul_1108, unsqueeze_810, unsqueeze_822, le_1, unsqueeze_834, unsqueeze_846, unsqueeze_858, le_3, unsqueeze_870, le_4, unsqueeze_882, unsqueeze_894, unsqueeze_906, unsqueeze_918, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('tf_mixnet_l', benchmark_compiled_module)
