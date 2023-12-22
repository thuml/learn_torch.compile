
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


# kernel path: /tmp/torchinductor_youkaichao/qi/cqiy56muln7nqbu53cnuh43mrshcnfmpcxe7gkl2jsylyovlsrk6.py
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_native_batch_norm_backward_threshold_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5632
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1408
    x1 = (xindex // 1408)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1408*r2) + (137984*x1)), rmask & xmask, eviction_policy='evict_first').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (x0 + (1408*(r2 // 49)) + (2816*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr2 + (x0 + (1408*r2) + (137984*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr4 + (x0 + (1408*r2) + (137984*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = 49.0
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
        tmp18 = tmp16 - tmp17
        tmp19 = tmp5 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask & xmask, tmp22, _tmp21)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp14, xmask)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/be/cbemwmibyvzmxxdb3dxiguqqwqcztrgnarmljavro6kqbsvg6nea.py
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
    xnumel = 1408
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1408*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ic/cic5stfqjyud232mqhzselk4mgvp3ydctq4c4bz25siq5lmkkofq.py
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
    xnumel = 1408
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1408*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6o/c6ot7dnbubi5x3c2xebk5qjrezaymyazfpffg7kmftvpjpazg2u6.py
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(15,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 551936
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1408
    x2 = (xindex // 68992)
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + (1408*x2)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x3), xmask)
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr8 + (x3), xmask)
    tmp24 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr10 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr11 + (x0), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr12 + (x0), xmask, eviction_policy='evict_last')
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
    tmp25 = tmp23 - tmp24
    tmp27 = tmp26 * tmp10
    tmp29 = tmp28 * tmp28
    tmp30 = tmp27 * tmp29
    tmp31 = tmp25 * tmp30
    tmp32 = tmp5 - tmp31
    tmp33 = tmp32 - tmp18
    tmp35 = tmp28 * tmp34
    tmp36 = tmp33 * tmp35
    tl.store(out_ptr0 + (x3), tmp22, xmask)
    tl.store(out_ptr1 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3t/c3tvvus77umjcwg25ztito56veu23klzfmdhkgbdomyizywo6jo2.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6w/c6wvkcypfzdsljkupyfjq3gh64astdvutprdkfufnpl7a35rhdbf.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp35 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (384*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((196*x1) + (75264*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr2 + ((196*x1) + (75264*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = tl.where(tmp5, tmp4, tmp8)
        tmp10 = tl.load(in_ptr3 + (x1 + (384*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr4 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp9 * tmp12
        tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
        tmp15 = tl.where(tmp2, tmp13, tmp14)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
        tmp19 = tl.load(in_ptr5 + (x1 + (384*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.load(in_ptr6 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tmp19 - tmp20
        tmp22 = tmp9 * tmp21
        tmp23 = tl.full(tmp22.shape, 0, tmp22.dtype)
        tmp24 = tl.where(tmp2, tmp22, tmp23)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
        tmp28 = tl.load(in_ptr7 + (x1 + (384*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp29 = tl.load(in_ptr8 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp30 = tmp28 - tmp29
        tmp31 = tmp9 * tmp30
        tmp32 = tl.full(tmp31.shape, 0, tmp31.dtype)
        tmp33 = tl.where(tmp2, tmp31, tmp32)
        tmp34 = tl.broadcast_to(tmp33, [XBLOCK, RBLOCK])
        tmp36 = _tmp35 + tmp34
        _tmp35 = tl.where(rmask & xmask, tmp36, _tmp35)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp26, xmask)
    tmp35 = tl.sum(_tmp35, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp35, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6h/c6h4kvpbaatwr5654qxosciyoopbfw542ctidwuzdev4a22cz7xr.py
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
    size_hints=[512, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
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


# kernel path: /tmp/torchinductor_youkaichao/5h/c5hdcgk6pwzu53efxcoy2dau7jsgxtpbablzlhbkvbglwbc56q4q.py
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
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32', 17: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16, 17))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr12 + (x2), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr13 + (x2), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp23, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (384*y3)), tmp37, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/36/c362cs3jirc6cphybg3btrxk5suq4bnpydtnju6vmnzq6g2ru2bd.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hc/chctmbzkgoewkeyfmgqe3o67caa7tux22iitagkgzdit3cjysg7m.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: 'i32', 21: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(20, 21))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp13 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp20 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp27 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    _tmp31 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr3 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr4 + (x0 + (384*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr6 + (x0 + (384*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp26 = tl.load(in_ptr8 + (x0 + (384*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tl.where(tmp2, tmp1, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp8 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
        tmp21 = tmp19 - tmp20
        tmp22 = tmp8 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
        tmp28 = tmp26 - tmp27
        tmp29 = tmp8 * tmp28
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp32 = _tmp31 + tmp30
        _tmp31 = tl.where(rmask & xmask, tmp32, _tmp31)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp17, xmask)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp24, xmask)
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    tl.store(out_ptr3 + (x0), tmp31, xmask)
    tmp33 = tl.load(in_ptr10 + (x0), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr11 + (x0), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr12 + (x0), xmask, eviction_policy='evict_last')
    tmp34 = tmp17 * tmp33
    tmp36 = tmp24 * tmp35
    tmp38 = tmp31 * tmp37
    tl.store(out_ptr4 + (x0), tmp34, xmask)
    tl.store(out_ptr5 + (x0), tmp36, xmask)
    tl.store(out_ptr6 + (x0), tmp38, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ha/chan5ergv6prdnh6isk4nht6gx4wx6varmrzrsus73hy2yuu4iql.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: 'i32', 23: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(22,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, out_ptr2, out_ptr3, out_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr9 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr10 + (y0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr11 + (y0), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr12 + (y0), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr13 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr14 + (y0), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr15 + (y0), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr16 + (y0), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr17 + (y0), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr18 + (y0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp2, tmp1, tmp7)
    tmp11 = tmp9 - tmp10
    tmp13 = 0.0006377551020408163
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
    tmp36 = tmp34 - tmp35
    tmp38 = tmp37 * tmp13
    tmp40 = tmp39 * tmp39
    tmp41 = tmp38 * tmp40
    tmp42 = tmp36 * tmp41
    tmp43 = tmp8 - tmp42
    tmp44 = tmp43 - tmp21
    tmp46 = tmp15 * tmp45
    tmp47 = tmp22 * tmp46
    tmp49 = tmp28 * tmp48
    tmp50 = tmp33 * tmp49
    tl.store(out_ptr2 + (x2 + (196*y3)), tmp44, xmask)
    tl.store(out_ptr3 + (y0 + (384*x2) + (75264*y1)), tmp47, xmask)
    tl.store(out_ptr4 + (y0 + (384*x2) + (75264*y1)), tmp50, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2s/c2swz4ulfsb7ibupmq3eduquoutw5ao6e3le2denyr6klm7w7cv5.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
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
    tmp0 = tl.load(in_ptr0 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 * tmp9
    tmp11 = tmp5 + tmp10
    tmp12 = tl.where(tmp2, tmp1, tmp11)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ea/ceapydplrawtrsqlcf5yfwz7hnuwfccp6bmldjlgh5iiytpaigqi.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/da/cda2inqpcxeurml6t7b6qktjuiwdwgwyyk7y5z4mthdsiuakrq4j.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp29 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (75264*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (384*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp3 * tmp6
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
        tmp13 = tl.load(in_ptr3 + (x1 + (384*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr4 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 - tmp14
        tmp16 = tmp3 * tmp15
        tmp17 = tl.full(tmp16.shape, 0, tmp16.dtype)
        tmp18 = tl.where(tmp2, tmp16, tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
        tmp22 = tl.load(in_ptr5 + (x1 + (384*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr6 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tmp22 - tmp23
        tmp25 = tmp3 * tmp24
        tmp26 = tl.full(tmp25.shape, 0, tmp25.dtype)
        tmp27 = tl.where(tmp2, tmp25, tmp26)
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
        tmp30 = _tmp29 + tmp28
        _tmp29 = tl.where(rmask & xmask, tmp30, _tmp29)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp20, xmask)
    tmp29 = tl.sum(_tmp29, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6l/c6lu3fws5wxkobdcnbkt37i3gr2bf5cmihziyhbn77cpsael2pmb.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14, 15))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 384
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (384*y3)), tmp31, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ht/chthadzft63rjaerwrcu7733pqgwwe4xu4r72dqzdur6xwwo464y.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
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
    tmp21 = tmp2 + tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v5/cv5sdldjip4wmm4pzv27zrkhq5fwpseq3ipee27ei2fykrho2moa.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
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
        tmp3 = tl.load(in_ptr0 + (x1 + (384*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((196*x1) + (75264*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rh/crh5c4orvjbhsqa5yiycgs25orctytxhl6fxfeu6q4yeijhbyvtp.py
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
    size_hints=[512, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
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


# kernel path: /tmp/torchinductor_youkaichao/4h/c4hmmwmevkp4u2qgp34y7cm5macermdcqcznif5gfeqqxta4ghze.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp33 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((196*x0) + (75264*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.load(in_ptr2 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp7 * tmp10
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
        tmp17 = tl.load(in_ptr4 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tmp17 - tmp18
        tmp20 = tmp7 * tmp19
        tmp21 = tl.full(tmp20.shape, 0, tmp20.dtype)
        tmp22 = tl.where(tmp2, tmp20, tmp21)
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
        tmp26 = tl.load(in_ptr6 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp27 = tl.load(in_ptr7 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tmp26 - tmp27
        tmp29 = tmp7 * tmp28
        tmp30 = tl.full(tmp29.shape, 0, tmp29.dtype)
        tmp31 = tl.where(tmp2, tmp29, tmp30)
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
        tmp34 = _tmp33 + tmp32
        _tmp33 = tl.where(rmask & xmask, tmp34, _tmp33)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp24, xmask)
    tmp33 = tl.sum(_tmp33, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp33, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/df/cdffrlgjljgb6c4qv4civqj2s7tvvakmhezkmg2w4bujxvoltdjv.py
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
    size_hints=[512, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 13
    RBLOCK: tl.constexpr = 16
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


# kernel path: /tmp/torchinductor_youkaichao/4l/c4lc46goazzyo3ocuhgqd3jmszg4yeftt4m2ecnqzy2sdpvnk5if.py
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
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: 'i32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(15, 16))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr12 + (x2), xmask, eviction_policy='evict_last')
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
    tmp24 = tmp22 - tmp23
    tmp26 = tmp25 * tmp9
    tmp28 = tmp27 * tmp27
    tmp29 = tmp26 * tmp28
    tmp30 = tmp24 * tmp29
    tmp31 = tmp4 - tmp30
    tmp32 = tmp31 - tmp17
    tmp34 = tmp27 * tmp33
    tmp35 = tmp32 * tmp34
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp21, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (384*y3)), tmp35, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k2/ck2d46rnlu4i4msf6wt5sg4sisifwgpeakz5vccpmuoxeneemavj.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tmp7 = tl.where(tmp5, tmp4, tmp6)
    tmp10 = tmp8 - tmp9
    tmp12 = 0.0006377551020408163
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp10 * tmp16
    tmp18 = tmp7 - tmp17
    tmp20 = tmp19 * tmp12
    tmp21 = tmp18 - tmp20
    tmp23 = tmp14 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp2 + tmp24
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m4/cm45lgnebbcd5gbsgavhbjiwp6kyqdak26zopwisqqnunietwmhp.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_23', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (y0 + (384*x2) + (75264*y1)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tmp7 = tl.where(tmp5, tmp4, tmp6)
    tmp10 = tmp8 - tmp9
    tmp12 = 0.0006377551020408163
    tmp13 = tmp11 * tmp12
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 * tmp15
    tmp17 = tmp10 * tmp16
    tmp18 = tmp7 - tmp17
    tmp20 = tmp19 * tmp12
    tmp21 = tmp18 - tmp20
    tmp23 = tmp14 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp2 + tmp24
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qx/cqxnosgznvpkcnckmaqtnoqdo3p26e47l3tdvnx7ef756vn2zkug.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((196*x0) + (75264*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.load(in_ptr2 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp7 * tmp10
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
        tmp17 = tl.load(in_ptr4 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tmp17 - tmp18
        tmp20 = tmp7 * tmp19
        tmp21 = tl.full(tmp20.shape, 0, tmp20.dtype)
        tmp22 = tl.where(tmp2, tmp20, tmp21)
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qc/cqcou7ajpemrv4whmcj62ae3zbmyiecpfxvzapxc3jo3utfdtul5.py
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
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (784*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/72/c72ea6yxklkqb22x2inrlcykk3wzbebs5vzm26wpdv3vpqsfu7i5.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    tmp8 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp15 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp22 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (192*r2) + (24576*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((784*x1) + (150528*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((784*x1) + (150528*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x1 + (192*r2) + (24576*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr5 + (x1 + (192*r2) + (24576*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr7 + (x1 + (192*r2) + (24576*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp6 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
        tmp23 = tmp21 - tmp22
        tmp24 = tmp6 * tmp23
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp19, xmask)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fs/cfsq4q5wac26ebdcw733mq7fghq4dkqwkyjx2vgyyiwmon4nkyaz.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
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


# kernel path: /tmp/torchinductor_youkaichao/n4/cn4sdftqcs4vpzjgddo5s7qxmcsbihull27uld4vkyddvr4fr7ck.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32', 17: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16, 17))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (784*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (784*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr12 + (x2), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr13 + (x2), xmask, eviction_policy='evict_last')
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
    tmp26 = tmp24 - tmp25
    tmp28 = tmp27 * tmp11
    tmp30 = tmp29 * tmp29
    tmp31 = tmp28 * tmp30
    tmp32 = tmp26 * tmp31
    tmp33 = tmp6 - tmp32
    tmp34 = tmp33 - tmp19
    tmp36 = tmp29 * tmp35
    tmp37 = tmp34 * tmp36
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp23, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (192*y3)), tmp37, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pc/cpcz6qn2oyqmyxttsxeutynqqs4fxbwnxrltke2b2tjrngxc54gc.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_29', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (y0 + (192*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (y0 + (192*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (784*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/62/c62notnlqoklhuqaew6og4ejw3vzmdgbvqw34aix674selxnpjcc.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: 'i32', 21: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(20, 21))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp13 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp20 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp27 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    _tmp31 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r1 + (784*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (784*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr3 + (r1 + (784*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr4 + (x0 + (192*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr6 + (x0 + (192*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp26 = tl.load(in_ptr8 + (x0 + (192*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tl.where(tmp2, tmp1, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp8 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
        tmp21 = tmp19 - tmp20
        tmp22 = tmp8 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
        tmp28 = tmp26 - tmp27
        tmp29 = tmp8 * tmp28
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp32 = _tmp31 + tmp30
        _tmp31 = tl.where(rmask & xmask, tmp32, _tmp31)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp17, xmask)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp24, xmask)
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    tl.store(out_ptr3 + (x0), tmp31, xmask)
    tmp33 = tl.load(in_ptr10 + (x0), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr11 + (x0), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr12 + (x0), xmask, eviction_policy='evict_last')
    tmp34 = tmp17 * tmp33
    tmp36 = tmp24 * tmp35
    tmp38 = tmp31 * tmp37
    tl.store(out_ptr4 + (x0), tmp34, xmask)
    tl.store(out_ptr5 + (x0), tmp36, xmask)
    tl.store(out_ptr6 + (x0), tmp38, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/my/cmyaiy77b3hqiouiciswxukfaye7ahkfik2n5r3ywc2ojqeaomgr.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: 'i32', 23: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(22, 23))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, out_ptr2, out_ptr3, out_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (y0 + (192*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0 + (192*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr9 + (y0 + (192*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr10 + (y0), ymask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr11 + (y0), ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr12 + (y0), ymask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr13 + (y0 + (192*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr14 + (y0), ymask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr15 + (y0), ymask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr16 + (y0), ymask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr17 + (y0), ymask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr18 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp2, tmp1, tmp7)
    tmp11 = tmp9 - tmp10
    tmp13 = 0.00015943877551020407
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
    tmp36 = tmp34 - tmp35
    tmp38 = tmp37 * tmp13
    tmp40 = tmp39 * tmp39
    tmp41 = tmp38 * tmp40
    tmp42 = tmp36 * tmp41
    tmp43 = tmp8 - tmp42
    tmp44 = tmp43 - tmp21
    tmp46 = tmp15 * tmp45
    tmp47 = tmp22 * tmp46
    tmp49 = tmp28 * tmp48
    tmp50 = tmp33 * tmp49
    tl.store(out_ptr2 + (x2 + (784*y3)), tmp44, xmask & ymask)
    tl.store(out_ptr3 + (y0 + (192*x2) + (150528*y1)), tmp47, xmask & ymask)
    tl.store(out_ptr4 + (y0 + (192*x2) + (150528*y1)), tmp50, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b6/cb6kgylvxqznklbgwtunrqanfokltsp3l7cifzxogrqhq6nq5wdl.py
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
    size_hints=[2048, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 784
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
    tmp0 = tl.load(in_ptr0 + (y0 + (192*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 * tmp9
    tmp11 = tmp5 + tmp10
    tmp12 = tl.where(tmp2, tmp1, tmp11)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (784*y3)), tmp12, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ju/cjuacb2g6ipv32mavbf5a5gudt4trfrgspa6o64esm5xvan7ae2c.py
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
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uo/cuotoo4xnq2tvbptqna3gokhzvmzi4mjapkvt6oasdkwbbmrf6yl.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
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
    tmp9 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp16 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((784*x1) + (150528*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (192*r2) + (24576*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x1 + (192*r2) + (24576*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr5 + (x1 + (192*r2) + (24576*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp0 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
        tmp17 = tmp15 - tmp16
        tmp18 = tmp0 * tmp17
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, xmask)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/up/cupikpjdjxx3u6ageij5qq67hy7xgbevu2agta4crqhqjhoqpzjm.py
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
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14, 15))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 192
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp17, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (192*y3)), tmp31, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rz/crzk7xd2hg7zvumikc4lomqgocdwphnyebnomtok77jxa5bvwvc2.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_36', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (192*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
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
    tmp21 = tmp2 + tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (784*y3)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7x/c7xi7je2tplhmvyg5jzus24fmrfbbbuyf4i3zemleaz7jm6h6xip.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 49
    x1 = (xindex // 49)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (192*r2) + (24576*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((784*x1) + (150528*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r2/cr2sdtlw2j47kghojjunhqnzppzdsj3kuhfrazvbs2gt4qqe2ff2.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
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


# kernel path: /tmp/torchinductor_youkaichao/lp/clpzczmlkstzop7wpgl2qmac5vfrmb6t7buamyfgjkugalhezjvg.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 192
    x1 = (xindex // 192)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp13 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((784*x0) + (150528*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr4 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp4 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i2/ci25ev3gsrm4aoaz3nr6uhbp6pliw3g74e7qrjn46kfifk52bysg.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 49
    RBLOCK: tl.constexpr = 64
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


# kernel path: /tmp/torchinductor_youkaichao/t7/ct7xa7ixp4tgg35zxotalmpmk7s4sudichj4klhixkxz4obhfaqt.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: 'i32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(15, 16))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (784*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr12 + (x2), xmask, eviction_policy='evict_last')
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
    tmp24 = tmp22 - tmp23
    tmp26 = tmp25 * tmp9
    tmp28 = tmp27 * tmp27
    tmp29 = tmp26 * tmp28
    tmp30 = tmp24 * tmp29
    tmp31 = tmp4 - tmp30
    tmp32 = tmp31 - tmp17
    tmp34 = tmp27 * tmp33
    tmp35 = tmp32 * tmp34
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp21, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (192*y3)), tmp35, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/26/c26cn33wbsic2nkjjjxahfnrvkmzqknjqaghypzrwoe44piiw4ja.py
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
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 96
    x1 = (xindex // 96)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (96*r2) + (602112*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (301056*(r2 // 3136)) + (602112*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((3136*x0) + (301056*(r2 // 3136)) + (602112*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kz/ckzz7cqubopiriusfybit7mim6vcaecawsraj2vtreqiecinsnzp.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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


# kernel path: /tmp/torchinductor_youkaichao/br/cbrm64bgbneeclpuzoalo6uuddisy5fyt3curai35uzmamakq5c6.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18816
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    tmp8 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp15 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp22 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (96*r2) + (12288*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x1) + (301056*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((3136*x1) + (301056*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x1 + (96*r2) + (12288*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr5 + (x1 + (96*r2) + (12288*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr7 + (x1 + (96*r2) + (12288*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp6 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
        tmp23 = tmp21 - tmp22
        tmp24 = tmp6 * tmp23
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp19, xmask)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qr/cqr7jq7fdvfitdzwgizlyhk4sg6olxwmjkfz2tqp5r2f2ix2z444.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
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


# kernel path: /tmp/torchinductor_youkaichao/5m/c5mx4d6mlyqkl7sm7en6frsdriybtlxc4rs3m2ddy2ct6vmmjzag.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32', 17: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16, 17))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 96
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
    tmp0 = tl.load(in_ptr0 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (3136*x2) + (301056*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (3136*x2) + (301056*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr12 + (x2), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr13 + (x2), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (96*y3)), tmp23, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (96*y3)), tmp37, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a7/ca7d73d5pnyv2fzn4mwcmzdtrjw6lzsz3buygtc5oig34fhd7bj6.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_47', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 3136
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
    tmp0 = tl.load(in_ptr0 + (y0 + (96*x2) + (301056*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (y0 + (96*x2) + (301056*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (3136*y3)), tmp23, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/en/cen7xu3phdfkyvfa3ubduzunay5f42wlwajh2r5yo7vdj2lcuyy3.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_48', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 96
    x1 = (xindex // 96)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp13 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp20 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (96*r2) + (602112*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (301056*(r2 // 3136)) + (602112*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((3136*x0) + (301056*(r2 // 3136)) + (602112*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr3 + ((3136*x0) + (301056*(r2 // 3136)) + (602112*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr4 + (x0 + (96*r2) + (602112*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr6 + (x0 + (96*r2) + (602112*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tl.where(tmp2, tmp1, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp8 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
        tmp21 = tmp19 - tmp20
        tmp22 = tmp8 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp17, xmask)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fd/cfd6pmufwhwa5enhw2hu4oi7cvuhieykbcbbhjgn3hmrnjizgpua.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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


# kernel path: /tmp/torchinductor_youkaichao/5o/c5o6s35jiomsi4lfl5j3uomw2saqprrlxd6mv4gj7bmimlrzfia5.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: 'i32', 18: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(17, 18))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 3136
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
    tmp0 = tl.load(in_ptr0 + (y0 + (96*x2) + (301056*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0 + (96*x2) + (301056*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr9 + (y0 + (96*x2) + (301056*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr10 + (y0), ymask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr11 + (y0), ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr12 + (y0), ymask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr13 + (y0), ymask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr14 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tl.where(tmp2, tmp1, tmp7)
    tmp11 = tmp9 - tmp10
    tmp13 = 3.985969387755102e-05
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
    tl.store(out_ptr2 + (y0 + (96*x2) + (301056*y1)), tmp36, xmask & ymask)
    tl.store(out_ptr3 + (y0 + (96*x2) + (301056*y1)), tmp39, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bc/cbc35qp53hdcklbzeovez5mffw7jw575gvodcaaogq34qg357r7a.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_native_batch_norm_backward_threshold_backward_51 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 832
    rnumel = 7720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7720*x0)
        tmp1 = tl.full([1, 1], 100352, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (64*((r2 + (7720*x0)) % 100352))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((12544*x1) + (802816*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr2 + ((12544*x1) + (802816*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = tl.where(tmp5, tmp4, tmp8)
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g3/cg3zsaiyflwjgppdgmn2orm2zyhoidsk5mnrdei5f4c36iwxrboi.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_52', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/wv/cwvy6pyovcqgwzehy555ulor5w26tbv6igsuaa2rrsxvshgcnwbp.py
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
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_threshold_backward_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    tmp8 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp15 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((12544*x1) + (802816*((r2 + (128*x0)) // 12544)) + ((r2 + (128*x0)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + ((12544*x1) + (802816*((r2 + (128*x0)) // 12544)) + ((r2 + (128*x0)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x1 + (64*r2) + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr5 + (x1 + (64*r2) + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tl.where(tmp2, tmp1, tmp5)
        tmp9 = tmp7 - tmp8
        tmp10 = tmp6 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp6 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rb/crbb2hpnpojhfuaj4cylki4qjwud2vcvsbwsr3cpjgvru3tiwr5m.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_native_batch_norm_backward_threshold_backward_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_threshold_backward_54', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 64
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7z/c7z2rp72dsjuseqhgl4cil53d22dseeuj5rfp6fis5jg75v7k6dp.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_native_batch_norm_backward_threshold_backward_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32', 17: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16, 17))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_threshold_backward_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 100352
    xnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (12544*x2) + (802816*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0 + (12544*x2) + (802816*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr12 + (x2), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr13 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tmp9 = tmp7 - tmp8
    tmp11 = 9.964923469387754e-06
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
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp23, xmask)
    tl.store(out_ptr1 + (x2 + (64*y3)), tmp37, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_352, convolution, squeeze_1, convolution_1, squeeze_4, relu, convolution_2, squeeze_7, convolution_3, squeeze_10, relu_1, squeeze_13, convolution_4, squeeze_16, convolution_5, squeeze_19, relu_2, convolution_6, squeeze_22, convolution_7, squeeze_25, relu_3, squeeze_28, convolution_8, squeeze_31, convolution_9, squeeze_34, relu_4, squeeze_37, convolution_10, squeeze_40, convolution_11, squeeze_43, relu_5, squeeze_46, convolution_12, squeeze_49, convolution_13, squeeze_52, relu_6, convolution_14, squeeze_55, convolution_15, squeeze_58, relu_7, squeeze_61, convolution_16, squeeze_64, convolution_17, squeeze_67, relu_8, squeeze_70, convolution_18, squeeze_73, convolution_19, squeeze_76, relu_9, squeeze_79, convolution_20, squeeze_82, convolution_21, squeeze_85, relu_10, squeeze_88, convolution_22, squeeze_91, convolution_23, squeeze_94, relu_11, squeeze_97, convolution_24, squeeze_100, convolution_25, squeeze_103, relu_12, squeeze_106, convolution_26, squeeze_109, convolution_27, squeeze_112, relu_13, squeeze_115, convolution_28, squeeze_118, convolution_29, squeeze_121, relu_14, squeeze_124, convolution_30, squeeze_127, convolution_31, squeeze_130, relu_15, squeeze_133, convolution_32, squeeze_136, convolution_33, squeeze_139, relu_16, squeeze_142, convolution_34, squeeze_145, convolution_35, squeeze_148, relu_17, squeeze_151, convolution_36, squeeze_154, convolution_37, squeeze_157, relu_18, squeeze_160, convolution_38, squeeze_163, convolution_39, squeeze_166, relu_19, squeeze_169, convolution_40, squeeze_172, convolution_41, squeeze_175, relu_20, convolution_42, squeeze_178, convolution_43, squeeze_181, clone, permute_1, le, unsqueeze_246, unsqueeze_258, unsqueeze_270, unsqueeze_282, unsqueeze_294, unsqueeze_306, unsqueeze_318, unsqueeze_330, unsqueeze_342, unsqueeze_354, unsqueeze_366, unsqueeze_378, unsqueeze_390, unsqueeze_402, unsqueeze_414, unsqueeze_426, unsqueeze_438, unsqueeze_450, unsqueeze_462, unsqueeze_474, unsqueeze_486, unsqueeze_498, unsqueeze_510, unsqueeze_522, unsqueeze_534, unsqueeze_546, unsqueeze_558, unsqueeze_570, unsqueeze_582, unsqueeze_594, unsqueeze_606, unsqueeze_618, unsqueeze_630, unsqueeze_642, unsqueeze_654, unsqueeze_666, unsqueeze_678, unsqueeze_690, unsqueeze_702, unsqueeze_714, unsqueeze_726, unsqueeze_738, unsqueeze_750, unsqueeze_762, unsqueeze_774, unsqueeze_786, unsqueeze_798, unsqueeze_810, unsqueeze_822, unsqueeze_834, unsqueeze_846, unsqueeze_858, unsqueeze_870, unsqueeze_882, unsqueeze_894, unsqueeze_906, unsqueeze_918, unsqueeze_930, unsqueeze_942, unsqueeze_954, unsqueeze_966, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (64, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_5, (96, ), (1, ))
    assert_size_stride(primals_7, (96, ), (1, ))
    assert_size_stride(primals_9, (96, ), (1, ))
    assert_size_stride(primals_11, (96, ), (1, ))
    assert_size_stride(primals_13, (96, ), (1, ))
    assert_size_stride(primals_15, (192, ), (1, ))
    assert_size_stride(primals_17, (192, ), (1, ))
    assert_size_stride(primals_19, (192, ), (1, ))
    assert_size_stride(primals_21, (192, ), (1, ))
    assert_size_stride(primals_23, (192, ), (1, ))
    assert_size_stride(primals_25, (192, ), (1, ))
    assert_size_stride(primals_27, (192, ), (1, ))
    assert_size_stride(primals_29, (192, ), (1, ))
    assert_size_stride(primals_31, (192, ), (1, ))
    assert_size_stride(primals_33, (192, ), (1, ))
    assert_size_stride(primals_35, (192, ), (1, ))
    assert_size_stride(primals_37, (384, ), (1, ))
    assert_size_stride(primals_39, (384, ), (1, ))
    assert_size_stride(primals_41, (384, ), (1, ))
    assert_size_stride(primals_43, (384, ), (1, ))
    assert_size_stride(primals_45, (384, ), (1, ))
    assert_size_stride(primals_47, (384, ), (1, ))
    assert_size_stride(primals_49, (384, ), (1, ))
    assert_size_stride(primals_51, (384, ), (1, ))
    assert_size_stride(primals_53, (384, ), (1, ))
    assert_size_stride(primals_55, (384, ), (1, ))
    assert_size_stride(primals_57, (384, ), (1, ))
    assert_size_stride(primals_59, (384, ), (1, ))
    assert_size_stride(primals_61, (384, ), (1, ))
    assert_size_stride(primals_63, (384, ), (1, ))
    assert_size_stride(primals_65, (384, ), (1, ))
    assert_size_stride(primals_67, (384, ), (1, ))
    assert_size_stride(primals_69, (384, ), (1, ))
    assert_size_stride(primals_71, (384, ), (1, ))
    assert_size_stride(primals_73, (384, ), (1, ))
    assert_size_stride(primals_75, (384, ), (1, ))
    assert_size_stride(primals_77, (384, ), (1, ))
    assert_size_stride(primals_79, (384, ), (1, ))
    assert_size_stride(primals_81, (384, ), (1, ))
    assert_size_stride(primals_83, (384, ), (1, ))
    assert_size_stride(primals_85, (384, ), (1, ))
    assert_size_stride(primals_87, (384, ), (1, ))
    assert_size_stride(primals_89, (384, ), (1, ))
    assert_size_stride(primals_91, (384, ), (1, ))
    assert_size_stride(primals_93, (384, ), (1, ))
    assert_size_stride(primals_95, (384, ), (1, ))
    assert_size_stride(primals_97, (384, ), (1, ))
    assert_size_stride(primals_99, (384, ), (1, ))
    assert_size_stride(primals_101, (384, ), (1, ))
    assert_size_stride(primals_103, (384, ), (1, ))
    assert_size_stride(primals_105, (384, ), (1, ))
    assert_size_stride(primals_107, (384, ), (1, ))
    assert_size_stride(primals_109, (384, ), (1, ))
    assert_size_stride(primals_111, (384, ), (1, ))
    assert_size_stride(primals_113, (384, ), (1, ))
    assert_size_stride(primals_115, (384, ), (1, ))
    assert_size_stride(primals_117, (384, ), (1, ))
    assert_size_stride(primals_119, (1408, ), (1, ))
    assert_size_stride(primals_121, (1408, ), (1, ))
    assert_size_stride(primals_123, (64, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(primals_124, (64, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_125, (96, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_126, (96, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_127, (96, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_128, (96, 96, 3, 3), (864, 1, 288, 96))
    assert_size_stride(primals_129, (192, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_130, (192, 96, 3, 3), (864, 1, 288, 96))
    assert_size_stride(primals_131, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_132, (192, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_133, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_134, (192, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_135, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_136, (192, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_137, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_138, (384, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_139, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_140, (384, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_141, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_142, (384, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_143, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_144, (384, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_145, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_146, (384, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_147, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_148, (384, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_149, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_150, (384, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_151, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_152, (384, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_153, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_154, (384, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_155, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_156, (384, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_157, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_158, (384, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_159, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_160, (384, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_161, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_162, (384, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_163, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_164, (384, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_165, (1408, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_166, (1408, 384, 3, 3), (3456, 1, 1152, 384))
    assert_size_stride(primals_352, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(squeeze_1, (64, ), (1, ))
    assert_size_stride(convolution_1, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(squeeze_4, (64, ), (1, ))
    assert_size_stride(relu, (8, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(convolution_2, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(squeeze_7, (96, ), (1, ))
    assert_size_stride(convolution_3, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(squeeze_10, (96, ), (1, ))
    assert_size_stride(relu_1, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(squeeze_13, (96, ), (1, ))
    assert_size_stride(convolution_4, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(squeeze_16, (96, ), (1, ))
    assert_size_stride(convolution_5, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(squeeze_19, (96, ), (1, ))
    assert_size_stride(relu_2, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(convolution_6, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_22, (192, ), (1, ))
    assert_size_stride(convolution_7, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_25, (192, ), (1, ))
    assert_size_stride(relu_3, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_28, (192, ), (1, ))
    assert_size_stride(convolution_8, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_31, (192, ), (1, ))
    assert_size_stride(convolution_9, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_34, (192, ), (1, ))
    assert_size_stride(relu_4, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_37, (192, ), (1, ))
    assert_size_stride(convolution_10, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_40, (192, ), (1, ))
    assert_size_stride(convolution_11, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_43, (192, ), (1, ))
    assert_size_stride(relu_5, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_46, (192, ), (1, ))
    assert_size_stride(convolution_12, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_49, (192, ), (1, ))
    assert_size_stride(convolution_13, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(squeeze_52, (192, ), (1, ))
    assert_size_stride(relu_6, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(convolution_14, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_55, (384, ), (1, ))
    assert_size_stride(convolution_15, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_58, (384, ), (1, ))
    assert_size_stride(relu_7, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_61, (384, ), (1, ))
    assert_size_stride(convolution_16, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_64, (384, ), (1, ))
    assert_size_stride(convolution_17, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_67, (384, ), (1, ))
    assert_size_stride(relu_8, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_70, (384, ), (1, ))
    assert_size_stride(convolution_18, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_73, (384, ), (1, ))
    assert_size_stride(convolution_19, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_76, (384, ), (1, ))
    assert_size_stride(relu_9, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_79, (384, ), (1, ))
    assert_size_stride(convolution_20, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_82, (384, ), (1, ))
    assert_size_stride(convolution_21, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_85, (384, ), (1, ))
    assert_size_stride(relu_10, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_88, (384, ), (1, ))
    assert_size_stride(convolution_22, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_91, (384, ), (1, ))
    assert_size_stride(convolution_23, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_94, (384, ), (1, ))
    assert_size_stride(relu_11, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_97, (384, ), (1, ))
    assert_size_stride(convolution_24, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_100, (384, ), (1, ))
    assert_size_stride(convolution_25, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_103, (384, ), (1, ))
    assert_size_stride(relu_12, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_106, (384, ), (1, ))
    assert_size_stride(convolution_26, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_109, (384, ), (1, ))
    assert_size_stride(convolution_27, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_112, (384, ), (1, ))
    assert_size_stride(relu_13, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_115, (384, ), (1, ))
    assert_size_stride(convolution_28, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_118, (384, ), (1, ))
    assert_size_stride(convolution_29, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_121, (384, ), (1, ))
    assert_size_stride(relu_14, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_124, (384, ), (1, ))
    assert_size_stride(convolution_30, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_127, (384, ), (1, ))
    assert_size_stride(convolution_31, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_130, (384, ), (1, ))
    assert_size_stride(relu_15, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_133, (384, ), (1, ))
    assert_size_stride(convolution_32, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_136, (384, ), (1, ))
    assert_size_stride(convolution_33, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_139, (384, ), (1, ))
    assert_size_stride(relu_16, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_142, (384, ), (1, ))
    assert_size_stride(convolution_34, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_145, (384, ), (1, ))
    assert_size_stride(convolution_35, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_148, (384, ), (1, ))
    assert_size_stride(relu_17, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_151, (384, ), (1, ))
    assert_size_stride(convolution_36, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_154, (384, ), (1, ))
    assert_size_stride(convolution_37, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_157, (384, ), (1, ))
    assert_size_stride(relu_18, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_160, (384, ), (1, ))
    assert_size_stride(convolution_38, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_163, (384, ), (1, ))
    assert_size_stride(convolution_39, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_166, (384, ), (1, ))
    assert_size_stride(relu_19, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_169, (384, ), (1, ))
    assert_size_stride(convolution_40, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_172, (384, ), (1, ))
    assert_size_stride(convolution_41, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(squeeze_175, (384, ), (1, ))
    assert_size_stride(relu_20, (8, 384, 14, 14), (75264, 1, 5376, 384))
    assert_size_stride(convolution_42, (8, 1408, 7, 7), (68992, 1, 9856, 1408))
    assert_size_stride(squeeze_178, (1408, ), (1, ))
    assert_size_stride(convolution_43, (8, 1408, 7, 7), (68992, 1, 9856, 1408))
    assert_size_stride(squeeze_181, (1408, ), (1, ))
    assert_size_stride(clone, (8, 1408), (1408, 1))
    assert_size_stride(permute_1, (1000, 1408), (1408, 1))
    assert_size_stride(le, (8, 1408, 7, 7), (68992, 1, 9856, 1408))
    assert_size_stride(unsqueeze_246, (1, 1408, 1, 1), (1408, 1, 1, 1))
    assert_size_stride(unsqueeze_258, (1, 1408, 1, 1), (1408, 1, 1, 1))
    assert_size_stride(unsqueeze_270, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_282, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_294, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_306, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_318, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_330, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_342, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_354, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_366, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_378, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_390, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_402, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_414, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_426, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_438, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_450, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_462, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_474, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_486, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_498, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_510, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_522, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_534, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_546, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_558, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_570, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_582, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_594, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_606, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_618, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_630, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_642, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_654, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_666, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_678, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_690, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_702, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_714, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_726, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_738, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_750, (1, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(unsqueeze_762, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_774, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_786, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_798, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_810, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_822, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_834, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_846, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_858, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_870, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_882, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_894, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_906, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_918, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_930, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_942, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_954, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_966, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 1408), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_1, out=buf0)
        del permute_1
        buf1 = empty((1000, 1408), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone, out=buf1)
        del clone
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf3 = empty_strided((1408, 4), (1, 1408), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((1408, 4), (1, 1408), device='cuda', dtype=torch.float32)
        buf12 = empty_strided((1408, 4), (1, 1408), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_div_native_batch_norm_backward_threshold_backward_1.run(le, buf0, convolution_43, unsqueeze_246, convolution_42, unsqueeze_258, buf3, buf5, buf12, 5632, 98, grid=grid(5632), stream=stream0)
        buf4 = empty((1408, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_2.run(buf3, buf4, 1408, 4, grid=grid(1408), stream=stream0)
        del buf3
        buf6 = empty((1408, ), device='cuda', dtype=torch.float32)
        buf7 = empty((1408, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_3.run(buf5, squeeze_181, buf6, buf7, 1408, 4, grid=grid(1408), stream=stream0)
        del buf5
        buf13 = empty((1408, ), device='cuda', dtype=torch.float32)
        buf14 = empty((1408, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_div_native_batch_norm_backward_threshold_backward_3.run(buf12, squeeze_178, buf13, buf14, 1408, 4, grid=grid(1408), stream=stream0)
        del buf12
        buf8 = empty_strided((8, 1408, 7, 7), (68992, 1, 9856, 1408), device='cuda', dtype=torch.float32)
        buf15 = empty_strided((8, 1408, 7, 7), (68992, 1, 9856, 1408), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_div_native_batch_norm_backward_threshold_backward_4.run(le, buf0, convolution_43, unsqueeze_246, buf6, squeeze_181, buf4, primals_121, convolution_42, unsqueeze_258, buf13, squeeze_178, primals_119, buf8, buf15, 551936, grid=grid(551936), stream=stream0)
        del buf0
        del buf13
        del buf6
        del convolution_42
        del convolution_43
        del le
        del primals_119
        del primals_121
        del squeeze_178
        del squeeze_181
        del unsqueeze_246
        del unsqueeze_258
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        buf9 = aten.convolution_backward(buf8, relu_20, primals_166, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf8
        del primals_166
        buf10 = buf9[0]
        buf11 = buf9[1]
        del buf9
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
        buf16 = aten.convolution_backward(buf15, relu_20, primals_165, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf15
        del primals_165
        buf17 = buf16[0]
        buf18 = buf16[1]
        del buf16
        buf19 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_5.run(relu_20, buf10, buf17, buf19, 384, 1568, grid=grid(384), stream=stream0)
        buf20 = empty((384, 13), device='cuda', dtype=torch.float32)
        buf27 = empty((384, 13), device='cuda', dtype=torch.float32)
        buf34 = empty((384, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_6.run(relu_20, buf10, buf17, convolution_41, unsqueeze_270, convolution_40, unsqueeze_282, relu_19, unsqueeze_294, buf20, buf27, buf34, 4992, 121, grid=grid(4992), stream=stream0)
        buf21 = empty((384, ), device='cuda', dtype=torch.float32)
        buf23 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_7.run(buf20, squeeze_175, buf21, buf23, 384, 13, grid=grid(384), stream=stream0)
        buf28 = empty((384, ), device='cuda', dtype=torch.float32)
        buf30 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_7.run(buf27, squeeze_172, buf28, buf30, 384, 13, grid=grid(384), stream=stream0)
        buf22 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        buf29 = empty_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_8.run(relu_20, buf10, buf17, convolution_41, unsqueeze_270, buf21, squeeze_175, buf19, primals_117, convolution_40, unsqueeze_282, buf28, squeeze_172, primals_115, buf22, buf29, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del convolution_40
        del convolution_41
        del primals_115
        del primals_117
        del squeeze_172
        del squeeze_175
        del unsqueeze_270
        del unsqueeze_282
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf24 = aten.convolution_backward(buf22, relu_19, primals_164, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_164
        buf25 = buf24[0]
        buf26 = buf24[1]
        del buf24
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf31 = aten.convolution_backward(buf29, relu_19, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_163
        buf32 = buf31[0]
        buf33 = buf31[1]
        del buf31
        buf35 = buf28; del buf28  # reuse
        buf37 = buf21; del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_7.run(buf34, squeeze_169, buf35, buf37, 384, 13, grid=grid(384), stream=stream0)
        buf36 = buf10; del buf10  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_9.run(buf36, relu_20, buf17, relu_19, unsqueeze_294, buf35, squeeze_169, buf19, primals_113, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_113
        del relu_20
        del squeeze_169
        del unsqueeze_294
        buf38 = buf35; del buf35  # reuse
        buf39 = empty((384, ), device='cuda', dtype=torch.float32)
        buf46 = empty((384, ), device='cuda', dtype=torch.float32)
        buf53 = empty((384, ), device='cuda', dtype=torch.float32)
        buf41 = empty((384, ), device='cuda', dtype=torch.float32)
        buf48 = empty((384, ), device='cuda', dtype=torch.float32)
        buf55 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_10.run(relu_19, buf25, buf32, buf36, convolution_39, unsqueeze_306, convolution_38, unsqueeze_318, relu_18, unsqueeze_330, squeeze_166, squeeze_163, squeeze_160, buf38, buf39, buf46, buf53, buf41, buf48, buf55, 384, 1568, grid=grid(384), stream=stream0)
        buf54 = buf17; del buf17  # reuse
        buf42 = buf29; del buf29  # reuse
        buf49 = buf22; del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_11.run(relu_19, buf25, buf32, buf36, convolution_39, unsqueeze_306, buf39, squeeze_166, buf38, convolution_38, unsqueeze_318, buf46, squeeze_163, relu_18, unsqueeze_330, buf53, squeeze_160, primals_111, primals_109, buf54, buf42, buf49, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del buf25
        del buf32
        del buf36
        del convolution_38
        del convolution_39
        del primals_109
        del primals_111
        del relu_19
        del squeeze_163
        del squeeze_166
        del unsqueeze_306
        del unsqueeze_318
        del unsqueeze_330
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf43 = aten.convolution_backward(buf42, relu_18, primals_162, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf42
        del primals_162
        buf44 = buf43[0]
        buf45 = buf43[1]
        del buf43
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf50 = aten.convolution_backward(buf49, relu_18, primals_161, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf49
        del primals_161
        buf51 = buf50[0]
        buf52 = buf50[1]
        del buf50
        buf56 = buf44; del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_12.run(buf56, relu_18, buf51, buf54, squeeze_160, primals_107, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_107
        del relu_18
        del squeeze_160
        buf57 = buf53; del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_13.run(buf56, buf57, 384, 1568, grid=grid(384), stream=stream0)
        buf58 = buf34; del buf34  # reuse
        buf65 = buf27; del buf27  # reuse
        buf72 = buf20; del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_14.run(buf56, convolution_37, unsqueeze_342, convolution_36, unsqueeze_354, relu_17, unsqueeze_366, buf58, buf65, buf72, 4992, 121, grid=grid(4992), stream=stream0)
        buf59 = buf46; del buf46  # reuse
        buf60 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_7.run(buf58, squeeze_157, buf59, buf60, 384, 13, grid=grid(384), stream=stream0)
        buf66 = empty((384, ), device='cuda', dtype=torch.float32)
        buf67 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_7.run(buf65, squeeze_154, buf66, buf67, 384, 13, grid=grid(384), stream=stream0)
        buf61 = reinterpret_tensor(buf54, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf54  # reuse
        buf68 = reinterpret_tensor(buf51, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf51  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_15.run(buf56, convolution_37, unsqueeze_342, buf59, squeeze_157, buf57, primals_105, convolution_36, unsqueeze_354, buf66, squeeze_154, primals_103, buf61, buf68, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del convolution_36
        del convolution_37
        del primals_103
        del primals_105
        del squeeze_154
        del squeeze_157
        del unsqueeze_342
        del unsqueeze_354
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf62 = aten.convolution_backward(buf61, relu_17, primals_160, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf61
        del primals_160
        buf63 = buf62[0]
        buf64 = buf62[1]
        del buf62
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf69 = aten.convolution_backward(buf68, relu_17, primals_159, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf68
        del primals_159
        buf70 = buf69[0]
        buf71 = buf69[1]
        del buf69
        buf73 = buf66; del buf66  # reuse
        buf74 = buf59; del buf59  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_7.run(buf72, squeeze_151, buf73, buf74, 384, 13, grid=grid(384), stream=stream0)
        buf75 = buf56; del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_16.run(buf75, buf63, buf70, relu_17, unsqueeze_366, buf73, squeeze_151, buf57, primals_101, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_101
        del squeeze_151
        del unsqueeze_366
        buf76 = buf72; del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_17.run(relu_17, buf75, buf76, 4992, 121, grid=grid(4992), stream=stream0)
        buf77 = buf73; del buf73  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_18.run(buf76, buf77, 384, 13, grid=grid(384), stream=stream0)
        buf78 = reinterpret_tensor(buf76, (384, 13), (1, 384), 0); del buf76  # reuse
        buf85 = reinterpret_tensor(buf65, (384, 13), (1, 384), 0); del buf65  # reuse
        buf92 = reinterpret_tensor(buf58, (384, 13), (1, 384), 0); del buf58  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_19.run(relu_17, buf75, convolution_35, unsqueeze_378, convolution_34, unsqueeze_390, relu_16, unsqueeze_402, buf78, buf85, buf92, 4992, 121, grid=grid(4992), stream=stream0)
        buf79 = empty((384, ), device='cuda', dtype=torch.float32)
        buf80 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf78, squeeze_148, buf79, buf80, 384, 13, grid=grid(384), stream=stream0)
        buf86 = empty((384, ), device='cuda', dtype=torch.float32)
        buf87 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf85, squeeze_145, buf86, buf87, 384, 13, grid=grid(384), stream=stream0)
        buf81 = reinterpret_tensor(buf70, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf70  # reuse
        buf88 = reinterpret_tensor(buf63, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf63  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21.run(relu_17, buf75, convolution_35, unsqueeze_378, buf79, squeeze_148, buf77, primals_99, convolution_34, unsqueeze_390, buf86, squeeze_145, primals_97, buf81, buf88, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del convolution_34
        del convolution_35
        del primals_97
        del primals_99
        del squeeze_145
        del squeeze_148
        del unsqueeze_378
        del unsqueeze_390
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf82 = aten.convolution_backward(buf81, relu_16, primals_158, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf81
        del primals_158
        buf83 = buf82[0]
        buf84 = buf82[1]
        del buf82
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf89 = aten.convolution_backward(buf88, relu_16, primals_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf88
        del primals_157
        buf90 = buf89[0]
        buf91 = buf89[1]
        del buf89
        buf93 = buf86; del buf86  # reuse
        buf94 = buf79; del buf79  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf92, squeeze_142, buf93, buf94, 384, 13, grid=grid(384), stream=stream0)
        buf95 = buf75; del buf75  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_22.run(buf95, buf83, buf90, relu_17, relu_16, unsqueeze_402, buf93, squeeze_142, buf77, primals_95, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_95
        del relu_17
        del squeeze_142
        del unsqueeze_402
        buf96 = reinterpret_tensor(buf92, (384, 13), (13, 1), 0); del buf92  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_17.run(relu_16, buf95, buf96, 4992, 121, grid=grid(4992), stream=stream0)
        buf97 = buf93; del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_18.run(buf96, buf97, 384, 13, grid=grid(384), stream=stream0)
        buf98 = reinterpret_tensor(buf96, (384, 13), (1, 384), 0); del buf96  # reuse
        buf105 = buf85; del buf85  # reuse
        buf112 = buf78; del buf78  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_19.run(relu_16, buf95, convolution_33, unsqueeze_414, convolution_32, unsqueeze_426, relu_15, unsqueeze_438, buf98, buf105, buf112, 4992, 121, grid=grid(4992), stream=stream0)
        buf99 = empty((384, ), device='cuda', dtype=torch.float32)
        buf100 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf98, squeeze_139, buf99, buf100, 384, 13, grid=grid(384), stream=stream0)
        buf106 = empty((384, ), device='cuda', dtype=torch.float32)
        buf107 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf105, squeeze_136, buf106, buf107, 384, 13, grid=grid(384), stream=stream0)
        buf101 = reinterpret_tensor(buf90, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf90  # reuse
        buf108 = reinterpret_tensor(buf83, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf83  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21.run(relu_16, buf95, convolution_33, unsqueeze_414, buf99, squeeze_139, buf97, primals_93, convolution_32, unsqueeze_426, buf106, squeeze_136, primals_91, buf101, buf108, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del convolution_32
        del convolution_33
        del primals_91
        del primals_93
        del squeeze_136
        del squeeze_139
        del unsqueeze_414
        del unsqueeze_426
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf102 = aten.convolution_backward(buf101, relu_15, primals_156, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf101
        del primals_156
        buf103 = buf102[0]
        buf104 = buf102[1]
        del buf102
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf109 = aten.convolution_backward(buf108, relu_15, primals_155, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf108
        del primals_155
        buf110 = buf109[0]
        buf111 = buf109[1]
        del buf109
        buf113 = buf99; del buf99  # reuse
        buf114 = buf106; del buf106  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf112, squeeze_133, buf113, buf114, 384, 13, grid=grid(384), stream=stream0)
        buf115 = buf103; del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_23.run(buf115, buf110, relu_16, buf95, relu_15, unsqueeze_438, buf113, squeeze_133, buf97, primals_89, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_89
        del relu_16
        del squeeze_133
        del unsqueeze_438
        buf116 = reinterpret_tensor(buf112, (384, 13), (13, 1), 0); del buf112  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_17.run(relu_15, buf115, buf116, 4992, 121, grid=grid(4992), stream=stream0)
        buf117 = buf113; del buf113  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_18.run(buf116, buf117, 384, 13, grid=grid(384), stream=stream0)
        buf118 = reinterpret_tensor(buf116, (384, 13), (1, 384), 0); del buf116  # reuse
        buf125 = buf105; del buf105  # reuse
        buf132 = buf98; del buf98  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_19.run(relu_15, buf115, convolution_31, unsqueeze_450, convolution_30, unsqueeze_462, relu_14, unsqueeze_474, buf118, buf125, buf132, 4992, 121, grid=grid(4992), stream=stream0)
        buf119 = empty((384, ), device='cuda', dtype=torch.float32)
        buf120 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf118, squeeze_130, buf119, buf120, 384, 13, grid=grid(384), stream=stream0)
        buf126 = empty((384, ), device='cuda', dtype=torch.float32)
        buf127 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf125, squeeze_127, buf126, buf127, 384, 13, grid=grid(384), stream=stream0)
        buf121 = reinterpret_tensor(buf95, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf95  # reuse
        buf128 = reinterpret_tensor(buf110, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf110  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21.run(relu_15, buf115, convolution_31, unsqueeze_450, buf119, squeeze_130, buf117, primals_87, convolution_30, unsqueeze_462, buf126, squeeze_127, primals_85, buf121, buf128, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del convolution_30
        del convolution_31
        del primals_85
        del primals_87
        del squeeze_127
        del squeeze_130
        del unsqueeze_450
        del unsqueeze_462
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf122 = aten.convolution_backward(buf121, relu_14, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf121
        del primals_154
        buf123 = buf122[0]
        buf124 = buf122[1]
        del buf122
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf129 = aten.convolution_backward(buf128, relu_14, primals_153, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf128
        del primals_153
        buf130 = buf129[0]
        buf131 = buf129[1]
        del buf129
        buf133 = buf126; del buf126  # reuse
        buf134 = buf119; del buf119  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf132, squeeze_124, buf133, buf134, 384, 13, grid=grid(384), stream=stream0)
        buf135 = buf115; del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_22.run(buf135, buf123, buf130, relu_15, relu_14, unsqueeze_474, buf133, squeeze_124, buf117, primals_83, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_83
        del relu_15
        del squeeze_124
        del unsqueeze_474
        buf136 = reinterpret_tensor(buf132, (384, 13), (13, 1), 0); del buf132  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_17.run(relu_14, buf135, buf136, 4992, 121, grid=grid(4992), stream=stream0)
        buf137 = buf133; del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_18.run(buf136, buf137, 384, 13, grid=grid(384), stream=stream0)
        buf138 = reinterpret_tensor(buf136, (384, 13), (1, 384), 0); del buf136  # reuse
        buf145 = buf125; del buf125  # reuse
        buf152 = buf118; del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_19.run(relu_14, buf135, convolution_29, unsqueeze_486, convolution_28, unsqueeze_498, relu_13, unsqueeze_510, buf138, buf145, buf152, 4992, 121, grid=grid(4992), stream=stream0)
        buf139 = empty((384, ), device='cuda', dtype=torch.float32)
        buf140 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf138, squeeze_121, buf139, buf140, 384, 13, grid=grid(384), stream=stream0)
        buf146 = empty((384, ), device='cuda', dtype=torch.float32)
        buf147 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf145, squeeze_118, buf146, buf147, 384, 13, grid=grid(384), stream=stream0)
        buf141 = reinterpret_tensor(buf130, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf130  # reuse
        buf148 = reinterpret_tensor(buf123, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21.run(relu_14, buf135, convolution_29, unsqueeze_486, buf139, squeeze_121, buf137, primals_81, convolution_28, unsqueeze_498, buf146, squeeze_118, primals_79, buf141, buf148, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del convolution_28
        del convolution_29
        del primals_79
        del primals_81
        del squeeze_118
        del squeeze_121
        del unsqueeze_486
        del unsqueeze_498
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf142 = aten.convolution_backward(buf141, relu_13, primals_152, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf141
        del primals_152
        buf143 = buf142[0]
        buf144 = buf142[1]
        del buf142
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf149 = aten.convolution_backward(buf148, relu_13, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf148
        del primals_151
        buf150 = buf149[0]
        buf151 = buf149[1]
        del buf149
        buf153 = buf146; del buf146  # reuse
        buf154 = buf139; del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf152, squeeze_115, buf153, buf154, 384, 13, grid=grid(384), stream=stream0)
        buf155 = buf135; del buf135  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_22.run(buf155, buf143, buf150, relu_14, relu_13, unsqueeze_510, buf153, squeeze_115, buf137, primals_77, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_77
        del relu_14
        del squeeze_115
        del unsqueeze_510
        buf156 = reinterpret_tensor(buf152, (384, 13), (13, 1), 0); del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_17.run(relu_13, buf155, buf156, 4992, 121, grid=grid(4992), stream=stream0)
        buf157 = buf153; del buf153  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_18.run(buf156, buf157, 384, 13, grid=grid(384), stream=stream0)
        buf158 = reinterpret_tensor(buf156, (384, 13), (1, 384), 0); del buf156  # reuse
        buf165 = buf145; del buf145  # reuse
        buf172 = buf138; del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_19.run(relu_13, buf155, convolution_27, unsqueeze_522, convolution_26, unsqueeze_534, relu_12, unsqueeze_546, buf158, buf165, buf172, 4992, 121, grid=grid(4992), stream=stream0)
        buf159 = empty((384, ), device='cuda', dtype=torch.float32)
        buf160 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf158, squeeze_112, buf159, buf160, 384, 13, grid=grid(384), stream=stream0)
        buf166 = empty((384, ), device='cuda', dtype=torch.float32)
        buf167 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf165, squeeze_109, buf166, buf167, 384, 13, grid=grid(384), stream=stream0)
        buf161 = reinterpret_tensor(buf150, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf150  # reuse
        buf168 = reinterpret_tensor(buf143, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf143  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21.run(relu_13, buf155, convolution_27, unsqueeze_522, buf159, squeeze_112, buf157, primals_75, convolution_26, unsqueeze_534, buf166, squeeze_109, primals_73, buf161, buf168, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del convolution_26
        del convolution_27
        del primals_73
        del primals_75
        del squeeze_109
        del squeeze_112
        del unsqueeze_522
        del unsqueeze_534
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf162 = aten.convolution_backward(buf161, relu_12, primals_150, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf161
        del primals_150
        buf163 = buf162[0]
        buf164 = buf162[1]
        del buf162
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf169 = aten.convolution_backward(buf168, relu_12, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf168
        del primals_149
        buf170 = buf169[0]
        buf171 = buf169[1]
        del buf169
        buf173 = buf166; del buf166  # reuse
        buf174 = buf159; del buf159  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf172, squeeze_106, buf173, buf174, 384, 13, grid=grid(384), stream=stream0)
        buf175 = buf155; del buf155  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_22.run(buf175, buf163, buf170, relu_13, relu_12, unsqueeze_546, buf173, squeeze_106, buf157, primals_71, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_71
        del relu_13
        del squeeze_106
        del unsqueeze_546
        buf176 = reinterpret_tensor(buf172, (384, 13), (13, 1), 0); del buf172  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_17.run(relu_12, buf175, buf176, 4992, 121, grid=grid(4992), stream=stream0)
        buf177 = buf173; del buf173  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_18.run(buf176, buf177, 384, 13, grid=grid(384), stream=stream0)
        buf178 = reinterpret_tensor(buf176, (384, 13), (1, 384), 0); del buf176  # reuse
        buf185 = buf165; del buf165  # reuse
        buf192 = buf158; del buf158  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_19.run(relu_12, buf175, convolution_25, unsqueeze_558, convolution_24, unsqueeze_570, relu_11, unsqueeze_582, buf178, buf185, buf192, 4992, 121, grid=grid(4992), stream=stream0)
        buf179 = empty((384, ), device='cuda', dtype=torch.float32)
        buf180 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf178, squeeze_103, buf179, buf180, 384, 13, grid=grid(384), stream=stream0)
        buf186 = empty((384, ), device='cuda', dtype=torch.float32)
        buf187 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf185, squeeze_100, buf186, buf187, 384, 13, grid=grid(384), stream=stream0)
        buf181 = reinterpret_tensor(buf170, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf170  # reuse
        buf188 = reinterpret_tensor(buf163, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21.run(relu_12, buf175, convolution_25, unsqueeze_558, buf179, squeeze_103, buf177, primals_69, convolution_24, unsqueeze_570, buf186, squeeze_100, primals_67, buf181, buf188, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del convolution_24
        del convolution_25
        del primals_67
        del primals_69
        del squeeze_100
        del squeeze_103
        del unsqueeze_558
        del unsqueeze_570
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf182 = aten.convolution_backward(buf181, relu_11, primals_148, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf181
        del primals_148
        buf183 = buf182[0]
        buf184 = buf182[1]
        del buf182
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf189 = aten.convolution_backward(buf188, relu_11, primals_147, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf188
        del primals_147
        buf190 = buf189[0]
        buf191 = buf189[1]
        del buf189
        buf193 = buf186; del buf186  # reuse
        buf194 = buf179; del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf192, squeeze_97, buf193, buf194, 384, 13, grid=grid(384), stream=stream0)
        buf195 = buf175; del buf175  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_22.run(buf195, buf183, buf190, relu_12, relu_11, unsqueeze_582, buf193, squeeze_97, buf177, primals_65, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_65
        del relu_12
        del squeeze_97
        del unsqueeze_582
        buf196 = reinterpret_tensor(buf192, (384, 13), (13, 1), 0); del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_17.run(relu_11, buf195, buf196, 4992, 121, grid=grid(4992), stream=stream0)
        buf197 = buf193; del buf193  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_18.run(buf196, buf197, 384, 13, grid=grid(384), stream=stream0)
        buf198 = reinterpret_tensor(buf196, (384, 13), (1, 384), 0); del buf196  # reuse
        buf205 = buf185; del buf185  # reuse
        buf212 = buf178; del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_19.run(relu_11, buf195, convolution_23, unsqueeze_594, convolution_22, unsqueeze_606, relu_10, unsqueeze_618, buf198, buf205, buf212, 4992, 121, grid=grid(4992), stream=stream0)
        buf199 = empty((384, ), device='cuda', dtype=torch.float32)
        buf200 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf198, squeeze_94, buf199, buf200, 384, 13, grid=grid(384), stream=stream0)
        buf206 = empty((384, ), device='cuda', dtype=torch.float32)
        buf207 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf205, squeeze_91, buf206, buf207, 384, 13, grid=grid(384), stream=stream0)
        buf201 = reinterpret_tensor(buf190, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf190  # reuse
        buf208 = reinterpret_tensor(buf183, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21.run(relu_11, buf195, convolution_23, unsqueeze_594, buf199, squeeze_94, buf197, primals_63, convolution_22, unsqueeze_606, buf206, squeeze_91, primals_61, buf201, buf208, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del convolution_22
        del convolution_23
        del primals_61
        del primals_63
        del squeeze_91
        del squeeze_94
        del unsqueeze_594
        del unsqueeze_606
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf202 = aten.convolution_backward(buf201, relu_10, primals_146, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf201
        del primals_146
        buf203 = buf202[0]
        buf204 = buf202[1]
        del buf202
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf209 = aten.convolution_backward(buf208, relu_10, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf208
        del primals_145
        buf210 = buf209[0]
        buf211 = buf209[1]
        del buf209
        buf213 = buf206; del buf206  # reuse
        buf214 = buf199; del buf199  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf212, squeeze_88, buf213, buf214, 384, 13, grid=grid(384), stream=stream0)
        buf215 = buf195; del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_22.run(buf215, buf203, buf210, relu_11, relu_10, unsqueeze_618, buf213, squeeze_88, buf197, primals_59, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_59
        del relu_11
        del squeeze_88
        del unsqueeze_618
        buf216 = reinterpret_tensor(buf212, (384, 13), (13, 1), 0); del buf212  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_17.run(relu_10, buf215, buf216, 4992, 121, grid=grid(4992), stream=stream0)
        buf217 = buf213; del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_18.run(buf216, buf217, 384, 13, grid=grid(384), stream=stream0)
        buf218 = reinterpret_tensor(buf216, (384, 13), (1, 384), 0); del buf216  # reuse
        buf225 = buf205; del buf205  # reuse
        buf232 = buf198; del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_19.run(relu_10, buf215, convolution_21, unsqueeze_630, convolution_20, unsqueeze_642, relu_9, unsqueeze_654, buf218, buf225, buf232, 4992, 121, grid=grid(4992), stream=stream0)
        buf219 = empty((384, ), device='cuda', dtype=torch.float32)
        buf220 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf218, squeeze_85, buf219, buf220, 384, 13, grid=grid(384), stream=stream0)
        buf226 = empty((384, ), device='cuda', dtype=torch.float32)
        buf227 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf225, squeeze_82, buf226, buf227, 384, 13, grid=grid(384), stream=stream0)
        buf221 = reinterpret_tensor(buf210, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf210  # reuse
        buf228 = reinterpret_tensor(buf203, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf203  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21.run(relu_10, buf215, convolution_21, unsqueeze_630, buf219, squeeze_85, buf217, primals_57, convolution_20, unsqueeze_642, buf226, squeeze_82, primals_55, buf221, buf228, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del convolution_20
        del convolution_21
        del primals_55
        del primals_57
        del squeeze_82
        del squeeze_85
        del unsqueeze_630
        del unsqueeze_642
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf222 = aten.convolution_backward(buf221, relu_9, primals_144, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf221
        del primals_144
        buf223 = buf222[0]
        buf224 = buf222[1]
        del buf222
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf229 = aten.convolution_backward(buf228, relu_9, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf228
        del primals_143
        buf230 = buf229[0]
        buf231 = buf229[1]
        del buf229
        buf233 = buf226; del buf226  # reuse
        buf234 = buf219; del buf219  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf232, squeeze_79, buf233, buf234, 384, 13, grid=grid(384), stream=stream0)
        buf235 = buf215; del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_22.run(buf235, buf223, buf230, relu_10, relu_9, unsqueeze_654, buf233, squeeze_79, buf217, primals_53, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_53
        del relu_10
        del squeeze_79
        del unsqueeze_654
        buf236 = reinterpret_tensor(buf232, (384, 13), (13, 1), 0); del buf232  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_17.run(relu_9, buf235, buf236, 4992, 121, grid=grid(4992), stream=stream0)
        buf237 = buf233; del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_18.run(buf236, buf237, 384, 13, grid=grid(384), stream=stream0)
        buf238 = reinterpret_tensor(buf236, (384, 13), (1, 384), 0); del buf236  # reuse
        buf245 = buf225; del buf225  # reuse
        buf252 = buf218; del buf218  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_19.run(relu_9, buf235, convolution_19, unsqueeze_666, convolution_18, unsqueeze_678, relu_8, unsqueeze_690, buf238, buf245, buf252, 4992, 121, grid=grid(4992), stream=stream0)
        buf239 = empty((384, ), device='cuda', dtype=torch.float32)
        buf240 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf238, squeeze_76, buf239, buf240, 384, 13, grid=grid(384), stream=stream0)
        buf246 = empty((384, ), device='cuda', dtype=torch.float32)
        buf247 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf245, squeeze_73, buf246, buf247, 384, 13, grid=grid(384), stream=stream0)
        buf241 = reinterpret_tensor(buf230, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf230  # reuse
        buf248 = reinterpret_tensor(buf223, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf223  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21.run(relu_9, buf235, convolution_19, unsqueeze_666, buf239, squeeze_76, buf237, primals_51, convolution_18, unsqueeze_678, buf246, squeeze_73, primals_49, buf241, buf248, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del convolution_18
        del convolution_19
        del primals_49
        del primals_51
        del squeeze_73
        del squeeze_76
        del unsqueeze_666
        del unsqueeze_678
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf242 = aten.convolution_backward(buf241, relu_8, primals_142, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf241
        del primals_142
        buf243 = buf242[0]
        buf244 = buf242[1]
        del buf242
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf249 = aten.convolution_backward(buf248, relu_8, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf248
        del primals_141
        buf250 = buf249[0]
        buf251 = buf249[1]
        del buf249
        buf253 = buf246; del buf246  # reuse
        buf254 = buf239; del buf239  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf252, squeeze_70, buf253, buf254, 384, 13, grid=grid(384), stream=stream0)
        buf255 = buf235; del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_22.run(buf255, buf243, buf250, relu_9, relu_8, unsqueeze_690, buf253, squeeze_70, buf237, primals_47, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_47
        del relu_9
        del squeeze_70
        del unsqueeze_690
        buf256 = reinterpret_tensor(buf252, (384, 13), (13, 1), 0); del buf252  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_17.run(relu_8, buf255, buf256, 4992, 121, grid=grid(4992), stream=stream0)
        buf257 = buf253; del buf253  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_18.run(buf256, buf257, 384, 13, grid=grid(384), stream=stream0)
        buf258 = reinterpret_tensor(buf256, (384, 13), (1, 384), 0); del buf256  # reuse
        buf265 = buf245; del buf245  # reuse
        buf272 = buf238; del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_19.run(relu_8, buf255, convolution_17, unsqueeze_702, convolution_16, unsqueeze_714, relu_7, unsqueeze_726, buf258, buf265, buf272, 4992, 121, grid=grid(4992), stream=stream0)
        buf259 = empty((384, ), device='cuda', dtype=torch.float32)
        buf260 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf258, squeeze_67, buf259, buf260, 384, 13, grid=grid(384), stream=stream0)
        del buf258
        buf266 = empty((384, ), device='cuda', dtype=torch.float32)
        buf267 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf265, squeeze_64, buf266, buf267, 384, 13, grid=grid(384), stream=stream0)
        buf261 = reinterpret_tensor(buf250, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf250  # reuse
        buf268 = reinterpret_tensor(buf243, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf243  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21.run(relu_8, buf255, convolution_17, unsqueeze_702, buf259, squeeze_67, buf257, primals_45, convolution_16, unsqueeze_714, buf266, squeeze_64, primals_43, buf261, buf268, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del convolution_16
        del convolution_17
        del primals_43
        del primals_45
        del squeeze_64
        del squeeze_67
        del unsqueeze_702
        del unsqueeze_714
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf262 = aten.convolution_backward(buf261, relu_7, primals_140, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf261
        del primals_140
        buf263 = buf262[0]
        buf264 = buf262[1]
        del buf262
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf269 = aten.convolution_backward(buf268, relu_7, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf268
        del primals_139
        buf270 = buf269[0]
        buf271 = buf269[1]
        del buf269
        buf273 = buf266; del buf266  # reuse
        buf274 = buf259; del buf259  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf272, squeeze_61, buf273, buf274, 384, 13, grid=grid(384), stream=stream0)
        buf275 = buf255; del buf255  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_22.run(buf275, buf263, buf270, relu_8, relu_7, unsqueeze_726, buf273, squeeze_61, buf257, primals_41, 3072, 196, grid=grid(3072, 196), stream=stream0)
        del primals_41
        del relu_8
        del squeeze_61
        del unsqueeze_726
        buf276 = reinterpret_tensor(buf272, (384, 13), (13, 1), 0); del buf272  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_17.run(relu_7, buf275, buf276, 4992, 121, grid=grid(4992), stream=stream0)
        buf277 = buf273; del buf273  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_18.run(buf276, buf277, 384, 13, grid=grid(384), stream=stream0)
        buf278 = reinterpret_tensor(buf276, (384, 13), (1, 384), 0); del buf276  # reuse
        buf285 = buf265; del buf265  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_24.run(relu_7, buf275, convolution_15, unsqueeze_738, convolution_14, unsqueeze_750, buf278, buf285, 4992, 121, grid=grid(4992), stream=stream0)
        buf279 = empty((384, ), device='cuda', dtype=torch.float32)
        buf280 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf278, squeeze_58, buf279, buf280, 384, 13, grid=grid(384), stream=stream0)
        del buf278
        buf286 = empty((384, ), device='cuda', dtype=torch.float32)
        buf287 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_20.run(buf285, squeeze_55, buf286, buf287, 384, 13, grid=grid(384), stream=stream0)
        del buf285
        buf281 = reinterpret_tensor(buf270, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf270  # reuse
        buf288 = reinterpret_tensor(buf263, (8, 384, 14, 14), (75264, 1, 5376, 384), 0); del buf263  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21.run(relu_7, buf275, convolution_15, unsqueeze_738, buf279, squeeze_58, buf277, primals_39, convolution_14, unsqueeze_750, buf286, squeeze_55, primals_37, buf281, buf288, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del buf275
        del convolution_14
        del convolution_15
        del primals_37
        del primals_39
        del relu_7
        del squeeze_55
        del squeeze_58
        del unsqueeze_738
        del unsqueeze_750
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf282 = aten.convolution_backward(buf281, relu_6, primals_138, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf281
        del primals_138
        buf283 = buf282[0]
        buf284 = buf282[1]
        del buf282
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf289 = aten.convolution_backward(buf288, relu_6, primals_137, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf288
        del primals_137
        buf290 = buf289[0]
        buf291 = buf289[1]
        del buf289
        buf292 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_25.run(relu_6, buf283, buf290, buf292, 192, 6272, grid=grid(192), stream=stream0)
        buf293 = empty((192, 49), device='cuda', dtype=torch.float32)
        buf300 = empty((192, 49), device='cuda', dtype=torch.float32)
        buf307 = empty((192, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_26.run(relu_6, buf283, buf290, convolution_13, unsqueeze_762, convolution_12, unsqueeze_774, relu_5, unsqueeze_786, buf293, buf300, buf307, 9408, 128, grid=grid(9408), stream=stream0)
        buf294 = empty((192, ), device='cuda', dtype=torch.float32)
        buf296 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_27.run(buf293, squeeze_52, buf294, buf296, 192, 49, grid=grid(192), stream=stream0)
        buf301 = empty((192, ), device='cuda', dtype=torch.float32)
        buf303 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_27.run(buf300, squeeze_49, buf301, buf303, 192, 49, grid=grid(192), stream=stream0)
        buf295 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        buf302 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_28.run(relu_6, buf283, buf290, convolution_13, unsqueeze_762, buf294, squeeze_52, buf292, primals_35, convolution_12, unsqueeze_774, buf301, squeeze_49, primals_33, buf295, buf302, 6272, 192, grid=grid(6272, 192), stream=stream0)
        del convolution_12
        del convolution_13
        del primals_33
        del primals_35
        del squeeze_49
        del squeeze_52
        del unsqueeze_762
        del unsqueeze_774
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf297 = aten.convolution_backward(buf295, relu_5, primals_136, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_136
        buf298 = buf297[0]
        buf299 = buf297[1]
        del buf297
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf304 = aten.convolution_backward(buf302, relu_5, primals_135, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_135
        buf305 = buf304[0]
        buf306 = buf304[1]
        del buf304
        buf308 = buf301; del buf301  # reuse
        buf310 = buf294; del buf294  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_27.run(buf307, squeeze_46, buf308, buf310, 192, 49, grid=grid(192), stream=stream0)
        buf309 = buf283; del buf283  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_29.run(buf309, relu_6, buf290, relu_5, unsqueeze_786, buf308, squeeze_46, buf292, primals_31, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del primals_31
        del relu_6
        del squeeze_46
        del unsqueeze_786
        buf311 = buf308; del buf308  # reuse
        buf312 = empty((192, ), device='cuda', dtype=torch.float32)
        buf319 = empty((192, ), device='cuda', dtype=torch.float32)
        buf326 = empty((192, ), device='cuda', dtype=torch.float32)
        buf314 = empty((192, ), device='cuda', dtype=torch.float32)
        buf321 = empty((192, ), device='cuda', dtype=torch.float32)
        buf328 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_30.run(relu_5, buf298, buf305, buf309, convolution_11, unsqueeze_798, convolution_10, unsqueeze_810, relu_4, unsqueeze_822, squeeze_43, squeeze_40, squeeze_37, buf311, buf312, buf319, buf326, buf314, buf321, buf328, 192, 6272, grid=grid(192), stream=stream0)
        buf327 = buf290; del buf290  # reuse
        buf315 = buf302; del buf302  # reuse
        buf322 = buf295; del buf295  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_31.run(relu_5, buf298, buf305, buf309, convolution_11, unsqueeze_798, buf312, squeeze_43, buf311, convolution_10, unsqueeze_810, buf319, squeeze_40, relu_4, unsqueeze_822, buf326, squeeze_37, primals_29, primals_27, buf327, buf315, buf322, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del buf298
        del buf305
        del buf309
        del convolution_10
        del convolution_11
        del primals_27
        del primals_29
        del relu_5
        del squeeze_40
        del squeeze_43
        del unsqueeze_798
        del unsqueeze_810
        del unsqueeze_822
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf316 = aten.convolution_backward(buf315, relu_4, primals_134, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf315
        del primals_134
        buf317 = buf316[0]
        buf318 = buf316[1]
        del buf316
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf323 = aten.convolution_backward(buf322, relu_4, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf322
        del primals_133
        buf324 = buf323[0]
        buf325 = buf323[1]
        del buf323
        buf329 = buf317; del buf317  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_32.run(buf329, relu_4, buf324, buf327, squeeze_37, primals_25, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del primals_25
        del relu_4
        del squeeze_37
        buf330 = buf326; del buf326  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_33.run(buf329, buf330, 192, 6272, grid=grid(192), stream=stream0)
        buf331 = buf307; del buf307  # reuse
        buf338 = buf300; del buf300  # reuse
        buf345 = buf293; del buf293  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_34.run(buf329, convolution_9, unsqueeze_834, convolution_8, unsqueeze_846, relu_3, unsqueeze_858, buf331, buf338, buf345, 9408, 128, grid=grid(9408), stream=stream0)
        buf332 = buf319; del buf319  # reuse
        buf333 = buf312; del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_27.run(buf331, squeeze_34, buf332, buf333, 192, 49, grid=grid(192), stream=stream0)
        del buf331
        buf339 = empty((192, ), device='cuda', dtype=torch.float32)
        buf340 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_27.run(buf338, squeeze_31, buf339, buf340, 192, 49, grid=grid(192), stream=stream0)
        buf334 = reinterpret_tensor(buf327, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf327  # reuse
        buf341 = reinterpret_tensor(buf324, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf324  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_35.run(buf329, convolution_9, unsqueeze_834, buf332, squeeze_34, buf330, primals_23, convolution_8, unsqueeze_846, buf339, squeeze_31, primals_21, buf334, buf341, 6272, 192, grid=grid(6272, 192), stream=stream0)
        del convolution_8
        del convolution_9
        del primals_21
        del primals_23
        del squeeze_31
        del squeeze_34
        del unsqueeze_834
        del unsqueeze_846
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf335 = aten.convolution_backward(buf334, relu_3, primals_132, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf334
        del primals_132
        buf336 = buf335[0]
        buf337 = buf335[1]
        del buf335
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf342 = aten.convolution_backward(buf341, relu_3, primals_131, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf341
        del primals_131
        buf343 = buf342[0]
        buf344 = buf342[1]
        del buf342
        buf346 = buf339; del buf339  # reuse
        buf347 = buf332; del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_27.run(buf345, squeeze_28, buf346, buf347, 192, 49, grid=grid(192), stream=stream0)
        buf348 = buf329; del buf329  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_36.run(buf348, buf336, buf343, relu_3, unsqueeze_858, buf346, squeeze_28, buf330, primals_19, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del primals_19
        del squeeze_28
        del unsqueeze_858
        buf349 = buf345; del buf345  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_37.run(relu_3, buf348, buf349, 9408, 128, grid=grid(9408), stream=stream0)
        buf350 = buf346; del buf346  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_38.run(buf349, buf350, 192, 49, grid=grid(192), stream=stream0)
        buf351 = reinterpret_tensor(buf349, (192, 49), (1, 192), 0); del buf349  # reuse
        buf358 = reinterpret_tensor(buf338, (192, 49), (1, 192), 0); del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_39.run(relu_3, buf348, convolution_7, unsqueeze_870, convolution_6, unsqueeze_882, buf351, buf358, 9408, 128, grid=grid(9408), stream=stream0)
        buf352 = empty((192, ), device='cuda', dtype=torch.float32)
        buf353 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_40.run(buf351, squeeze_25, buf352, buf353, 192, 49, grid=grid(192), stream=stream0)
        del buf351
        buf359 = empty((192, ), device='cuda', dtype=torch.float32)
        buf360 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_40.run(buf358, squeeze_22, buf359, buf360, 192, 49, grid=grid(192), stream=stream0)
        del buf358
        buf354 = reinterpret_tensor(buf343, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf343  # reuse
        buf361 = reinterpret_tensor(buf336, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf336  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_41.run(relu_3, buf348, convolution_7, unsqueeze_870, buf352, squeeze_25, buf350, primals_17, convolution_6, unsqueeze_882, buf359, squeeze_22, primals_15, buf354, buf361, 6272, 192, grid=grid(6272, 192), stream=stream0)
        del buf348
        del buf352
        del buf359
        del convolution_6
        del convolution_7
        del primals_15
        del primals_17
        del relu_3
        del squeeze_22
        del squeeze_25
        del unsqueeze_870
        del unsqueeze_882
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf355 = aten.convolution_backward(buf354, relu_2, primals_130, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf354
        del primals_130
        buf356 = buf355[0]
        buf357 = buf355[1]
        del buf355
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf362 = aten.convolution_backward(buf361, relu_2, primals_129, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf361
        del primals_129
        buf363 = buf362[0]
        buf364 = buf362[1]
        del buf362
        buf365 = reinterpret_tensor(buf286, (96, 4), (1, 96), 0); del buf286  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_42.run(relu_2, buf356, buf363, buf365, 384, 6272, grid=grid(384), stream=stream0)
        buf366 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_43.run(buf365, buf366, 96, 4, grid=grid(96), stream=stream0)
        buf367 = empty((96, 196), device='cuda', dtype=torch.float32)
        buf374 = empty((96, 196), device='cuda', dtype=torch.float32)
        buf381 = empty((96, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_44.run(relu_2, buf356, buf363, convolution_5, unsqueeze_894, convolution_4, unsqueeze_906, relu_1, unsqueeze_918, buf367, buf374, buf381, 18816, 128, grid=grid(18816), stream=stream0)
        buf368 = empty((96, ), device='cuda', dtype=torch.float32)
        buf370 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_45.run(buf367, squeeze_19, buf368, buf370, 96, 196, grid=grid(96), stream=stream0)
        del buf367
        buf375 = empty((96, ), device='cuda', dtype=torch.float32)
        buf377 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_45.run(buf374, squeeze_16, buf375, buf377, 96, 196, grid=grid(96), stream=stream0)
        del buf374
        buf369 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        buf376 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_46.run(relu_2, buf356, buf363, convolution_5, unsqueeze_894, buf368, squeeze_19, buf366, primals_13, convolution_4, unsqueeze_906, buf375, squeeze_16, primals_11, buf369, buf376, 25088, 96, grid=grid(25088, 96), stream=stream0)
        del convolution_4
        del convolution_5
        del primals_11
        del primals_13
        del squeeze_16
        del squeeze_19
        del unsqueeze_894
        del unsqueeze_906
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf371 = aten.convolution_backward(buf369, relu_1, primals_128, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf369
        del primals_128
        buf372 = buf371[0]
        buf373 = buf371[1]
        del buf371
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf378 = aten.convolution_backward(buf376, relu_1, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_127
        buf379 = buf378[0]
        buf380 = buf378[1]
        del buf378
        buf382 = buf375; del buf375  # reuse
        buf384 = buf368; del buf368  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_45.run(buf381, squeeze_13, buf382, buf384, 96, 196, grid=grid(96), stream=stream0)
        del buf381
        buf383 = buf356; del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_47.run(buf383, relu_2, buf363, relu_1, unsqueeze_918, buf382, squeeze_13, buf366, primals_9, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del primals_9
        del relu_2
        del squeeze_13
        del unsqueeze_918
        buf385 = buf365; del buf365  # reuse
        buf387 = reinterpret_tensor(buf279, (96, 4), (1, 96), 0); del buf279  # reuse
        buf395 = empty_strided((96, 4), (1, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_48.run(relu_1, buf372, buf379, buf383, convolution_3, unsqueeze_930, convolution_2, unsqueeze_942, buf385, buf387, buf395, 384, 6272, grid=grid(384), stream=stream0)
        buf386 = buf382; del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_43.run(buf385, buf386, 96, 4, grid=grid(96), stream=stream0)
        del buf385
        buf388 = empty((96, ), device='cuda', dtype=torch.float32)
        buf390 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_49.run(buf387, squeeze_10, buf388, buf390, 96, 4, grid=grid(96), stream=stream0)
        del buf387
        buf396 = empty((96, ), device='cuda', dtype=torch.float32)
        buf398 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_49.run(buf395, squeeze_7, buf396, buf398, 96, 4, grid=grid(96), stream=stream0)
        del buf395
        buf391 = reinterpret_tensor(buf363, (8, 96, 56, 56), (301056, 1, 5376, 96), 0); del buf363  # reuse
        buf399 = buf376; del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_50.run(relu_1, buf372, buf379, buf383, convolution_3, unsqueeze_930, buf388, squeeze_10, buf386, convolution_2, unsqueeze_942, buf396, squeeze_7, primals_7, primals_5, buf391, buf399, 768, 3136, grid=grid(768, 3136), stream=stream0)
        del buf372
        del buf379
        del buf383
        del buf388
        del buf396
        del convolution_2
        del convolution_3
        del primals_5
        del primals_7
        del relu_1
        del squeeze_10
        del squeeze_7
        del unsqueeze_930
        del unsqueeze_942
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf392 = aten.convolution_backward(buf391, relu, primals_126, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf391
        del primals_126
        buf393 = buf392[0]
        buf394 = buf392[1]
        del buf392
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf400 = aten.convolution_backward(buf399, relu, primals_125, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf399
        del primals_125
        buf401 = buf400[0]
        buf402 = buf400[1]
        del buf400
        buf403 = empty((64, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_51.run(relu, buf393, buf401, buf403, 832, 7720, grid=grid(832), stream=stream0)
        buf404 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_52.run(buf403, buf404, 64, 13, grid=grid(64), stream=stream0)
        del buf403
        buf405 = empty((64, 784), device='cuda', dtype=torch.float32)
        buf411 = empty((64, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_native_batch_norm_backward_threshold_backward_53.run(relu, buf393, buf401, convolution_1, unsqueeze_954, convolution, unsqueeze_966, buf405, buf411, 50176, 128, grid=grid(50176), stream=stream0)
        buf406 = empty((64, ), device='cuda', dtype=torch.float32)
        buf408 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_54.run(buf405, squeeze_4, buf406, buf408, 64, 784, grid=grid(64), stream=stream0)
        del buf405
        buf412 = empty((64, ), device='cuda', dtype=torch.float32)
        buf414 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_native_batch_norm_backward_threshold_backward_54.run(buf411, squeeze_1, buf412, buf414, 64, 784, grid=grid(64), stream=stream0)
        del buf411
        buf407 = empty_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda', dtype=torch.float32)
        buf413 = empty_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_native_batch_norm_backward_threshold_backward_55.run(relu, buf393, buf401, convolution_1, unsqueeze_954, buf406, squeeze_4, buf404, primals_3, convolution, unsqueeze_966, buf412, squeeze_1, primals_1, buf407, buf413, 100352, 64, grid=grid(100352, 64), stream=stream0)
        del buf393
        del buf401
        del buf406
        del buf412
        del convolution
        del convolution_1
        del primals_1
        del primals_3
        del relu
        del squeeze_1
        del squeeze_4
        del unsqueeze_954
        del unsqueeze_966
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf409 = aten.convolution_backward(buf407, primals_352, primals_124, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf407
        del primals_124
        buf410 = buf409[1]
        del buf409
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf415 = aten.convolution_backward(buf413, primals_352, primals_123, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf413
        del primals_123
        del primals_352
        buf416 = buf415[1]
        return (buf414, buf404, buf408, buf404, buf398, buf386, buf390, buf386, buf384, buf366, buf377, buf366, buf370, buf366, buf360, buf350, buf353, buf350, buf347, buf330, buf340, buf330, buf333, buf330, buf328, buf311, buf321, buf311, buf314, buf311, buf310, buf292, buf303, buf292, buf296, buf292, buf287, buf277, buf280, buf277, buf274, buf257, buf267, buf257, buf260, buf257, buf254, buf237, buf247, buf237, buf240, buf237, buf234, buf217, buf227, buf217, buf220, buf217, buf214, buf197, buf207, buf197, buf200, buf197, buf194, buf177, buf187, buf177, buf180, buf177, buf174, buf157, buf167, buf157, buf160, buf157, buf154, buf137, buf147, buf137, buf140, buf137, buf134, buf117, buf127, buf117, buf120, buf117, buf114, buf97, buf107, buf97, buf100, buf97, buf94, buf77, buf87, buf77, buf80, buf77, buf74, buf57, buf67, buf57, buf60, buf57, buf55, buf38, buf48, buf38, buf41, buf38, buf37, buf19, buf30, buf19, buf23, buf19, buf14, buf4, buf7, buf4, buf416, buf410, buf402, buf394, buf380, buf373, buf364, buf357, buf344, buf337, buf325, buf318, buf306, buf299, buf291, buf284, buf271, buf264, buf251, buf244, buf231, buf224, buf211, buf204, buf191, buf184, buf171, buf164, buf151, buf144, buf131, buf124, buf111, buf104, buf91, buf84, buf71, buf64, buf52, buf45, buf33, buf26, buf18, buf11, reinterpret_tensor(buf1, (1000, 1408), (1408, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((64, 3, 1, 1), (3, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((64, 3, 3, 3), (27, 1, 9, 3), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((96, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((96, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((96, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((96, 96, 3, 3), (864, 1, 288, 96), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((192, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((192, 96, 3, 3), (864, 1, 288, 96), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((384, 192, 3, 3), (1728, 1, 576, 192), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((1408, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((1408, 384, 3, 3), (3456, 1, 1152, 384), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_6 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_7 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_70 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_9 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_82 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_10 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_88 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_94 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_11 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_97 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_100 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_103 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_12 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_106 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_109 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_112 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_13 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_115 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_118 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_121 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_14 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_124 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_127 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_130 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_15 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_133 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_136 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_139 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_16 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_142 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_145 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_148 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_17 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_151 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_154 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_157 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_18 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_160 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_163 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_166 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_19 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_169 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_172 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    squeeze_175 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_20 = rand_strided((8, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((8, 1408, 7, 7), (68992, 1, 9856, 1408), device='cuda:0', dtype=torch.float32)
    squeeze_178 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((8, 1408, 7, 7), (68992, 1, 9856, 1408), device='cuda:0', dtype=torch.float32)
    squeeze_181 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone = rand_strided((8, 1408), (1408, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1408), (1408, 1), device='cuda:0', dtype=torch.float32)
    le = rand_strided((8, 1408, 7, 7), (68992, 1, 9856, 1408), device='cuda:0', dtype=torch.bool)
    unsqueeze_246 = rand_strided((1, 1408, 1, 1), (1408, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_258 = rand_strided((1, 1408, 1, 1), (1408, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_270 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_282 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_294 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_306 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_318 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_330 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_342 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_354 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_366 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_378 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_390 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_402 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_414 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_426 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_438 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_450 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_462 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_474 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_486 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_498 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_510 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_522 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_534 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_546 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_558 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_570 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_582 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_594 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_606 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_618 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_630 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_642 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_654 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_666 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_678 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_690 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_702 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_714 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_726 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_738 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_750 = rand_strided((1, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_762 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_774 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_786 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_798 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_810 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_822 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_834 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_846 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_858 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_870 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_882 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_894 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_906 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_918 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_930 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_942 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_954 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_966 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_352, convolution, squeeze_1, convolution_1, squeeze_4, relu, convolution_2, squeeze_7, convolution_3, squeeze_10, relu_1, squeeze_13, convolution_4, squeeze_16, convolution_5, squeeze_19, relu_2, convolution_6, squeeze_22, convolution_7, squeeze_25, relu_3, squeeze_28, convolution_8, squeeze_31, convolution_9, squeeze_34, relu_4, squeeze_37, convolution_10, squeeze_40, convolution_11, squeeze_43, relu_5, squeeze_46, convolution_12, squeeze_49, convolution_13, squeeze_52, relu_6, convolution_14, squeeze_55, convolution_15, squeeze_58, relu_7, squeeze_61, convolution_16, squeeze_64, convolution_17, squeeze_67, relu_8, squeeze_70, convolution_18, squeeze_73, convolution_19, squeeze_76, relu_9, squeeze_79, convolution_20, squeeze_82, convolution_21, squeeze_85, relu_10, squeeze_88, convolution_22, squeeze_91, convolution_23, squeeze_94, relu_11, squeeze_97, convolution_24, squeeze_100, convolution_25, squeeze_103, relu_12, squeeze_106, convolution_26, squeeze_109, convolution_27, squeeze_112, relu_13, squeeze_115, convolution_28, squeeze_118, convolution_29, squeeze_121, relu_14, squeeze_124, convolution_30, squeeze_127, convolution_31, squeeze_130, relu_15, squeeze_133, convolution_32, squeeze_136, convolution_33, squeeze_139, relu_16, squeeze_142, convolution_34, squeeze_145, convolution_35, squeeze_148, relu_17, squeeze_151, convolution_36, squeeze_154, convolution_37, squeeze_157, relu_18, squeeze_160, convolution_38, squeeze_163, convolution_39, squeeze_166, relu_19, squeeze_169, convolution_40, squeeze_172, convolution_41, squeeze_175, relu_20, convolution_42, squeeze_178, convolution_43, squeeze_181, clone, permute_1, le, unsqueeze_246, unsqueeze_258, unsqueeze_270, unsqueeze_282, unsqueeze_294, unsqueeze_306, unsqueeze_318, unsqueeze_330, unsqueeze_342, unsqueeze_354, unsqueeze_366, unsqueeze_378, unsqueeze_390, unsqueeze_402, unsqueeze_414, unsqueeze_426, unsqueeze_438, unsqueeze_450, unsqueeze_462, unsqueeze_474, unsqueeze_486, unsqueeze_498, unsqueeze_510, unsqueeze_522, unsqueeze_534, unsqueeze_546, unsqueeze_558, unsqueeze_570, unsqueeze_582, unsqueeze_594, unsqueeze_606, unsqueeze_618, unsqueeze_630, unsqueeze_642, unsqueeze_654, unsqueeze_666, unsqueeze_678, unsqueeze_690, unsqueeze_702, unsqueeze_714, unsqueeze_726, unsqueeze_738, unsqueeze_750, unsqueeze_762, unsqueeze_774, unsqueeze_786, unsqueeze_798, unsqueeze_810, unsqueeze_822, unsqueeze_834, unsqueeze_846, unsqueeze_858, unsqueeze_870, unsqueeze_882, unsqueeze_894, unsqueeze_906, unsqueeze_918, unsqueeze_930, unsqueeze_942, unsqueeze_954, unsqueeze_966, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('repvgg_a2', benchmark_compiled_module)
