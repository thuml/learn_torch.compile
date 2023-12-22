
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


# kernel path: /tmp/torchinductor_youkaichao/gg/cgg463u26szk2koomsmwdhziq76kysbb7oomrurjsyrliqijjwz4.py
# Source Nodes: [], Original ATen: [aten.sum, aten.view]

triton_poi_fused_sum_view_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sum_view_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (1000 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (2000 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (3000 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/sb/csb3yvgovzw4zmttlziq5wcm3ylegwpatj6auyzk5eiqd2u6oudj.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward]

triton_poi_fused_hardswish_backward_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_backward_1', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp5 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = tmp0 / tmp3
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp4, tmp9, tmp5)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp11, tmp10)
    tl.store(in_out_ptr0 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7y/c7ybvyd7t4uoytazpb7axoebkjgnvr3gj5uj4cvvdwy6bhfrblss.py
# Source Nodes: [], Original ATen: [aten.sum, aten.view]

triton_poi_fused_sum_view_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sum_view_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (1280 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (2560 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (3840 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e7/ce7zhfela27oic4qwoum6adsg6krfbmspvp77uefx6r5rh5q6bdr.py
# Source Nodes: [], Original ATen: [aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_div_hardswish_backward_native_batch_norm_backward_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_hardswish_backward_native_batch_norm_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 960
    x1 = (xindex // 960)
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp19 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (960*r2) + (94080*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr1 + (x0 + (960*(r2 // 49)) + (1920*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr2 + (x0 + (960*r2) + (94080*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = 49.0
        tmp7 = tmp5 / tmp6
        tmp8 = tmp0 / tmp3
        tmp9 = 0.5
        tmp10 = tmp8 + tmp9
        tmp11 = tmp7 * tmp10
        tmp12 = tl.where(tmp4, tmp11, tmp7)
        tmp13 = 0.0
        tmp14 = tl.where(tmp2, tmp13, tmp12)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
        tmp20 = tmp18 - tmp19
        tmp21 = tmp14 * tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp16, xmask)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/si/csigogyna2m6tuwuctwy3m2chfnbks5njoxn6shl2s5wivoesc2r.py
# Source Nodes: [], Original ATen: [aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_div_hardswish_backward_native_batch_norm_backward_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_hardswish_backward_native_batch_norm_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    rnumel = 2
    RBLOCK: tl.constexpr = 2
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


# kernel path: /tmp/torchinductor_youkaichao/di/cdih2smytgqbu7qsafurjupgc2av5vjisdu33vk7kuqjvvvm4pul.py
# Source Nodes: [], Original ATen: [aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_div_hardswish_backward_native_batch_norm_backward_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_hardswish_backward_native_batch_norm_backward_5', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    rnumel = 2
    RBLOCK: tl.constexpr = 2
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
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ur/cur4cmses4z72e3xa7mzesrpwbobr36s5x53yyvgr4rp47og2dmv.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_div_hardswish_backward_native_batch_norm_backward_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_hardswish_backward_native_batch_norm_backward_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 188160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 960
    x2 = (xindex // 47040)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp5 = tl.load(in_ptr1 + (x0 + (960*x2)), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = 49.0
    tmp7 = tmp5 / tmp6
    tmp8 = tmp0 / tmp3
    tmp9 = 0.5
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = tl.where(tmp4, tmp11, tmp7)
    tmp13 = 0.0
    tmp14 = tl.where(tmp2, tmp13, tmp12)
    tmp16 = 0.001
    tmp17 = tmp15 + tmp16
    tmp18 = tl.math.rsqrt(tmp17)
    tmp20 = tmp18 * tmp19
    tmp21 = tmp14 * tmp20
    tl.store(out_ptr0 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6q/c6q5scd56bup2dmm6quucyu6hgue56pxwzha5wi65glvd4xrembz.py
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
    size_hints=[256, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (7840*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a4/ca4jn5trbr4j2eda2dj2pwbgpdx3eitkn4zbj4qf4opglist6jsu.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 320
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 160
    x1 = (xindex // 160)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((49*x0) + (7840*(r2 // 49)) + (15680*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (160*r2) + (15680*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vk/cvkfrf7khabioyhwnionbmfre4ktbpghqkfvguvycsk36smag3qo.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 2],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_9', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 2
    RBLOCK: tl.constexpr = 2
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
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o6/co6n4uiwmqsgp2whphi5je333k2snfzsgkyfpujuy7gy463sdsxe.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 31360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 160
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = 0.001
    tmp3 = tmp1 + tmp2
    tmp4 = tl.math.rsqrt(tmp3)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 * tmp6
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e4/ce4klhfgs5isquob5zfhuark26cefz2u4k7iycvkhwk6x5vtjk2m.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]

triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_11', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3840
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
    tmp7 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = 0.16666666666666666
    tmp9 = tmp6 * tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp7, tmp9, tmp10)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ca/ccar7yy2wqeqszmvfpky7mj5dm4voxrjom5vhkz5zg6nunjqyojh.py
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
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (960 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (1920 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (2880 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zm/czm7a2tvazfv75buerzljrov7a5el6fs2p4mt72bn6ackxemz5te.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.threshold_backward]

triton_poi_fused_hardswish_backward_threshold_backward_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_backward_threshold_backward_13', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j7/cj7ze7rxtpamwewll7szi4pvbacwvp6winio4inufpp64amz62pr.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (240 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (480 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (720 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5g/c5gedivuz2qw3estxfuzutohitdzw3zirbpwmr3iintieqmui6zn.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 960
    x1 = (xindex // 960)
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp23 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (960*r2) + (94080*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((49*x0) + (47040*(r2 // 49)) + (94080*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (960*(r2 // 49)) + (1920*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (960*(r2 // 49)) + (1920*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp22 = tl.load(in_ptr4 + (x0 + (960*r2) + (94080*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp7 = tmp5 * tmp6
        tmp9 = 49.0
        tmp10 = tmp8 / tmp9
        tmp11 = tmp7 + tmp10
        tmp12 = tmp0 / tmp3
        tmp13 = 0.5
        tmp14 = tmp12 + tmp13
        tmp15 = tmp11 * tmp14
        tmp16 = tl.where(tmp4, tmp15, tmp11)
        tmp17 = 0.0
        tmp18 = tl.where(tmp2, tmp17, tmp16)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
        tmp24 = tmp22 - tmp23
        tmp25 = tmp18 * tmp24
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(rmask & xmask, tmp28, _tmp27)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp20, xmask)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z2/cz2ri7mwxtazwkfgwmeyxipbtstcshw6onixqrohiqbjq52mt5zg.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 196
    xnumel = 960
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 49
    y1 = (yindex // 49)
    tmp0 = tl.load(in_ptr0 + (x2 + (960*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (49*x2) + (47040*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (960*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (960*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp7 = tmp5 * tmp6
    tmp9 = 49.0
    tmp10 = tmp8 / tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = tmp0 / tmp3
    tmp13 = 0.5
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp4, tmp15, tmp11)
    tmp17 = 0.0
    tmp18 = tl.where(tmp2, tmp17, tmp16)
    tmp20 = 0.001
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp24 = tmp22 * tmp23
    tmp25 = tmp18 * tmp24
    tl.store(out_ptr0 + (x2 + (960*y3)), tmp25, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gw/cgwk5g6quu6isqczjqwtyuid57z35du2erdbsr5is4goqob5lkqh.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 960
    x1 = (xindex // 960)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp17 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (960*r2) + (94080*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((49*x0) + (47040*(r2 // 49)) + (94080*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr2 + (x0 + (960*r2) + (94080*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = tmp0 / tmp3
        tmp7 = 0.5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.where(tmp4, tmp9, tmp5)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, tmp10)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
        tmp18 = tmp16 - tmp17
        tmp19 = tmp12 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask & xmask, tmp22, _tmp21)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/br/cbr24vmecfgdw47jrfqt6ntrzwxba33f6nvwcd2iv3n27f7lxw6b.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 196
    xnumel = 960
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 49
    y1 = (yindex // 49)
    tmp0 = tl.load(in_ptr0 + (x2 + (960*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (49*x2) + (47040*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = tmp0 / tmp3
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp4, tmp9, tmp5)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp11, tmp10)
    tmp14 = 0.001
    tmp15 = tmp13 + tmp14
    tmp16 = tl.math.rsqrt(tmp15)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp12 * tmp18
    tl.store(out_ptr0 + (x2 + (960*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ds/cds6uju3h7324cnkrp676xnhpgj3ulpsbef7omeksgbmtmofwqft.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 31360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 160
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.math.rsqrt(tmp5)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp2 * tmp8
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xj/cxjqkjeepqhqrwkkb2x2fdewfdbg74dkiq5vlfl6wabie3r75bmv.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_20', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    r3 = rindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (7840*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (49*x0) + (7840*r2)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (x0 + (160*r3)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (r1 + (49*x0) + (7840*r2)), rmask & xmask, other=0.0)
    tmp21 = tl.load(in_ptr5 + (x0 + (160*r3)), rmask & xmask, other=0.0)
    tmp22 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp9 = tmp7 - tmp8
    tmp10 = tmp2 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp16 = tmp2 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp23 = tmp21 - tmp22
    tmp24 = tmp16 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp27 = tl.where(rmask & xmask, tmp25, 0)
    tmp28 = tl.sum(tmp27, 1)[:, None]
    tmp30 = 0.001
    tmp31 = tmp29 + tmp30
    tmp32 = tl.math.rsqrt(tmp31)
    tmp33 = tmp14 * tmp32
    tmp35 = tmp34 + tmp30
    tmp36 = tl.math.rsqrt(tmp35)
    tmp37 = tmp28 * tmp36
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp33, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp37, xmask)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7y/c7yni2anal4rumyk3gxkl55j73aegbj4sbbo63p4jfvsv6iqymfc.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 31360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 160
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(in_out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ao/cao3wxq5vsmtjqtucw4xfhng2kwisy23jhf4tklvztgpqqktryfs.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]

triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_22', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2688
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 672
    x1 = (xindex // 672)
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (672*r2) + (32928*x1)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = 0.16666666666666666
    tmp9 = tmp6 * tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp7, tmp9, tmp10)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q3/cq3l3wcia5je2of2apij334hvmpqb6sedhd5qmsf5xbpkqp7fsql.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (672 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (1344 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (2016 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b5/cb5lky4ew4wpjj3z4n6dqshj2wuvhnp7jim75iuqpwrs2dtmpdak.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.threshold_backward]

triton_poi_fused_hardswish_backward_threshold_backward_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_backward_threshold_backward_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pv/cpv256pbsqyzidlnfzryzxatuoiojcgoxzix3zfqbriiyoiljg4a.py
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
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (168 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (336 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (504 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/25/c25t5hlnxe5icq5b2sgudumaywbu5q7irsmgyiixijzixe3ffjs7.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1344
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 672
    x1 = (xindex // 672)
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp23 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (672*r2) + (65856*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((49*x0) + (32928*(r2 // 49)) + (65856*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (672*(r2 // 49)) + (1344*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (672*(r2 // 49)) + (1344*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp22 = tl.load(in_ptr4 + (x0 + (672*r2) + (65856*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp7 = tmp5 * tmp6
        tmp9 = 49.0
        tmp10 = tmp8 / tmp9
        tmp11 = tmp7 + tmp10
        tmp12 = tmp0 / tmp3
        tmp13 = 0.5
        tmp14 = tmp12 + tmp13
        tmp15 = tmp11 * tmp14
        tmp16 = tl.where(tmp4, tmp15, tmp11)
        tmp17 = 0.0
        tmp18 = tl.where(tmp2, tmp17, tmp16)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
        tmp24 = tmp22 - tmp23
        tmp25 = tmp18 * tmp24
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(rmask & xmask, tmp28, _tmp27)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp20, xmask)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/di/cdiuist2vm55kfc33my5w66sfwdgjyd6kk3cxl3efgfhkagrjfz7.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 672
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (672*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x7/cx7ty4nkrbqyzmcjj7txauwrrha3tx5ohlygbuol5idw57spzwy2.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_28', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 672
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (672*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pq/cpqb6xbkwdujcm2e33hx2hjysxoyogfam4x43xiprjg7obypehxb.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 196
    xnumel = 672
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 49
    y1 = (yindex // 49)
    tmp0 = tl.load(in_ptr0 + (x2 + (672*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (49*x2) + (32928*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (672*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (672*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp7 = tmp5 * tmp6
    tmp9 = 49.0
    tmp10 = tmp8 / tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = tmp0 / tmp3
    tmp13 = 0.5
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp4, tmp15, tmp11)
    tmp17 = 0.0
    tmp18 = tl.where(tmp2, tmp17, tmp16)
    tmp20 = 0.001
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp24 = tmp22 * tmp23
    tmp25 = tmp18 * tmp24
    tl.store(out_ptr0 + (x2 + (672*y3)), tmp25, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qm/cqm5hzh4mauuhaa752rmqrnnwz7sacwhkazhwnkq2wy5awlevjus.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (672*r2) + (75264*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((14*(((r2 + (112*x0)) // 14) % 14)) + (196*x1) + (131712*((r2 + (112*x0)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = tmp0 / tmp3
        tmp7 = 0.5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.where(tmp4, tmp9, tmp5)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, tmp10)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rt/crtebdxemmkdgfztsfoadyjwwgfh6z2ue4lyhefrjxmvesupes7z.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 672
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6g/c6gdeywtlmtxlrka7ighuwyyxxqgepgkjnv65jyuskxlnbr2oxl2.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 672
    x1 = (xindex // 672)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (672*r2) + (75264*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((14*(((r2 + (112*x1)) // 14) % 14)) + (196*x0) + (131712*((r2 + (112*x1)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr2 + (x0 + (672*r2) + (75264*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = tmp0 / tmp3
        tmp7 = 0.5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.where(tmp4, tmp9, tmp5)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, tmp10)
        tmp15 = tmp13 - tmp14
        tmp16 = tmp12 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yy/cyyltj72xneovc4yj2c6i5d3wlxnfcduqimem3suhsi3kqqqmk5y.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_33', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 672
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (672*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5h/c5hg44mk5ddcgsjz6twibbtflmcqlhjztlj3o4joiqeepf7ewb7y.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 784
    xnumel = 672
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
    tmp0 = tl.load(in_ptr0 + (x2 + (672*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (196*x2) + (131712*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = tmp0 / tmp3
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp4, tmp9, tmp5)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp11, tmp10)
    tmp14 = 0.001
    tmp15 = tmp13 + tmp14
    tmp16 = tl.math.rsqrt(tmp15)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp12 * tmp18
    tl.store(out_ptr0 + (x2 + (672*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lf/clffnq2bn4qjecxyubg2ybsogvkryu753r2sjloxzp7v2uuvh7fl.py
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
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 87808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 112
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = 0.001
    tmp3 = tmp1 + tmp2
    tmp4 = tl.math.rsqrt(tmp3)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 * tmp6
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pu/cpupuxwfd5rqos7mmv4joa7jus62mpkhgzjixhxvkgty2jva5fif.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_red_fused_mul_sum_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_36', 'mutated_arg_names': []}
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
    x1 = (xindex // 2) % 672
    x2 = (xindex // 1344)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (98*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (672*r3) + (65856*x0) + (131712*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3s/c3s7ullzb6ja2x3ft4qte7k5l3hanimc3bhwrmu7z6524rf3tou6.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]

triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_37', 'mutated_arg_names': ['in_out_ptr0']}
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 0.16666666666666666
    tmp7 = tmp4 * tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp5, tmp7, tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5q/c5qffdi2dy34aocuuks36o45xu6j2jloawlgzngplz7fbyi5vlkh.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 784
    xnumel = 672
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
    tmp0 = tl.load(in_ptr0 + (x2 + (672*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (196*x2) + (131712*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (672*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (672*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp7 = tmp5 * tmp6
    tmp9 = 196.0
    tmp10 = tmp8 / tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = tmp0 / tmp3
    tmp13 = 0.5
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp4, tmp15, tmp11)
    tmp17 = 0.0
    tmp18 = tl.where(tmp2, tmp17, tmp16)
    tmp20 = 0.001
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp24 = tmp22 * tmp23
    tmp25 = tmp18 * tmp24
    tl.store(out_ptr0 + (x2 + (672*y3)), tmp25, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7r/c7rbxu434r7h4whdtmgvkxnvna5ddtchs6wptm6g2mrm4o7rp73v.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_39', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 112
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 196
    r2 = (rindex // 196)
    x0 = xindex
    r3 = rindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (21952*r2)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (196*x0) + (21952*r2)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr2 + (x0 + (112*r3)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tmp0 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp13 = tmp11 - tmp12
    tmp14 = tmp6 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = 0.001
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp18 * tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp23, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/54/c54nfz6b574cigrckac6ytac4ex6w5ukkony4jommrrcpsdg5kuw.py
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
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 784
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((14*(((r2 + (112*x0)) // 14) % 14)) + (196*x1) + (21952*((r2 + (112*x0)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (112*r2) + (12544*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yt/cytb6hkkl3kgzdpo6zhyvvbmrp2gh62smkvkve4alvtqkbke3dfs.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_41', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 112
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
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yx/cyx44dsi3yyjcpljzg2254qjqavs4tddmwe5pbqelij6mglprrzr.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (672*r2) + (75264*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((14*(((r2 + (112*x0)) // 14) % 14)) + (196*x1) + (131712*((r2 + (112*x0)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x1 + (672*((r2 + (112*x0)) // 196))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x1 + (672*((r2 + (112*x0)) // 196))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp7 = tmp5 * tmp6
        tmp9 = 196.0
        tmp10 = tmp8 / tmp9
        tmp11 = tmp7 + tmp10
        tmp12 = tmp0 / tmp3
        tmp13 = 0.5
        tmp14 = tmp12 + tmp13
        tmp15 = tmp11 * tmp14
        tmp16 = tl.where(tmp4, tmp15, tmp11)
        tmp17 = 0.0
        tmp18 = tl.where(tmp2, tmp17, tmp16)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/su/csuq5gx7airm2u4upqs46pwrclrzwiy7snzxbo4xeeryhicpx6dv.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 672
    x1 = (xindex // 672)
    tmp20 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (672*r2) + (75264*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((14*(((r2 + (112*x1)) // 14) % 14)) + (196*x0) + (131712*((r2 + (112*x1)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (672*((r2 + (112*x1)) // 196))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (672*((r2 + (112*x1)) // 196))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr4 + (x0 + (672*r2) + (75264*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp7 = tmp5 * tmp6
        tmp9 = 196.0
        tmp10 = tmp8 / tmp9
        tmp11 = tmp7 + tmp10
        tmp12 = tmp0 / tmp3
        tmp13 = 0.5
        tmp14 = tmp12 + tmp13
        tmp15 = tmp11 * tmp14
        tmp16 = tl.where(tmp4, tmp15, tmp11)
        tmp17 = 0.0
        tmp18 = tl.where(tmp2, tmp17, tmp16)
        tmp21 = tmp19 - tmp20
        tmp22 = tmp18 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o4/co42a7jqkrfvume25snmngaruqfhhg3k5gv6nqlt5v4ekllzgzcj.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 672
    x1 = (xindex // 672)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (672*r2) + (75264*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((14*(((r2 + (112*x1)) // 14) % 14)) + (196*x0) + (131712*((r2 + (112*x1)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr2 + (x0 + (672*r2) + (75264*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = tmp0 / tmp3
        tmp7 = 0.5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.where(tmp4, tmp9, tmp5)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, tmp10)
        tmp15 = tmp13 - tmp14
        tmp16 = tmp12 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/po/cpoisjbcdmam7qdvbxkdy2gzoyd2bckrqa2p22v5rmcfagpqxz32.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_45', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 87808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 112
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.math.rsqrt(tmp5)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp2 * tmp8
    tl.store(in_out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rt/crt35cbmk33xwhbgg6jye2bu7rxbseepnepx4frdutt2y6ftccec.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_red_fused_mul_sum_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 2
    x1 = (xindex // 2) % 480
    x2 = (xindex // 960)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (98*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (480*r3) + (47040*x0) + (94080*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sx/csxjl5hxbig4ne2glxtsnvqcm7r42xijhml3vuw2kty2xqnkviw6.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]

triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_47', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 0.16666666666666666
    tmp7 = tmp4 * tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp5, tmp7, tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qh/cqhzqy3dwqpue46aftbv6747tb6rvwl7tbobvz2ezrc54ve3wtc5.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (480 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (960 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (1440 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z2/cz2saopfo2zzwulnji5cuywyw76e7j2gxjloq5kmtjocvrqmvqmp.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.threshold_backward]

triton_poi_fused_hardswish_backward_threshold_backward_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_backward_threshold_backward_49', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lg/clgsaq6jimu7fvrrfpxhvwmtq7ezmjdxpwdfa7hlaqyj3w2j7yag.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_50 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (120 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (240 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (360 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b7/cb7htpkolid7vkcdtgp7b6uleykw7zpawhvsz3t3tpv46vmj3jrk.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3360
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (480*r2) + (53760*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((14*(((r2 + (112*x0)) // 14) % 14)) + (196*x1) + (94080*((r2 + (112*x0)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x1 + (480*((r2 + (112*x0)) // 196))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x1 + (480*((r2 + (112*x0)) // 196))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp7 = tmp5 * tmp6
        tmp9 = 196.0
        tmp10 = tmp8 / tmp9
        tmp11 = tmp7 + tmp10
        tmp12 = tmp0 / tmp3
        tmp13 = 0.5
        tmp14 = tmp12 + tmp13
        tmp15 = tmp11 * tmp14
        tmp16 = tl.where(tmp4, tmp15, tmp11)
        tmp17 = 0.0
        tmp18 = tl.where(tmp2, tmp17, tmp16)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/in/cinsdzghznvsiheykqwndltmlltp7pn5mndmsdotr2hzregezscs.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_52', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 480
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dx/cdx6ty457ieemdcsayirzg3j6no7itlfqgpxyg3cpzvgy4ie7ptp.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3360
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 480
    x1 = (xindex // 480)
    tmp20 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (480*r2) + (53760*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((14*(((r2 + (112*x1)) // 14) % 14)) + (196*x0) + (94080*((r2 + (112*x1)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (480*((r2 + (112*x1)) // 196))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (480*((r2 + (112*x1)) // 196))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr4 + (x0 + (480*r2) + (53760*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp7 = tmp5 * tmp6
        tmp9 = 196.0
        tmp10 = tmp8 / tmp9
        tmp11 = tmp7 + tmp10
        tmp12 = tmp0 / tmp3
        tmp13 = 0.5
        tmp14 = tmp12 + tmp13
        tmp15 = tmp11 * tmp14
        tmp16 = tl.where(tmp4, tmp15, tmp11)
        tmp17 = 0.0
        tmp18 = tl.where(tmp2, tmp17, tmp16)
        tmp21 = tmp19 - tmp20
        tmp22 = tmp18 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6h/c6hkljkj3xeddacczye4sfnn6nseeaiwpv22ktakbkhbcctg3dzx.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_54', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 480
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (480*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/75/c75utwe6wsynvdrvqpdyjtatvshhlv7aw7azq5pdzmbfwjj4ebtg.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 784
    xnumel = 480
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
    tmp0 = tl.load(in_ptr0 + (x2 + (480*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (196*x2) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (480*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (480*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp7 = tmp5 * tmp6
    tmp9 = 196.0
    tmp10 = tmp8 / tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = tmp0 / tmp3
    tmp13 = 0.5
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 * tmp14
    tmp16 = tl.where(tmp4, tmp15, tmp11)
    tmp17 = 0.0
    tmp18 = tl.where(tmp2, tmp17, tmp16)
    tmp20 = 0.001
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp24 = tmp22 * tmp23
    tmp25 = tmp18 * tmp24
    tl.store(out_ptr0 + (x2 + (480*y3)), tmp25, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fo/cfori22xqw5zwwgri7p3moyhufb4gsf3nczy6wnogvswdnkhemyb.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_56', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3360
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (480*r2) + (53760*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((14*(((r2 + (112*x0)) // 14) % 14)) + (196*x1) + (94080*((r2 + (112*x0)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = tmp0 / tmp3
        tmp7 = 0.5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.where(tmp4, tmp9, tmp5)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, tmp10)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hh/chheqtsr2223tffgiva4dyq2scfqzcugyrwoseefsn2ar2uzw7nv.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_57', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3360
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 480
    x1 = (xindex // 480)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (480*r2) + (53760*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((14*(((r2 + (112*x1)) // 14) % 14)) + (196*x0) + (94080*((r2 + (112*x1)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr2 + (x0 + (480*r2) + (53760*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = tmp0 / tmp3
        tmp7 = 0.5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.where(tmp4, tmp9, tmp5)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, tmp10)
        tmp15 = tmp13 - tmp14
        tmp16 = tmp12 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/es/ceszbjcb4u25bsr57lm6st3pcoawjey7pch7dgncoiyz7ezx2snc.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_58 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_58', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 784
    xnumel = 480
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
    tmp0 = tl.load(in_ptr0 + (x2 + (480*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (196*x2) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = tmp0 / tmp3
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp4, tmp9, tmp5)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp11, tmp10)
    tmp14 = 0.001
    tmp15 = tmp13 + tmp14
    tmp16 = tl.math.rsqrt(tmp15)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp12 * tmp18
    tl.store(out_ptr0 + (x2 + (480*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5s/c5scfzrg3m5e5g3bjas4ucoif7d6mgkdsmzmw5mt4p6tsx4lgs4v.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_59 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_59', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 80
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 196
    r2 = (rindex // 196)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (15680*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3m/c3mj4gokqqgrrz72epfwdz2g7fooi7rvx4qw3j66zebjcro4wfqv.py
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
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_60', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 560
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((14*(((r2 + (112*x0)) // 14) % 14)) + (196*x1) + (15680*((r2 + (112*x0)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (80*r2) + (8960*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ro/crogqcupgatudvffhn7aphr73v437v5uoe2ic6ppruhh7bqflprs.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_61 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_61', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 80
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
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qi/cqi7tycwqsaq672sxvzask2he7gufugnubpenz447tsnjpuf7k4u.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_62 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_62', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 62720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 80
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = 0.001
    tmp3 = tmp1 + tmp2
    tmp4 = tl.math.rsqrt(tmp3)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 * tmp6
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qu/cquywnm5qzp5hrmatlqnn4zv5iixz2kwtdon4cmobnl6agrkwxww.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_63 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_63', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1288
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (184*r2) + (20608*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((14*(((r2 + (112*x0)) // 14) % 14)) + (196*x1) + (36064*((r2 + (112*x0)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = tmp0 / tmp3
        tmp7 = 0.5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.where(tmp4, tmp9, tmp5)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, tmp10)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wg/cwglhhgwgn7tra2dlvayo2vaigmafh4civsx4dyd6z5bs74vw5ij.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_64 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_64', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 184
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vl/cvlh3ptfrkgi3fuhg7wpfck3kiitmv3evep74rm7jyvny46dizkx.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_65 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_65', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1288
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 184
    x1 = (xindex // 184)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (184*r2) + (20608*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((14*(((r2 + (112*x1)) // 14) % 14)) + (196*x0) + (36064*((r2 + (112*x1)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr2 + (x0 + (184*r2) + (20608*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = tmp0 / tmp3
        tmp7 = 0.5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.where(tmp4, tmp9, tmp5)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, tmp10)
        tmp15 = tmp13 - tmp14
        tmp16 = tmp12 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zo/czogv4a7nlht2cpwhjwcwx5wb3b5gfctn72zfjmhyng6f4xcipxf.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_66 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_66', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 184
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (184*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ok/cokihw5vsaau7anqcqma7tvirqssamuu3ck3wkpmkarrcgbi2h2o.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_67 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_67', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 784
    xnumel = 184
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
    tmp0 = tl.load(in_ptr0 + (x2 + (184*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (196*x2) + (36064*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = tmp0 / tmp3
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp4, tmp9, tmp5)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp11, tmp10)
    tmp14 = 0.001
    tmp15 = tmp13 + tmp14
    tmp16 = tl.math.rsqrt(tmp15)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp12 * tmp18
    tl.store(out_ptr0 + (x2 + (184*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c3/cc3fncjnfy4yn3m3aeanmjdpmjnn4fby3gvaqsapve7ipydraz36.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_68 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_68', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 62720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 80
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.math.rsqrt(tmp5)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp2 * tmp8
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ic/cicfph36lt5agv3fn6swnpd4hfqy7gpukw2qyz4qrkdceetp73t5.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_69 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_69', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 62720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 80
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp3 = tl.load(in_ptr2 + (x3), xmask)
    tmp5 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vq/cvqeac4clkpywl3mgdro6hsqhbpq6zz7mefo6r5wibnymjjh5d42.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_70 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_70', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 784
    xnumel = 200
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
    tmp0 = tl.load(in_ptr0 + (x2 + (200*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (196*x2) + (39200*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = tmp0 / tmp3
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp4, tmp9, tmp5)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp11, tmp10)
    tmp14 = 0.001
    tmp15 = tmp13 + tmp14
    tmp16 = tl.math.rsqrt(tmp15)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp12 * tmp18
    tl.store(out_ptr0 + (x2 + (200*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ex/cexetb2h5d45smywvrol7mm7dgaecbfdoj57bpcrpjyzyd6zbec2.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_71 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: 'i32', 20: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(19, 20))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_71', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 80
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 196
    r2 = (rindex // 196)
    x0 = xindex
    r3 = rindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (15680*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (15680*r2)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (x0 + (80*r3)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (r1 + (196*x0) + (15680*r2)), rmask & xmask, other=0.0)
    tmp21 = tl.load(in_ptr5 + (x0 + (80*r3)), rmask & xmask, other=0.0)
    tmp22 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr7 + (r1 + (196*x0) + (15680*r2)), rmask & xmask, other=0.0)
    tmp35 = tl.load(in_ptr8 + (x0 + (80*r3)), rmask & xmask, other=0.0)
    tmp36 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr10 + (x0), xmask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr11 + (x0), xmask, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr12 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp9 = tmp7 - tmp8
    tmp10 = tmp2 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = tmp2 + tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp23 = tmp21 - tmp22
    tmp24 = tmp16 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp27 = tl.where(rmask & xmask, tmp25, 0)
    tmp28 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp30 = tmp16 + tmp29
    tmp31 = tl.broadcast_to(tmp30, [RBLOCK])
    tmp33 = tl.where(rmask & xmask, tmp31, 0)
    tmp34 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp37 = tmp35 - tmp36
    tmp38 = tmp30 * tmp37
    tmp39 = tl.broadcast_to(tmp38, [RBLOCK])
    tmp41 = tl.where(rmask & xmask, tmp39, 0)
    tmp42 = triton_helpers.promote_to_tensor(tl.sum(tmp41, 0))
    tmp44 = 0.001
    tmp45 = tmp43 + tmp44
    tmp46 = tl.math.rsqrt(tmp45)
    tmp47 = tmp14 * tmp46
    tmp49 = tmp48 + tmp44
    tmp50 = tl.math.rsqrt(tmp49)
    tmp51 = tmp28 * tmp50
    tmp53 = tmp52 + tmp44
    tmp54 = tl.math.rsqrt(tmp53)
    tmp55 = tmp42 * tmp54
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp47, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp51, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr2 + (x0), tmp55, xmask)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp20, xmask)
    tl.store(out_ptr2 + (x0), tmp34, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tg/ctgh52jeytll66u3zhpnfmgtk63rl333gw7ya44aj7obnilts73n.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_72 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_72', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1288
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 184
    x1 = (xindex // 184)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (184*r2) + (20608*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((14*(((r2 + (112*x1)) // 14) % 14)) + (196*x0) + (36064*((r2 + (112*x1)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr2 + (x0 + (184*r2) + (20608*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = tmp0 / tmp3
        tmp7 = 0.5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.where(tmp4, tmp9, tmp5)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, tmp10)
        tmp15 = tmp13 - tmp14
        tmp16 = tmp12 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yc/cyck3ug4vhs3duzusxeoxnf35bqnzbvxdu4dzvzcnlhmkgi5poui.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_73 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_73', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1400
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (200*r2) + (22400*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((14*(((r2 + (112*x0)) // 14) % 14)) + (196*x1) + (39200*((r2 + (112*x0)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = tmp0 / tmp3
        tmp7 = 0.5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.where(tmp4, tmp9, tmp5)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, tmp10)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7s/c7s3jekhkvvhua3v3n5ojzf6og5iri5igtb2mrqjd65gfdhgi3or.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_74 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_74', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 200
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2o/c2omkm4cn3tm3gck7kofjbqxryv75p5pzmdeqlwyigowgtmfjnua.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_75 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_75', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1400
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 200
    x1 = (xindex // 200)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (200*r2) + (22400*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((14*(((r2 + (112*x1)) // 14) % 14)) + (196*x0) + (39200*((r2 + (112*x1)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr2 + (x0 + (200*r2) + (22400*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = tmp0 / tmp3
        tmp7 = 0.5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.where(tmp4, tmp9, tmp5)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, tmp10)
        tmp15 = tmp13 - tmp14
        tmp16 = tmp12 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j6/cj6qffrln4cwf5drs4pahznn6ozgppko72nmcq2bsbvv4zvsudz6.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_76 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_76', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 200
    rnumel = 7
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (200*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3h/c3hiitesfj3sktf2nz5lc6in3knzjeavi72fsxpcuiuk45lpxmmu.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_77 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_77', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 62720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 80
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp5 = tl.load(in_ptr2 + (x3), xmask)
    tmp7 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = 0.001
    tmp9 = tmp7 + tmp8
    tmp10 = tl.math.rsqrt(tmp9)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 * tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/43/c43cyf75pgnueoc4vl3hmj5dikmelykdtclcsom2qmxi6kj4kryl.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_78 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_78', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1680
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (240*r2) + (26880*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((14*(((r2 + (112*x0)) // 14) % 14)) + (196*x1) + (47040*((r2 + (112*x0)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = tmp0 / tmp3
        tmp7 = 0.5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.where(tmp4, tmp9, tmp5)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, tmp10)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/au/caugdf5hoxzfvonakdtxlg7bpg5ukklwhteaxtzcu4yo75qojdl5.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_79 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_79', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ts/ctsz2ekutmxunwx5e6dbdxei4pdawvdseixybvpjvb5eydoxmdxd.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_80 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_80', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1680
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 240
    x1 = (xindex // 240)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (240*r2) + (26880*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((14*(((r2 + (112*x1)) // 14) % 14)) + (196*x0) + (47040*((r2 + (112*x1)) // 196)) + (r2 % 14)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr2 + (x0 + (240*r2) + (26880*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = tmp0 / tmp3
        tmp7 = 0.5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.where(tmp4, tmp9, tmp5)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, tmp10)
        tmp15 = tmp13 - tmp14
        tmp16 = tmp12 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/je/cje7xtzbd3j33dm6sr36rjvuux6i2b34bzwecmojau5boxtjuhzi.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_81 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_81', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    rnumel = 7
    RBLOCK: tl.constexpr = 8
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
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ov/covgfcdudyykxksd6hl5nclju6fat2gbvl7gzeguwmuaxqyrwikv.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_82 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_82', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 784
    xnumel = 240
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
    tmp0 = tl.load(in_ptr0 + (x2 + (240*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (196*x2) + (47040*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = tmp0 / tmp3
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp4, tmp9, tmp5)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp11, tmp10)
    tmp14 = 0.001
    tmp15 = tmp13 + tmp14
    tmp16 = tl.math.rsqrt(tmp15)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp12 * tmp18
    tl.store(out_ptr0 + (x2 + (240*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ck/ccko2rw4roqhsfnjmsw67dafkak4zdcxxpkbql5oyb4vzcqsuyv6.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_83 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_83', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6000
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 25
    x1 = (xindex // 25)
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x0)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (240*((r2 + (126*x0)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + ((784*x1) + (188160*(((r2 + (126*x0)) // 784) % 4)) + ((r2 + (126*x0)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp3 / tmp6
        tmp10 = 0.5
        tmp11 = tmp9 + tmp10
        tmp12 = tmp8 * tmp11
        tmp13 = tl.where(tmp7, tmp12, tmp8)
        tmp14 = 0.0
        tmp15 = tl.where(tmp5, tmp14, tmp13)
        tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
        tmp17 = tl.where(tmp2, tmp15, tmp16)
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ff/cffsyczwusryxwub5ztcedg5y7fmms4dokwss5a6defaxqsrpefg.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_84 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_84', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (25*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3u/c3umtwgp5rx44kcxc7e2xhqnaucmbsxk3m2i6p6xksz2cr53dvtg.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_85 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_85', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6000
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 240)
    x0 = xindex % 240
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (240*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + ((784*x0) + (188160*(((r2 + (126*x1)) // 784) % 4)) + ((r2 + (126*x1)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp3 / tmp6
        tmp10 = 0.5
        tmp11 = tmp9 + tmp10
        tmp12 = tmp8 * tmp11
        tmp13 = tl.where(tmp7, tmp12, tmp8)
        tmp14 = 0.0
        tmp15 = tl.where(tmp5, tmp14, tmp13)
        tmp16 = tl.load(in_ptr2 + (x0 + (240*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tmp16 - tmp17
        tmp19 = tmp15 * tmp18
        tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
        tmp21 = tl.where(tmp2, tmp19, tmp20)
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ej/cejyylfelh6fedr6as6ig3bk5xptdl627lnqvkcif4qmz63lqqqr.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_86 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_86', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    rnumel = 25
    RBLOCK: tl.constexpr = 32
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
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6u/c6u7os2iiwmjqfht5565mzvlyahtzuacq2xd3jmhhvkytenjwdd2.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_87 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_87', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3136
    xnumel = 240
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
    tmp0 = tl.load(in_ptr0 + (x2 + (240*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (y0 + (784*x2) + (188160*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = tmp0 / tmp3
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp4, tmp9, tmp5)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp11, tmp10)
    tmp14 = 0.001
    tmp15 = tmp13 + tmp14
    tmp16 = tl.math.rsqrt(tmp15)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp12 * tmp18
    tl.store(out_ptr0 + (x2 + (240*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l5/cl5mwponmwqqkd6hepmdx5dqbgogjmzpqqukj7hhpejjpi6cnz63.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_88 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_88', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 40
    rnumel = 3136
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (31360*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/je/cjer65x75oryydf4b6fier6mf2kv2xcl7fbzqb36r5zulxbbkufd.py
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
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_89', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1000
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 25
    x1 = (xindex // 25)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x0)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((784*x1) + (31360*(((r2 + (126*x0)) // 784) % 4)) + ((r2 + (126*x0)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (40*((r2 + (126*x0)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/gu/cgukqumqf3ejuhxaszpyfflbosgov4qzgvgdgzmfhtcdrgb5icdy.py
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
    size_hints=[64, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_90', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 40
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (25*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/35/c35qrzi7lunpssuyhhxkfjj4ni5ax3mhqxdu6xgvtqshmijb7hrv.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_91 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_91', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 40
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = 0.001
    tmp3 = tmp1 + tmp2
    tmp4 = tl.math.rsqrt(tmp3)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 * tmp6
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zq/czqaarqothucb3sk5lnelyfdgmvn35at42xdimgaqakzvz3ghtmm.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_red_fused_mul_sum_92 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_92', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3360
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 7
    x1 = (xindex // 7) % 120
    x2 = (xindex // 840)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (112*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (120*r3) + (13440*x0) + (94080*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nb/cnbonlba7zolgkqynd5ko7zjqx6csyb7mwydng447nxvxqw6x445.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]

triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_93 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_93', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 480
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 0.16666666666666666
    tmp7 = tmp4 * tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp5, tmp7, tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lp/clppfus6g6ooal4ewskau6qrqhv4osdzqxbtcrr7oiake3l7x3uk.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.threshold_backward]

triton_poi_fused_hardswish_backward_threshold_backward_94 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_backward_threshold_backward_94', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jn/cjnrezh4u6wj5gpgozonbnefgpxz653a6c5rosskeg3i4kh5y237.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_95 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_95', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (32 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (64 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (96 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gg/cgguwkqw7meujc7aekhs3y7ebb3h5x5ssy52kfrljxbit6kqargj.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_96 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_96', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3000
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 25
    x1 = (xindex // 25)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x0)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (120*((r2 + (126*x0)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((784*x1) + (94080*(((r2 + (126*x0)) // 784) % 4)) + ((r2 + (126*x0)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr2 + (x1 + (120*(((r2 + (126*x0)) // 784) % 4))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 * tmp7
        tmp9 = tl.load(in_ptr3 + (x1 + (120*(((r2 + (126*x0)) // 784) % 4))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = 784.0
        tmp11 = tmp9 / tmp10
        tmp12 = tmp8 + tmp11
        tmp13 = tl.where(tmp5, tmp4, tmp12)
        tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
        tmp15 = tl.where(tmp2, tmp13, tmp14)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/la/clabvn6t4hr5xa4sfbu5lhrpjrpv7go6koucmkbhebmlo2etnklp.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_97 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_97', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 120
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (25*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ja/cja3ie75r5tookv4u7sqhn4cgtowz5p5twkjn5ge46czgtt2krxz.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_98 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_98', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3000
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 120)
    x0 = xindex % 120
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (120*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((784*x0) + (94080*(((r2 + (126*x1)) // 784) % 4)) + ((r2 + (126*x1)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr2 + (x0 + (120*(((r2 + (126*x1)) // 784) % 4))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 * tmp7
        tmp9 = tl.load(in_ptr3 + (x0 + (120*(((r2 + (126*x1)) // 784) % 4))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = 784.0
        tmp11 = tmp9 / tmp10
        tmp12 = tmp8 + tmp11
        tmp13 = tl.where(tmp5, tmp4, tmp12)
        tmp14 = tl.load(in_ptr4 + (x0 + (120*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp13 * tmp16
        tmp18 = tl.full(tmp17.shape, 0, tmp17.dtype)
        tmp19 = tl.where(tmp2, tmp17, tmp18)
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask & xmask, tmp22, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hr/chreiebwnlnsnnmauoa5lwdcqt2l2wx5jk7pzuvkvgo2ww4oykfc.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_99 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_99', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 120
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (120*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4a/c4atiwzwqodimiziatjydk35itkb7uxryfrm55xass37crpc26uo.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_100 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_100', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3136
    xnumel = 120
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
    tmp0 = tl.load(in_ptr0 + (x2 + (120*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (784*x2) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (120*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (120*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 * tmp4
    tmp7 = 784.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp12 = 0.001
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp10 * tmp16
    tl.store(out_ptr0 + (x2 + (120*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ug/cugcwpt4hpo3jlfckvnr4j4xbfcsb4mdkha33huj75z4kiki5oew.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_101 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_101', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3000
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 25
    x1 = (xindex // 25)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x0)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (120*((r2 + (126*x0)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((784*x1) + (94080*(((r2 + (126*x0)) // 784) % 4)) + ((r2 + (126*x0)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4x/c4xeuk4zgjvmnebhkxcqcks5h6pnmuow6qzpgofyuhzh5er7fey3.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_102 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_102', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3000
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 120)
    x0 = xindex % 120
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (120*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((784*x0) + (94080*(((r2 + (126*x1)) // 784) % 4)) + ((r2 + (126*x1)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.load(in_ptr2 + (x0 + (120*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/mj/cmjho7quur7omfmf5ossorz56uzzf5xxhxfnsxq3b6lnpvsliqu3.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_103 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_103', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3136
    xnumel = 120
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
    tmp0 = tl.load(in_ptr0 + (x2 + (120*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (784*x2) + (94080*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x2 + (120*y3)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zf/czfdo5nks7xszvz6yv7zv7hl57bngzqeu62tuv4n4jzeybmqnwa5.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_104 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_104', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 40
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x3), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.math.rsqrt(tmp5)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp2 * tmp8
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/44/c44m2onb3mz2fw5djnjuwuehpzdfcw3rcf5ybm4nqyhlvrovpyzo.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_105 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_105', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 40
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (31360*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (784*x0) + (31360*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (40*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr4 + (r1 + (784*x0) + (31360*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr5 + (x0 + (40*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
        tmp14 = tmp2 + tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
        tmp20 = tmp18 - tmp19
        tmp21 = tmp14 * tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp16, xmask)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tmp25 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = 0.001
    tmp27 = tmp25 + tmp26
    tmp28 = tl.math.rsqrt(tmp27)
    tmp29 = tmp11 * tmp28
    tmp31 = tmp30 + tmp26
    tmp32 = tl.math.rsqrt(tmp31)
    tmp33 = tmp23 * tmp32
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp29, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp33, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6f/c6f4u56gs347w56qf77vokik4dv5vsn4hunz5suasaxop75cbu4m.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_106 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_106', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3000
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 120)
    x0 = xindex % 120
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (120*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((784*x0) + (94080*(((r2 + (126*x1)) // 784) % 4)) + ((r2 + (126*x1)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr2 + (x0 + (120*(((r2 + (126*x1)) // 784) % 4))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tmp6 * tmp7
        tmp9 = tl.load(in_ptr3 + (x0 + (120*(((r2 + (126*x1)) // 784) % 4))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = 784.0
        tmp11 = tmp9 / tmp10
        tmp12 = tmp8 + tmp11
        tmp13 = tl.where(tmp5, tmp4, tmp12)
        tmp14 = tl.load(in_ptr4 + (x0 + (120*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp13 * tmp16
        tmp18 = tl.full(tmp17.shape, 0, tmp17.dtype)
        tmp19 = tl.where(tmp2, tmp17, tmp18)
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask & xmask, tmp22, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c3/cc3ipds7vkm527wjnidxjldvzwbviqy2zbwr6og3e6fkjkql7khd.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_107 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_107', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3000
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 120)
    x0 = xindex % 120
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (120*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((784*x0) + (94080*(((r2 + (126*x1)) // 784) % 4)) + ((r2 + (126*x1)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.load(in_ptr2 + (x0 + (120*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/nn/cnn6jkijnsfswvhlghdwb6za3xm3hfjtl2mxrddltjplmaploskz.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_108 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_108', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 40
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp5 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(in_out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/54/c54g3axcamx527nalpuuddbobokznm7ggjat6og675d6faa2mck3.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_red_fused_mul_sum_109 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_109', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2016
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 7
    x1 = (xindex // 7) % 72
    x2 = (xindex // 504)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (112*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (72*r3) + (8064*x0) + (56448*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tl/ctlvlbbhi76p6pq6kpn5nnlzdg6lnaciqi46te6hc5s5mz7dknhe.py
# Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]

triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_110 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_110', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 288
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 0.16666666666666666
    tmp7 = tmp4 * tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp5, tmp7, tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y2/cy234xtvjywlf44dux6ah7f7oi35ego7fa4zw7n2kqptboncqqbb.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_111 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_111', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 72
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (72 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (144 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (216 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xk/cxkcbdlwv2653wk4skvjpsxbpp4ejegqu5n32ubth2bh3v2jqqje.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.threshold_backward]

triton_poi_fused_hardswish_backward_threshold_backward_112 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_backward_threshold_backward_112', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qx/cqx6xeh4ublhz7pe25c3cgbs32jcegzmbzd5nswz6o4rygkkfgjc.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_poi_fused_convolution_backward_113 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_113', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (24 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (48 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (72 + x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vy/cvyku7gc64f5z3nbjhae3nu4tqoyw4s2k4dkorcgeffm3dri7hlc.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_114 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_114', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1800
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 25
    x1 = (xindex // 25)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x0)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x1 + (72*((r2 + (126*x0)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((784*x1) + (56448*(((r2 + (126*x0)) // 784) % 4)) + ((r2 + (126*x0)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr2 + (x1 + (72*(((r2 + (126*x0)) // 784) % 4))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 * tmp7
        tmp9 = tl.load(in_ptr3 + (x1 + (72*(((r2 + (126*x0)) // 784) % 4))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = 784.0
        tmp11 = tmp9 / tmp10
        tmp12 = tmp8 + tmp11
        tmp13 = tl.where(tmp5, tmp4, tmp12)
        tmp14 = tl.full(tmp13.shape, 0, tmp13.dtype)
        tmp15 = tl.where(tmp2, tmp13, tmp14)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oh/cohlq7tvh4pt6dg52wbqpuojmvmtpptoiva47qgqclw6emiij5fi.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_115 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_115', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 72
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (25*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ss/cssyigvoj3fjcdmrwadfipjceih6vtjymu7o4kra77c4lvy6nyvn.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_116 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_116', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1800
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 72)
    x0 = xindex % 72
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 3136, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (72*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((784*x0) + (56448*(((r2 + (126*x1)) // 784) % 4)) + ((r2 + (126*x1)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr2 + (x0 + (72*(((r2 + (126*x1)) // 784) % 4))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 * tmp7
        tmp9 = tl.load(in_ptr3 + (x0 + (72*(((r2 + (126*x1)) // 784) % 4))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = 784.0
        tmp11 = tmp9 / tmp10
        tmp12 = tmp8 + tmp11
        tmp13 = tl.where(tmp5, tmp4, tmp12)
        tmp14 = tl.load(in_ptr4 + (x0 + (72*((r2 + (126*x1)) % 3136))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp13 * tmp16
        tmp18 = tl.full(tmp17.shape, 0, tmp17.dtype)
        tmp19 = tl.where(tmp2, tmp17, tmp18)
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask & xmask, tmp22, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t2/ct2ixv7g63ydk6pwbotk3x3v3xmg4vwcvshsss5i6xudwg6nim2a.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_117 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_117', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 72
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (72*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3c/c3cd3tgp5ywg7wz45uaabzs7gnpufbflr6s462mpvt3vaxfkmw6k.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_118 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_118', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3136
    xnumel = 72
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
    tmp0 = tl.load(in_ptr0 + (x2 + (72*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (784*x2) + (56448*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (72*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2 + (72*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 * tmp4
    tmp7 = 784.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.where(tmp2, tmp1, tmp9)
    tmp12 = 0.001
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp10 * tmp16
    tl.store(out_ptr0 + (x2 + (72*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cb/ccba3mjqfavzzcheewnknojituvw3rczwd6jkkl2imi7ydgzomnx.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_119 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_119', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7056
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 98
    x1 = (xindex // 98)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (72*r2) + (9216*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x1) + (225792*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6t/c6tt4d6gslo2cal5mscmsokdb6ak25wi6s2qv35rwryd7w22ngrb.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_120 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_120', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 72
    rnumel = 98
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (98*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7i/c7ie5ywmlalgmq6boref2j42v7p2z7o6bzfpb6in2ljcz672mtcy.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_121 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_121', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7056
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 72
    x1 = (xindex // 72)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (72*r2) + (9216*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (225792*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (72*r2) + (9216*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/mz/cmzlzlawb76oqswqztyqe2icpji66h54jxtchx6wvy3wtpxn4vzr.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_122 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_122', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 72
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x0 + (72*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = 0.001
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)
    tmp8 = tmp2 * tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/45/c45yda3uu6i5zjbaprftfqrxpkwimtmlif43tnuboqu47wsbyl3v.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_123 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_123', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12544
    xnumel = 72
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
    tmp0 = tl.load(in_ptr0 + (x2 + (72*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (3136*x2) + (225792*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x2 + (72*y3)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fu/cfurrgwysrvwngjdvejyyoldrbosbevsta7fwo43jidexc6p7oga.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_124 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_124', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 24
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp2 = 0.001
    tmp3 = tmp1 + tmp2
    tmp4 = tl.math.rsqrt(tmp3)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 * tmp6
    tl.store(out_ptr0 + (x3), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2x/c2xdgwhuxqadux4iyrie7nhs72qrw3vig2fvc4kptz2wjummwaav.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_125 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_125', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 48
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 24
    x1 = (xindex // 24)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (75264*(r2 // 3136)) + (150528*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + ((3136*x0) + (75264*(r2 // 3136)) + (150528*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr2 + (x0 + (24*r2) + (150528*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp5 = tmp0 + tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp5 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp7, xmask)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t7/ct7kw6tofaebxprwct2uqhyowyzhuefxmlovxhpgkaah2orqejgp.py
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
    size_hints=[32, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_126', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (24*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6l/c6lqm2pscnruk6w5t66bj7lyylt43q4npcb2rhc3cj734tmgfenn.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_127 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_127', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2352
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 98
    x1 = (xindex // 98)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x1) + (75264*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (24*r2) + (3072*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gc/cgccdx3w7rbcig4i7kbgtxno7op72cwogcsvoj6gizd46iy7mbtk.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_128 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_128', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 98
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (98*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pj/cpjhbqpca4tserwbuxxsmrww7tuuyxiydzugzmqcpvmaeyyhav6z.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_129 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_129', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7056
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 72
    x1 = (xindex // 72)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (72*r2) + (9216*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (225792*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (72*r2) + (9216*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/3i/c3ipay3q7cdvej7gj3udz4kj65ebp726au4flojmwgvhpin5lo54.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_130 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_130', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (24*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6m/c6mlidejmmyjihq2ncoj32uq6lvlovlhhbglyr6wqvyt4jvsyoib.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_131 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_131', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 24
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = tl.math.rsqrt(tmp5)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp2 * tmp8
    tl.store(in_out_ptr0 + (x3), tmp9, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bp/cbpzsiapxnvy6tyt2lkoku5tuzt6hykaxhnicar26dmazyfz4m7d.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_132 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_132', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 98
    x1 = (xindex // 98)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x1) + (200704*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nw/cnwz5kxn6rrkhodveen5656zylqrafect32eqnbs7friyyhpfnxf.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_133 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_133', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 98
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (98*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7u/c7ud2zfkoh2skiw5be2etx447vbh3un25issp7ngxvuurxcytucn.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_134 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_134', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((3136*x0) + (200704*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/3l/c3ldagkr2n4attrgphshcwvhrgvylpoothspox6z3vq5u44fuk2l.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_135 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_135', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 98
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
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = 0.001
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)
    tmp8 = tmp2 * tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zv/czvme5gobomajbctfrqeyykoqgnvvrsuovkqey4yeyxbbxztdipq.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_136 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_136', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12544
    xnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (3136*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dr/cdr2oxgkaxbmwp34zd6uuy5jtupr5glrbz4ahgzxw4rfqro4rifu.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_137 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_137', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 392
    x1 = (xindex // 392)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (64*r2) + (8192*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((12544*x1) + (802816*((r2 + (128*x0)) // 12544)) + ((r2 + (128*x0)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u2/cu2cy3gisxgysbtsdxwj36dgderhr4hm6vgyc3cwggfjbfruazz4.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_138 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_138', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 64
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (392*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3a/c3apwn22rbrglndfzeuuleb5o2kostneob6npr5ui75aw7hrvbjg.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_139 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_139', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((12544*x0) + (802816*((r2 + (128*x1)) // 12544)) + ((r2 + (128*x1)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/a7/ca7h22j3sdwapbswunmqv6jtu5abm4kvvzcpeyn6zwj2eofihxvq.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_140 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 512],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_140', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 392
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
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = 0.001
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)
    tmp8 = tmp2 * tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ef/cef4pr6nywaeej5ewjtnbw2geecv377jb24ttmmdxzudkaz5ryph.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_141 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_141', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 50176
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
    tmp5 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5w/c5wyzdptswjbgngz2u2xjafpnha5ja6i35y27yorqabebdarqsyw.py
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
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_142', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 16
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp2 = 0.001
    tmp3 = tmp1 + tmp2
    tmp4 = tl.math.rsqrt(tmp3)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 * tmp6
    tl.store(out_ptr0 + (x3), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zp/czpln4iyytsjyahpvrdvvxz3jpmfaycc6i4kekxjsiuv2x7wrtnc.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_143 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_143', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 50176
    xnumel = 16
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
    tmp0 = tl.load(in_ptr0 + (x2 + (16*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (12544*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp4 * tmp10
    tl.store(out_ptr0 + (x2 + (16*y3)), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/73/c73wpemvbpr3ekt5bkbfwfdqwlmu7takti5fzpxgtspshwhnyc3d.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_add_hardswish_backward_native_batch_norm_backward_144 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_hardswish_backward_native_batch_norm_backward_144', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 112
    rnumel = 7168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((112*(((r2 + (7168*x0)) // 112) % 112)) + (12544*x1) + (200704*((r2 + (7168*x0)) // 12544)) + (r2 % 112)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (16*r2) + (114688*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr2 + ((112*(((r2 + (7168*x0)) // 112) % 112)) + (12544*x1) + (200704*((r2 + (7168*x0)) // 12544)) + (r2 % 112)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp5 = -3.0
        tmp6 = tmp4 < tmp5
        tmp7 = 3.0
        tmp8 = tmp4 <= tmp7
        tmp10 = tmp0 + tmp9
        tmp11 = tmp4 / tmp7
        tmp12 = 0.5
        tmp13 = tmp11 + tmp12
        tmp14 = tmp10 * tmp13
        tmp15 = tl.where(tmp8, tmp14, tmp10)
        tmp16 = 0.0
        tmp17 = tl.where(tmp6, tmp16, tmp15)
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cn/ccn7fbvonl6nmfrmn3s63fekvcjlotybx7nukrgopacaqrqoqh4q.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_145 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_145', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pg/cpgp7qorql24ttabzwsipq37bjsfodkd2bjcvsuew4237suxfyfp.py
# Source Nodes: [], Original ATen: [aten.add, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_add_hardswish_backward_native_batch_norm_backward_146 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_hardswish_backward_native_batch_norm_backward_146', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 392
    x1 = (xindex // 392)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp23 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((12544*x1) + (200704*((r2 + (128*x0)) // 12544)) + ((r2 + (128*x0)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (16*r2) + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x1 + (16*r2) + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.load(in_ptr4 + ((12544*x1) + (200704*((r2 + (128*x0)) // 12544)) + ((r2 + (128*x0)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tl.load(in_ptr5 + (x1 + (16*r2) + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp9 = -3.0
        tmp10 = tmp8 < tmp9
        tmp11 = 3.0
        tmp12 = tmp8 <= tmp11
        tmp14 = tmp0 + tmp13
        tmp15 = tmp8 / tmp11
        tmp16 = 0.5
        tmp17 = tmp15 + tmp16
        tmp18 = tmp14 * tmp17
        tmp19 = tl.where(tmp12, tmp18, tmp14)
        tmp20 = 0.0
        tmp21 = tl.where(tmp10, tmp20, tmp19)
        tmp24 = tmp22 - tmp23
        tmp25 = tmp21 * tmp24
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(rmask & xmask, tmp28, _tmp27)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp27, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/aw/cawypvhnwnelulomexfainmkvp4k5ufw2rnvqrxhvwasmbuvvc5c.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_147 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_147', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel):
    xnumel = 16
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (392*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dw/cdwevun7vj5apqeclcw7g4r4ylmr6rijkozkedftwxjxfytip56m.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_148 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_148', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 392
    x1 = (xindex // 392)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (16*r2) + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((12544*x1) + (200704*((r2 + (128*x0)) // 12544)) + ((r2 + (128*x0)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 <= tmp1
        tmp4 = tl.where(tmp2, tmp1, tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zx/czx6cozznvwjz5goqazjudbqdi7r63nzxsxjyzialosnnmyseskz.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_149 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_149', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 16
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (392*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pr/cprdckn55j7sazyrkyzukiotqpa5eibib5jqacitocex6z3j6baw.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_150 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_150', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (16*r2) + (2048*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + ((12544*x0) + (200704*((r2 + (128*x1)) // 12544)) + ((r2 + (128*x1)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (16*r2) + (2048*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/wa/cwajkplr54ph4whfvnp7xgjux2cs625aotwhnynxsi3gbpqegzwq.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_151 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16, 512],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_151', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 392
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
        tmp0 = tl.load(in_ptr0 + (x0 + (16*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = 0.001
    tmp6 = tmp4 + tmp5
    tmp7 = tl.math.rsqrt(tmp6)
    tmp8 = tmp2 * tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fi/cfijuxkqdjjcl6pryh36c6qmqvmg232fummhxttdqhox6d67rc3g.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_hardswish_backward_native_batch_norm_backward_152 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_hardswish_backward_native_batch_norm_backward_152', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (16*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp3
    tmp9 = 0.5
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 * tmp10
    tmp12 = tl.where(tmp4, tmp11, tmp7)
    tmp13 = 0.0
    tmp14 = tl.where(tmp2, tmp13, tmp12)
    tmp16 = 0.001
    tmp17 = tmp15 + tmp16
    tmp18 = tl.math.rsqrt(tmp17)
    tmp20 = tmp18 * tmp19
    tmp21 = tmp14 * tmp20
    tl.store(out_ptr0 + (y0 + (16*x2) + (200704*y1)), tmp21, xmask & ymask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_36, primals_38, primals_39, primals_41, primals_42, primals_44, primals_45, primals_47, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_60, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_111, primals_113, primals_114, primals_116, primals_117, primals_119, primals_120, primals_122, primals_124, primals_126, primals_127, primals_129, primals_130, primals_132, primals_133, primals_135, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_150, primals_152, primals_153, primals_155, primals_156, primals_158, primals_159, primals_161, primals_163, primals_165, primals_166, primals_168, primals_169, primals_175, primals_176, primals_178, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, primals_190, primals_191, primals_193, primals_194, primals_196, primals_197, primals_199, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_221, primals_223, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_233, primals_235, primals_236, primals_238, primals_239, primals_241, primals_242, primals_244, primals_245, primals_247, primals_248, primals_250, primals_251, primals_253, primals_254, primals_256, primals_257, primals_259, primals_260, primals_262, primals_263, primals_265, primals_266, primals_268, primals_269, primals_271, primals_272, primals_274, primals_275, primals_277, primals_278, primals_280, primals_281, primals_283, primals_284, primals_286, primals_287, primals_289, primals_290, primals_292, primals_293, primals_295, primals_296, primals_298, primals_299, primals_301, primals_302, primals_304, primals_305, primals_307, primals_308, primals_310, primals_311, primals_313, convolution, clone, div, convolution_1, relu, convolution_2, add_7, convolution_3, relu_1, convolution_4, relu_2, convolution_5, add_13, convolution_6, relu_3, convolution_7, relu_4, convolution_8, add_20, convolution_9, relu_5, convolution_10, relu_6, mean, relu_7, div_1, mul_34, convolution_13, add_27, convolution_14, relu_8, convolution_15, relu_9, mean_1, relu_10, div_2, mul_44, convolution_18, add_35, convolution_19, relu_11, convolution_20, relu_12, mean_2, relu_13, div_3, mul_54, convolution_23, add_43, convolution_24, clone_1, div_4, convolution_25, clone_2, div_5, convolution_26, add_51, convolution_27, clone_3, div_6, convolution_28, clone_4, div_7, convolution_29, add_60, convolution_30, clone_5, div_8, convolution_31, clone_6, div_9, convolution_32, add_69, convolution_33, clone_7, div_10, convolution_34, clone_8, div_11, convolution_35, add_78, convolution_36, clone_9, div_12, convolution_37, clone_10, div_13, mean_3, relu_14, div_14, mul_110, convolution_40, add_87, convolution_41, clone_11, div_15, convolution_42, clone_12, div_16, mean_4, relu_15, div_17, mul_122, convolution_45, add_97, convolution_46, clone_13, div_18, convolution_47, clone_14, div_19, mean_5, relu_16, div_20, mul_134, convolution_50, add_106, convolution_51, clone_15, div_21, convolution_52, clone_16, div_22, mean_6, relu_17, div_23, mul_146, convolution_55, add_116, convolution_56, clone_17, div_24, convolution_57, clone_18, div_25, mean_7, relu_18, div_26, mul_158, convolution_60, add_126, convolution_61, clone_19, view, addmm, div_28, permute_2, permute_6, bitwise_and, bitwise_and_1, bitwise_and_2, bitwise_and_3, bitwise_and_4, bitwise_and_5, bitwise_and_6, bitwise_and_7, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_2, (16, ), (1, ))
    assert_size_stride(primals_4, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_7, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_8, (16, ), (1, ))
    assert_size_stride(primals_10, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_13, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_16, (24, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_17, (24, ), (1, ))
    assert_size_stride(primals_19, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_20, (72, ), (1, ))
    assert_size_stride(primals_22, (72, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_23, (72, ), (1, ))
    assert_size_stride(primals_25, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_26, (24, ), (1, ))
    assert_size_stride(primals_28, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_29, (72, ), (1, ))
    assert_size_stride(primals_31, (72, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_32, (72, ), (1, ))
    assert_size_stride(primals_34, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_36, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_38, (40, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_39, (40, ), (1, ))
    assert_size_stride(primals_41, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_42, (120, ), (1, ))
    assert_size_stride(primals_44, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_45, (120, ), (1, ))
    assert_size_stride(primals_47, (32, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_49, (120, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_51, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_52, (40, ), (1, ))
    assert_size_stride(primals_54, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_55, (120, ), (1, ))
    assert_size_stride(primals_57, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_58, (120, ), (1, ))
    assert_size_stride(primals_60, (32, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_62, (120, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_64, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_65, (40, ), (1, ))
    assert_size_stride(primals_67, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_68, (240, ), (1, ))
    assert_size_stride(primals_70, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_71, (240, ), (1, ))
    assert_size_stride(primals_73, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_74, (80, ), (1, ))
    assert_size_stride(primals_76, (200, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_77, (200, ), (1, ))
    assert_size_stride(primals_79, (200, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_80, (200, ), (1, ))
    assert_size_stride(primals_82, (80, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(primals_83, (80, ), (1, ))
    assert_size_stride(primals_85, (184, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_86, (184, ), (1, ))
    assert_size_stride(primals_88, (184, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_89, (184, ), (1, ))
    assert_size_stride(primals_91, (80, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_92, (80, ), (1, ))
    assert_size_stride(primals_94, (184, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_95, (184, ), (1, ))
    assert_size_stride(primals_97, (184, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_98, (184, ), (1, ))
    assert_size_stride(primals_100, (80, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_101, (80, ), (1, ))
    assert_size_stride(primals_103, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_104, (480, ), (1, ))
    assert_size_stride(primals_106, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_107, (480, ), (1, ))
    assert_size_stride(primals_109, (120, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_111, (480, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_113, (112, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_114, (112, ), (1, ))
    assert_size_stride(primals_116, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_117, (672, ), (1, ))
    assert_size_stride(primals_119, (672, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_120, (672, ), (1, ))
    assert_size_stride(primals_122, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_124, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_126, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_127, (112, ), (1, ))
    assert_size_stride(primals_129, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_130, (672, ), (1, ))
    assert_size_stride(primals_132, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_133, (672, ), (1, ))
    assert_size_stride(primals_135, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_137, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_139, (160, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_140, (160, ), (1, ))
    assert_size_stride(primals_142, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_143, (960, ), (1, ))
    assert_size_stride(primals_145, (960, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_146, (960, ), (1, ))
    assert_size_stride(primals_148, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_150, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_152, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_153, (160, ), (1, ))
    assert_size_stride(primals_155, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_156, (960, ), (1, ))
    assert_size_stride(primals_158, (960, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_159, (960, ), (1, ))
    assert_size_stride(primals_161, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_163, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_165, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_166, (160, ), (1, ))
    assert_size_stride(primals_168, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_169, (960, ), (1, ))
    assert_size_stride(primals_175, (16, ), (1, ))
    assert_size_stride(primals_176, (16, ), (1, ))
    assert_size_stride(primals_178, (16, ), (1, ))
    assert_size_stride(primals_179, (16, ), (1, ))
    assert_size_stride(primals_181, (16, ), (1, ))
    assert_size_stride(primals_182, (16, ), (1, ))
    assert_size_stride(primals_184, (64, ), (1, ))
    assert_size_stride(primals_185, (64, ), (1, ))
    assert_size_stride(primals_187, (64, ), (1, ))
    assert_size_stride(primals_188, (64, ), (1, ))
    assert_size_stride(primals_190, (24, ), (1, ))
    assert_size_stride(primals_191, (24, ), (1, ))
    assert_size_stride(primals_193, (72, ), (1, ))
    assert_size_stride(primals_194, (72, ), (1, ))
    assert_size_stride(primals_196, (72, ), (1, ))
    assert_size_stride(primals_197, (72, ), (1, ))
    assert_size_stride(primals_199, (24, ), (1, ))
    assert_size_stride(primals_200, (24, ), (1, ))
    assert_size_stride(primals_202, (72, ), (1, ))
    assert_size_stride(primals_203, (72, ), (1, ))
    assert_size_stride(primals_205, (72, ), (1, ))
    assert_size_stride(primals_206, (72, ), (1, ))
    assert_size_stride(primals_208, (40, ), (1, ))
    assert_size_stride(primals_209, (40, ), (1, ))
    assert_size_stride(primals_211, (120, ), (1, ))
    assert_size_stride(primals_212, (120, ), (1, ))
    assert_size_stride(primals_214, (120, ), (1, ))
    assert_size_stride(primals_215, (120, ), (1, ))
    assert_size_stride(primals_217, (40, ), (1, ))
    assert_size_stride(primals_218, (40, ), (1, ))
    assert_size_stride(primals_220, (120, ), (1, ))
    assert_size_stride(primals_221, (120, ), (1, ))
    assert_size_stride(primals_223, (120, ), (1, ))
    assert_size_stride(primals_224, (120, ), (1, ))
    assert_size_stride(primals_226, (40, ), (1, ))
    assert_size_stride(primals_227, (40, ), (1, ))
    assert_size_stride(primals_229, (240, ), (1, ))
    assert_size_stride(primals_230, (240, ), (1, ))
    assert_size_stride(primals_232, (240, ), (1, ))
    assert_size_stride(primals_233, (240, ), (1, ))
    assert_size_stride(primals_235, (80, ), (1, ))
    assert_size_stride(primals_236, (80, ), (1, ))
    assert_size_stride(primals_238, (200, ), (1, ))
    assert_size_stride(primals_239, (200, ), (1, ))
    assert_size_stride(primals_241, (200, ), (1, ))
    assert_size_stride(primals_242, (200, ), (1, ))
    assert_size_stride(primals_244, (80, ), (1, ))
    assert_size_stride(primals_245, (80, ), (1, ))
    assert_size_stride(primals_247, (184, ), (1, ))
    assert_size_stride(primals_248, (184, ), (1, ))
    assert_size_stride(primals_250, (184, ), (1, ))
    assert_size_stride(primals_251, (184, ), (1, ))
    assert_size_stride(primals_253, (80, ), (1, ))
    assert_size_stride(primals_254, (80, ), (1, ))
    assert_size_stride(primals_256, (184, ), (1, ))
    assert_size_stride(primals_257, (184, ), (1, ))
    assert_size_stride(primals_259, (184, ), (1, ))
    assert_size_stride(primals_260, (184, ), (1, ))
    assert_size_stride(primals_262, (80, ), (1, ))
    assert_size_stride(primals_263, (80, ), (1, ))
    assert_size_stride(primals_265, (480, ), (1, ))
    assert_size_stride(primals_266, (480, ), (1, ))
    assert_size_stride(primals_268, (480, ), (1, ))
    assert_size_stride(primals_269, (480, ), (1, ))
    assert_size_stride(primals_271, (112, ), (1, ))
    assert_size_stride(primals_272, (112, ), (1, ))
    assert_size_stride(primals_274, (672, ), (1, ))
    assert_size_stride(primals_275, (672, ), (1, ))
    assert_size_stride(primals_277, (672, ), (1, ))
    assert_size_stride(primals_278, (672, ), (1, ))
    assert_size_stride(primals_280, (112, ), (1, ))
    assert_size_stride(primals_281, (112, ), (1, ))
    assert_size_stride(primals_283, (672, ), (1, ))
    assert_size_stride(primals_284, (672, ), (1, ))
    assert_size_stride(primals_286, (672, ), (1, ))
    assert_size_stride(primals_287, (672, ), (1, ))
    assert_size_stride(primals_289, (160, ), (1, ))
    assert_size_stride(primals_290, (160, ), (1, ))
    assert_size_stride(primals_292, (960, ), (1, ))
    assert_size_stride(primals_293, (960, ), (1, ))
    assert_size_stride(primals_295, (960, ), (1, ))
    assert_size_stride(primals_296, (960, ), (1, ))
    assert_size_stride(primals_298, (160, ), (1, ))
    assert_size_stride(primals_299, (160, ), (1, ))
    assert_size_stride(primals_301, (960, ), (1, ))
    assert_size_stride(primals_302, (960, ), (1, ))
    assert_size_stride(primals_304, (960, ), (1, ))
    assert_size_stride(primals_305, (960, ), (1, ))
    assert_size_stride(primals_307, (160, ), (1, ))
    assert_size_stride(primals_308, (160, ), (1, ))
    assert_size_stride(primals_310, (960, ), (1, ))
    assert_size_stride(primals_311, (960, ), (1, ))
    assert_size_stride(primals_313, (4, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (4, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(clone, (4, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(div, (4, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_1, (4, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(relu, (4, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_2, (4, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(add_7, (4, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_3, (4, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(relu_1, (4, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(convolution_4, (4, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(relu_2, (4, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_5, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(add_13, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_6, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(relu_3, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(convolution_7, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(relu_4, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(convolution_8, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(add_20, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_9, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(relu_5, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(convolution_10, (4, 72, 28, 28), (56448, 1, 2016, 72))
    assert_size_stride(relu_6, (4, 72, 28, 28), (56448, 1, 2016, 72))
    assert_size_stride(mean, (4, 72, 1, 1), (72, 1, 72, 72))
    assert_size_stride(relu_7, (4, 24, 1, 1), (24, 1, 24, 24))
    assert_size_stride(div_1, (4, 72, 1, 1), (72, 1, 72, 72))
    assert_size_stride(mul_34, (4, 72, 28, 28), (56448, 1, 2016, 72))
    assert_size_stride(convolution_13, (4, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(add_27, (4, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_14, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(relu_8, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_15, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(relu_9, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(mean_1, (4, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(relu_10, (4, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_2, (4, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(mul_44, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_18, (4, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(add_35, (4, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_19, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(relu_11, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_20, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(relu_12, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(mean_2, (4, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(relu_13, (4, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_3, (4, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(mul_54, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_23, (4, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(add_43, (4, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_24, (4, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(clone_1, (4, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(div_4, (4, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(convolution_25, (4, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(clone_2, (4, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(div_5, (4, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(convolution_26, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(add_51, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_27, (4, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(clone_3, (4, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(div_6, (4, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(convolution_28, (4, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(clone_4, (4, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(div_7, (4, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(convolution_29, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(add_60, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_30, (4, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(clone_5, (4, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(div_8, (4, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(convolution_31, (4, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(clone_6, (4, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(div_9, (4, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(convolution_32, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(add_69, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_33, (4, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(clone_7, (4, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(div_10, (4, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(convolution_34, (4, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(clone_8, (4, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(div_11, (4, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(convolution_35, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(add_78, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_36, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(clone_9, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(div_12, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_37, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(clone_10, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(div_13, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(mean_3, (4, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(relu_14, (4, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(div_14, (4, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(mul_110, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_40, (4, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(add_87, (4, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(convolution_41, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(clone_11, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(div_15, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(convolution_42, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(clone_12, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(div_16, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(mean_4, (4, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(relu_15, (4, 168, 1, 1), (168, 1, 168, 168))
    assert_size_stride(div_17, (4, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(mul_122, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(convolution_45, (4, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(add_97, (4, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(convolution_46, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(clone_13, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(div_18, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(convolution_47, (4, 672, 7, 7), (32928, 1, 4704, 672))
    assert_size_stride(clone_14, (4, 672, 7, 7), (32928, 1, 4704, 672))
    assert_size_stride(div_19, (4, 672, 7, 7), (32928, 1, 4704, 672))
    assert_size_stride(mean_5, (4, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(relu_16, (4, 168, 1, 1), (168, 1, 168, 168))
    assert_size_stride(div_20, (4, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(mul_134, (4, 672, 7, 7), (32928, 1, 4704, 672))
    assert_size_stride(convolution_50, (4, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(add_106, (4, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(convolution_51, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(clone_15, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(div_21, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_52, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(clone_16, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(div_22, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(mean_6, (4, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(relu_17, (4, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(div_23, (4, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(mul_146, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_55, (4, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(add_116, (4, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(convolution_56, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(clone_17, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(div_24, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_57, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(clone_18, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(div_25, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(mean_7, (4, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(relu_18, (4, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(div_26, (4, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(mul_158, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_60, (4, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(add_126, (4, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(convolution_61, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(clone_19, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(view, (4, 960), (960, 1))
    assert_size_stride(addmm, (4, 1280), (1280, 1))
    assert_size_stride(div_28, (4, 1280), (1280, 1))
    assert_size_stride(permute_2, (1000, 1280), (1280, 1))
    assert_size_stride(permute_6, (1280, 960), (960, 1))
    assert_size_stride(bitwise_and, (4, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(bitwise_and_1, (4, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(bitwise_and_2, (4, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(bitwise_and_3, (4, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(bitwise_and_4, (4, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(bitwise_and_5, (4, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(bitwise_and_6, (4, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(bitwise_and_7, (4, 72, 1, 1), (72, 1, 72, 72))
    assert_size_stride(tangents_1, (4, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((4, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_2, out=buf0)
        del permute_2
        buf1 = empty((1000, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 4), (1, 1000), 0), div_28, out=buf1)
        del div_28
        buf2 = empty((1000, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_sum_view_0.run(tangents_1, buf2, 1000, grid=grid(1000), stream=stream0)
        del tangents_1
        buf3 = buf0; del buf0  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward]
        triton_poi_fused_hardswish_backward_1.run(buf3, addmm, 5120, grid=grid(5120), stream=stream0)
        del addmm
        buf4 = empty((4, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf3, permute_6, out=buf4)
        del permute_6
        buf5 = empty((1280, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf3, (1280, 4), (1, 1280), 0), view, out=buf5)
        del view
        buf6 = empty((1280, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        triton_poi_fused_sum_view_2.run(buf3, buf6, 1280, grid=grid(1280), stream=stream0)
        del buf3
        buf7 = empty_strided((960, 2), (1, 960), device='cuda', dtype=torch.float32)
        buf9 = empty_strided((960, 2), (1, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_div_hardswish_backward_native_batch_norm_backward_3.run(clone_19, buf4, convolution_61, primals_310, buf7, buf9, 1920, 98, grid=grid(1920), stream=stream0)
        del convolution_61
        del primals_310
        buf8 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_div_hardswish_backward_native_batch_norm_backward_4.run(buf7, buf8, 960, 2, grid=grid(960), stream=stream0)
        buf10 = empty((960, ), device='cuda', dtype=torch.float32)
        buf11 = buf10; del buf10  # reuse
        # Source Nodes: [], Original ATen: [aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_div_hardswish_backward_native_batch_norm_backward_5.run(buf11, buf9, primals_311, 960, 2, grid=grid(960), stream=stream0)
        buf12 = empty_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_div_hardswish_backward_native_batch_norm_backward_6.run(clone_19, buf4, primals_311, primals_169, buf12, 188160, grid=grid(188160), stream=stream0)
        del clone_19
        del primals_169
        del primals_311
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf13 = aten.convolution_backward(buf12, add_126, primals_168, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_126
        del primals_168
        buf14 = buf13[0]
        buf15 = buf13[1]
        del buf13
        buf16 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_7.run(buf14, buf16, 160, 196, grid=grid(160), stream=stream0)
        buf17 = empty_strided((160, 2), (1, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_8.run(buf14, convolution_60, primals_307, buf17, 320, 98, grid=grid(320), stream=stream0)
        del convolution_60
        del primals_307
        buf18 = empty((160, ), device='cuda', dtype=torch.float32)
        buf19 = buf18; del buf18  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_9.run(buf19, buf17, primals_308, 160, 2, grid=grid(160), stream=stream0)
        del buf17
        buf20 = empty((4, 160, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_10.run(buf14, primals_308, primals_166, buf20, 31360, grid=grid(31360), stream=stream0)
        del primals_166
        del primals_308
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf21 = aten.convolution_backward(buf20, mul_158, primals_165, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_158
        del primals_165
        buf22 = buf21[0]
        buf23 = buf21[1]
        del buf21
        buf24 = reinterpret_tensor(buf4, (4, 960, 1, 1), (960, 1, 3840, 3840), 0); del buf4  # reuse
        buf25 = reinterpret_tensor(buf24, (4, 960, 1, 1), (960, 1, 960, 960), 0); del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_11.run(buf25, buf22, div_25, bitwise_and, 3840, 49, grid=grid(3840), stream=stream0)
        del bitwise_and
        del div_25
        buf26 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_12.run(buf25, buf26, 960, grid=grid(960), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf27 = aten.convolution_backward(buf25, relu_18, primals_163, [960], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf25
        del primals_163
        buf28 = buf27[0]
        buf29 = buf27[1]
        del buf27
        buf30 = buf28; del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.threshold_backward]
        triton_poi_fused_hardswish_backward_threshold_backward_13.run(buf30, relu_18, 960, grid=grid(960), stream=stream0)
        del relu_18
        buf31 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_14.run(buf30, buf31, 240, grid=grid(240), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf32 = aten.convolution_backward(buf30, mean_7, primals_161, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_7
        del primals_161
        buf33 = buf32[0]
        buf34 = buf32[1]
        del buf32
        buf35 = buf9; del buf9  # reuse
        buf37 = buf7; del buf7  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_15.run(clone_18, buf22, div_26, buf33, convolution_57, primals_304, buf35, buf37, 1920, 98, grid=grid(1920), stream=stream0)
        del convolution_57
        del primals_304
        buf36 = reinterpret_tensor(buf30, (960, ), (1, ), 0); del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_div_hardswish_backward_native_batch_norm_backward_4.run(buf35, buf36, 960, 2, grid=grid(960), stream=stream0)
        buf38 = empty((960, ), device='cuda', dtype=torch.float32)
        buf39 = buf38; del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_div_hardswish_backward_native_batch_norm_backward_5.run(buf39, buf37, primals_305, 960, 2, grid=grid(960), stream=stream0)
        buf40 = buf12; del buf12  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_16.run(clone_18, buf22, div_26, buf33, primals_305, primals_159, buf40, 196, 960, grid=grid(196, 960), stream=stream0)
        del buf22
        del clone_18
        del div_26
        del primals_159
        del primals_305
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        buf41 = aten.convolution_backward(buf40, div_24, primals_158, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 960, [True, True, False])
        del div_24
        del primals_158
        buf42 = buf41[0]
        buf43 = buf41[1]
        del buf41
        buf44 = buf37; del buf37  # reuse
        buf46 = buf35; del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_17.run(clone_17, buf42, convolution_56, primals_301, buf44, buf46, 1920, 98, grid=grid(1920), stream=stream0)
        del convolution_56
        del primals_301
        buf45 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_div_hardswish_backward_native_batch_norm_backward_4.run(buf44, buf45, 960, 2, grid=grid(960), stream=stream0)
        buf47 = empty((960, ), device='cuda', dtype=torch.float32)
        buf48 = buf47; del buf47  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_div_hardswish_backward_native_batch_norm_backward_5.run(buf48, buf46, primals_302, 960, 2, grid=grid(960), stream=stream0)
        buf49 = buf40; del buf40  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_18.run(clone_17, buf42, primals_302, primals_156, buf49, 196, 960, grid=grid(196, 960), stream=stream0)
        del buf42
        del clone_17
        del primals_156
        del primals_302
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf50 = aten.convolution_backward(buf49, add_116, primals_155, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_116
        del primals_155
        buf51 = buf50[0]
        buf52 = buf50[1]
        del buf50
        buf56 = buf20; del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_19.run(buf14, buf51, primals_299, primals_153, buf56, 31360, grid=grid(31360), stream=stream0)
        del primals_153
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf57 = aten.convolution_backward(buf56, mul_146, primals_152, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf56
        del mul_146
        del primals_152
        buf58 = buf57[0]
        buf60 = reinterpret_tensor(buf33, (4, 960, 1, 1), (960, 1, 3840, 3840), 0); del buf33  # reuse
        buf61 = reinterpret_tensor(buf60, (4, 960, 1, 1), (960, 1, 960, 960), 0); del buf60  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_11.run(buf61, buf58, div_22, bitwise_and_1, 3840, 49, grid=grid(3840), stream=stream0)
        del bitwise_and_1
        del div_22
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf63 = aten.convolution_backward(buf61, relu_17, primals_150, [960], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_150
        buf64 = buf63[0]
        buf66 = buf64; del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.threshold_backward]
        triton_poi_fused_hardswish_backward_threshold_backward_13.run(buf66, relu_17, 960, grid=grid(960), stream=stream0)
        del relu_17
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf68 = aten.convolution_backward(buf66, mean_6, primals_148, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_6
        del primals_148
        buf69 = buf68[0]
        buf76 = buf49; del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_16.run(clone_16, buf58, div_23, buf69, primals_296, primals_146, buf76, 196, 960, grid=grid(196, 960), stream=stream0)
        del primals_146
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        buf77 = aten.convolution_backward(buf76, div_21, primals_145, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 960, [True, True, False])
        del div_21
        del primals_145
        buf78 = buf77[0]
        buf85 = buf76; del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_18.run(clone_15, buf78, primals_293, primals_143, buf85, 196, 960, grid=grid(196, 960), stream=stream0)
        del primals_143
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf86 = aten.convolution_backward(buf85, add_106, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_106
        del buf85
        del primals_142
        buf87 = buf86[0]
        buf53 = empty((160, ), device='cuda', dtype=torch.float32)
        buf54 = empty((160, ), device='cuda', dtype=torch.float32)
        buf89 = empty((160, ), device='cuda', dtype=torch.float32)
        buf90 = empty((160, ), device='cuda', dtype=torch.float32)
        buf55 = buf54; del buf54  # reuse
        buf91 = buf90; del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_20.run(buf55, buf91, buf14, buf51, convolution_55, primals_298, buf87, convolution_50, primals_289, primals_299, primals_290, buf53, buf89, 160, 196, grid=grid(160), stream=stream0)
        del convolution_50
        del convolution_55
        del primals_289
        del primals_298
        del primals_299
        buf59 = buf57[1]
        del buf57
        buf62 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_12.run(buf61, buf62, 960, grid=grid(960), stream=stream0)
        del buf61
        buf65 = buf63[1]
        del buf63
        buf67 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_14.run(buf66, buf67, 240, grid=grid(240), stream=stream0)
        buf70 = buf68[1]
        del buf68
        buf71 = buf46; del buf46  # reuse
        buf73 = buf44; del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_15.run(clone_16, buf58, div_23, buf69, convolution_52, primals_295, buf71, buf73, 1920, 98, grid=grid(1920), stream=stream0)
        del buf58
        del clone_16
        del convolution_52
        del div_23
        del primals_295
        buf72 = reinterpret_tensor(buf66, (960, ), (1, ), 0); del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_div_hardswish_backward_native_batch_norm_backward_4.run(buf71, buf72, 960, 2, grid=grid(960), stream=stream0)
        buf74 = empty((960, ), device='cuda', dtype=torch.float32)
        buf75 = buf74; del buf74  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_div_hardswish_backward_native_batch_norm_backward_5.run(buf75, buf73, primals_296, 960, 2, grid=grid(960), stream=stream0)
        del primals_296
        buf79 = buf77[1]
        del buf77
        buf80 = buf73; del buf73  # reuse
        buf82 = buf71; del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_17.run(clone_15, buf78, convolution_51, primals_292, buf80, buf82, 1920, 98, grid=grid(1920), stream=stream0)
        del clone_15
        del convolution_51
        del primals_292
        buf81 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_div_hardswish_backward_native_batch_norm_backward_4.run(buf80, buf81, 960, 2, grid=grid(960), stream=stream0)
        del buf80
        buf83 = empty((960, ), device='cuda', dtype=torch.float32)
        buf84 = buf83; del buf83  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_div_hardswish_backward_native_batch_norm_backward_5.run(buf84, buf82, primals_293, 960, 2, grid=grid(960), stream=stream0)
        del primals_293
        buf88 = buf86[1]
        del buf86
        buf92 = buf14; del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_21.run(buf92, buf51, buf87, primals_290, primals_140, 31360, grid=grid(31360), stream=stream0)
        del buf51
        del buf87
        del primals_140
        del primals_290
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf93 = aten.convolution_backward(buf92, mul_134, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf92
        del mul_134
        del primals_139
        buf94 = buf93[0]
        buf95 = buf93[1]
        del buf93
        buf96 = empty_strided((4, 672, 1, 1), (672, 1, 2688, 2688), device='cuda', dtype=torch.float32)
        buf97 = reinterpret_tensor(buf96, (4, 672, 1, 1), (672, 1, 672, 672), 0); del buf96  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_22.run(buf97, buf94, div_19, bitwise_and_2, 2688, 49, grid=grid(2688), stream=stream0)
        del bitwise_and_2
        del div_19
        buf98 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_23.run(buf97, buf98, 672, grid=grid(672), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf99 = aten.convolution_backward(buf97, relu_16, primals_137, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf97
        del primals_137
        buf100 = buf99[0]
        buf101 = buf99[1]
        del buf99
        buf102 = buf100; del buf100  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.threshold_backward]
        triton_poi_fused_hardswish_backward_threshold_backward_24.run(buf102, relu_16, 672, grid=grid(672), stream=stream0)
        del relu_16
        buf103 = empty((168, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_25.run(buf102, buf103, 168, grid=grid(168), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf104 = aten.convolution_backward(buf102, mean_5, primals_135, [168], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_5
        del primals_135
        buf105 = buf104[0]
        buf106 = buf104[1]
        del buf104
        buf107 = empty_strided((672, 2), (1, 672), device='cuda', dtype=torch.float32)
        buf109 = empty_strided((672, 2), (1, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_26.run(clone_14, buf94, div_20, buf105, convolution_47, primals_286, buf107, buf109, 1344, 98, grid=grid(1344), stream=stream0)
        del convolution_47
        del primals_286
        buf108 = reinterpret_tensor(buf102, (672, ), (1, ), 0); del buf102  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_27.run(buf107, buf108, 672, 2, grid=grid(672), stream=stream0)
        del buf107
        buf110 = empty((672, ), device='cuda', dtype=torch.float32)
        buf111 = buf110; del buf110  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_28.run(buf111, buf109, primals_287, 672, 2, grid=grid(672), stream=stream0)
        del buf109
        buf112 = empty_strided((4, 672, 7, 7), (32928, 1, 4704, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_29.run(clone_14, buf94, div_20, buf105, primals_287, primals_133, buf112, 196, 672, grid=grid(196, 672), stream=stream0)
        del buf94
        del clone_14
        del div_20
        del primals_133
        del primals_287
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        buf113 = aten.convolution_backward(buf112, div_18, primals_132, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False])
        del buf112
        del div_18
        del primals_132
        buf114 = buf113[0]
        buf115 = buf113[1]
        del buf113
        buf116 = empty((672, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_30.run(clone_13, buf114, buf116, 4704, 112, grid=grid(4704), stream=stream0)
        buf117 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_31.run(buf116, buf117, 672, 7, grid=grid(672), stream=stream0)
        buf118 = reinterpret_tensor(buf116, (672, 7), (1, 672), 0); del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_32.run(clone_13, buf114, convolution_46, primals_283, buf118, 4704, 112, grid=grid(4704), stream=stream0)
        del convolution_46
        del primals_283
        buf119 = empty((672, ), device='cuda', dtype=torch.float32)
        buf120 = buf119; del buf119  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_33.run(buf120, buf118, primals_284, 672, 7, grid=grid(672), stream=stream0)
        buf121 = empty_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_34.run(clone_13, buf114, primals_284, primals_130, buf121, 784, 672, grid=grid(784, 672), stream=stream0)
        del buf114
        del clone_13
        del primals_130
        del primals_284
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf122 = aten.convolution_backward(buf121, add_97, primals_129, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_97
        del primals_129
        buf123 = buf122[0]
        buf124 = buf122[1]
        del buf122
        buf129 = empty((4, 112, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_35.run(buf123, primals_281, primals_127, buf129, 87808, grid=grid(87808), stream=stream0)
        del primals_127
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf130 = aten.convolution_backward(buf129, mul_122, primals_126, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf129
        del mul_122
        del primals_126
        buf131 = buf130[0]
        buf133 = empty_strided((4, 672, 1, 1, 2), (1344, 2, 5376, 5376, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_36.run(buf131, div_16, buf133, 5376, 98, grid=grid(5376), stream=stream0)
        del div_16
        buf134 = reinterpret_tensor(buf105, (4, 672, 1, 1), (672, 1, 2688, 2688), 0); del buf105  # reuse
        buf135 = reinterpret_tensor(buf134, (4, 672, 1, 1), (672, 1, 672, 672), 0); del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_37.run(buf135, buf133, bitwise_and_3, 2688, 2, grid=grid(2688), stream=stream0)
        del bitwise_and_3
        del buf133
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf137 = aten.convolution_backward(buf135, relu_15, primals_124, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_124
        buf138 = buf137[0]
        buf140 = buf138; del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.threshold_backward]
        triton_poi_fused_hardswish_backward_threshold_backward_24.run(buf140, relu_15, 672, grid=grid(672), stream=stream0)
        del relu_15
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf142 = aten.convolution_backward(buf140, mean_4, primals_122, [168], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_4
        del primals_122
        buf143 = buf142[0]
        buf150 = buf121; del buf121  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_38.run(clone_12, buf131, div_17, buf143, primals_278, primals_120, buf150, 784, 672, grid=grid(784, 672), stream=stream0)
        del primals_120
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        buf151 = aten.convolution_backward(buf150, div_15, primals_119, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 672, [True, True, False])
        del div_15
        del primals_119
        buf152 = buf151[0]
        buf159 = buf150; del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_34.run(clone_11, buf152, primals_275, primals_117, buf159, 784, 672, grid=grid(784, 672), stream=stream0)
        del primals_117
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf160 = aten.convolution_backward(buf159, add_87, primals_116, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_87
        del buf159
        del primals_116
        buf161 = buf160[0]
        buf125 = empty((112, ), device='cuda', dtype=torch.float32)
        buf163 = empty((112, ), device='cuda', dtype=torch.float32)
        buf164 = empty((112, ), device='cuda', dtype=torch.float32)
        buf165 = buf164; del buf164  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_39.run(buf165, buf123, buf161, convolution_40, primals_271, primals_272, buf125, buf163, 112, 784, grid=grid(112), stream=stream0)
        del convolution_40
        del primals_271
        buf126 = empty((112, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_40.run(buf123, convolution_45, primals_280, buf126, 784, 112, grid=grid(784), stream=stream0)
        del convolution_45
        del primals_280
        buf127 = empty((112, ), device='cuda', dtype=torch.float32)
        buf128 = buf127; del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_41.run(buf128, buf126, primals_281, 112, 7, grid=grid(112), stream=stream0)
        del buf126
        del primals_281
        buf132 = buf130[1]
        del buf130
        buf136 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_23.run(buf135, buf136, 672, grid=grid(672), stream=stream0)
        del buf135
        buf139 = buf137[1]
        del buf137
        buf141 = empty((168, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_25.run(buf140, buf141, 168, grid=grid(168), stream=stream0)
        buf144 = buf142[1]
        del buf142
        buf145 = reinterpret_tensor(buf118, (672, 7), (7, 1), 0); del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_42.run(clone_12, buf131, div_17, buf143, buf145, 4704, 112, grid=grid(4704), stream=stream0)
        buf146 = reinterpret_tensor(buf140, (672, ), (1, ), 0); del buf140  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_31.run(buf145, buf146, 672, 7, grid=grid(672), stream=stream0)
        buf147 = reinterpret_tensor(buf145, (672, 7), (1, 672), 0); del buf145  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_43.run(clone_12, buf131, div_17, buf143, convolution_42, primals_277, buf147, 4704, 112, grid=grid(4704), stream=stream0)
        del buf131
        del buf143
        del clone_12
        del convolution_42
        del div_17
        del primals_277
        buf148 = empty((672, ), device='cuda', dtype=torch.float32)
        buf149 = buf148; del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_33.run(buf149, buf147, primals_278, 672, 7, grid=grid(672), stream=stream0)
        del primals_278
        buf153 = buf151[1]
        del buf151
        buf154 = reinterpret_tensor(buf147, (672, 7), (7, 1), 0); del buf147  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_30.run(clone_11, buf152, buf154, 4704, 112, grid=grid(4704), stream=stream0)
        buf155 = empty((672, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_31.run(buf154, buf155, 672, 7, grid=grid(672), stream=stream0)
        buf156 = reinterpret_tensor(buf154, (672, 7), (1, 672), 0); del buf154  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_44.run(clone_11, buf152, convolution_41, primals_274, buf156, 4704, 112, grid=grid(4704), stream=stream0)
        del buf152
        del clone_11
        del convolution_41
        del primals_274
        buf157 = empty((672, ), device='cuda', dtype=torch.float32)
        buf158 = buf157; del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_33.run(buf158, buf156, primals_275, 672, 7, grid=grid(672), stream=stream0)
        del buf156
        del primals_275
        buf162 = buf160[1]
        del buf160
        buf166 = buf123; del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_45.run(buf166, buf161, primals_272, primals_114, 87808, grid=grid(87808), stream=stream0)
        del buf161
        del primals_114
        del primals_272
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf167 = aten.convolution_backward(buf166, mul_110, primals_113, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf166
        del mul_110
        del primals_113
        buf168 = buf167[0]
        buf169 = buf167[1]
        del buf167
        buf170 = reinterpret_tensor(buf69, (4, 480, 1, 1, 2), (960, 2, 3840, 3840, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_46.run(buf168, div_13, buf170, 3840, 98, grid=grid(3840), stream=stream0)
        del div_13
        buf171 = reinterpret_tensor(buf82, (4, 480, 1, 1), (480, 1, 1920, 1920), 0); del buf82  # reuse
        buf172 = reinterpret_tensor(buf171, (4, 480, 1, 1), (480, 1, 480, 480), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_47.run(buf172, buf170, bitwise_and_4, 1920, 2, grid=grid(1920), stream=stream0)
        del bitwise_and_4
        del buf170
        buf173 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_48.run(buf172, buf173, 480, grid=grid(480), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf174 = aten.convolution_backward(buf172, relu_14, primals_111, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf172
        del primals_111
        buf175 = buf174[0]
        buf176 = buf174[1]
        del buf174
        buf177 = buf175; del buf175  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.threshold_backward]
        triton_poi_fused_hardswish_backward_threshold_backward_49.run(buf177, relu_14, 480, grid=grid(480), stream=stream0)
        del relu_14
        buf178 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_50.run(buf177, buf178, 120, grid=grid(120), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf179 = aten.convolution_backward(buf177, mean_3, primals_109, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_3
        del primals_109
        buf180 = buf179[0]
        buf181 = buf179[1]
        del buf179
        buf182 = empty((480, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_51.run(clone_10, buf168, div_14, buf180, buf182, 3360, 112, grid=grid(3360), stream=stream0)
        buf183 = reinterpret_tensor(buf177, (480, ), (1, ), 0); del buf177  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_52.run(buf182, buf183, 480, 7, grid=grid(480), stream=stream0)
        buf184 = reinterpret_tensor(buf182, (480, 7), (1, 480), 0); del buf182  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_53.run(clone_10, buf168, div_14, buf180, convolution_37, primals_268, buf184, 3360, 112, grid=grid(3360), stream=stream0)
        del convolution_37
        del primals_268
        buf185 = empty((480, ), device='cuda', dtype=torch.float32)
        buf186 = buf185; del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_54.run(buf186, buf184, primals_269, 480, 7, grid=grid(480), stream=stream0)
        buf187 = empty_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_55.run(clone_10, buf168, div_14, buf180, primals_269, primals_107, buf187, 784, 480, grid=grid(784, 480), stream=stream0)
        del buf168
        del buf180
        del clone_10
        del div_14
        del primals_107
        del primals_269
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
        buf188 = aten.convolution_backward(buf187, div_12, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False])
        del div_12
        del primals_106
        buf189 = buf188[0]
        buf190 = buf188[1]
        del buf188
        buf191 = reinterpret_tensor(buf184, (480, 7), (7, 1), 0); del buf184  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_56.run(clone_9, buf189, buf191, 3360, 112, grid=grid(3360), stream=stream0)
        buf192 = empty((480, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_52.run(buf191, buf192, 480, 7, grid=grid(480), stream=stream0)
        buf193 = reinterpret_tensor(buf191, (480, 7), (1, 480), 0); del buf191  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_57.run(clone_9, buf189, convolution_36, primals_265, buf193, 3360, 112, grid=grid(3360), stream=stream0)
        del convolution_36
        del primals_265
        buf194 = empty((480, ), device='cuda', dtype=torch.float32)
        buf195 = buf194; del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_54.run(buf195, buf193, primals_266, 480, 7, grid=grid(480), stream=stream0)
        buf196 = buf187; del buf187  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_58.run(clone_9, buf189, primals_266, primals_104, buf196, 784, 480, grid=grid(784, 480), stream=stream0)
        del buf189
        del clone_9
        del primals_104
        del primals_266
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf197 = aten.convolution_backward(buf196, add_78, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_78
        del primals_103
        buf198 = buf197[0]
        buf199 = buf197[1]
        del buf197
        buf200 = empty((80, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_59.run(buf198, buf200, 80, 784, grid=grid(80), stream=stream0)
        buf201 = empty((80, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_60.run(buf198, convolution_35, primals_262, buf201, 560, 112, grid=grid(560), stream=stream0)
        del convolution_35
        del primals_262
        buf202 = empty((80, ), device='cuda', dtype=torch.float32)
        buf203 = buf202; del buf202  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_61.run(buf203, buf201, primals_263, 80, 7, grid=grid(80), stream=stream0)
        del buf201
        buf204 = empty((4, 80, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_62.run(buf198, primals_263, primals_101, buf204, 62720, grid=grid(62720), stream=stream0)
        del primals_101
        del primals_263
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf205 = aten.convolution_backward(buf204, div_11, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del div_11
        del primals_100
        buf206 = buf205[0]
        buf207 = buf205[1]
        del buf205
        buf208 = empty((184, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_63.run(clone_8, buf206, buf208, 1288, 112, grid=grid(1288), stream=stream0)
        buf209 = empty((184, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_64.run(buf208, buf209, 184, 7, grid=grid(184), stream=stream0)
        buf210 = reinterpret_tensor(buf208, (184, 7), (1, 184), 0); del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_65.run(clone_8, buf206, convolution_34, primals_259, buf210, 1288, 112, grid=grid(1288), stream=stream0)
        del convolution_34
        del primals_259
        buf211 = empty((184, ), device='cuda', dtype=torch.float32)
        buf212 = buf211; del buf211  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_66.run(buf212, buf210, primals_260, 184, 7, grid=grid(184), stream=stream0)
        buf213 = empty_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_67.run(clone_8, buf206, primals_260, primals_98, buf213, 784, 184, grid=grid(784, 184), stream=stream0)
        del buf206
        del clone_8
        del primals_260
        del primals_98
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf214 = aten.convolution_backward(buf213, div_10, primals_97, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 184, [True, True, False])
        del div_10
        del primals_97
        buf215 = buf214[0]
        buf216 = buf214[1]
        del buf214
        buf217 = reinterpret_tensor(buf210, (184, 7), (7, 1), 0); del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_63.run(clone_7, buf215, buf217, 1288, 112, grid=grid(1288), stream=stream0)
        buf218 = empty((184, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_64.run(buf217, buf218, 184, 7, grid=grid(184), stream=stream0)
        buf219 = reinterpret_tensor(buf217, (184, 7), (1, 184), 0); del buf217  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_65.run(clone_7, buf215, convolution_33, primals_256, buf219, 1288, 112, grid=grid(1288), stream=stream0)
        del convolution_33
        del primals_256
        buf220 = empty((184, ), device='cuda', dtype=torch.float32)
        buf221 = buf220; del buf220  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_66.run(buf221, buf219, primals_257, 184, 7, grid=grid(184), stream=stream0)
        buf222 = buf213; del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_67.run(clone_7, buf215, primals_257, primals_95, buf222, 784, 184, grid=grid(784, 184), stream=stream0)
        del buf215
        del clone_7
        del primals_257
        del primals_95
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf223 = aten.convolution_backward(buf222, add_69, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_69
        del primals_94
        buf224 = buf223[0]
        buf225 = buf223[1]
        del buf223
        buf229 = buf204; del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_68.run(buf198, buf224, primals_254, primals_92, buf229, 62720, grid=grid(62720), stream=stream0)
        del primals_92
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf230 = aten.convolution_backward(buf229, div_9, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del div_9
        del primals_91
        buf231 = buf230[0]
        buf238 = buf222; del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_67.run(clone_6, buf231, primals_251, primals_89, buf238, 784, 184, grid=grid(784, 184), stream=stream0)
        del primals_89
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf239 = aten.convolution_backward(buf238, div_8, primals_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 184, [True, True, False])
        del div_8
        del primals_88
        buf240 = buf239[0]
        buf247 = buf238; del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_67.run(clone_5, buf240, primals_248, primals_86, buf247, 784, 184, grid=grid(784, 184), stream=stream0)
        del primals_86
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf248 = aten.convolution_backward(buf247, add_60, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_60
        del buf247
        del primals_85
        buf249 = buf248[0]
        buf254 = buf229; del buf229  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_69.run(buf198, buf224, buf249, primals_245, primals_83, buf254, 62720, grid=grid(62720), stream=stream0)
        del primals_83
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf255 = aten.convolution_backward(buf254, div_7, primals_82, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf254
        del div_7
        del primals_82
        buf256 = buf255[0]
        buf263 = empty_strided((4, 200, 14, 14), (39200, 1, 2800, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_70.run(clone_4, buf256, primals_242, primals_80, buf263, 784, 200, grid=grid(784, 200), stream=stream0)
        del primals_80
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf264 = aten.convolution_backward(buf263, div_6, primals_79, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 200, [True, True, False])
        del div_6
        del primals_79
        buf265 = buf264[0]
        buf272 = buf263; del buf263  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_70.run(clone_3, buf265, primals_239, primals_77, buf272, 784, 200, grid=grid(784, 200), stream=stream0)
        del primals_77
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf273 = aten.convolution_backward(buf272, add_51, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_51
        del buf272
        del primals_76
        buf274 = buf273[0]
        buf226 = empty((80, ), device='cuda', dtype=torch.float32)
        buf227 = empty((80, ), device='cuda', dtype=torch.float32)
        buf251 = empty((80, ), device='cuda', dtype=torch.float32)
        buf252 = empty((80, ), device='cuda', dtype=torch.float32)
        buf276 = empty((80, ), device='cuda', dtype=torch.float32)
        buf277 = empty((80, ), device='cuda', dtype=torch.float32)
        buf228 = buf227; del buf227  # reuse
        buf253 = buf252; del buf252  # reuse
        buf278 = buf277; del buf277  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_71.run(buf228, buf253, buf278, buf198, buf224, convolution_32, primals_253, buf249, convolution_29, primals_244, buf274, convolution_26, primals_235, primals_254, primals_245, primals_236, buf226, buf251, buf276, 80, 784, grid=grid(80), stream=stream0)
        del convolution_26
        del convolution_29
        del convolution_32
        del primals_235
        del primals_244
        del primals_245
        del primals_253
        del primals_254
        buf232 = buf230[1]
        del buf230
        buf233 = reinterpret_tensor(buf219, (184, 7), (7, 1), 0); del buf219  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_63.run(clone_6, buf231, buf233, 1288, 112, grid=grid(1288), stream=stream0)
        buf234 = empty((184, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_64.run(buf233, buf234, 184, 7, grid=grid(184), stream=stream0)
        buf235 = reinterpret_tensor(buf233, (184, 7), (1, 184), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_72.run(clone_6, buf231, convolution_31, primals_250, buf235, 1288, 112, grid=grid(1288), stream=stream0)
        del buf231
        del clone_6
        del convolution_31
        del primals_250
        buf236 = empty((184, ), device='cuda', dtype=torch.float32)
        buf237 = buf236; del buf236  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_66.run(buf237, buf235, primals_251, 184, 7, grid=grid(184), stream=stream0)
        del primals_251
        buf241 = buf239[1]
        del buf239
        buf242 = reinterpret_tensor(buf235, (184, 7), (7, 1), 0); del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_63.run(clone_5, buf240, buf242, 1288, 112, grid=grid(1288), stream=stream0)
        buf243 = empty((184, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_64.run(buf242, buf243, 184, 7, grid=grid(184), stream=stream0)
        buf244 = reinterpret_tensor(buf242, (184, 7), (1, 184), 0); del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_72.run(clone_5, buf240, convolution_30, primals_247, buf244, 1288, 112, grid=grid(1288), stream=stream0)
        del buf240
        del clone_5
        del convolution_30
        del primals_247
        buf245 = empty((184, ), device='cuda', dtype=torch.float32)
        buf246 = buf245; del buf245  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_66.run(buf246, buf244, primals_248, 184, 7, grid=grid(184), stream=stream0)
        del buf244
        del primals_248
        buf250 = buf248[1]
        del buf248
        buf257 = buf255[1]
        del buf255
        buf258 = empty((200, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_73.run(clone_4, buf256, buf258, 1400, 112, grid=grid(1400), stream=stream0)
        buf259 = empty((200, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_74.run(buf258, buf259, 200, 7, grid=grid(200), stream=stream0)
        buf260 = reinterpret_tensor(buf258, (200, 7), (1, 200), 0); del buf258  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_75.run(clone_4, buf256, convolution_28, primals_241, buf260, 1400, 112, grid=grid(1400), stream=stream0)
        del buf256
        del clone_4
        del convolution_28
        del primals_241
        buf261 = empty((200, ), device='cuda', dtype=torch.float32)
        buf262 = buf261; del buf261  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_76.run(buf262, buf260, primals_242, 200, 7, grid=grid(200), stream=stream0)
        del primals_242
        buf266 = buf264[1]
        del buf264
        buf267 = reinterpret_tensor(buf260, (200, 7), (7, 1), 0); del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_73.run(clone_3, buf265, buf267, 1400, 112, grid=grid(1400), stream=stream0)
        buf268 = empty((200, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_74.run(buf267, buf268, 200, 7, grid=grid(200), stream=stream0)
        buf269 = reinterpret_tensor(buf267, (200, 7), (1, 200), 0); del buf267  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_75.run(clone_3, buf265, convolution_27, primals_238, buf269, 1400, 112, grid=grid(1400), stream=stream0)
        del buf265
        del clone_3
        del convolution_27
        del primals_238
        buf270 = empty((200, ), device='cuda', dtype=torch.float32)
        buf271 = buf270; del buf270  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_76.run(buf271, buf269, primals_239, 200, 7, grid=grid(200), stream=stream0)
        del buf269
        del primals_239
        buf275 = buf273[1]
        del buf273
        buf279 = buf198; del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_77.run(buf279, buf224, buf249, buf274, primals_236, primals_74, 62720, grid=grid(62720), stream=stream0)
        del buf224
        del buf249
        del buf274
        del primals_236
        del primals_74
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf280 = aten.convolution_backward(buf279, div_5, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf279
        del div_5
        del primals_73
        buf281 = buf280[0]
        buf282 = buf280[1]
        del buf280
        buf283 = empty((240, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_78.run(clone_2, buf281, buf283, 1680, 112, grid=grid(1680), stream=stream0)
        buf284 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_79.run(buf283, buf284, 240, 7, grid=grid(240), stream=stream0)
        buf285 = reinterpret_tensor(buf283, (240, 7), (1, 240), 0); del buf283  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_80.run(clone_2, buf281, convolution_25, primals_232, buf285, 1680, 112, grid=grid(1680), stream=stream0)
        del convolution_25
        del primals_232
        buf286 = empty((240, ), device='cuda', dtype=torch.float32)
        buf287 = buf286; del buf286  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_81.run(buf287, buf285, primals_233, 240, 7, grid=grid(240), stream=stream0)
        del buf285
        buf288 = reinterpret_tensor(buf78, (4, 240, 14, 14), (47040, 1, 3360, 240), 0); del buf78  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_82.run(clone_2, buf281, primals_233, primals_71, buf288, 784, 240, grid=grid(784, 240), stream=stream0)
        del buf281
        del clone_2
        del primals_233
        del primals_71
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf289 = aten.convolution_backward(buf288, div_4, primals_70, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 240, [True, True, False])
        del buf288
        del div_4
        del primals_70
        buf290 = buf289[0]
        buf291 = buf289[1]
        del buf289
        buf292 = empty((240, 25), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_83.run(clone_1, buf290, buf292, 6000, 126, grid=grid(6000), stream=stream0)
        buf293 = empty((240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_84.run(buf292, buf293, 240, 25, grid=grid(240), stream=stream0)
        buf294 = reinterpret_tensor(buf292, (240, 25), (1, 240), 0); del buf292  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_85.run(clone_1, buf290, convolution_24, primals_229, buf294, 6000, 126, grid=grid(6000), stream=stream0)
        del convolution_24
        del primals_229
        buf295 = empty((240, ), device='cuda', dtype=torch.float32)
        buf296 = buf295; del buf295  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_86.run(buf296, buf294, primals_230, 240, 25, grid=grid(240), stream=stream0)
        del buf294
        buf297 = empty_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_87.run(clone_1, buf290, primals_230, primals_68, buf297, 3136, 240, grid=grid(3136, 240), stream=stream0)
        del buf290
        del clone_1
        del primals_230
        del primals_68
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf298 = aten.convolution_backward(buf297, add_43, primals_67, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_43
        del buf297
        del primals_67
        buf299 = buf298[0]
        buf300 = buf298[1]
        del buf298
        buf301 = empty((40, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_88.run(buf299, buf301, 40, 3136, grid=grid(40), stream=stream0)
        buf302 = empty((40, 25), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_89.run(buf299, convolution_23, primals_226, buf302, 1000, 126, grid=grid(1000), stream=stream0)
        del convolution_23
        del primals_226
        buf303 = empty((40, ), device='cuda', dtype=torch.float32)
        buf304 = buf303; del buf303  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_90.run(buf304, buf302, primals_227, 40, 25, grid=grid(40), stream=stream0)
        del buf302
        buf305 = empty((4, 40, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_91.run(buf299, primals_227, primals_65, buf305, 125440, grid=grid(125440), stream=stream0)
        del primals_227
        del primals_65
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf306 = aten.convolution_backward(buf305, mul_54, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_54
        del primals_64
        buf307 = buf306[0]
        buf308 = buf306[1]
        del buf306
        buf309 = reinterpret_tensor(buf193, (4, 120, 1, 1, 7), (840, 7, 3360, 3360, 1), 0); del buf193  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_92.run(buf307, relu_12, buf309, 3360, 112, grid=grid(3360), stream=stream0)
        buf310 = empty_strided((4, 120, 1, 1), (120, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf311 = reinterpret_tensor(buf310, (4, 120, 1, 1), (120, 1, 120, 120), 0); del buf310  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_93.run(buf311, buf309, bitwise_and_5, 480, 7, grid=grid(480), stream=stream0)
        del bitwise_and_5
        buf312 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_50.run(buf311, buf312, 120, grid=grid(120), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf313 = aten.convolution_backward(buf311, relu_13, primals_62, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf311
        del primals_62
        buf314 = buf313[0]
        buf315 = buf313[1]
        del buf313
        buf316 = buf314; del buf314  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.threshold_backward]
        triton_poi_fused_hardswish_backward_threshold_backward_94.run(buf316, relu_13, 128, grid=grid(128), stream=stream0)
        del relu_13
        buf317 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_95.run(buf316, buf317, 32, grid=grid(32), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf318 = aten.convolution_backward(buf316, mean_2, primals_60, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf316
        del mean_2
        del primals_60
        buf319 = buf318[0]
        buf320 = buf318[1]
        del buf318
        buf321 = empty((120, 25), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_96.run(relu_12, buf307, div_3, buf319, buf321, 3000, 126, grid=grid(3000), stream=stream0)
        buf322 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_97.run(buf321, buf322, 120, 25, grid=grid(120), stream=stream0)
        buf323 = reinterpret_tensor(buf321, (120, 25), (1, 120), 0); del buf321  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_98.run(relu_12, buf307, div_3, buf319, convolution_20, primals_223, buf323, 3000, 126, grid=grid(3000), stream=stream0)
        del convolution_20
        del primals_223
        buf324 = empty((120, ), device='cuda', dtype=torch.float32)
        buf325 = buf324; del buf324  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_99.run(buf325, buf323, primals_224, 120, 25, grid=grid(120), stream=stream0)
        buf326 = reinterpret_tensor(buf196, (4, 120, 28, 28), (94080, 1, 3360, 120), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_100.run(relu_12, buf307, div_3, buf319, primals_224, primals_58, buf326, 3136, 120, grid=grid(3136, 120), stream=stream0)
        del buf307
        del div_3
        del primals_224
        del primals_58
        del relu_12
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        buf327 = aten.convolution_backward(buf326, relu_11, primals_57, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
        del primals_57
        buf328 = buf327[0]
        buf329 = buf327[1]
        del buf327
        buf330 = reinterpret_tensor(buf323, (120, 25), (25, 1), 0); del buf323  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_101.run(relu_11, buf328, buf330, 3000, 126, grid=grid(3000), stream=stream0)
        buf331 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_97.run(buf330, buf331, 120, 25, grid=grid(120), stream=stream0)
        buf332 = reinterpret_tensor(buf330, (120, 25), (1, 120), 0); del buf330  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_102.run(relu_11, buf328, convolution_19, primals_220, buf332, 3000, 126, grid=grid(3000), stream=stream0)
        del convolution_19
        del primals_220
        buf333 = empty((120, ), device='cuda', dtype=torch.float32)
        buf334 = buf333; del buf333  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_99.run(buf334, buf332, primals_221, 120, 25, grid=grid(120), stream=stream0)
        buf335 = buf326; del buf326  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_103.run(relu_11, buf328, primals_221, primals_55, buf335, 3136, 120, grid=grid(3136, 120), stream=stream0)
        del buf328
        del primals_221
        del primals_55
        del relu_11
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf336 = aten.convolution_backward(buf335, add_35, primals_54, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_35
        del primals_54
        buf337 = buf336[0]
        buf338 = buf336[1]
        del buf336
        buf342 = buf305; del buf305  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_104.run(buf299, buf337, primals_218, primals_52, buf342, 125440, grid=grid(125440), stream=stream0)
        del primals_52
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf343 = aten.convolution_backward(buf342, mul_44, primals_51, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf342
        del mul_44
        del primals_51
        buf344 = buf343[0]
        buf346 = buf309; del buf309  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_92.run(buf344, relu_9, buf346, 3360, 112, grid=grid(3360), stream=stream0)
        buf347 = reinterpret_tensor(buf319, (4, 120, 1, 1), (120, 1, 480, 480), 0); del buf319  # reuse
        buf348 = reinterpret_tensor(buf347, (4, 120, 1, 1), (120, 1, 120, 120), 0); del buf347  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_93.run(buf348, buf346, bitwise_and_6, 480, 7, grid=grid(480), stream=stream0)
        del bitwise_and_6
        del buf346
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf350 = aten.convolution_backward(buf348, relu_10, primals_49, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_49
        buf351 = buf350[0]
        buf353 = buf351; del buf351  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.threshold_backward]
        triton_poi_fused_hardswish_backward_threshold_backward_94.run(buf353, relu_10, 128, grid=grid(128), stream=stream0)
        del relu_10
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf355 = aten.convolution_backward(buf353, mean_1, primals_47, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_1
        del primals_47
        buf356 = buf355[0]
        buf363 = buf335; del buf335  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_100.run(relu_9, buf344, div_2, buf356, primals_215, primals_45, buf363, 3136, 120, grid=grid(3136, 120), stream=stream0)
        del primals_45
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        buf364 = aten.convolution_backward(buf363, relu_8, primals_44, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
        del primals_44
        buf365 = buf364[0]
        buf372 = buf363; del buf363  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_103.run(relu_8, buf365, primals_212, primals_42, buf372, 3136, 120, grid=grid(3136, 120), stream=stream0)
        del primals_42
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf373 = aten.convolution_backward(buf372, add_27, primals_41, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_27
        del buf372
        del primals_41
        buf374 = buf373[0]
        buf339 = empty((40, ), device='cuda', dtype=torch.float32)
        buf340 = empty((40, ), device='cuda', dtype=torch.float32)
        buf376 = empty((40, ), device='cuda', dtype=torch.float32)
        buf377 = empty((40, ), device='cuda', dtype=torch.float32)
        buf341 = buf340; del buf340  # reuse
        buf378 = buf377; del buf377  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_105.run(buf341, buf378, buf299, buf337, convolution_18, primals_217, buf374, convolution_13, primals_208, primals_218, primals_209, buf339, buf376, 40, 3136, grid=grid(40), stream=stream0)
        del convolution_13
        del convolution_18
        del primals_208
        del primals_217
        del primals_218
        buf345 = buf343[1]
        del buf343
        buf349 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_50.run(buf348, buf349, 120, grid=grid(120), stream=stream0)
        del buf348
        buf352 = buf350[1]
        del buf350
        buf354 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_95.run(buf353, buf354, 32, grid=grid(32), stream=stream0)
        del buf353
        buf357 = buf355[1]
        del buf355
        buf358 = reinterpret_tensor(buf332, (120, 25), (25, 1), 0); del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_96.run(relu_9, buf344, div_2, buf356, buf358, 3000, 126, grid=grid(3000), stream=stream0)
        buf359 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_97.run(buf358, buf359, 120, 25, grid=grid(120), stream=stream0)
        buf360 = reinterpret_tensor(buf358, (120, 25), (1, 120), 0); del buf358  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_106.run(relu_9, buf344, div_2, buf356, convolution_15, primals_214, buf360, 3000, 126, grid=grid(3000), stream=stream0)
        del buf344
        del buf356
        del convolution_15
        del div_2
        del primals_214
        del relu_9
        buf361 = empty((120, ), device='cuda', dtype=torch.float32)
        buf362 = buf361; del buf361  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_99.run(buf362, buf360, primals_215, 120, 25, grid=grid(120), stream=stream0)
        del primals_215
        buf366 = buf364[1]
        del buf364
        buf367 = reinterpret_tensor(buf360, (120, 25), (25, 1), 0); del buf360  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_101.run(relu_8, buf365, buf367, 3000, 126, grid=grid(3000), stream=stream0)
        buf368 = empty((120, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_97.run(buf367, buf368, 120, 25, grid=grid(120), stream=stream0)
        buf369 = reinterpret_tensor(buf367, (120, 25), (1, 120), 0); del buf367  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_107.run(relu_8, buf365, convolution_14, primals_211, buf369, 3000, 126, grid=grid(3000), stream=stream0)
        del buf365
        del convolution_14
        del primals_211
        del relu_8
        buf370 = empty((120, ), device='cuda', dtype=torch.float32)
        buf371 = buf370; del buf370  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_99.run(buf371, buf369, primals_212, 120, 25, grid=grid(120), stream=stream0)
        del buf369
        del primals_212
        buf375 = buf373[1]
        del buf373
        buf379 = buf299; del buf299  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_108.run(buf379, buf337, buf374, primals_209, primals_39, 125440, grid=grid(125440), stream=stream0)
        del buf337
        del buf374
        del primals_209
        del primals_39
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf380 = aten.convolution_backward(buf379, mul_34, primals_38, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf379
        del mul_34
        del primals_38
        buf381 = buf380[0]
        buf382 = buf380[1]
        del buf380
        buf383 = empty_strided((4, 72, 1, 1, 7), (504, 7, 2016, 2016, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_109.run(buf381, relu_6, buf383, 2016, 112, grid=grid(2016), stream=stream0)
        buf384 = empty_strided((4, 72, 1, 1), (72, 1, 288, 288), device='cuda', dtype=torch.float32)
        buf385 = reinterpret_tensor(buf384, (4, 72, 1, 1), (72, 1, 72, 72), 0); del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.hardsigmoid_backward, aten.hardswish_backward, aten.mul, aten.sum]
        triton_per_fused_hardsigmoid_backward_hardswish_backward_mul_sum_110.run(buf385, buf383, bitwise_and_7, 288, 7, grid=grid(288), stream=stream0)
        del bitwise_and_7
        del buf383
        buf386 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_111.run(buf385, buf386, 72, grid=grid(72), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf387 = aten.convolution_backward(buf385, relu_7, primals_36, [72], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf385
        del primals_36
        buf388 = buf387[0]
        buf389 = buf387[1]
        del buf387
        buf390 = buf388; del buf388  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.threshold_backward]
        triton_poi_fused_hardswish_backward_threshold_backward_112.run(buf390, relu_7, 96, grid=grid(96), stream=stream0)
        del relu_7
        buf391 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_poi_fused_convolution_backward_113.run(buf390, buf391, 24, grid=grid(24), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf392 = aten.convolution_backward(buf390, mean, primals_34, [24], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf390
        del mean
        del primals_34
        buf393 = buf392[0]
        buf394 = buf392[1]
        del buf392
        buf395 = empty((72, 25), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_114.run(relu_6, buf381, div_1, buf393, buf395, 1800, 126, grid=grid(1800), stream=stream0)
        buf396 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_115.run(buf395, buf396, 72, 25, grid=grid(72), stream=stream0)
        buf397 = reinterpret_tensor(buf395, (72, 25), (1, 72), 0); del buf395  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_116.run(relu_6, buf381, div_1, buf393, convolution_10, primals_205, buf397, 1800, 126, grid=grid(1800), stream=stream0)
        del convolution_10
        del primals_205
        buf398 = empty((72, ), device='cuda', dtype=torch.float32)
        buf399 = buf398; del buf398  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_117.run(buf399, buf397, primals_206, 72, 25, grid=grid(72), stream=stream0)
        del buf397
        buf400 = empty_strided((4, 72, 28, 28), (56448, 1, 2016, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_118.run(relu_6, buf381, div_1, buf393, primals_206, primals_32, buf400, 3136, 72, grid=grid(3136, 72), stream=stream0)
        del buf381
        del buf393
        del div_1
        del primals_206
        del primals_32
        del relu_6
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
        buf401 = aten.convolution_backward(buf400, relu_5, primals_31, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 72, [True, True, False])
        del buf400
        del primals_31
        buf402 = buf401[0]
        buf403 = buf401[1]
        del buf401
        buf404 = empty((72, 98), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_119.run(relu_5, buf402, buf404, 7056, 128, grid=grid(7056), stream=stream0)
        buf405 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_120.run(buf404, buf405, 72, 98, grid=grid(72), stream=stream0)
        buf406 = reinterpret_tensor(buf404, (72, 98), (1, 72), 0); del buf404  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_121.run(relu_5, buf402, convolution_9, primals_202, buf406, 7056, 128, grid=grid(7056), stream=stream0)
        del convolution_9
        del primals_202
        buf407 = empty((72, ), device='cuda', dtype=torch.float32)
        buf408 = buf407; del buf407  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_122.run(buf408, buf406, primals_203, 72, 98, grid=grid(72), stream=stream0)
        buf409 = empty_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_123.run(relu_5, buf402, primals_203, primals_29, buf409, 12544, 72, grid=grid(12544, 72), stream=stream0)
        del buf402
        del primals_203
        del primals_29
        del relu_5
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf410 = aten.convolution_backward(buf409, add_20, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_20
        del primals_28
        buf411 = buf410[0]
        buf412 = buf410[1]
        del buf410
        buf418 = empty((4, 24, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_124.run(buf411, primals_200, primals_26, buf418, 301056, grid=grid(301056), stream=stream0)
        del primals_26
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf419 = aten.convolution_backward(buf418, relu_4, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf418
        del primals_25
        buf420 = buf419[0]
        buf427 = buf409; del buf409  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_123.run(relu_4, buf420, primals_197, primals_23, buf427, 12544, 72, grid=grid(12544, 72), stream=stream0)
        del primals_23
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf428 = aten.convolution_backward(buf427, relu_3, primals_22, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 72, [True, True, False])
        del primals_22
        buf429 = buf428[0]
        buf436 = buf427; del buf427  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_123.run(relu_3, buf429, primals_194, primals_20, buf436, 12544, 72, grid=grid(12544, 72), stream=stream0)
        del primals_20
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf437 = aten.convolution_backward(buf436, add_13, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_13
        del buf436
        del primals_19
        buf438 = buf437[0]
        buf413 = empty_strided((24, 2), (1, 24), device='cuda', dtype=torch.float32)
        buf440 = empty_strided((24, 2), (1, 24), device='cuda', dtype=torch.float32)
        buf442 = empty_strided((24, 2), (1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_125.run(buf411, buf438, convolution_5, primals_190, buf413, buf440, buf442, 48, 6272, grid=grid(48), stream=stream0)
        del convolution_5
        del primals_190
        buf414 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_126.run(buf413, buf414, 24, 2, grid=grid(24), stream=stream0)
        del buf413
        buf415 = empty((24, 98), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_127.run(buf411, convolution_8, primals_199, buf415, 2352, 128, grid=grid(2352), stream=stream0)
        del convolution_8
        del primals_199
        buf416 = empty((24, ), device='cuda', dtype=torch.float32)
        buf417 = buf416; del buf416  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_128.run(buf417, buf415, primals_200, 24, 98, grid=grid(24), stream=stream0)
        del buf415
        del primals_200
        buf421 = buf419[1]
        del buf419
        buf422 = reinterpret_tensor(buf406, (72, 98), (98, 1), 0); del buf406  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_119.run(relu_4, buf420, buf422, 7056, 128, grid=grid(7056), stream=stream0)
        buf423 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_120.run(buf422, buf423, 72, 98, grid=grid(72), stream=stream0)
        buf424 = reinterpret_tensor(buf422, (72, 98), (1, 72), 0); del buf422  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_129.run(relu_4, buf420, convolution_7, primals_196, buf424, 7056, 128, grid=grid(7056), stream=stream0)
        del buf420
        del convolution_7
        del primals_196
        del relu_4
        buf425 = empty((72, ), device='cuda', dtype=torch.float32)
        buf426 = buf425; del buf425  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_122.run(buf426, buf424, primals_197, 72, 98, grid=grid(72), stream=stream0)
        del primals_197
        buf430 = buf428[1]
        del buf428
        buf431 = reinterpret_tensor(buf424, (72, 98), (98, 1), 0); del buf424  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_119.run(relu_3, buf429, buf431, 7056, 128, grid=grid(7056), stream=stream0)
        buf432 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_120.run(buf431, buf432, 72, 98, grid=grid(72), stream=stream0)
        buf433 = reinterpret_tensor(buf431, (72, 98), (1, 72), 0); del buf431  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_129.run(relu_3, buf429, convolution_6, primals_193, buf433, 7056, 128, grid=grid(7056), stream=stream0)
        del buf429
        del convolution_6
        del primals_193
        del relu_3
        buf434 = empty((72, ), device='cuda', dtype=torch.float32)
        buf435 = buf434; del buf434  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_122.run(buf435, buf433, primals_194, 72, 98, grid=grid(72), stream=stream0)
        del buf433
        del primals_194
        buf439 = buf437[1]
        del buf437
        buf441 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_126.run(buf440, buf441, 24, 2, grid=grid(24), stream=stream0)
        del buf440
        buf443 = empty((24, ), device='cuda', dtype=torch.float32)
        buf444 = buf443; del buf443  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_130.run(buf444, buf442, primals_191, 24, 2, grid=grid(24), stream=stream0)
        del buf442
        buf445 = buf411; del buf411  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_131.run(buf445, buf438, primals_191, primals_17, 301056, grid=grid(301056), stream=stream0)
        del buf438
        del primals_17
        del primals_191
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf446 = aten.convolution_backward(buf445, relu_2, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf445
        del primals_16
        buf447 = buf446[0]
        buf448 = buf446[1]
        del buf446
        buf449 = empty((64, 98), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_132.run(relu_2, buf447, buf449, 6272, 128, grid=grid(6272), stream=stream0)
        buf450 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_133.run(buf449, buf450, 64, 98, grid=grid(64), stream=stream0)
        buf451 = reinterpret_tensor(buf449, (64, 98), (1, 64), 0); del buf449  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_134.run(relu_2, buf447, convolution_4, primals_187, buf451, 6272, 128, grid=grid(6272), stream=stream0)
        del convolution_4
        del primals_187
        buf452 = empty((64, ), device='cuda', dtype=torch.float32)
        buf453 = buf452; del buf452  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_135.run(buf453, buf451, primals_188, 64, 98, grid=grid(64), stream=stream0)
        buf454 = empty_strided((4, 64, 56, 56), (200704, 1, 3584, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_136.run(relu_2, buf447, primals_188, primals_14, buf454, 12544, 64, grid=grid(12544, 64), stream=stream0)
        del buf447
        del primals_14
        del primals_188
        del relu_2
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf455 = aten.convolution_backward(buf454, relu_1, primals_13, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False])
        del primals_13
        buf456 = buf455[0]
        buf457 = buf455[1]
        del buf455
        buf458 = empty((64, 392), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_137.run(relu_1, buf456, buf458, 25088, 128, grid=grid(25088), stream=stream0)
        buf459 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_138.run(buf458, buf459, 64, 392, grid=grid(64), stream=stream0)
        buf460 = reinterpret_tensor(buf458, (64, 392), (1, 64), 0); del buf458  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_139.run(relu_1, buf456, convolution_3, primals_184, buf460, 25088, 128, grid=grid(25088), stream=stream0)
        del convolution_3
        del primals_184
        buf461 = empty((64, ), device='cuda', dtype=torch.float32)
        buf462 = buf461; del buf461  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_140.run(buf462, buf460, primals_185, 64, 392, grid=grid(64), stream=stream0)
        del buf460
        buf463 = empty_strided((4, 64, 112, 112), (802816, 1, 7168, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_141.run(relu_1, buf456, primals_185, primals_11, buf463, 50176, 64, grid=grid(50176, 64), stream=stream0)
        del buf456
        del primals_11
        del primals_185
        del relu_1
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf464 = aten.convolution_backward(buf463, add_7, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_7
        del buf463
        del primals_10
        buf465 = buf464[0]
        buf466 = buf464[1]
        del buf464
        buf472 = reinterpret_tensor(buf454, (4, 16, 112, 112), (200704, 12544, 112, 1), 0); del buf454  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_142.run(buf465, primals_182, primals_8, buf472, 802816, grid=grid(802816), stream=stream0)
        del primals_8
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf473 = aten.convolution_backward(buf472, relu, primals_7, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_7
        buf474 = buf473[0]
        buf481 = reinterpret_tensor(buf472, (4, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf472  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_143.run(relu, buf474, primals_179, primals_5, buf481, 50176, 16, grid=grid(50176, 16), stream=stream0)
        del primals_5
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf482 = aten.convolution_backward(buf481, div, primals_4, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False])
        del buf481
        del div
        del primals_4
        buf483 = buf482[0]
        buf467 = empty((16, 7), device='cuda', dtype=torch.float32)
        buf485 = empty((16, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_add_hardswish_backward_native_batch_norm_backward_144.run(buf465, clone, buf483, buf467, buf485, 112, 7168, grid=grid(112), stream=stream0)
        buf468 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_145.run(buf467, buf468, 16, 7, grid=grid(16), stream=stream0)
        del buf467
        buf469 = reinterpret_tensor(buf451, (16, 392), (392, 1), 0); del buf451  # reuse
        buf487 = empty((16, 392), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_add_hardswish_backward_native_batch_norm_backward_146.run(buf465, convolution_2, primals_181, clone, buf483, convolution, primals_175, buf469, buf487, 6272, 128, grid=grid(6272), stream=stream0)
        del convolution
        del convolution_2
        del primals_175
        del primals_181
        buf470 = empty((16, ), device='cuda', dtype=torch.float32)
        buf471 = buf470; del buf470  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_147.run(buf471, buf469, primals_182, 16, 392, grid=grid(16), stream=stream0)
        del primals_182
        buf475 = buf473[1]
        del buf473
        buf476 = buf469; del buf469  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_148.run(relu, buf474, buf476, 6272, 128, grid=grid(6272), stream=stream0)
        buf477 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_149.run(buf476, buf477, 16, 392, grid=grid(16), stream=stream0)
        buf478 = reinterpret_tensor(buf476, (16, 392), (1, 16), 0); del buf476  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_150.run(relu, buf474, convolution_1, primals_178, buf478, 6272, 128, grid=grid(6272), stream=stream0)
        del convolution_1
        del primals_178
        del relu
        buf479 = empty((16, ), device='cuda', dtype=torch.float32)
        buf480 = buf479; del buf479  # reuse
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_151.run(buf480, buf478, primals_179, 16, 392, grid=grid(16), stream=stream0)
        del buf478
        del primals_179
        buf484 = buf482[1]
        del buf482
        buf486 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_145.run(buf485, buf486, 16, 7, grid=grid(16), stream=stream0)
        del buf485
        buf488 = empty((16, ), device='cuda', dtype=torch.float32)
        buf489 = buf488; del buf488  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_147.run(buf489, buf487, primals_176, 16, 392, grid=grid(16), stream=stream0)
        del buf487
        buf490 = reinterpret_tensor(buf474, (4, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf474  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_hardswish_backward_native_batch_norm_backward_152.run(clone, buf465, buf483, primals_176, primals_2, buf490, 64, 12544, grid=grid(64, 12544), stream=stream0)
        del buf465
        del buf483
        del clone
        del primals_176
        del primals_2
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf491 = aten.convolution_backward(buf490, primals_313, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf490
        del primals_1
        del primals_313
        buf492 = buf491[1]
        return (buf492, buf489, buf486, buf484, buf480, buf477, buf475, buf471, buf468, buf466, buf462, buf459, buf457, buf453, buf450, buf448, buf444, buf441, buf439, buf435, buf432, buf430, buf426, buf423, buf421, buf417, buf414, buf412, buf408, buf405, buf403, buf399, buf396, buf394, buf391, buf389, buf386, buf382, buf378, buf376, buf375, buf371, buf368, buf366, buf362, buf359, buf357, buf354, buf352, buf349, buf345, buf341, buf339, buf338, buf334, buf331, buf329, buf325, buf322, buf320, buf317, buf315, buf312, buf308, buf304, buf301, buf300, buf296, buf293, buf291, buf287, buf284, buf282, buf278, buf276, buf275, buf271, buf268, buf266, buf262, buf259, buf257, buf253, buf251, buf250, buf246, buf243, buf241, buf237, buf234, buf232, buf228, buf226, buf225, buf221, buf218, buf216, buf212, buf209, buf207, buf203, buf200, buf199, buf195, buf192, buf190, buf186, buf183, buf181, buf178, buf176, buf173, buf169, buf165, buf163, buf162, buf158, buf155, buf153, buf149, buf146, buf144, buf141, buf139, buf136, buf132, buf128, buf125, buf124, buf120, buf117, buf115, buf111, buf108, buf106, buf103, buf101, buf98, buf95, buf91, buf89, buf88, buf84, buf81, buf79, buf75, buf72, buf70, buf67, buf65, buf62, buf59, buf55, buf53, buf52, buf48, buf45, buf43, buf39, buf36, buf34, buf31, buf29, buf26, buf23, buf19, buf16, buf15, buf11, buf8, reinterpret_tensor(buf5, (1280, 960), (960, 1), 0), buf6, reinterpret_tensor(buf1, (1000, 1280), (1280, 1), 0), buf2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((24, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((72, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((72, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((40, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((32, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((120, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((32, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((120, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((200, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((200, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((80, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((184, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((184, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((80, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((184, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((184, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((80, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((120, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((480, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((112, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((672, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((160, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((960, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((960, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    clone = rand_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    div = rand_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    add_7 = rand_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((4, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((4, 64, 112, 112), (802816, 1, 7168, 64), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((4, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((4, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    add_13 = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    add_20 = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((4, 72, 28, 28), (56448, 1, 2016, 72), device='cuda:0', dtype=torch.float32)
    relu_6 = rand_strided((4, 72, 28, 28), (56448, 1, 2016, 72), device='cuda:0', dtype=torch.float32)
    mean = rand_strided((4, 72, 1, 1), (72, 1, 72, 72), device='cuda:0', dtype=torch.float32)
    relu_7 = rand_strided((4, 24, 1, 1), (24, 1, 24, 24), device='cuda:0', dtype=torch.float32)
    div_1 = rand_strided((4, 72, 1, 1), (72, 1, 72, 72), device='cuda:0', dtype=torch.float32)
    mul_34 = rand_strided((4, 72, 28, 28), (56448, 1, 2016, 72), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    add_27 = rand_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    relu_9 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    mean_1 = rand_strided((4, 120, 1, 1), (120, 1, 120, 120), device='cuda:0', dtype=torch.float32)
    relu_10 = rand_strided((4, 32, 1, 1), (32, 1, 32, 32), device='cuda:0', dtype=torch.float32)
    div_2 = rand_strided((4, 120, 1, 1), (120, 1, 120, 120), device='cuda:0', dtype=torch.float32)
    mul_44 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    add_35 = rand_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    relu_11 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    relu_12 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    mean_2 = rand_strided((4, 120, 1, 1), (120, 1, 120, 120), device='cuda:0', dtype=torch.float32)
    relu_13 = rand_strided((4, 32, 1, 1), (32, 1, 32, 32), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((4, 120, 1, 1), (120, 1, 120, 120), device='cuda:0', dtype=torch.float32)
    mul_54 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    add_43 = rand_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cuda:0', dtype=torch.float32)
    clone_1 = rand_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cuda:0', dtype=torch.float32)
    div_4 = rand_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((4, 240, 14, 14), (47040, 1, 3360, 240), device='cuda:0', dtype=torch.float32)
    clone_2 = rand_strided((4, 240, 14, 14), (47040, 1, 3360, 240), device='cuda:0', dtype=torch.float32)
    div_5 = rand_strided((4, 240, 14, 14), (47040, 1, 3360, 240), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    add_51 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((4, 200, 14, 14), (39200, 1, 2800, 200), device='cuda:0', dtype=torch.float32)
    clone_3 = rand_strided((4, 200, 14, 14), (39200, 1, 2800, 200), device='cuda:0', dtype=torch.float32)
    div_6 = rand_strided((4, 200, 14, 14), (39200, 1, 2800, 200), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((4, 200, 14, 14), (39200, 1, 2800, 200), device='cuda:0', dtype=torch.float32)
    clone_4 = rand_strided((4, 200, 14, 14), (39200, 1, 2800, 200), device='cuda:0', dtype=torch.float32)
    div_7 = rand_strided((4, 200, 14, 14), (39200, 1, 2800, 200), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    add_60 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cuda:0', dtype=torch.float32)
    clone_5 = rand_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cuda:0', dtype=torch.float32)
    div_8 = rand_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cuda:0', dtype=torch.float32)
    clone_6 = rand_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cuda:0', dtype=torch.float32)
    div_9 = rand_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    add_69 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cuda:0', dtype=torch.float32)
    clone_7 = rand_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cuda:0', dtype=torch.float32)
    div_10 = rand_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cuda:0', dtype=torch.float32)
    clone_8 = rand_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cuda:0', dtype=torch.float32)
    div_11 = rand_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    add_78 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    clone_9 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    div_12 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    clone_10 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    div_13 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    mean_3 = rand_strided((4, 480, 1, 1), (480, 1, 480, 480), device='cuda:0', dtype=torch.float32)
    relu_14 = rand_strided((4, 120, 1, 1), (120, 1, 120, 120), device='cuda:0', dtype=torch.float32)
    div_14 = rand_strided((4, 480, 1, 1), (480, 1, 480, 480), device='cuda:0', dtype=torch.float32)
    mul_110 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((4, 112, 14, 14), (21952, 1, 1568, 112), device='cuda:0', dtype=torch.float32)
    add_87 = rand_strided((4, 112, 14, 14), (21952, 1, 1568, 112), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    clone_11 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    div_15 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    clone_12 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    div_16 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    mean_4 = rand_strided((4, 672, 1, 1), (672, 1, 672, 672), device='cuda:0', dtype=torch.float32)
    relu_15 = rand_strided((4, 168, 1, 1), (168, 1, 168, 168), device='cuda:0', dtype=torch.float32)
    div_17 = rand_strided((4, 672, 1, 1), (672, 1, 672, 672), device='cuda:0', dtype=torch.float32)
    mul_122 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((4, 112, 14, 14), (21952, 1, 1568, 112), device='cuda:0', dtype=torch.float32)
    add_97 = rand_strided((4, 112, 14, 14), (21952, 1, 1568, 112), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    clone_13 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    div_18 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cuda:0', dtype=torch.float32)
    convolution_47 = rand_strided((4, 672, 7, 7), (32928, 1, 4704, 672), device='cuda:0', dtype=torch.float32)
    clone_14 = rand_strided((4, 672, 7, 7), (32928, 1, 4704, 672), device='cuda:0', dtype=torch.float32)
    div_19 = rand_strided((4, 672, 7, 7), (32928, 1, 4704, 672), device='cuda:0', dtype=torch.float32)
    mean_5 = rand_strided((4, 672, 1, 1), (672, 1, 672, 672), device='cuda:0', dtype=torch.float32)
    relu_16 = rand_strided((4, 168, 1, 1), (168, 1, 168, 168), device='cuda:0', dtype=torch.float32)
    div_20 = rand_strided((4, 672, 1, 1), (672, 1, 672, 672), device='cuda:0', dtype=torch.float32)
    mul_134 = rand_strided((4, 672, 7, 7), (32928, 1, 4704, 672), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((4, 160, 7, 7), (7840, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    add_106 = rand_strided((4, 160, 7, 7), (7840, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    clone_15 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    div_21 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    convolution_52 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    clone_16 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    div_22 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    mean_6 = rand_strided((4, 960, 1, 1), (960, 1, 960, 960), device='cuda:0', dtype=torch.float32)
    relu_17 = rand_strided((4, 240, 1, 1), (240, 1, 240, 240), device='cuda:0', dtype=torch.float32)
    div_23 = rand_strided((4, 960, 1, 1), (960, 1, 960, 960), device='cuda:0', dtype=torch.float32)
    mul_146 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    convolution_55 = rand_strided((4, 160, 7, 7), (7840, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    add_116 = rand_strided((4, 160, 7, 7), (7840, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    convolution_56 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    clone_17 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    div_24 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    convolution_57 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    clone_18 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    div_25 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    mean_7 = rand_strided((4, 960, 1, 1), (960, 1, 960, 960), device='cuda:0', dtype=torch.float32)
    relu_18 = rand_strided((4, 240, 1, 1), (240, 1, 240, 240), device='cuda:0', dtype=torch.float32)
    div_26 = rand_strided((4, 960, 1, 1), (960, 1, 960, 960), device='cuda:0', dtype=torch.float32)
    mul_158 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    convolution_60 = rand_strided((4, 160, 7, 7), (7840, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    add_126 = rand_strided((4, 160, 7, 7), (7840, 1, 1120, 160), device='cuda:0', dtype=torch.float32)
    convolution_61 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    clone_19 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cuda:0', dtype=torch.float32)
    view = rand_strided((4, 960), (960, 1), device='cuda:0', dtype=torch.float32)
    addmm = rand_strided((4, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    div_28 = rand_strided((4, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_2 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_6 = rand_strided((1280, 960), (960, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and = rand_strided((4, 960, 1, 1), (960, 1, 960, 960), device='cuda:0', dtype=torch.bool)
    bitwise_and_1 = rand_strided((4, 960, 1, 1), (960, 1, 960, 960), device='cuda:0', dtype=torch.bool)
    bitwise_and_2 = rand_strided((4, 672, 1, 1), (672, 1, 672, 672), device='cuda:0', dtype=torch.bool)
    bitwise_and_3 = rand_strided((4, 672, 1, 1), (672, 1, 672, 672), device='cuda:0', dtype=torch.bool)
    bitwise_and_4 = rand_strided((4, 480, 1, 1), (480, 1, 480, 480), device='cuda:0', dtype=torch.bool)
    bitwise_and_5 = rand_strided((4, 120, 1, 1), (120, 1, 120, 120), device='cuda:0', dtype=torch.bool)
    bitwise_and_6 = rand_strided((4, 120, 1, 1), (120, 1, 120, 120), device='cuda:0', dtype=torch.bool)
    bitwise_and_7 = rand_strided((4, 72, 1, 1), (72, 1, 72, 72), device='cuda:0', dtype=torch.bool)
    tangents_1 = rand_strided((4, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_36, primals_38, primals_39, primals_41, primals_42, primals_44, primals_45, primals_47, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_60, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_111, primals_113, primals_114, primals_116, primals_117, primals_119, primals_120, primals_122, primals_124, primals_126, primals_127, primals_129, primals_130, primals_132, primals_133, primals_135, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_150, primals_152, primals_153, primals_155, primals_156, primals_158, primals_159, primals_161, primals_163, primals_165, primals_166, primals_168, primals_169, primals_175, primals_176, primals_178, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, primals_190, primals_191, primals_193, primals_194, primals_196, primals_197, primals_199, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_221, primals_223, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_233, primals_235, primals_236, primals_238, primals_239, primals_241, primals_242, primals_244, primals_245, primals_247, primals_248, primals_250, primals_251, primals_253, primals_254, primals_256, primals_257, primals_259, primals_260, primals_262, primals_263, primals_265, primals_266, primals_268, primals_269, primals_271, primals_272, primals_274, primals_275, primals_277, primals_278, primals_280, primals_281, primals_283, primals_284, primals_286, primals_287, primals_289, primals_290, primals_292, primals_293, primals_295, primals_296, primals_298, primals_299, primals_301, primals_302, primals_304, primals_305, primals_307, primals_308, primals_310, primals_311, primals_313, convolution, clone, div, convolution_1, relu, convolution_2, add_7, convolution_3, relu_1, convolution_4, relu_2, convolution_5, add_13, convolution_6, relu_3, convolution_7, relu_4, convolution_8, add_20, convolution_9, relu_5, convolution_10, relu_6, mean, relu_7, div_1, mul_34, convolution_13, add_27, convolution_14, relu_8, convolution_15, relu_9, mean_1, relu_10, div_2, mul_44, convolution_18, add_35, convolution_19, relu_11, convolution_20, relu_12, mean_2, relu_13, div_3, mul_54, convolution_23, add_43, convolution_24, clone_1, div_4, convolution_25, clone_2, div_5, convolution_26, add_51, convolution_27, clone_3, div_6, convolution_28, clone_4, div_7, convolution_29, add_60, convolution_30, clone_5, div_8, convolution_31, clone_6, div_9, convolution_32, add_69, convolution_33, clone_7, div_10, convolution_34, clone_8, div_11, convolution_35, add_78, convolution_36, clone_9, div_12, convolution_37, clone_10, div_13, mean_3, relu_14, div_14, mul_110, convolution_40, add_87, convolution_41, clone_11, div_15, convolution_42, clone_12, div_16, mean_4, relu_15, div_17, mul_122, convolution_45, add_97, convolution_46, clone_13, div_18, convolution_47, clone_14, div_19, mean_5, relu_16, div_20, mul_134, convolution_50, add_106, convolution_51, clone_15, div_21, convolution_52, clone_16, div_22, mean_6, relu_17, div_23, mul_146, convolution_55, add_116, convolution_56, clone_17, div_24, convolution_57, clone_18, div_25, mean_7, relu_18, div_26, mul_158, convolution_60, add_126, convolution_61, clone_19, view, addmm, div_28, permute_2, permute_6, bitwise_and, bitwise_and_1, bitwise_and_2, bitwise_and_3, bitwise_and_4, bitwise_and_5, bitwise_and_6, bitwise_and_7, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mobilenet_v3_large', benchmark_compiled_module)
