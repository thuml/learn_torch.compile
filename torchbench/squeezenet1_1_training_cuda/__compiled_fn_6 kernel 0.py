
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


# kernel path: /tmp/torchinductor_youkaichao/46/c467635w5q6phnvctmnlgdk2gbnoflchbqndcrgsdyhcgp3rskvz.py
# Source Nodes: [], Original ATen: [aten.div, aten.threshold_backward]

triton_poi_fused_div_threshold_backward_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_threshold_backward_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 676000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1000
    x2 = (xindex // 169000)
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + (1000*x2)), xmask, eviction_policy='evict_last')
    tmp2 = 169.0
    tmp3 = tmp1 / tmp2
    tmp4 = 0.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/eq/ceqbrrggzueqnyxqqlk7csncrmcrzn3mwf56epxqqih36win4iiy.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6000
    rnumel = 113
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 1000)
    x0 = xindex % 1000
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (113*x1)
        tmp1 = tl.full([1, 1], 676, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (1000*((r2 + (113*x1)) % 676))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x6/cx665o7psywfrm2f5rjvknaz5ovdjsynhqwfatpfvmv6zgin5ghy.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1000
    rnumel = 6
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


# kernel path: /tmp/torchinductor_youkaichao/vv/cvvuubxpt6hp5nvirzpjazxuzcoydjyvfbsibkntbrqku3wt23hj.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 676
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 169
    y1 = (yindex // 169)
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (43264 + y0 + (169*x2) + (86528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp3, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4u/c4u2hh2z254klcftrkpq6loilc6jzznigkcmvsuwbojg4bmiosmv.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 113
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 256)
    x0 = xindex % 256
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (113*x1)
        tmp1 = tl.full([1, 1], 676, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (256*((r2 + (113*x1)) % 676))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7t/c7t42lzsxviuq56p2eb72yn3dfwwocrcckfvsrcfhckw4uvvsr5a.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 6
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ja/cjalox3hnzm76e7pp5a5sjng6vktecfg6oqa22gkl5ht3lwry2x4.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 676
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 169
    y1 = (yindex // 169)
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (169*x2) + (86528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp3, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j4/cj4esldepy4u3kzb724lbflqdc43qsqkvhsxm4tjltjq3qh7san7.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 169
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
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (10816*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (169*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (169*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tl.store(out_ptr0 + (y0 + (64*x2) + (10816*y1)), tmp6, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n2/cn2k4ope6hkrcdttbhqiqzt5b4cs3tnv4rvststx3hwsdxpdqpin.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 113
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (113*x1)
        tmp1 = tl.full([1, 1], 676, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (64*((r2 + (113*x1)) % 676))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ei/cei226iuriy7j5slk736ksdaxs6frajdeklhklytugjdccyohyly.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 6
    RBLOCK: tl.constexpr = 8
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


# kernel path: /tmp/torchinductor_youkaichao/qc/cqcjmv77ozokh2kz5veoagnc6x57zznpwqwib77pfs7q6o4pmv64.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 676
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 169
    y1 = (yindex // 169)
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (32448 + y0 + (169*x2) + (64896*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp3, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bp/cbpj2sqxmdlwi7lzyq2si5lrznxqyzx3tyajhqsoemnutfyvu447.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1152
    rnumel = 113
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 192)
    x0 = xindex % 192
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (113*x1)
        tmp1 = tl.full([1, 1], 676, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (192*((r2 + (113*x1)) % 676))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j5/cj5k4zxuec4liavfxr523vsr4l4oh5krdko74ina2tb6nke6phdq.py
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
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 6
    RBLOCK: tl.constexpr = 8
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


# kernel path: /tmp/torchinductor_youkaichao/my/cmyvkks3ihpsb56yqobyn7bxehoftsygb5cw5fffhaufkjcfcrqw.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 676
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 169
    y1 = (yindex // 169)
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (169*x2) + (64896*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp3, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/75/c75xs4wh45xaviij7pu2gebma4darbjqczebsl7chds7qhtd7n7q.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 169
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 48
    y1 = (yindex // 48)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (48*x2) + (8112*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (169*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (169*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tl.store(out_ptr0 + (y0 + (48*x2) + (8112*y1)), tmp6, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zd/czdjfzrsc7plmwq2sfwb3rvewbgr44jmjulxrrhpmyj2cnjad32g.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 288
    rnumel = 113
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 48)
    x0 = xindex % 48
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (113*x1)
        tmp1 = tl.full([1, 1], 676, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (48*((r2 + (113*x1)) % 676))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/27/c273xkq5h65rjyfcsmdcq7ie4hgoow65m7tkqk3oi2gxlvz5hz6y.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 8],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 48
    rnumel = 6
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (48*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wn/cwnaxlv2h2fjqf45xwn4qk5vapq5yttkyh4wffcqwo3mqcxuxvqw.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 373248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (128 + x0 + (256*x1)), xmask)
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tl.store(out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6d/c6d3h4ccfs5rsxzqjowgbujavw3z4gnv42e6rbcpihl7jyijqzzm.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2944
    rnumel = 127
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 128)
    x0 = xindex % 128
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (127*x1)
        tmp1 = tl.full([1, 1], 2916, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (128*((r2 + (127*x1)) % 2916))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qy/cqyg7ecf4fsxq7mizhamy4wut4td2n45vycibvrgryfccoi2abxh.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 23
    RBLOCK: tl.constexpr = 32
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


# kernel path: /tmp/torchinductor_youkaichao/ar/careqndc2q4xjspxccnwp6ajuttj3eg7dsdlikcl2m4lyv2t4loh.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 373248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + (256*x1)), xmask)
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tl.store(out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sl/cslluhx5bvcurofq5zudcexrmremynddi57n35e5tkwm5e43rpqe.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 729
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
    tmp0 = tl.load(in_ptr0 + (y0 + (32*x2) + (23328*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (729*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (729*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tl.store(out_ptr0 + (y0 + (32*x2) + (23328*y1)), tmp6, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vg/cvgbp2a23hvqdtmhn3xn2xjcihu6jmrorakqvordn6dpc6uluiwp.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 736
    rnumel = 127
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 32)
    x0 = xindex % 32
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (127*x1)
        tmp1 = tl.full([1, 1], 2916, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (32*((r2 + (127*x1)) % 2916))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yy/cyydfq6eikkkdzegev67svib4xy7zwlor2v5x3ke5pc4xdmqxact.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 23
    RBLOCK: tl.constexpr = 32
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


# kernel path: /tmp/torchinductor_youkaichao/fw/cfwqf2a2ltvaoqpeyontbj4rtgxzbyxumowv3oytb6lk7kx3bwaj.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2916
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 729
    y1 = (yindex // 729)
    tmp0 = tl.load(in_ptr0 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (93312 + y0 + (729*x2) + (186624*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp3, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5t/c5tfy3tke5tttuikxt3zyjr3vfqi7jolrwhgcusneg7wtextojxm.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2916
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 729
    y1 = (yindex // 729)
    tmp0 = tl.load(in_ptr0 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (729*x2) + (186624*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp3, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3q/c3qnnvxziyoqihwy6gc3rzjzty37mjfphsptkqhqewfpc4zy6baf.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 774400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (64 + x0 + (128*x1)), xmask)
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tl.store(out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jc/cjcgoxekfw3moo5yoon57dawaqa4qrhrv4zd4c3w5dqpnxeuan2q.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6080
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 12100, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (64*((r2 + (128*x1)) % 12100))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m3/cm3gcjob7h5y2f3dqgfsrin5dpkkeuo4gjlck7uhcgpatt62f352.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 95
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


# kernel path: /tmp/torchinductor_youkaichao/nd/cndljji5mpn7lchc3odhrjyj3vaxbnnzb2tb7oc2refqa6oodx2x.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 774400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x0 + (128*x1)), xmask)
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tl.store(out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4k/c4kq2wwzfyckk4pkrfl3k7nihzrpvcknq6qj3wsq62rqdksh5db2.py
# Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]

triton_poi_fused_add_threshold_backward_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_threshold_backward_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 3025
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
    tmp0 = tl.load(in_ptr0 + (y0 + (16*x2) + (48400*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (3025*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (3025*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.where(tmp2, tmp1, tmp5)
    tl.store(out_ptr0 + (y0 + (16*x2) + (48400*y1)), tmp6, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oq/coqznmpaqxgwcsx6wcwsumjtvtsdpccbyjtq5w6aqthvfetudpud.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1520
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 16)
    x0 = xindex % 16
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 12100, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (16*((r2 + (128*x1)) % 12100))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yo/cyo2avaxxqk254veyizxw6bl3g2qnoundlod54r7zous2d5iz36v.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16, 128],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 95
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
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fx/cfx3zswvy3lli37nmtggipefr5zjbg5tihtzmcyraf6hipxn3gvt.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12100
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3025
    y1 = (yindex // 3025)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (193600 + y0 + (3025*x2) + (387200*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp3, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ex/cexjhfvpsffavp5rg6favj7ujtodn6v6fnpav7veohg3uogyrrv5.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12100
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3025
    y1 = (yindex // 3025)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (3025*x2) + (387200*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tl.where(tmp0, tmp2, tmp1)
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp3, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3p/c3peylbjr7hz73neyf2dazzauqiovsszdtuzbkpqlzhkcm2n6ctr.py
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
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_35', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3154176
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


# kernel path: /tmp/torchinductor_youkaichao/2w/c2winipgt4oadbrawfdwfqruxl4gnbto7skmbcy3y6usun4lpuof.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24704
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 49284, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (64*((r2 + (128*x1)) % 49284))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6s/c6skl7orvptvc3ujr7yeo7xb5omqi7l7f7n2bdl6segolr2oelu6.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 386
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, relu, getitem, getitem_1, relu_1, cat, relu_4, cat_1, getitem_2, getitem_3, relu_7, cat_2, relu_10, cat_3, getitem_4, getitem_5, relu_13, cat_4, relu_16, cat_5, relu_19, cat_6, relu_22, clone, le, le_1, le_2, le_4, le_5, le_7, le_8, le_10, le_11, le_13, le_14, le_16, le_17, le_19, le_20, le_22, le_23, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_3, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_5, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_7, (64, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_9, (16, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_11, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_13, (64, 16, 3, 3), (144, 1, 48, 16))
    assert_size_stride(primals_15, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_17, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_19, (128, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_21, (32, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_23, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_25, (128, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_27, (48, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_29, (192, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_31, (192, 48, 3, 3), (432, 1, 144, 48))
    assert_size_stride(primals_33, (48, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_35, (192, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_37, (192, 48, 3, 3), (432, 1, 144, 48))
    assert_size_stride(primals_39, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_41, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_43, (256, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_45, (64, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_47, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_49, (256, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_51, (1000, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_53, (4, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(relu, (4, 64, 111, 111), (788544, 1, 7104, 64))
    assert_size_stride(getitem, (4, 64, 55, 55), (193600, 1, 3520, 64))
    assert_size_stride(getitem_1, (4, 64, 55, 55), (193600, 1, 3520, 64))
    assert_size_stride(relu_1, (4, 16, 55, 55), (48400, 1, 880, 16))
    assert_size_stride(cat, (4, 128, 55, 55), (387200, 1, 7040, 128))
    assert_size_stride(relu_4, (4, 16, 55, 55), (48400, 1, 880, 16))
    assert_size_stride(cat_1, (4, 128, 55, 55), (387200, 1, 7040, 128))
    assert_size_stride(getitem_2, (4, 128, 27, 27), (93312, 1, 3456, 128))
    assert_size_stride(getitem_3, (4, 128, 27, 27), (93312, 1, 3456, 128))
    assert_size_stride(relu_7, (4, 32, 27, 27), (23328, 1, 864, 32))
    assert_size_stride(cat_2, (4, 256, 27, 27), (186624, 1, 6912, 256))
    assert_size_stride(relu_10, (4, 32, 27, 27), (23328, 1, 864, 32))
    assert_size_stride(cat_3, (4, 256, 27, 27), (186624, 1, 6912, 256))
    assert_size_stride(getitem_4, (4, 256, 13, 13), (43264, 1, 3328, 256))
    assert_size_stride(getitem_5, (4, 256, 13, 13), (43264, 1, 3328, 256))
    assert_size_stride(relu_13, (4, 48, 13, 13), (8112, 1, 624, 48))
    assert_size_stride(cat_4, (4, 384, 13, 13), (64896, 1, 4992, 384))
    assert_size_stride(relu_16, (4, 48, 13, 13), (8112, 1, 624, 48))
    assert_size_stride(cat_5, (4, 384, 13, 13), (64896, 1, 4992, 384))
    assert_size_stride(relu_19, (4, 64, 13, 13), (10816, 1, 832, 64))
    assert_size_stride(cat_6, (4, 512, 13, 13), (86528, 1, 6656, 512))
    assert_size_stride(relu_22, (4, 64, 13, 13), (10816, 1, 832, 64))
    assert_size_stride(clone, (4, 512, 13, 13), (86528, 1, 6656, 512))
    assert_size_stride(le, (4, 1000, 13, 13), (169000, 1, 13000, 1000))
    assert_size_stride(le_1, (4, 256, 13, 13), (43264, 1, 3328, 256))
    assert_size_stride(le_2, (4, 256, 13, 13), (43264, 1, 3328, 256))
    assert_size_stride(le_4, (4, 256, 13, 13), (43264, 1, 3328, 256))
    assert_size_stride(le_5, (4, 256, 13, 13), (43264, 1, 3328, 256))
    assert_size_stride(le_7, (4, 192, 13, 13), (32448, 1, 2496, 192))
    assert_size_stride(le_8, (4, 192, 13, 13), (32448, 1, 2496, 192))
    assert_size_stride(le_10, (4, 192, 13, 13), (32448, 1, 2496, 192))
    assert_size_stride(le_11, (4, 192, 13, 13), (32448, 1, 2496, 192))
    assert_size_stride(le_13, (4, 128, 27, 27), (93312, 1, 3456, 128))
    assert_size_stride(le_14, (4, 128, 27, 27), (93312, 1, 3456, 128))
    assert_size_stride(le_16, (4, 128, 27, 27), (93312, 1, 3456, 128))
    assert_size_stride(le_17, (4, 128, 27, 27), (93312, 1, 3456, 128))
    assert_size_stride(le_19, (4, 64, 55, 55), (193600, 1, 3520, 64))
    assert_size_stride(le_20, (4, 64, 55, 55), (193600, 1, 3520, 64))
    assert_size_stride(le_22, (4, 64, 55, 55), (193600, 1, 3520, 64))
    assert_size_stride(le_23, (4, 64, 55, 55), (193600, 1, 3520, 64))
    assert_size_stride(tangents_1, (4, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((4, 1000, 13, 13), (169000, 1, 13000, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.threshold_backward]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_div_threshold_backward_0.run(le, tangents_1, buf0, 676000, grid=grid(676000), stream=stream0)
        del le
        del tangents_1
        buf1 = empty_strided((1000, 6), (1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_1.run(buf0, buf1, 6000, 113, grid=grid(6000), stream=stream0)
        buf2 = empty((1000, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_2.run(buf1, buf2, 1000, 6, grid=grid(1000), stream=stream0)
        del buf1
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf3 = aten.convolution_backward(buf0, clone, primals_51, [1000], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf0
        del clone
        del primals_51
        buf4 = buf3[0]
        buf5 = buf3[1]
        del buf3
        buf6 = empty_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_3.run(le_1, buf4, buf6, 676, 256, grid=grid(676, 256), stream=stream0)
        del le_1
        buf7 = empty_strided((256, 6), (1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_4.run(buf6, buf7, 1536, 113, grid=grid(1536), stream=stream0)
        buf8 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_5.run(buf7, buf8, 256, 6, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf9 = aten.convolution_backward(buf6, relu_22, primals_49, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_49
        buf10 = buf9[0]
        buf11 = buf9[1]
        del buf9
        buf12 = buf6; del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_6.run(le_2, buf4, buf12, 676, 256, grid=grid(676, 256), stream=stream0)
        del buf4
        del le_2
        buf13 = buf7; del buf7  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_4.run(buf12, buf13, 1536, 113, grid=grid(1536), stream=stream0)
        buf14 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_5.run(buf13, buf14, 256, 6, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf15 = aten.convolution_backward(buf12, relu_22, primals_47, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_47
        buf16 = buf15[0]
        buf17 = buf15[1]
        del buf15
        buf18 = empty_strided((4, 64, 13, 13), (10816, 1, 832, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_7.run(relu_22, buf10, buf16, buf18, 256, 169, grid=grid(256, 169), stream=stream0)
        del buf10
        del buf16
        del relu_22
        buf19 = empty_strided((64, 6), (1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_8.run(buf18, buf19, 384, 113, grid=grid(384), stream=stream0)
        buf20 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_9.run(buf19, buf20, 64, 6, grid=grid(64), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf21 = aten.convolution_backward(buf18, cat_6, primals_45, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_6
        del primals_45
        buf22 = buf21[0]
        buf23 = buf21[1]
        del buf21
        buf24 = buf12; del buf12  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_3.run(le_4, buf22, buf24, 676, 256, grid=grid(676, 256), stream=stream0)
        del le_4
        buf25 = buf13; del buf13  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_4.run(buf24, buf25, 1536, 113, grid=grid(1536), stream=stream0)
        buf26 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_5.run(buf25, buf26, 256, 6, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf27 = aten.convolution_backward(buf24, relu_19, primals_43, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_43
        buf28 = buf27[0]
        buf29 = buf27[1]
        del buf27
        buf30 = buf24; del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_6.run(le_5, buf22, buf30, 676, 256, grid=grid(676, 256), stream=stream0)
        del buf22
        del le_5
        buf31 = buf25; del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_4.run(buf30, buf31, 1536, 113, grid=grid(1536), stream=stream0)
        buf32 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_5.run(buf31, buf32, 256, 6, grid=grid(256), stream=stream0)
        del buf31
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf33 = aten.convolution_backward(buf30, relu_19, primals_41, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf30
        del primals_41
        buf34 = buf33[0]
        buf35 = buf33[1]
        del buf33
        buf36 = buf18; del buf18  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_7.run(relu_19, buf28, buf34, buf36, 256, 169, grid=grid(256, 169), stream=stream0)
        del buf28
        del buf34
        del relu_19
        buf37 = buf19; del buf19  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_8.run(buf36, buf37, 384, 113, grid=grid(384), stream=stream0)
        buf38 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_9.run(buf37, buf38, 64, 6, grid=grid(64), stream=stream0)
        del buf37
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf39 = aten.convolution_backward(buf36, cat_5, primals_39, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf36
        del cat_5
        del primals_39
        buf40 = buf39[0]
        buf41 = buf39[1]
        del buf39
        buf42 = empty_strided((4, 192, 13, 13), (32448, 1, 2496, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_10.run(le_7, buf40, buf42, 676, 192, grid=grid(676, 192), stream=stream0)
        del le_7
        buf43 = empty_strided((192, 6), (1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_11.run(buf42, buf43, 1152, 113, grid=grid(1152), stream=stream0)
        buf44 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf43, buf44, 192, 6, grid=grid(192), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf45 = aten.convolution_backward(buf42, relu_16, primals_37, [192], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_37
        buf46 = buf45[0]
        buf47 = buf45[1]
        del buf45
        buf48 = buf42; del buf42  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_13.run(le_8, buf40, buf48, 676, 192, grid=grid(676, 192), stream=stream0)
        del buf40
        del le_8
        buf49 = buf43; del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_11.run(buf48, buf49, 1152, 113, grid=grid(1152), stream=stream0)
        buf50 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf49, buf50, 192, 6, grid=grid(192), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf51 = aten.convolution_backward(buf48, relu_16, primals_35, [192], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_35
        buf52 = buf51[0]
        buf53 = buf51[1]
        del buf51
        buf54 = empty_strided((4, 48, 13, 13), (8112, 1, 624, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_14.run(relu_16, buf46, buf52, buf54, 192, 169, grid=grid(192, 169), stream=stream0)
        del buf46
        del buf52
        del relu_16
        buf55 = empty_strided((48, 6), (1, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_15.run(buf54, buf55, 288, 113, grid=grid(288), stream=stream0)
        buf56 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_16.run(buf55, buf56, 48, 6, grid=grid(48), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf57 = aten.convolution_backward(buf54, cat_4, primals_33, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_4
        del primals_33
        buf58 = buf57[0]
        buf59 = buf57[1]
        del buf57
        buf60 = buf48; del buf48  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_10.run(le_10, buf58, buf60, 676, 192, grid=grid(676, 192), stream=stream0)
        del le_10
        buf61 = buf49; del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_11.run(buf60, buf61, 1152, 113, grid=grid(1152), stream=stream0)
        buf62 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf61, buf62, 192, 6, grid=grid(192), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf63 = aten.convolution_backward(buf60, relu_13, primals_31, [192], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_31
        buf64 = buf63[0]
        buf65 = buf63[1]
        del buf63
        buf66 = buf60; del buf60  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_13.run(le_11, buf58, buf66, 676, 192, grid=grid(676, 192), stream=stream0)
        del buf58
        del le_11
        buf67 = buf61; del buf61  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_11.run(buf66, buf67, 1152, 113, grid=grid(1152), stream=stream0)
        buf68 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_12.run(buf67, buf68, 192, 6, grid=grid(192), stream=stream0)
        del buf67
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf69 = aten.convolution_backward(buf66, relu_13, primals_29, [192], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf66
        del primals_29
        buf70 = buf69[0]
        buf71 = buf69[1]
        del buf69
        buf72 = buf54; del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_14.run(relu_13, buf64, buf70, buf72, 192, 169, grid=grid(192, 169), stream=stream0)
        del buf64
        del buf70
        del relu_13
        buf73 = buf55; del buf55  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_15.run(buf72, buf73, 288, 113, grid=grid(288), stream=stream0)
        buf74 = empty((48, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_16.run(buf73, buf74, 48, 6, grid=grid(48), stream=stream0)
        del buf73
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf75 = aten.convolution_backward(buf72, getitem_4, primals_27, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf72
        del getitem_4
        del primals_27
        buf76 = buf75[0]
        buf77 = buf75[1]
        del buf75
        # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
        buf78 = aten.max_pool2d_with_indices_backward(buf76, cat_3, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_5)
        del buf76
        del cat_3
        del getitem_5
        buf79 = buf78
        del buf78
        buf80 = empty_strided((4, 128, 27, 27), (93312, 1, 3456, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_17.run(le_13, buf79, buf80, 373248, grid=grid(373248), stream=stream0)
        del le_13
        buf81 = empty_strided((128, 23), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_18.run(buf80, buf81, 2944, 127, grid=grid(2944), stream=stream0)
        buf82 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_19.run(buf81, buf82, 128, 23, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf83 = aten.convolution_backward(buf80, relu_10, primals_25, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_25
        buf84 = buf83[0]
        buf85 = buf83[1]
        del buf83
        buf86 = buf80; del buf80  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_20.run(le_14, buf79, buf86, 373248, grid=grid(373248), stream=stream0)
        del buf79
        del le_14
        buf87 = buf81; del buf81  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_18.run(buf86, buf87, 2944, 127, grid=grid(2944), stream=stream0)
        buf88 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_19.run(buf87, buf88, 128, 23, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf89 = aten.convolution_backward(buf86, relu_10, primals_23, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_23
        buf90 = buf89[0]
        buf91 = buf89[1]
        del buf89
        buf92 = empty_strided((4, 32, 27, 27), (23328, 1, 864, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_21.run(relu_10, buf84, buf90, buf92, 128, 729, grid=grid(128, 729), stream=stream0)
        del buf84
        del buf90
        del relu_10
        buf93 = empty_strided((32, 23), (1, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_22.run(buf92, buf93, 736, 127, grid=grid(736), stream=stream0)
        buf94 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_23.run(buf93, buf94, 32, 23, grid=grid(32), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf95 = aten.convolution_backward(buf92, cat_2, primals_21, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat_2
        del primals_21
        buf96 = buf95[0]
        buf97 = buf95[1]
        del buf95
        buf98 = buf86; del buf86  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_24.run(le_16, buf96, buf98, 2916, 128, grid=grid(2916, 128), stream=stream0)
        del le_16
        buf99 = buf87; del buf87  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_18.run(buf98, buf99, 2944, 127, grid=grid(2944), stream=stream0)
        buf100 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_19.run(buf99, buf100, 128, 23, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf101 = aten.convolution_backward(buf98, relu_7, primals_19, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_19
        buf102 = buf101[0]
        buf103 = buf101[1]
        del buf101
        buf104 = buf98; del buf98  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_25.run(le_17, buf96, buf104, 2916, 128, grid=grid(2916, 128), stream=stream0)
        del buf96
        del le_17
        buf105 = buf99; del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_18.run(buf104, buf105, 2944, 127, grid=grid(2944), stream=stream0)
        buf106 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_19.run(buf105, buf106, 128, 23, grid=grid(128), stream=stream0)
        del buf105
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf107 = aten.convolution_backward(buf104, relu_7, primals_17, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf104
        del primals_17
        buf108 = buf107[0]
        buf109 = buf107[1]
        del buf107
        buf110 = buf92; del buf92  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_21.run(relu_7, buf102, buf108, buf110, 128, 729, grid=grid(128, 729), stream=stream0)
        del buf102
        del buf108
        del relu_7
        buf111 = buf93; del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_22.run(buf110, buf111, 736, 127, grid=grid(736), stream=stream0)
        buf112 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_23.run(buf111, buf112, 32, 23, grid=grid(32), stream=stream0)
        del buf111
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf113 = aten.convolution_backward(buf110, getitem_2, primals_15, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf110
        del getitem_2
        del primals_15
        buf114 = buf113[0]
        buf115 = buf113[1]
        del buf113
        # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
        buf116 = aten.max_pool2d_with_indices_backward(buf114, cat_1, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_3)
        del buf114
        del cat_1
        del getitem_3
        buf117 = buf116
        del buf116
        buf118 = empty_strided((4, 64, 55, 55), (193600, 1, 3520, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_26.run(le_19, buf117, buf118, 774400, grid=grid(774400), stream=stream0)
        del le_19
        buf119 = empty_strided((64, 95), (1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_27.run(buf118, buf119, 6080, 128, grid=grid(6080), stream=stream0)
        buf120 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_28.run(buf119, buf120, 64, 95, grid=grid(64), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf121 = aten.convolution_backward(buf118, relu_4, primals_13, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_13
        buf122 = buf121[0]
        buf123 = buf121[1]
        del buf121
        buf124 = buf118; del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_29.run(le_20, buf117, buf124, 774400, grid=grid(774400), stream=stream0)
        del buf117
        del le_20
        buf125 = buf119; del buf119  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_27.run(buf124, buf125, 6080, 128, grid=grid(6080), stream=stream0)
        buf126 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_28.run(buf125, buf126, 64, 95, grid=grid(64), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf127 = aten.convolution_backward(buf124, relu_4, primals_11, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_11
        buf128 = buf127[0]
        buf129 = buf127[1]
        del buf127
        buf130 = empty_strided((4, 16, 55, 55), (48400, 1, 880, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_30.run(relu_4, buf122, buf128, buf130, 64, 3025, grid=grid(64, 3025), stream=stream0)
        del buf122
        del buf128
        del relu_4
        buf131 = empty_strided((16, 95), (1, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf130, buf131, 1520, 128, grid=grid(1520), stream=stream0)
        buf132 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_32.run(buf131, buf132, 16, 95, grid=grid(16), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf133 = aten.convolution_backward(buf130, cat, primals_9, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del cat
        del primals_9
        buf134 = buf133[0]
        buf135 = buf133[1]
        del buf133
        buf136 = buf124; del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_33.run(le_22, buf134, buf136, 12100, 64, grid=grid(12100, 64), stream=stream0)
        del le_22
        buf137 = buf125; del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_27.run(buf136, buf137, 6080, 128, grid=grid(6080), stream=stream0)
        buf138 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_28.run(buf137, buf138, 64, 95, grid=grid(64), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf139 = aten.convolution_backward(buf136, relu_1, primals_7, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_7
        buf140 = buf139[0]
        buf141 = buf139[1]
        del buf139
        buf142 = buf136; del buf136  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_34.run(le_23, buf134, buf142, 12100, 64, grid=grid(12100, 64), stream=stream0)
        del buf134
        del le_23
        buf143 = buf137; del buf137  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_27.run(buf142, buf143, 6080, 128, grid=grid(6080), stream=stream0)
        buf144 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_28.run(buf143, buf144, 64, 95, grid=grid(64), stream=stream0)
        del buf143
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf145 = aten.convolution_backward(buf142, relu_1, primals_5, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf142
        del primals_5
        buf146 = buf145[0]
        buf147 = buf145[1]
        del buf145
        buf148 = buf130; del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.threshold_backward]
        triton_poi_fused_add_threshold_backward_30.run(relu_1, buf140, buf146, buf148, 64, 3025, grid=grid(64, 3025), stream=stream0)
        del buf140
        del buf146
        del relu_1
        buf149 = buf131; del buf131  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf148, buf149, 1520, 128, grid=grid(1520), stream=stream0)
        buf150 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_32.run(buf149, buf150, 16, 95, grid=grid(16), stream=stream0)
        del buf149
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf151 = aten.convolution_backward(buf148, getitem, primals_3, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf148
        del getitem
        del primals_3
        buf152 = buf151[0]
        buf153 = buf151[1]
        del buf151
        # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
        buf154 = aten.max_pool2d_with_indices_backward(buf152, relu, [3, 3], [2, 2], [0, 0], [1, 1], True, getitem_1)
        del buf152
        del getitem_1
        buf155 = buf154
        del buf154
        buf156 = buf155; del buf155  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_35.run(buf156, relu, 3154176, grid=grid(3154176), stream=stream0)
        del relu
        buf157 = empty_strided((64, 386), (1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_36.run(buf156, buf157, 24704, 128, grid=grid(24704), stream=stream0)
        buf158 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_37.run(buf157, buf158, 64, 386, grid=grid(64), stream=stream0)
        del buf157
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf159 = aten.convolution_backward(buf156, primals_53, primals_1, [64], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf156
        del primals_1
        del primals_53
        buf160 = buf159[1]
        return (buf160, buf158, buf153, buf150, buf147, buf144, buf141, buf138, buf135, buf132, buf129, buf126, buf123, buf120, buf115, buf112, buf109, buf106, buf103, buf100, buf97, buf94, buf91, buf88, buf85, buf82, buf77, buf74, buf71, buf68, buf65, buf62, buf59, buf56, buf53, buf50, buf47, buf44, buf41, buf38, buf35, buf32, buf29, buf26, buf23, buf20, buf17, buf14, buf11, buf8, buf5, buf2, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 3, 3), (27, 1, 9, 3), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 16, 3, 3), (144, 1, 48, 16), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((16, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, 16, 3, 3), (144, 1, 48, 16), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, 32, 3, 3), (288, 1, 96, 32), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((32, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, 32, 3, 3), (288, 1, 96, 32), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((48, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((192, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((192, 48, 3, 3), (432, 1, 144, 48), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((48, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((192, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((192, 48, 3, 3), (432, 1, 144, 48), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((256, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((64, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((256, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((1000, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((4, 64, 111, 111), (788544, 1, 7104, 64), device='cuda:0', dtype=torch.float32)
    getitem = rand_strided((4, 64, 55, 55), (193600, 1, 3520, 64), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((4, 64, 55, 55), (193600, 1, 3520, 64), device='cuda:0', dtype=torch.int64)
    relu_1 = rand_strided((4, 16, 55, 55), (48400, 1, 880, 16), device='cuda:0', dtype=torch.float32)
    cat = rand_strided((4, 128, 55, 55), (387200, 1, 7040, 128), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((4, 16, 55, 55), (48400, 1, 880, 16), device='cuda:0', dtype=torch.float32)
    cat_1 = rand_strided((4, 128, 55, 55), (387200, 1, 7040, 128), device='cuda:0', dtype=torch.float32)
    getitem_2 = rand_strided((4, 128, 27, 27), (93312, 1, 3456, 128), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((4, 128, 27, 27), (93312, 1, 3456, 128), device='cuda:0', dtype=torch.int64)
    relu_7 = rand_strided((4, 32, 27, 27), (23328, 1, 864, 32), device='cuda:0', dtype=torch.float32)
    cat_2 = rand_strided((4, 256, 27, 27), (186624, 1, 6912, 256), device='cuda:0', dtype=torch.float32)
    relu_10 = rand_strided((4, 32, 27, 27), (23328, 1, 864, 32), device='cuda:0', dtype=torch.float32)
    cat_3 = rand_strided((4, 256, 27, 27), (186624, 1, 6912, 256), device='cuda:0', dtype=torch.float32)
    getitem_4 = rand_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cuda:0', dtype=torch.float32)
    getitem_5 = rand_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cuda:0', dtype=torch.int64)
    relu_13 = rand_strided((4, 48, 13, 13), (8112, 1, 624, 48), device='cuda:0', dtype=torch.float32)
    cat_4 = rand_strided((4, 384, 13, 13), (64896, 1, 4992, 384), device='cuda:0', dtype=torch.float32)
    relu_16 = rand_strided((4, 48, 13, 13), (8112, 1, 624, 48), device='cuda:0', dtype=torch.float32)
    cat_5 = rand_strided((4, 384, 13, 13), (64896, 1, 4992, 384), device='cuda:0', dtype=torch.float32)
    relu_19 = rand_strided((4, 64, 13, 13), (10816, 1, 832, 64), device='cuda:0', dtype=torch.float32)
    cat_6 = rand_strided((4, 512, 13, 13), (86528, 1, 6656, 512), device='cuda:0', dtype=torch.float32)
    relu_22 = rand_strided((4, 64, 13, 13), (10816, 1, 832, 64), device='cuda:0', dtype=torch.float32)
    clone = rand_strided((4, 512, 13, 13), (86528, 1, 6656, 512), device='cuda:0', dtype=torch.float32)
    le = rand_strided((4, 1000, 13, 13), (169000, 1, 13000, 1000), device='cuda:0', dtype=torch.bool)
    le_1 = rand_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cuda:0', dtype=torch.bool)
    le_2 = rand_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cuda:0', dtype=torch.bool)
    le_4 = rand_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cuda:0', dtype=torch.bool)
    le_5 = rand_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cuda:0', dtype=torch.bool)
    le_7 = rand_strided((4, 192, 13, 13), (32448, 1, 2496, 192), device='cuda:0', dtype=torch.bool)
    le_8 = rand_strided((4, 192, 13, 13), (32448, 1, 2496, 192), device='cuda:0', dtype=torch.bool)
    le_10 = rand_strided((4, 192, 13, 13), (32448, 1, 2496, 192), device='cuda:0', dtype=torch.bool)
    le_11 = rand_strided((4, 192, 13, 13), (32448, 1, 2496, 192), device='cuda:0', dtype=torch.bool)
    le_13 = rand_strided((4, 128, 27, 27), (93312, 1, 3456, 128), device='cuda:0', dtype=torch.bool)
    le_14 = rand_strided((4, 128, 27, 27), (93312, 1, 3456, 128), device='cuda:0', dtype=torch.bool)
    le_16 = rand_strided((4, 128, 27, 27), (93312, 1, 3456, 128), device='cuda:0', dtype=torch.bool)
    le_17 = rand_strided((4, 128, 27, 27), (93312, 1, 3456, 128), device='cuda:0', dtype=torch.bool)
    le_19 = rand_strided((4, 64, 55, 55), (193600, 1, 3520, 64), device='cuda:0', dtype=torch.bool)
    le_20 = rand_strided((4, 64, 55, 55), (193600, 1, 3520, 64), device='cuda:0', dtype=torch.bool)
    le_22 = rand_strided((4, 64, 55, 55), (193600, 1, 3520, 64), device='cuda:0', dtype=torch.bool)
    le_23 = rand_strided((4, 64, 55, 55), (193600, 1, 3520, 64), device='cuda:0', dtype=torch.bool)
    tangents_1 = rand_strided((4, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, relu, getitem, getitem_1, relu_1, cat, relu_4, cat_1, getitem_2, getitem_3, relu_7, cat_2, relu_10, cat_3, getitem_4, getitem_5, relu_13, cat_4, relu_16, cat_5, relu_19, cat_6, relu_22, clone, le, le_1, le_2, le_4, le_5, le_7, le_8, le_10, le_11, le_13, le_14, le_16, le_17, le_19, le_20, le_22, le_23, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('squeezenet1_1', benchmark_compiled_module)
