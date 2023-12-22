
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


# kernel path: /tmp/torchinductor_youkaichao/fm/cfmfgch2qnmllijhuy4wk2j6tjc2sgmerfxxhbzfzqpukb34gfdj.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_layer_norm_backward]

triton_red_fused_div_native_layer_norm_backward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_native_layer_norm_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 196)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (256*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r2 + (256*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 196.0
        tmp2 = tmp0 / tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp9 = tmp4 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp13 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp14 = tl.load(in_ptr0 + (r2 + (256*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tl.load(in_ptr2 + (r2 + (256*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = 196.0
        tmp16 = tmp14 / tmp15
        tmp18 = tmp16 * tmp17
        tmp19 = 256.0
        tmp20 = tmp18 * tmp19
        tmp21 = tmp20 - tmp6
        tmp23 = tmp22 * tmp11
        tmp24 = tmp21 - tmp23
        tmp25 = tmp13 * tmp24
        tl.store(out_ptr2 + (r2 + (256*x3)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xd/cxd6jns4evb3r3xrxoi4uo2v646c6m2hie3vixzu356wmtt5vlg2.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_layer_norm_backward]

triton_red_fused_div_native_layer_norm_backward_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_native_layer_norm_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3328
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 256)
    x0 = xindex % 256
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (256*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 196.0
        tmp5 = tmp3 / tmp4
        tmp6 = tl.load(in_ptr1 + (x0 + (256*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tmp5 * tmp6
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pd/cpd5jjdn3rr6jsujz3armxmvnu3qszrzyxtvcv7nkkzhuj5loqf7.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_layer_norm_backward]

triton_per_fused_div_native_layer_norm_backward_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_layer_norm_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 13
    RBLOCK: tl.constexpr = 16
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


# kernel path: /tmp/torchinductor_youkaichao/gm/cgm7w5fz4xxmupxogwqceipa6yihivubgm3t6p7hmbxlark552cj.py
# Source Nodes: [], Original ATen: [aten.div, aten.native_layer_norm_backward]

triton_red_fused_div_native_layer_norm_backward_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 2048],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_native_layer_norm_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 196.0
        tmp2 = tmp0 / tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r3/cr36ftmpego7utqryqif6quflqf7psmi5fdnkjw2temm3e4nshbl.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3328
    rnumel = 121
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
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (256*r2) + (30976*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e6/ce6i4qlpgo4doopgfhgl36y7stqduxv4mh2i5xn2lcay4qpt5gug.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 196
    rnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 768
        r2 = (rindex // 768)
        tmp0 = tl.load(in_ptr0 + (r1 + (768*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (1536*x0) + (301056*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oj/cojamnfb64tbkbmdej73ienvazuo4r6ndjfpx75nbkwrudid3yz2.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (1536*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (y0 + (196*x2) + (150528*y1)), tmp2, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gj/cgjunwxppdxe2xctetvyspca6bxjpuaz7fuwcwwmqaotidoz7y2i.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x4 = (xindex // 196)
    x1 = (xindex // 196) % 6
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (196*r3) + (25088*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dd/cddxyjlwti4r6qvikt4ykrfyzqdnrzm6clmh25vcv7jcwedq6o2q.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 6
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (1176*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ro/croxr77r6ywyrdmljd3zfvrpp3ngehge55pzdtz5uk2fniiqvif4.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 6
    x1 = (xindex // 6) % 196
    x2 = (xindex // 1176)
    x5 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (196*r3) + (25088*x0) + (150528*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r3 + (128*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mm/cmmpa57n4ainuvtk66me6w6gnl55iyfh6rtapgo5i365gfjl7wvc.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 6
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (6*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lb/clb6ylian4j5uae5w2grfy4va3burepaix5tkkmxgnokfncn7x7t.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9984
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
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (150528*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (768*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yg/cyg72qqvth74n4bamxkraam2nge2rmy2dehlwdbhxucfxvs3wmpw.py
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
    size_hints=[1024, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
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


# kernel path: /tmp/torchinductor_youkaichao/cf/ccfvc2x2pety454dbhr5ib36yjngauxucb2e2ut5ltshiy7gyt23.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/og/cogihocmm6yqlbqwqm6j5lulj4gwtojeftxrtlnbmr2xkpw47adl.py
# Source Nodes: [x_237], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
# x_237 => add_205, erf_29, mul_235
triton_poi_fused_cat_gelu_gelu_backward_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_gelu_gelu_backward_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 1536
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
    tmp31 = tl.load(in_ptr9 + (x2 + (1536*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 768, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (768*y3)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (y0 + (196*x2) + (150528*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1, 1], 1536, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr3 + (tl.broadcast_to(y3, [XBLOCK, YBLOCK])), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + ((-150528) + y0 + (196*x2) + (150528*y1)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.load(in_ptr5 + (tl.broadcast_to((-768) + x2, [XBLOCK, YBLOCK])), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = 768.0
    tmp20 = tmp18 * tmp19
    tmp21 = tl.load(in_ptr6 + (tl.broadcast_to(y3, [XBLOCK, YBLOCK])), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 - tmp21
    tmp23 = tl.load(in_ptr7 + ((-768) + x2 + (768*y3)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr8 + (tl.broadcast_to(y3, [XBLOCK, YBLOCK])), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 - tmp25
    tmp27 = tmp15 * tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp12, tmp27, tmp28)
    tmp30 = tl.where(tmp4, tmp11, tmp29)
    tmp32 = 0.7071067811865476
    tmp33 = tmp31 * tmp32
    tmp34 = tl.math.erf(tmp33)
    tmp35 = 1.0
    tmp36 = tmp34 + tmp35
    tmp37 = 0.5
    tmp38 = tmp36 * tmp37
    tmp39 = tmp31 * tmp31
    tmp40 = -0.5
    tmp41 = tmp39 * tmp40
    tmp42 = tl.exp(tmp41)
    tmp43 = 0.3989422804014327
    tmp44 = tmp42 * tmp43
    tmp45 = tmp31 * tmp44
    tmp46 = tmp38 + tmp45
    tmp47 = tmp30 * tmp46
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (1536*y3)), tmp47, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rw/crwiryjaz6oizp2qomunmip2h4r4ufqpo3r7q7yugenohbzelqmc.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 19968
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 1536)
    x0 = xindex % 1536
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (1536*r2) + (185856*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uu/cuu73titlmdsecfdvipjqumjhsnyrgwkwb6setlfgss3tjbamxku.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 13
    RBLOCK: tl.constexpr = 16
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


# kernel path: /tmp/torchinductor_youkaichao/xr/cxrjzrp2h4ryztyylhpxfs4bnymav4hd5v5esxhgcvou5qr2xqiu.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_18', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 256.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(in_out_ptr0 + (r1 + (256*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qi/cqic6ose5wv2cohnjc2rzlhvl43iv55pq2drlcm2fsvv2qphxo66.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3328
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 256)
    x0 = xindex % 256
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (256*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (256*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp11 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp12 = tl.where(tmp2, tmp3, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ep/ceppn6chced5usvxh7ankvbaosa3f6eltifdacmwq4dwxgxxnr6q.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3328
    rnumel = 121
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
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (256*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_7, primals_10, primals_13, primals_17, primals_20, primals_23, primals_27, primals_30, primals_33, primals_37, primals_40, primals_43, primals_47, primals_50, primals_53, primals_57, primals_60, primals_63, primals_67, primals_70, primals_73, primals_77, primals_80, primals_83, primals_87, primals_90, primals_93, primals_97, primals_100, primals_103, primals_107, primals_110, primals_113, primals_117, primals_120, primals_123, primals_127, primals_130, primals_133, primals_137, primals_140, primals_143, primals_147, primals_150, primals_153, primals_157, primals_160, primals_163, primals_167, primals_170, primals_173, primals_177, primals_180, primals_183, primals_187, primals_190, primals_193, primals_197, primals_200, primals_203, primals_207, primals_210, primals_213, primals_217, primals_220, primals_223, primals_227, primals_230, primals_233, primals_237, primals_240, primals_243, primals_247, primals_250, primals_253, primals_257, primals_260, primals_263, primals_267, primals_270, primals_273, primals_277, primals_280, primals_283, primals_287, primals_290, primals_293, primals_297, primals_300, primals_303, primals_307, mul, view_1, addmm, getitem_2, mul_5, view_3, mm, view_5, mul_8, view_7, addmm_2, getitem_8, mul_13, view_9, mm_1, view_11, mul_16, view_13, addmm_4, getitem_14, mul_21, view_15, mm_2, view_17, mul_24, view_19, addmm_6, getitem_20, mul_29, view_21, mm_3, view_23, mul_32, view_25, addmm_8, getitem_26, mul_37, view_27, mm_4, view_29, mul_40, view_31, addmm_10, getitem_32, mul_45, view_33, mm_5, view_35, mul_48, view_37, addmm_12, getitem_38, mul_53, view_39, mm_6, view_41, mul_56, view_43, addmm_14, getitem_44, mul_61, view_45, mm_7, view_47, mul_64, view_49, addmm_16, getitem_50, mul_69, view_51, mm_8, view_53, mul_72, view_55, addmm_18, getitem_56, mul_77, view_57, mm_9, view_59, mul_80, view_61, addmm_20, getitem_62, mul_85, view_63, mm_10, view_65, mul_88, view_67, addmm_22, getitem_68, mul_93, view_69, mm_11, view_71, mul_96, view_73, addmm_24, getitem_74, mul_101, view_75, mm_12, view_77, mul_104, view_79, addmm_26, getitem_80, mul_109, view_81, mm_13, view_83, mul_112, view_85, addmm_28, getitem_86, mul_117, view_87, mm_14, view_89, mul_120, view_91, addmm_30, getitem_92, mul_125, view_93, mm_15, view_95, mul_128, view_97, addmm_32, getitem_98, mul_133, view_99, mm_16, view_101, mul_136, view_103, addmm_34, getitem_104, mul_141, view_105, mm_17, view_107, mul_144, view_109, addmm_36, getitem_110, mul_149, view_111, mm_18, view_113, mul_152, view_115, addmm_38, getitem_116, mul_157, view_117, mm_19, view_119, mul_160, view_121, addmm_40, getitem_122, mul_165, view_123, mm_20, view_125, mul_168, view_127, addmm_42, getitem_128, mul_173, view_129, mm_21, view_131, mul_176, view_133, addmm_44, getitem_134, mul_181, view_135, mm_22, view_137, mul_184, view_139, addmm_46, getitem_140, mul_189, view_141, mm_23, view_143, mul_192, view_145, addmm_48, getitem_146, mul_197, view_147, mm_24, view_149, mul_200, view_151, addmm_50, getitem_152, mul_205, view_153, mm_25, view_155, mul_208, view_157, addmm_52, getitem_158, mul_213, view_159, mm_26, view_161, mul_216, view_163, addmm_54, getitem_164, mul_221, view_165, mm_27, view_167, mul_224, view_169, addmm_56, getitem_170, mul_229, view_171, mm_28, view_173, mul_232, view_175, addmm_58, getitem_176, mul_237, view_177, mm_29, view_179, mul_240, clone_151, permute_152, div_1, permute_156, permute_163, div_2, permute_166, div_3, permute_170, permute_177, div_4, permute_180, div_5, permute_184, permute_191, div_6, permute_194, div_7, permute_198, permute_205, div_8, permute_208, div_9, permute_212, permute_219, div_10, permute_222, div_11, permute_226, permute_233, div_12, permute_236, div_13, permute_240, permute_247, div_14, permute_250, div_15, permute_254, permute_261, div_16, permute_264, div_17, permute_268, permute_275, div_18, permute_278, div_19, permute_282, permute_289, div_20, permute_292, div_21, permute_296, permute_303, div_22, permute_306, div_23, permute_310, permute_317, div_24, permute_320, div_25, permute_324, permute_331, div_26, permute_334, div_27, permute_338, permute_345, div_28, permute_348, div_29, permute_352, permute_359, div_30, permute_362, div_31, permute_366, permute_373, div_32, permute_376, div_33, permute_380, permute_387, div_34, permute_390, div_35, permute_394, permute_401, div_36, permute_404, div_37, permute_408, permute_415, div_38, permute_418, div_39, permute_422, permute_429, div_40, permute_432, div_41, permute_436, permute_443, div_42, permute_446, div_43, permute_450, permute_457, div_44, permute_460, div_45, permute_464, permute_471, div_46, permute_474, div_47, permute_478, permute_485, div_48, permute_488, div_49, permute_492, permute_499, div_50, permute_502, div_51, permute_506, permute_513, div_52, permute_516, div_53, permute_520, permute_527, div_54, permute_530, div_55, permute_534, permute_541, div_56, permute_544, div_57, permute_548, permute_555, div_58, permute_558, div_59, permute_562, permute_569, div_60, permute_572, div_61, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (256, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(primals_3, (256, ), (1, ))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_10, (196, ), (1, ))
    assert_size_stride(primals_13, (256, ), (1, ))
    assert_size_stride(primals_17, (768, ), (1, ))
    assert_size_stride(primals_20, (196, ), (1, ))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_27, (768, ), (1, ))
    assert_size_stride(primals_30, (196, ), (1, ))
    assert_size_stride(primals_33, (256, ), (1, ))
    assert_size_stride(primals_37, (768, ), (1, ))
    assert_size_stride(primals_40, (196, ), (1, ))
    assert_size_stride(primals_43, (256, ), (1, ))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_50, (196, ), (1, ))
    assert_size_stride(primals_53, (256, ), (1, ))
    assert_size_stride(primals_57, (768, ), (1, ))
    assert_size_stride(primals_60, (196, ), (1, ))
    assert_size_stride(primals_63, (256, ), (1, ))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_70, (196, ), (1, ))
    assert_size_stride(primals_73, (256, ), (1, ))
    assert_size_stride(primals_77, (768, ), (1, ))
    assert_size_stride(primals_80, (196, ), (1, ))
    assert_size_stride(primals_83, (256, ), (1, ))
    assert_size_stride(primals_87, (768, ), (1, ))
    assert_size_stride(primals_90, (196, ), (1, ))
    assert_size_stride(primals_93, (256, ), (1, ))
    assert_size_stride(primals_97, (768, ), (1, ))
    assert_size_stride(primals_100, (196, ), (1, ))
    assert_size_stride(primals_103, (256, ), (1, ))
    assert_size_stride(primals_107, (768, ), (1, ))
    assert_size_stride(primals_110, (196, ), (1, ))
    assert_size_stride(primals_113, (256, ), (1, ))
    assert_size_stride(primals_117, (768, ), (1, ))
    assert_size_stride(primals_120, (196, ), (1, ))
    assert_size_stride(primals_123, (256, ), (1, ))
    assert_size_stride(primals_127, (768, ), (1, ))
    assert_size_stride(primals_130, (196, ), (1, ))
    assert_size_stride(primals_133, (256, ), (1, ))
    assert_size_stride(primals_137, (768, ), (1, ))
    assert_size_stride(primals_140, (196, ), (1, ))
    assert_size_stride(primals_143, (256, ), (1, ))
    assert_size_stride(primals_147, (768, ), (1, ))
    assert_size_stride(primals_150, (196, ), (1, ))
    assert_size_stride(primals_153, (256, ), (1, ))
    assert_size_stride(primals_157, (768, ), (1, ))
    assert_size_stride(primals_160, (196, ), (1, ))
    assert_size_stride(primals_163, (256, ), (1, ))
    assert_size_stride(primals_167, (768, ), (1, ))
    assert_size_stride(primals_170, (196, ), (1, ))
    assert_size_stride(primals_173, (256, ), (1, ))
    assert_size_stride(primals_177, (768, ), (1, ))
    assert_size_stride(primals_180, (196, ), (1, ))
    assert_size_stride(primals_183, (256, ), (1, ))
    assert_size_stride(primals_187, (768, ), (1, ))
    assert_size_stride(primals_190, (196, ), (1, ))
    assert_size_stride(primals_193, (256, ), (1, ))
    assert_size_stride(primals_197, (768, ), (1, ))
    assert_size_stride(primals_200, (196, ), (1, ))
    assert_size_stride(primals_203, (256, ), (1, ))
    assert_size_stride(primals_207, (768, ), (1, ))
    assert_size_stride(primals_210, (196, ), (1, ))
    assert_size_stride(primals_213, (256, ), (1, ))
    assert_size_stride(primals_217, (768, ), (1, ))
    assert_size_stride(primals_220, (196, ), (1, ))
    assert_size_stride(primals_223, (256, ), (1, ))
    assert_size_stride(primals_227, (768, ), (1, ))
    assert_size_stride(primals_230, (196, ), (1, ))
    assert_size_stride(primals_233, (256, ), (1, ))
    assert_size_stride(primals_237, (768, ), (1, ))
    assert_size_stride(primals_240, (196, ), (1, ))
    assert_size_stride(primals_243, (256, ), (1, ))
    assert_size_stride(primals_247, (768, ), (1, ))
    assert_size_stride(primals_250, (196, ), (1, ))
    assert_size_stride(primals_253, (256, ), (1, ))
    assert_size_stride(primals_257, (768, ), (1, ))
    assert_size_stride(primals_260, (196, ), (1, ))
    assert_size_stride(primals_263, (256, ), (1, ))
    assert_size_stride(primals_267, (768, ), (1, ))
    assert_size_stride(primals_270, (196, ), (1, ))
    assert_size_stride(primals_273, (256, ), (1, ))
    assert_size_stride(primals_277, (768, ), (1, ))
    assert_size_stride(primals_280, (196, ), (1, ))
    assert_size_stride(primals_283, (256, ), (1, ))
    assert_size_stride(primals_287, (768, ), (1, ))
    assert_size_stride(primals_290, (196, ), (1, ))
    assert_size_stride(primals_293, (256, ), (1, ))
    assert_size_stride(primals_297, (768, ), (1, ))
    assert_size_stride(primals_300, (196, ), (1, ))
    assert_size_stride(primals_303, (256, ), (1, ))
    assert_size_stride(primals_307, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(mul, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_1, (1568, 256), (256, 1))
    assert_size_stride(addmm, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_2, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_5, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_3, (6144, 196), (196, 1))
    assert_size_stride(mm, (6144, 196), (196, 1))
    assert_size_stride(view_5, (1568, 768), (768, 1))
    assert_size_stride(mul_8, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_7, (1568, 256), (256, 1))
    assert_size_stride(addmm_2, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_8, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_13, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_9, (6144, 196), (196, 1))
    assert_size_stride(mm_1, (6144, 196), (196, 1))
    assert_size_stride(view_11, (1568, 768), (768, 1))
    assert_size_stride(mul_16, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_13, (1568, 256), (256, 1))
    assert_size_stride(addmm_4, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_14, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_21, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_15, (6144, 196), (196, 1))
    assert_size_stride(mm_2, (6144, 196), (196, 1))
    assert_size_stride(view_17, (1568, 768), (768, 1))
    assert_size_stride(mul_24, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_19, (1568, 256), (256, 1))
    assert_size_stride(addmm_6, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_20, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_29, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_21, (6144, 196), (196, 1))
    assert_size_stride(mm_3, (6144, 196), (196, 1))
    assert_size_stride(view_23, (1568, 768), (768, 1))
    assert_size_stride(mul_32, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_25, (1568, 256), (256, 1))
    assert_size_stride(addmm_8, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_26, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_37, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_27, (6144, 196), (196, 1))
    assert_size_stride(mm_4, (6144, 196), (196, 1))
    assert_size_stride(view_29, (1568, 768), (768, 1))
    assert_size_stride(mul_40, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_31, (1568, 256), (256, 1))
    assert_size_stride(addmm_10, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_32, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_45, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_33, (6144, 196), (196, 1))
    assert_size_stride(mm_5, (6144, 196), (196, 1))
    assert_size_stride(view_35, (1568, 768), (768, 1))
    assert_size_stride(mul_48, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_37, (1568, 256), (256, 1))
    assert_size_stride(addmm_12, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_38, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_53, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_39, (6144, 196), (196, 1))
    assert_size_stride(mm_6, (6144, 196), (196, 1))
    assert_size_stride(view_41, (1568, 768), (768, 1))
    assert_size_stride(mul_56, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_43, (1568, 256), (256, 1))
    assert_size_stride(addmm_14, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_44, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_61, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_45, (6144, 196), (196, 1))
    assert_size_stride(mm_7, (6144, 196), (196, 1))
    assert_size_stride(view_47, (1568, 768), (768, 1))
    assert_size_stride(mul_64, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_49, (1568, 256), (256, 1))
    assert_size_stride(addmm_16, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_50, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_69, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_51, (6144, 196), (196, 1))
    assert_size_stride(mm_8, (6144, 196), (196, 1))
    assert_size_stride(view_53, (1568, 768), (768, 1))
    assert_size_stride(mul_72, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_55, (1568, 256), (256, 1))
    assert_size_stride(addmm_18, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_56, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_77, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_57, (6144, 196), (196, 1))
    assert_size_stride(mm_9, (6144, 196), (196, 1))
    assert_size_stride(view_59, (1568, 768), (768, 1))
    assert_size_stride(mul_80, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_61, (1568, 256), (256, 1))
    assert_size_stride(addmm_20, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_62, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_85, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_63, (6144, 196), (196, 1))
    assert_size_stride(mm_10, (6144, 196), (196, 1))
    assert_size_stride(view_65, (1568, 768), (768, 1))
    assert_size_stride(mul_88, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_67, (1568, 256), (256, 1))
    assert_size_stride(addmm_22, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_68, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_93, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_69, (6144, 196), (196, 1))
    assert_size_stride(mm_11, (6144, 196), (196, 1))
    assert_size_stride(view_71, (1568, 768), (768, 1))
    assert_size_stride(mul_96, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_73, (1568, 256), (256, 1))
    assert_size_stride(addmm_24, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_74, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_101, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_75, (6144, 196), (196, 1))
    assert_size_stride(mm_12, (6144, 196), (196, 1))
    assert_size_stride(view_77, (1568, 768), (768, 1))
    assert_size_stride(mul_104, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_79, (1568, 256), (256, 1))
    assert_size_stride(addmm_26, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_80, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_109, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_81, (6144, 196), (196, 1))
    assert_size_stride(mm_13, (6144, 196), (196, 1))
    assert_size_stride(view_83, (1568, 768), (768, 1))
    assert_size_stride(mul_112, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_85, (1568, 256), (256, 1))
    assert_size_stride(addmm_28, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_86, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_117, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_87, (6144, 196), (196, 1))
    assert_size_stride(mm_14, (6144, 196), (196, 1))
    assert_size_stride(view_89, (1568, 768), (768, 1))
    assert_size_stride(mul_120, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_91, (1568, 256), (256, 1))
    assert_size_stride(addmm_30, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_92, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_125, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_93, (6144, 196), (196, 1))
    assert_size_stride(mm_15, (6144, 196), (196, 1))
    assert_size_stride(view_95, (1568, 768), (768, 1))
    assert_size_stride(mul_128, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_97, (1568, 256), (256, 1))
    assert_size_stride(addmm_32, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_98, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_133, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_99, (6144, 196), (196, 1))
    assert_size_stride(mm_16, (6144, 196), (196, 1))
    assert_size_stride(view_101, (1568, 768), (768, 1))
    assert_size_stride(mul_136, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_103, (1568, 256), (256, 1))
    assert_size_stride(addmm_34, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_104, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_141, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_105, (6144, 196), (196, 1))
    assert_size_stride(mm_17, (6144, 196), (196, 1))
    assert_size_stride(view_107, (1568, 768), (768, 1))
    assert_size_stride(mul_144, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_109, (1568, 256), (256, 1))
    assert_size_stride(addmm_36, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_110, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_149, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_111, (6144, 196), (196, 1))
    assert_size_stride(mm_18, (6144, 196), (196, 1))
    assert_size_stride(view_113, (1568, 768), (768, 1))
    assert_size_stride(mul_152, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_115, (1568, 256), (256, 1))
    assert_size_stride(addmm_38, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_116, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_157, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_117, (6144, 196), (196, 1))
    assert_size_stride(mm_19, (6144, 196), (196, 1))
    assert_size_stride(view_119, (1568, 768), (768, 1))
    assert_size_stride(mul_160, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_121, (1568, 256), (256, 1))
    assert_size_stride(addmm_40, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_122, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_165, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_123, (6144, 196), (196, 1))
    assert_size_stride(mm_20, (6144, 196), (196, 1))
    assert_size_stride(view_125, (1568, 768), (768, 1))
    assert_size_stride(mul_168, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_127, (1568, 256), (256, 1))
    assert_size_stride(addmm_42, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_128, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_173, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_129, (6144, 196), (196, 1))
    assert_size_stride(mm_21, (6144, 196), (196, 1))
    assert_size_stride(view_131, (1568, 768), (768, 1))
    assert_size_stride(mul_176, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_133, (1568, 256), (256, 1))
    assert_size_stride(addmm_44, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_134, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_181, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_135, (6144, 196), (196, 1))
    assert_size_stride(mm_22, (6144, 196), (196, 1))
    assert_size_stride(view_137, (1568, 768), (768, 1))
    assert_size_stride(mul_184, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_139, (1568, 256), (256, 1))
    assert_size_stride(addmm_46, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_140, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_189, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_141, (6144, 196), (196, 1))
    assert_size_stride(mm_23, (6144, 196), (196, 1))
    assert_size_stride(view_143, (1568, 768), (768, 1))
    assert_size_stride(mul_192, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_145, (1568, 256), (256, 1))
    assert_size_stride(addmm_48, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_146, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_197, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_147, (6144, 196), (196, 1))
    assert_size_stride(mm_24, (6144, 196), (196, 1))
    assert_size_stride(view_149, (1568, 768), (768, 1))
    assert_size_stride(mul_200, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_151, (1568, 256), (256, 1))
    assert_size_stride(addmm_50, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_152, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_205, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_153, (6144, 196), (196, 1))
    assert_size_stride(mm_25, (6144, 196), (196, 1))
    assert_size_stride(view_155, (1568, 768), (768, 1))
    assert_size_stride(mul_208, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_157, (1568, 256), (256, 1))
    assert_size_stride(addmm_52, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_158, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_213, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_159, (6144, 196), (196, 1))
    assert_size_stride(mm_26, (6144, 196), (196, 1))
    assert_size_stride(view_161, (1568, 768), (768, 1))
    assert_size_stride(mul_216, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_163, (1568, 256), (256, 1))
    assert_size_stride(addmm_54, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_164, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_221, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_165, (6144, 196), (196, 1))
    assert_size_stride(mm_27, (6144, 196), (196, 1))
    assert_size_stride(view_167, (1568, 768), (768, 1))
    assert_size_stride(mul_224, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_169, (1568, 256), (256, 1))
    assert_size_stride(addmm_56, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_170, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_229, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_171, (6144, 196), (196, 1))
    assert_size_stride(mm_28, (6144, 196), (196, 1))
    assert_size_stride(view_173, (1568, 768), (768, 1))
    assert_size_stride(mul_232, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_175, (1568, 256), (256, 1))
    assert_size_stride(addmm_58, (1568, 1536), (1536, 1))
    assert_size_stride(getitem_176, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(mul_237, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_177, (6144, 196), (196, 1))
    assert_size_stride(mm_29, (6144, 196), (196, 1))
    assert_size_stride(view_179, (1568, 768), (768, 1))
    assert_size_stride(mul_240, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(clone_151, (8, 256), (256, 1))
    assert_size_stride(permute_152, (1000, 256), (256, 1))
    assert_size_stride(div_1, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_156, (256, 768), (768, 1))
    assert_size_stride(permute_163, (196, 196), (196, 1))
    assert_size_stride(div_2, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_166, (1536, 256), (256, 1))
    assert_size_stride(div_3, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_170, (256, 768), (768, 1))
    assert_size_stride(permute_177, (196, 196), (196, 1))
    assert_size_stride(div_4, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_180, (1536, 256), (256, 1))
    assert_size_stride(div_5, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_184, (256, 768), (768, 1))
    assert_size_stride(permute_191, (196, 196), (196, 1))
    assert_size_stride(div_6, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_194, (1536, 256), (256, 1))
    assert_size_stride(div_7, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_198, (256, 768), (768, 1))
    assert_size_stride(permute_205, (196, 196), (196, 1))
    assert_size_stride(div_8, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_208, (1536, 256), (256, 1))
    assert_size_stride(div_9, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_212, (256, 768), (768, 1))
    assert_size_stride(permute_219, (196, 196), (196, 1))
    assert_size_stride(div_10, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_222, (1536, 256), (256, 1))
    assert_size_stride(div_11, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_226, (256, 768), (768, 1))
    assert_size_stride(permute_233, (196, 196), (196, 1))
    assert_size_stride(div_12, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_236, (1536, 256), (256, 1))
    assert_size_stride(div_13, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_240, (256, 768), (768, 1))
    assert_size_stride(permute_247, (196, 196), (196, 1))
    assert_size_stride(div_14, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_250, (1536, 256), (256, 1))
    assert_size_stride(div_15, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_254, (256, 768), (768, 1))
    assert_size_stride(permute_261, (196, 196), (196, 1))
    assert_size_stride(div_16, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_264, (1536, 256), (256, 1))
    assert_size_stride(div_17, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_268, (256, 768), (768, 1))
    assert_size_stride(permute_275, (196, 196), (196, 1))
    assert_size_stride(div_18, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_278, (1536, 256), (256, 1))
    assert_size_stride(div_19, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_282, (256, 768), (768, 1))
    assert_size_stride(permute_289, (196, 196), (196, 1))
    assert_size_stride(div_20, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_292, (1536, 256), (256, 1))
    assert_size_stride(div_21, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_296, (256, 768), (768, 1))
    assert_size_stride(permute_303, (196, 196), (196, 1))
    assert_size_stride(div_22, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_306, (1536, 256), (256, 1))
    assert_size_stride(div_23, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_310, (256, 768), (768, 1))
    assert_size_stride(permute_317, (196, 196), (196, 1))
    assert_size_stride(div_24, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_320, (1536, 256), (256, 1))
    assert_size_stride(div_25, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_324, (256, 768), (768, 1))
    assert_size_stride(permute_331, (196, 196), (196, 1))
    assert_size_stride(div_26, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_334, (1536, 256), (256, 1))
    assert_size_stride(div_27, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_338, (256, 768), (768, 1))
    assert_size_stride(permute_345, (196, 196), (196, 1))
    assert_size_stride(div_28, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_348, (1536, 256), (256, 1))
    assert_size_stride(div_29, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_352, (256, 768), (768, 1))
    assert_size_stride(permute_359, (196, 196), (196, 1))
    assert_size_stride(div_30, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_362, (1536, 256), (256, 1))
    assert_size_stride(div_31, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_366, (256, 768), (768, 1))
    assert_size_stride(permute_373, (196, 196), (196, 1))
    assert_size_stride(div_32, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_376, (1536, 256), (256, 1))
    assert_size_stride(div_33, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_380, (256, 768), (768, 1))
    assert_size_stride(permute_387, (196, 196), (196, 1))
    assert_size_stride(div_34, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_390, (1536, 256), (256, 1))
    assert_size_stride(div_35, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_394, (256, 768), (768, 1))
    assert_size_stride(permute_401, (196, 196), (196, 1))
    assert_size_stride(div_36, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_404, (1536, 256), (256, 1))
    assert_size_stride(div_37, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_408, (256, 768), (768, 1))
    assert_size_stride(permute_415, (196, 196), (196, 1))
    assert_size_stride(div_38, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_418, (1536, 256), (256, 1))
    assert_size_stride(div_39, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_422, (256, 768), (768, 1))
    assert_size_stride(permute_429, (196, 196), (196, 1))
    assert_size_stride(div_40, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_432, (1536, 256), (256, 1))
    assert_size_stride(div_41, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_436, (256, 768), (768, 1))
    assert_size_stride(permute_443, (196, 196), (196, 1))
    assert_size_stride(div_42, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_446, (1536, 256), (256, 1))
    assert_size_stride(div_43, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_450, (256, 768), (768, 1))
    assert_size_stride(permute_457, (196, 196), (196, 1))
    assert_size_stride(div_44, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_460, (1536, 256), (256, 1))
    assert_size_stride(div_45, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_464, (256, 768), (768, 1))
    assert_size_stride(permute_471, (196, 196), (196, 1))
    assert_size_stride(div_46, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_474, (1536, 256), (256, 1))
    assert_size_stride(div_47, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_478, (256, 768), (768, 1))
    assert_size_stride(permute_485, (196, 196), (196, 1))
    assert_size_stride(div_48, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_488, (1536, 256), (256, 1))
    assert_size_stride(div_49, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_492, (256, 768), (768, 1))
    assert_size_stride(permute_499, (196, 196), (196, 1))
    assert_size_stride(div_50, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_502, (1536, 256), (256, 1))
    assert_size_stride(div_51, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_506, (256, 768), (768, 1))
    assert_size_stride(permute_513, (196, 196), (196, 1))
    assert_size_stride(div_52, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_516, (1536, 256), (256, 1))
    assert_size_stride(div_53, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_520, (256, 768), (768, 1))
    assert_size_stride(permute_527, (196, 196), (196, 1))
    assert_size_stride(div_54, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_530, (1536, 256), (256, 1))
    assert_size_stride(div_55, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_534, (256, 768), (768, 1))
    assert_size_stride(permute_541, (196, 196), (196, 1))
    assert_size_stride(div_56, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_544, (1536, 256), (256, 1))
    assert_size_stride(div_57, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_548, (256, 768), (768, 1))
    assert_size_stride(permute_555, (196, 196), (196, 1))
    assert_size_stride(div_58, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_558, (1536, 256), (256, 1))
    assert_size_stride(div_59, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_562, (256, 768), (768, 1))
    assert_size_stride(permute_569, (196, 196), (196, 1))
    assert_size_stride(div_60, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_572, (1536, 256), (256, 1))
    assert_size_stride(div_61, (8, 196, 1), (196, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_152, out=buf0)
        del permute_152
        buf1 = empty((1000, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_151, out=buf1)
        del clone_151
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf5 = empty((8, 196, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_layer_norm_backward]
        triton_red_fused_div_native_layer_norm_backward_1.run(buf0, primals_303, mul_240, div_1, buf5, 1568, 256, grid=grid(1568), stream=stream0)
        del div_1
        del primals_303
        buf6 = empty_strided((256, 13), (1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_layer_norm_backward]
        triton_red_fused_div_native_layer_norm_backward_2.run(buf0, mul_240, buf6, 3328, 121, grid=grid(3328), stream=stream0)
        del mul_240
        buf7 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf6, buf7, 256, 13, grid=grid(256), stream=stream0)
        buf8 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_layer_norm_backward]
        triton_red_fused_div_native_layer_norm_backward_4.run(buf0, buf8, 256, 1568, grid=grid(256), stream=stream0)
        del buf0
        buf9 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (1568, 256), (256, 1), 0), permute_156, out=buf9)
        del permute_156
        buf10 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (256, 1568), (1, 256), 0), view_179, out=buf10)
        del view_179
        buf11 = reinterpret_tensor(buf6, (1, 256, 13), (3328, 1, 256), 0); del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf5, buf11, 3328, 121, grid=grid(3328), stream=stream0)
        buf12 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf11, buf12, 256, 13, grid=grid(256), stream=stream0)
        buf13 = empty((1, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf9, getitem_176, buf13, 196, 6144, grid=grid(196), stream=stream0)
        buf14 = empty((8, 768, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf9, getitem_176, buf14, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del getitem_176
        buf15 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf14, (196, 6144), (1, 196), 0), view_177, out=buf15)
        del view_177
        buf16 = empty((6144, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf14, (6144, 196), (196, 1), 0), permute_163, out=buf16)
        del permute_163
        buf17 = empty_strided((8, 196, 1, 6), (1176, 1, 9408, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf16, primals_297, buf17, 9408, 128, grid=grid(9408), stream=stream0)
        buf18 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf17, buf18, 1568, 6, grid=grid(1568), stream=stream0)
        buf19 = reinterpret_tensor(buf17, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf17  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf16, primals_297, mul_237, buf19, 9408, 128, grid=grid(9408), stream=stream0)
        buf20 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_11.run(buf19, buf20, 1568, 6, grid=grid(1568), stream=stream0)
        buf21 = empty((768, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf16, mul_237, buf21, 9984, 121, grid=grid(9984), stream=stream0)
        buf22 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf21, buf22, 768, 13, grid=grid(768), stream=stream0)
        buf23 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_14.run(buf16, buf23, 768, 1568, grid=grid(768), stream=stream0)
        buf24 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        buf25 = buf24; del buf24  # reuse
        # Source Nodes: [x_237], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
        triton_poi_fused_cat_gelu_gelu_backward_15.run(buf25, buf9, mm_29, primals_300, div_2, buf16, primals_297, buf18, mul_237, buf20, addmm_58, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del addmm_58
        del div_2
        del mm_29
        del mul_237
        del primals_297
        del primals_300
        buf26 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf25, (1568, 1536), (1536, 1), 0), permute_166, out=buf26)
        del permute_166
        buf27 = empty((1536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf25, (1536, 1568), (1, 1536), 0), view_175, out=buf27)
        del view_175
        buf28 = empty_strided((1, 1536, 13), (19968, 1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf25, buf28, 19968, 121, grid=grid(19968), stream=stream0)
        buf29 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf28, buf29, 1536, 13, grid=grid(1536), stream=stream0)
        buf36 = buf5; del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf36, buf26, primals_293, mul_232, div_3, 1568, 256, grid=grid(1568), stream=stream0)
        del div_3
        del primals_293
        buf32 = reinterpret_tensor(buf11, (256, 13), (1, 256), 0); del buf11  # reuse
        buf34 = empty_strided((256, 13), (1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf26, mul_232, buf32, buf34, 3328, 121, grid=grid(3328), stream=stream0)
        del mul_232
        buf33 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf32, buf33, 256, 13, grid=grid(256), stream=stream0)
        buf35 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf34, buf35, 256, 13, grid=grid(256), stream=stream0)
        buf37 = buf9; del buf9  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (1568, 256), (256, 1), 0), permute_170, out=buf37)
        del permute_170
        buf38 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (256, 1568), (1, 256), 0), view_173, out=buf38)
        del view_173
        buf39 = reinterpret_tensor(buf34, (1, 256, 13), (3328, 1, 256), 0); del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf36, buf39, 3328, 121, grid=grid(3328), stream=stream0)
        buf40 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf39, buf40, 256, 13, grid=grid(256), stream=stream0)
        buf41 = empty((1, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf37, getitem_170, buf41, 196, 6144, grid=grid(196), stream=stream0)
        buf42 = reinterpret_tensor(buf16, (8, 768, 196), (150528, 196, 1), 0); del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf37, getitem_170, buf42, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del getitem_170
        buf43 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf42, (196, 6144), (1, 196), 0), view_171, out=buf43)
        del view_171
        buf44 = reinterpret_tensor(buf14, (6144, 196), (196, 1), 0); del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf42, (6144, 196), (196, 1), 0), permute_177, out=buf44)
        del permute_177
        buf45 = reinterpret_tensor(buf19, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf19  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf44, primals_287, buf45, 9408, 128, grid=grid(9408), stream=stream0)
        buf46 = buf20; del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf45, buf46, 1568, 6, grid=grid(1568), stream=stream0)
        buf47 = reinterpret_tensor(buf45, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf45  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf44, primals_287, mul_229, buf47, 9408, 128, grid=grid(9408), stream=stream0)
        buf48 = buf18; del buf18  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_11.run(buf47, buf48, 1568, 6, grid=grid(1568), stream=stream0)
        buf49 = buf21; del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf44, mul_229, buf49, 9984, 121, grid=grid(9984), stream=stream0)
        buf50 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf49, buf50, 768, 13, grid=grid(768), stream=stream0)
        buf51 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_14.run(buf44, buf51, 768, 1568, grid=grid(768), stream=stream0)
        buf52 = buf25; del buf25  # reuse
        buf53 = buf52; del buf52  # reuse
        # Source Nodes: [x_229], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
        triton_poi_fused_cat_gelu_gelu_backward_15.run(buf53, buf37, mm_28, primals_290, div_4, buf44, primals_287, buf46, mul_229, buf48, addmm_56, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del addmm_56
        del div_4
        del mm_28
        del mul_229
        del primals_287
        del primals_290
        buf54 = buf26; del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf53, (1568, 1536), (1536, 1), 0), permute_180, out=buf54)
        del permute_180
        buf55 = empty((1536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf53, (1536, 1568), (1, 1536), 0), view_169, out=buf55)
        del view_169
        buf56 = buf28; del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf53, buf56, 19968, 121, grid=grid(19968), stream=stream0)
        buf57 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf56, buf57, 1536, 13, grid=grid(1536), stream=stream0)
        buf64 = buf36; del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf64, buf54, primals_283, mul_224, div_5, 1568, 256, grid=grid(1568), stream=stream0)
        del div_5
        del primals_283
        buf60 = reinterpret_tensor(buf39, (256, 13), (1, 256), 0); del buf39  # reuse
        buf62 = buf32; del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf54, mul_224, buf60, buf62, 3328, 121, grid=grid(3328), stream=stream0)
        del mul_224
        buf61 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf60, buf61, 256, 13, grid=grid(256), stream=stream0)
        buf63 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf62, buf63, 256, 13, grid=grid(256), stream=stream0)
        buf65 = reinterpret_tensor(buf44, (1568, 768), (768, 1), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (1568, 256), (256, 1), 0), permute_184, out=buf65)
        del permute_184
        buf66 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (256, 1568), (1, 256), 0), view_167, out=buf66)
        del view_167
        buf67 = reinterpret_tensor(buf62, (1, 256, 13), (3328, 1, 256), 0); del buf62  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf64, buf67, 3328, 121, grid=grid(3328), stream=stream0)
        buf68 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf67, buf68, 256, 13, grid=grid(256), stream=stream0)
        buf69 = empty((1, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf65, getitem_164, buf69, 196, 6144, grid=grid(196), stream=stream0)
        buf70 = reinterpret_tensor(buf37, (8, 768, 196), (150528, 196, 1), 0); del buf37  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf65, getitem_164, buf70, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del getitem_164
        buf71 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf70, (196, 6144), (1, 196), 0), view_165, out=buf71)
        del view_165
        buf72 = reinterpret_tensor(buf42, (6144, 196), (196, 1), 0); del buf42  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf70, (6144, 196), (196, 1), 0), permute_191, out=buf72)
        del permute_191
        buf73 = reinterpret_tensor(buf47, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf47  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf72, primals_277, buf73, 9408, 128, grid=grid(9408), stream=stream0)
        buf74 = buf48; del buf48  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf73, buf74, 1568, 6, grid=grid(1568), stream=stream0)
        buf75 = reinterpret_tensor(buf73, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf73  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf72, primals_277, mul_221, buf75, 9408, 128, grid=grid(9408), stream=stream0)
        buf76 = buf46; del buf46  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_11.run(buf75, buf76, 1568, 6, grid=grid(1568), stream=stream0)
        buf77 = buf49; del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf72, mul_221, buf77, 9984, 121, grid=grid(9984), stream=stream0)
        buf78 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf77, buf78, 768, 13, grid=grid(768), stream=stream0)
        buf79 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_14.run(buf72, buf79, 768, 1568, grid=grid(768), stream=stream0)
        buf80 = buf53; del buf53  # reuse
        buf81 = buf80; del buf80  # reuse
        # Source Nodes: [x_221], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
        triton_poi_fused_cat_gelu_gelu_backward_15.run(buf81, buf65, mm_27, primals_280, div_6, buf72, primals_277, buf74, mul_221, buf76, addmm_54, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del addmm_54
        del div_6
        del mm_27
        del mul_221
        del primals_277
        del primals_280
        buf82 = buf54; del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf81, (1568, 1536), (1536, 1), 0), permute_194, out=buf82)
        del permute_194
        buf83 = empty((1536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf81, (1536, 1568), (1, 1536), 0), view_163, out=buf83)
        del view_163
        buf84 = buf56; del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf81, buf84, 19968, 121, grid=grid(19968), stream=stream0)
        buf85 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf84, buf85, 1536, 13, grid=grid(1536), stream=stream0)
        buf92 = buf64; del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf92, buf82, primals_273, mul_216, div_7, 1568, 256, grid=grid(1568), stream=stream0)
        del div_7
        del primals_273
        buf88 = reinterpret_tensor(buf67, (256, 13), (1, 256), 0); del buf67  # reuse
        buf90 = buf60; del buf60  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf82, mul_216, buf88, buf90, 3328, 121, grid=grid(3328), stream=stream0)
        del mul_216
        buf89 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf88, buf89, 256, 13, grid=grid(256), stream=stream0)
        buf91 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf90, buf91, 256, 13, grid=grid(256), stream=stream0)
        buf93 = reinterpret_tensor(buf72, (1568, 768), (768, 1), 0); del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf92, (1568, 256), (256, 1), 0), permute_198, out=buf93)
        del permute_198
        buf94 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf92, (256, 1568), (1, 256), 0), view_161, out=buf94)
        del view_161
        buf95 = reinterpret_tensor(buf90, (1, 256, 13), (3328, 1, 256), 0); del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf92, buf95, 3328, 121, grid=grid(3328), stream=stream0)
        buf96 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf95, buf96, 256, 13, grid=grid(256), stream=stream0)
        buf97 = empty((1, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf93, getitem_158, buf97, 196, 6144, grid=grid(196), stream=stream0)
        buf98 = reinterpret_tensor(buf65, (8, 768, 196), (150528, 196, 1), 0); del buf65  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf93, getitem_158, buf98, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del getitem_158
        buf99 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf98, (196, 6144), (1, 196), 0), view_159, out=buf99)
        del view_159
        buf100 = reinterpret_tensor(buf70, (6144, 196), (196, 1), 0); del buf70  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf98, (6144, 196), (196, 1), 0), permute_205, out=buf100)
        del permute_205
        buf101 = reinterpret_tensor(buf75, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf75  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf100, primals_267, buf101, 9408, 128, grid=grid(9408), stream=stream0)
        buf102 = buf76; del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf101, buf102, 1568, 6, grid=grid(1568), stream=stream0)
        buf103 = reinterpret_tensor(buf101, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf101  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf100, primals_267, mul_213, buf103, 9408, 128, grid=grid(9408), stream=stream0)
        buf104 = buf74; del buf74  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_11.run(buf103, buf104, 1568, 6, grid=grid(1568), stream=stream0)
        buf105 = buf77; del buf77  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf100, mul_213, buf105, 9984, 121, grid=grid(9984), stream=stream0)
        buf106 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf105, buf106, 768, 13, grid=grid(768), stream=stream0)
        buf107 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_14.run(buf100, buf107, 768, 1568, grid=grid(768), stream=stream0)
        buf108 = buf81; del buf81  # reuse
        buf109 = buf108; del buf108  # reuse
        # Source Nodes: [x_213], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
        triton_poi_fused_cat_gelu_gelu_backward_15.run(buf109, buf93, mm_26, primals_270, div_8, buf100, primals_267, buf102, mul_213, buf104, addmm_52, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del addmm_52
        del div_8
        del mm_26
        del mul_213
        del primals_267
        del primals_270
        buf110 = buf82; del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf109, (1568, 1536), (1536, 1), 0), permute_208, out=buf110)
        del permute_208
        buf111 = empty((1536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf109, (1536, 1568), (1, 1536), 0), view_157, out=buf111)
        del view_157
        buf112 = buf84; del buf84  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf109, buf112, 19968, 121, grid=grid(19968), stream=stream0)
        buf113 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf112, buf113, 1536, 13, grid=grid(1536), stream=stream0)
        buf120 = buf92; del buf92  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf120, buf110, primals_263, mul_208, div_9, 1568, 256, grid=grid(1568), stream=stream0)
        del div_9
        del primals_263
        buf116 = reinterpret_tensor(buf95, (256, 13), (1, 256), 0); del buf95  # reuse
        buf118 = buf88; del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf110, mul_208, buf116, buf118, 3328, 121, grid=grid(3328), stream=stream0)
        del mul_208
        buf117 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf116, buf117, 256, 13, grid=grid(256), stream=stream0)
        buf119 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf118, buf119, 256, 13, grid=grid(256), stream=stream0)
        buf121 = buf93; del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf120, (1568, 256), (256, 1), 0), permute_212, out=buf121)
        del permute_212
        buf122 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf120, (256, 1568), (1, 256), 0), view_155, out=buf122)
        del view_155
        buf123 = reinterpret_tensor(buf118, (1, 256, 13), (3328, 1, 256), 0); del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf120, buf123, 3328, 121, grid=grid(3328), stream=stream0)
        buf124 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf123, buf124, 256, 13, grid=grid(256), stream=stream0)
        buf125 = empty((1, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf121, getitem_152, buf125, 196, 6144, grid=grid(196), stream=stream0)
        buf126 = reinterpret_tensor(buf100, (8, 768, 196), (150528, 196, 1), 0); del buf100  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf121, getitem_152, buf126, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del getitem_152
        buf127 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf126, (196, 6144), (1, 196), 0), view_153, out=buf127)
        del view_153
        buf128 = reinterpret_tensor(buf98, (6144, 196), (196, 1), 0); del buf98  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf126, (6144, 196), (196, 1), 0), permute_219, out=buf128)
        del permute_219
        buf129 = reinterpret_tensor(buf103, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf128, primals_257, buf129, 9408, 128, grid=grid(9408), stream=stream0)
        buf130 = buf104; del buf104  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf129, buf130, 1568, 6, grid=grid(1568), stream=stream0)
        buf131 = reinterpret_tensor(buf129, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf129  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf128, primals_257, mul_205, buf131, 9408, 128, grid=grid(9408), stream=stream0)
        buf132 = buf102; del buf102  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_11.run(buf131, buf132, 1568, 6, grid=grid(1568), stream=stream0)
        buf133 = buf105; del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf128, mul_205, buf133, 9984, 121, grid=grid(9984), stream=stream0)
        buf134 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf133, buf134, 768, 13, grid=grid(768), stream=stream0)
        buf135 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_14.run(buf128, buf135, 768, 1568, grid=grid(768), stream=stream0)
        buf136 = buf109; del buf109  # reuse
        buf137 = buf136; del buf136  # reuse
        # Source Nodes: [x_205], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
        triton_poi_fused_cat_gelu_gelu_backward_15.run(buf137, buf121, mm_25, primals_260, div_10, buf128, primals_257, buf130, mul_205, buf132, addmm_50, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del addmm_50
        del div_10
        del mm_25
        del mul_205
        del primals_257
        del primals_260
        buf138 = buf110; del buf110  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf137, (1568, 1536), (1536, 1), 0), permute_222, out=buf138)
        del permute_222
        buf139 = empty((1536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf137, (1536, 1568), (1, 1536), 0), view_151, out=buf139)
        del view_151
        buf140 = buf112; del buf112  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf137, buf140, 19968, 121, grid=grid(19968), stream=stream0)
        buf141 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf140, buf141, 1536, 13, grid=grid(1536), stream=stream0)
        buf148 = buf120; del buf120  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf148, buf138, primals_253, mul_200, div_11, 1568, 256, grid=grid(1568), stream=stream0)
        del div_11
        del primals_253
        buf144 = reinterpret_tensor(buf123, (256, 13), (1, 256), 0); del buf123  # reuse
        buf146 = buf116; del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf138, mul_200, buf144, buf146, 3328, 121, grid=grid(3328), stream=stream0)
        del mul_200
        buf145 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf144, buf145, 256, 13, grid=grid(256), stream=stream0)
        buf147 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf146, buf147, 256, 13, grid=grid(256), stream=stream0)
        buf149 = reinterpret_tensor(buf128, (1568, 768), (768, 1), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf148, (1568, 256), (256, 1), 0), permute_226, out=buf149)
        del permute_226
        buf150 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf148, (256, 1568), (1, 256), 0), view_149, out=buf150)
        del view_149
        buf151 = reinterpret_tensor(buf146, (1, 256, 13), (3328, 1, 256), 0); del buf146  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf148, buf151, 3328, 121, grid=grid(3328), stream=stream0)
        buf152 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf151, buf152, 256, 13, grid=grid(256), stream=stream0)
        buf153 = empty((1, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf149, getitem_146, buf153, 196, 6144, grid=grid(196), stream=stream0)
        buf154 = reinterpret_tensor(buf121, (8, 768, 196), (150528, 196, 1), 0); del buf121  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf149, getitem_146, buf154, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del getitem_146
        buf155 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (196, 6144), (1, 196), 0), view_147, out=buf155)
        del view_147
        buf156 = reinterpret_tensor(buf126, (6144, 196), (196, 1), 0); del buf126  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (6144, 196), (196, 1), 0), permute_233, out=buf156)
        del permute_233
        buf157 = reinterpret_tensor(buf131, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf131  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf156, primals_247, buf157, 9408, 128, grid=grid(9408), stream=stream0)
        buf158 = buf132; del buf132  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf157, buf158, 1568, 6, grid=grid(1568), stream=stream0)
        buf159 = reinterpret_tensor(buf157, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf156, primals_247, mul_197, buf159, 9408, 128, grid=grid(9408), stream=stream0)
        buf160 = buf130; del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_11.run(buf159, buf160, 1568, 6, grid=grid(1568), stream=stream0)
        buf161 = buf133; del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf156, mul_197, buf161, 9984, 121, grid=grid(9984), stream=stream0)
        buf162 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf161, buf162, 768, 13, grid=grid(768), stream=stream0)
        buf163 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_14.run(buf156, buf163, 768, 1568, grid=grid(768), stream=stream0)
        buf164 = buf137; del buf137  # reuse
        buf165 = buf164; del buf164  # reuse
        # Source Nodes: [x_197], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
        triton_poi_fused_cat_gelu_gelu_backward_15.run(buf165, buf149, mm_24, primals_250, div_12, buf156, primals_247, buf158, mul_197, buf160, addmm_48, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del addmm_48
        del div_12
        del mm_24
        del mul_197
        del primals_247
        del primals_250
        buf166 = buf138; del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (1568, 1536), (1536, 1), 0), permute_236, out=buf166)
        del permute_236
        buf167 = empty((1536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (1536, 1568), (1, 1536), 0), view_145, out=buf167)
        del view_145
        buf168 = buf140; del buf140  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf165, buf168, 19968, 121, grid=grid(19968), stream=stream0)
        buf169 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf168, buf169, 1536, 13, grid=grid(1536), stream=stream0)
        buf176 = buf148; del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf176, buf166, primals_243, mul_192, div_13, 1568, 256, grid=grid(1568), stream=stream0)
        del div_13
        del primals_243
        buf172 = reinterpret_tensor(buf151, (256, 13), (1, 256), 0); del buf151  # reuse
        buf174 = buf144; del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf166, mul_192, buf172, buf174, 3328, 121, grid=grid(3328), stream=stream0)
        del mul_192
        buf173 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf172, buf173, 256, 13, grid=grid(256), stream=stream0)
        buf175 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf174, buf175, 256, 13, grid=grid(256), stream=stream0)
        buf177 = reinterpret_tensor(buf156, (1568, 768), (768, 1), 0); del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf176, (1568, 256), (256, 1), 0), permute_240, out=buf177)
        del permute_240
        buf178 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf176, (256, 1568), (1, 256), 0), view_143, out=buf178)
        del view_143
        buf179 = reinterpret_tensor(buf174, (1, 256, 13), (3328, 1, 256), 0); del buf174  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf176, buf179, 3328, 121, grid=grid(3328), stream=stream0)
        buf180 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf179, buf180, 256, 13, grid=grid(256), stream=stream0)
        buf181 = empty((1, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf177, getitem_140, buf181, 196, 6144, grid=grid(196), stream=stream0)
        buf182 = reinterpret_tensor(buf149, (8, 768, 196), (150528, 196, 1), 0); del buf149  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf177, getitem_140, buf182, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del getitem_140
        buf183 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf182, (196, 6144), (1, 196), 0), view_141, out=buf183)
        del view_141
        buf184 = reinterpret_tensor(buf154, (6144, 196), (196, 1), 0); del buf154  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf182, (6144, 196), (196, 1), 0), permute_247, out=buf184)
        del permute_247
        buf185 = reinterpret_tensor(buf159, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf159  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf184, primals_237, buf185, 9408, 128, grid=grid(9408), stream=stream0)
        buf186 = buf160; del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf185, buf186, 1568, 6, grid=grid(1568), stream=stream0)
        buf187 = reinterpret_tensor(buf185, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf184, primals_237, mul_189, buf187, 9408, 128, grid=grid(9408), stream=stream0)
        buf188 = buf158; del buf158  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_11.run(buf187, buf188, 1568, 6, grid=grid(1568), stream=stream0)
        buf189 = buf161; del buf161  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf184, mul_189, buf189, 9984, 121, grid=grid(9984), stream=stream0)
        buf190 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf189, buf190, 768, 13, grid=grid(768), stream=stream0)
        buf191 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_14.run(buf184, buf191, 768, 1568, grid=grid(768), stream=stream0)
        buf192 = buf165; del buf165  # reuse
        buf193 = buf192; del buf192  # reuse
        # Source Nodes: [x_189], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
        triton_poi_fused_cat_gelu_gelu_backward_15.run(buf193, buf177, mm_23, primals_240, div_14, buf184, primals_237, buf186, mul_189, buf188, addmm_46, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del addmm_46
        del div_14
        del mm_23
        del mul_189
        del primals_237
        del primals_240
        buf194 = buf166; del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf193, (1568, 1536), (1536, 1), 0), permute_250, out=buf194)
        del permute_250
        buf195 = empty((1536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf193, (1536, 1568), (1, 1536), 0), view_139, out=buf195)
        del view_139
        buf196 = buf168; del buf168  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf193, buf196, 19968, 121, grid=grid(19968), stream=stream0)
        buf197 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf196, buf197, 1536, 13, grid=grid(1536), stream=stream0)
        buf204 = buf176; del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf204, buf194, primals_233, mul_184, div_15, 1568, 256, grid=grid(1568), stream=stream0)
        del div_15
        del primals_233
        buf200 = reinterpret_tensor(buf179, (256, 13), (1, 256), 0); del buf179  # reuse
        buf202 = buf172; del buf172  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf194, mul_184, buf200, buf202, 3328, 121, grid=grid(3328), stream=stream0)
        del mul_184
        buf201 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf200, buf201, 256, 13, grid=grid(256), stream=stream0)
        buf203 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf202, buf203, 256, 13, grid=grid(256), stream=stream0)
        buf205 = reinterpret_tensor(buf184, (1568, 768), (768, 1), 0); del buf184  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf204, (1568, 256), (256, 1), 0), permute_254, out=buf205)
        del permute_254
        buf206 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf204, (256, 1568), (1, 256), 0), view_137, out=buf206)
        del view_137
        buf207 = reinterpret_tensor(buf202, (1, 256, 13), (3328, 1, 256), 0); del buf202  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf204, buf207, 3328, 121, grid=grid(3328), stream=stream0)
        buf208 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf207, buf208, 256, 13, grid=grid(256), stream=stream0)
        buf209 = empty((1, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf205, getitem_134, buf209, 196, 6144, grid=grid(196), stream=stream0)
        buf210 = reinterpret_tensor(buf177, (8, 768, 196), (150528, 196, 1), 0); del buf177  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf205, getitem_134, buf210, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del getitem_134
        buf211 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf210, (196, 6144), (1, 196), 0), view_135, out=buf211)
        del view_135
        buf212 = reinterpret_tensor(buf182, (6144, 196), (196, 1), 0); del buf182  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf210, (6144, 196), (196, 1), 0), permute_261, out=buf212)
        del permute_261
        buf213 = reinterpret_tensor(buf187, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf187  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf212, primals_227, buf213, 9408, 128, grid=grid(9408), stream=stream0)
        buf214 = buf188; del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf213, buf214, 1568, 6, grid=grid(1568), stream=stream0)
        buf215 = reinterpret_tensor(buf213, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf212, primals_227, mul_181, buf215, 9408, 128, grid=grid(9408), stream=stream0)
        buf216 = buf186; del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_11.run(buf215, buf216, 1568, 6, grid=grid(1568), stream=stream0)
        buf217 = buf189; del buf189  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf212, mul_181, buf217, 9984, 121, grid=grid(9984), stream=stream0)
        buf218 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf217, buf218, 768, 13, grid=grid(768), stream=stream0)
        buf219 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_14.run(buf212, buf219, 768, 1568, grid=grid(768), stream=stream0)
        buf220 = buf193; del buf193  # reuse
        buf221 = buf220; del buf220  # reuse
        # Source Nodes: [x_181], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
        triton_poi_fused_cat_gelu_gelu_backward_15.run(buf221, buf205, mm_22, primals_230, div_16, buf212, primals_227, buf214, mul_181, buf216, addmm_44, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del addmm_44
        del div_16
        del mm_22
        del mul_181
        del primals_227
        del primals_230
        buf222 = buf194; del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf221, (1568, 1536), (1536, 1), 0), permute_264, out=buf222)
        del permute_264
        buf223 = empty((1536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf221, (1536, 1568), (1, 1536), 0), view_133, out=buf223)
        del view_133
        buf224 = buf196; del buf196  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf221, buf224, 19968, 121, grid=grid(19968), stream=stream0)
        buf225 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf224, buf225, 1536, 13, grid=grid(1536), stream=stream0)
        buf232 = buf204; del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf232, buf222, primals_223, mul_176, div_17, 1568, 256, grid=grid(1568), stream=stream0)
        del div_17
        del primals_223
        buf228 = reinterpret_tensor(buf207, (256, 13), (1, 256), 0); del buf207  # reuse
        buf230 = buf200; del buf200  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf222, mul_176, buf228, buf230, 3328, 121, grid=grid(3328), stream=stream0)
        del mul_176
        buf229 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf228, buf229, 256, 13, grid=grid(256), stream=stream0)
        buf231 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf230, buf231, 256, 13, grid=grid(256), stream=stream0)
        buf233 = reinterpret_tensor(buf212, (1568, 768), (768, 1), 0); del buf212  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf232, (1568, 256), (256, 1), 0), permute_268, out=buf233)
        del permute_268
        buf234 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf232, (256, 1568), (1, 256), 0), view_131, out=buf234)
        del view_131
        buf235 = reinterpret_tensor(buf230, (1, 256, 13), (3328, 1, 256), 0); del buf230  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf232, buf235, 3328, 121, grid=grid(3328), stream=stream0)
        buf236 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf235, buf236, 256, 13, grid=grid(256), stream=stream0)
        buf237 = empty((1, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf233, getitem_128, buf237, 196, 6144, grid=grid(196), stream=stream0)
        buf238 = reinterpret_tensor(buf205, (8, 768, 196), (150528, 196, 1), 0); del buf205  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf233, getitem_128, buf238, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del getitem_128
        buf239 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf238, (196, 6144), (1, 196), 0), view_129, out=buf239)
        del view_129
        buf240 = reinterpret_tensor(buf210, (6144, 196), (196, 1), 0); del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf238, (6144, 196), (196, 1), 0), permute_275, out=buf240)
        del permute_275
        buf241 = reinterpret_tensor(buf215, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf240, primals_217, buf241, 9408, 128, grid=grid(9408), stream=stream0)
        buf242 = buf216; del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf241, buf242, 1568, 6, grid=grid(1568), stream=stream0)
        buf243 = reinterpret_tensor(buf241, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf241  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf240, primals_217, mul_173, buf243, 9408, 128, grid=grid(9408), stream=stream0)
        buf244 = buf214; del buf214  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_11.run(buf243, buf244, 1568, 6, grid=grid(1568), stream=stream0)
        buf245 = buf217; del buf217  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf240, mul_173, buf245, 9984, 121, grid=grid(9984), stream=stream0)
        buf246 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf245, buf246, 768, 13, grid=grid(768), stream=stream0)
        buf247 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_14.run(buf240, buf247, 768, 1568, grid=grid(768), stream=stream0)
        buf248 = buf221; del buf221  # reuse
        buf249 = buf248; del buf248  # reuse
        # Source Nodes: [x_173], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
        triton_poi_fused_cat_gelu_gelu_backward_15.run(buf249, buf233, mm_21, primals_220, div_18, buf240, primals_217, buf242, mul_173, buf244, addmm_42, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del addmm_42
        del div_18
        del mm_21
        del mul_173
        del primals_217
        del primals_220
        buf250 = buf222; del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf249, (1568, 1536), (1536, 1), 0), permute_278, out=buf250)
        del permute_278
        buf251 = empty((1536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf249, (1536, 1568), (1, 1536), 0), view_127, out=buf251)
        del view_127
        buf252 = buf224; del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf249, buf252, 19968, 121, grid=grid(19968), stream=stream0)
        buf253 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf252, buf253, 1536, 13, grid=grid(1536), stream=stream0)
        buf260 = buf232; del buf232  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf260, buf250, primals_213, mul_168, div_19, 1568, 256, grid=grid(1568), stream=stream0)
        del div_19
        del primals_213
        buf256 = reinterpret_tensor(buf235, (256, 13), (1, 256), 0); del buf235  # reuse
        buf258 = buf228; del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf250, mul_168, buf256, buf258, 3328, 121, grid=grid(3328), stream=stream0)
        del mul_168
        buf257 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf256, buf257, 256, 13, grid=grid(256), stream=stream0)
        buf259 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf258, buf259, 256, 13, grid=grid(256), stream=stream0)
        buf261 = reinterpret_tensor(buf240, (1568, 768), (768, 1), 0); del buf240  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf260, (1568, 256), (256, 1), 0), permute_282, out=buf261)
        del permute_282
        buf262 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf260, (256, 1568), (1, 256), 0), view_125, out=buf262)
        del view_125
        buf263 = reinterpret_tensor(buf258, (1, 256, 13), (3328, 1, 256), 0); del buf258  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf260, buf263, 3328, 121, grid=grid(3328), stream=stream0)
        buf264 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf263, buf264, 256, 13, grid=grid(256), stream=stream0)
        buf265 = empty((1, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf261, getitem_122, buf265, 196, 6144, grid=grid(196), stream=stream0)
        buf266 = reinterpret_tensor(buf233, (8, 768, 196), (150528, 196, 1), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf261, getitem_122, buf266, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del getitem_122
        buf267 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf266, (196, 6144), (1, 196), 0), view_123, out=buf267)
        del view_123
        buf268 = reinterpret_tensor(buf238, (6144, 196), (196, 1), 0); del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf266, (6144, 196), (196, 1), 0), permute_289, out=buf268)
        del permute_289
        buf269 = reinterpret_tensor(buf243, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf243  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf268, primals_207, buf269, 9408, 128, grid=grid(9408), stream=stream0)
        buf270 = buf244; del buf244  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf269, buf270, 1568, 6, grid=grid(1568), stream=stream0)
        buf271 = reinterpret_tensor(buf269, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf269  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf268, primals_207, mul_165, buf271, 9408, 128, grid=grid(9408), stream=stream0)
        buf272 = buf242; del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_11.run(buf271, buf272, 1568, 6, grid=grid(1568), stream=stream0)
        buf273 = buf245; del buf245  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf268, mul_165, buf273, 9984, 121, grid=grid(9984), stream=stream0)
        buf274 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf273, buf274, 768, 13, grid=grid(768), stream=stream0)
        buf275 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_14.run(buf268, buf275, 768, 1568, grid=grid(768), stream=stream0)
        buf276 = buf249; del buf249  # reuse
        buf277 = buf276; del buf276  # reuse
        # Source Nodes: [x_165], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
        triton_poi_fused_cat_gelu_gelu_backward_15.run(buf277, buf261, mm_20, primals_210, div_20, buf268, primals_207, buf270, mul_165, buf272, addmm_40, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del addmm_40
        del div_20
        del mm_20
        del mul_165
        del primals_207
        del primals_210
        buf278 = buf250; del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf277, (1568, 1536), (1536, 1), 0), permute_292, out=buf278)
        del permute_292
        buf279 = empty((1536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf277, (1536, 1568), (1, 1536), 0), view_121, out=buf279)
        del view_121
        buf280 = buf252; del buf252  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf277, buf280, 19968, 121, grid=grid(19968), stream=stream0)
        buf281 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf280, buf281, 1536, 13, grid=grid(1536), stream=stream0)
        buf288 = buf260; del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf288, buf278, primals_203, mul_160, div_21, 1568, 256, grid=grid(1568), stream=stream0)
        del div_21
        del primals_203
        buf284 = reinterpret_tensor(buf263, (256, 13), (1, 256), 0); del buf263  # reuse
        buf286 = buf256; del buf256  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf278, mul_160, buf284, buf286, 3328, 121, grid=grid(3328), stream=stream0)
        del mul_160
        buf285 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf284, buf285, 256, 13, grid=grid(256), stream=stream0)
        buf287 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf286, buf287, 256, 13, grid=grid(256), stream=stream0)
        buf289 = reinterpret_tensor(buf268, (1568, 768), (768, 1), 0); del buf268  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf288, (1568, 256), (256, 1), 0), permute_296, out=buf289)
        del permute_296
        buf290 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf288, (256, 1568), (1, 256), 0), view_119, out=buf290)
        del view_119
        buf291 = reinterpret_tensor(buf286, (1, 256, 13), (3328, 1, 256), 0); del buf286  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf288, buf291, 3328, 121, grid=grid(3328), stream=stream0)
        buf292 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf291, buf292, 256, 13, grid=grid(256), stream=stream0)
        buf293 = empty((1, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf289, getitem_116, buf293, 196, 6144, grid=grid(196), stream=stream0)
        buf294 = reinterpret_tensor(buf261, (8, 768, 196), (150528, 196, 1), 0); del buf261  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf289, getitem_116, buf294, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del getitem_116
        buf295 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf294, (196, 6144), (1, 196), 0), view_117, out=buf295)
        del view_117
        buf296 = reinterpret_tensor(buf266, (6144, 196), (196, 1), 0); del buf266  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf294, (6144, 196), (196, 1), 0), permute_303, out=buf296)
        del permute_303
        buf297 = reinterpret_tensor(buf271, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf271  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf296, primals_197, buf297, 9408, 128, grid=grid(9408), stream=stream0)
        buf298 = buf272; del buf272  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf297, buf298, 1568, 6, grid=grid(1568), stream=stream0)
        buf299 = reinterpret_tensor(buf297, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf297  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf296, primals_197, mul_157, buf299, 9408, 128, grid=grid(9408), stream=stream0)
        buf300 = buf270; del buf270  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_11.run(buf299, buf300, 1568, 6, grid=grid(1568), stream=stream0)
        buf301 = buf273; del buf273  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf296, mul_157, buf301, 9984, 121, grid=grid(9984), stream=stream0)
        buf302 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf301, buf302, 768, 13, grid=grid(768), stream=stream0)
        buf303 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_14.run(buf296, buf303, 768, 1568, grid=grid(768), stream=stream0)
        buf304 = buf277; del buf277  # reuse
        buf305 = buf304; del buf304  # reuse
        # Source Nodes: [x_157], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
        triton_poi_fused_cat_gelu_gelu_backward_15.run(buf305, buf289, mm_19, primals_200, div_22, buf296, primals_197, buf298, mul_157, buf300, addmm_38, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del addmm_38
        del div_22
        del mm_19
        del mul_157
        del primals_197
        del primals_200
        buf306 = buf278; del buf278  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf305, (1568, 1536), (1536, 1), 0), permute_306, out=buf306)
        del permute_306
        buf307 = empty((1536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf305, (1536, 1568), (1, 1536), 0), view_115, out=buf307)
        del view_115
        buf308 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf305, buf308, 19968, 121, grid=grid(19968), stream=stream0)
        buf309 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf308, buf309, 1536, 13, grid=grid(1536), stream=stream0)
        buf316 = buf288; del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf316, buf306, primals_193, mul_152, div_23, 1568, 256, grid=grid(1568), stream=stream0)
        del div_23
        del primals_193
        buf312 = reinterpret_tensor(buf291, (256, 13), (1, 256), 0); del buf291  # reuse
        buf314 = buf284; del buf284  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf306, mul_152, buf312, buf314, 3328, 121, grid=grid(3328), stream=stream0)
        del mul_152
        buf313 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf312, buf313, 256, 13, grid=grid(256), stream=stream0)
        buf315 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf314, buf315, 256, 13, grid=grid(256), stream=stream0)
        buf317 = reinterpret_tensor(buf296, (1568, 768), (768, 1), 0); del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf316, (1568, 256), (256, 1), 0), permute_310, out=buf317)
        del permute_310
        buf318 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf316, (256, 1568), (1, 256), 0), view_113, out=buf318)
        del view_113
        buf319 = reinterpret_tensor(buf314, (1, 256, 13), (3328, 1, 256), 0); del buf314  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf316, buf319, 3328, 121, grid=grid(3328), stream=stream0)
        buf320 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf319, buf320, 256, 13, grid=grid(256), stream=stream0)
        buf321 = empty((1, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf317, getitem_110, buf321, 196, 6144, grid=grid(196), stream=stream0)
        buf322 = reinterpret_tensor(buf289, (8, 768, 196), (150528, 196, 1), 0); del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf317, getitem_110, buf322, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del getitem_110
        buf323 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf322, (196, 6144), (1, 196), 0), view_111, out=buf323)
        del view_111
        buf324 = reinterpret_tensor(buf294, (6144, 196), (196, 1), 0); del buf294  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf322, (6144, 196), (196, 1), 0), permute_317, out=buf324)
        del permute_317
        buf325 = reinterpret_tensor(buf299, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf299  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf324, primals_187, buf325, 9408, 128, grid=grid(9408), stream=stream0)
        buf326 = buf300; del buf300  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf325, buf326, 1568, 6, grid=grid(1568), stream=stream0)
        buf327 = reinterpret_tensor(buf325, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf325  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf324, primals_187, mul_149, buf327, 9408, 128, grid=grid(9408), stream=stream0)
        buf328 = buf298; del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_11.run(buf327, buf328, 1568, 6, grid=grid(1568), stream=stream0)
        buf329 = buf301; del buf301  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf324, mul_149, buf329, 9984, 121, grid=grid(9984), stream=stream0)
        buf330 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf329, buf330, 768, 13, grid=grid(768), stream=stream0)
        buf331 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_14.run(buf324, buf331, 768, 1568, grid=grid(768), stream=stream0)
        buf332 = buf305; del buf305  # reuse
        buf333 = buf332; del buf332  # reuse
        # Source Nodes: [x_149], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
        triton_poi_fused_cat_gelu_gelu_backward_15.run(buf333, buf317, mm_18, primals_190, div_24, buf324, primals_187, buf326, mul_149, buf328, addmm_36, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del addmm_36
        del div_24
        del mm_18
        del mul_149
        del primals_187
        del primals_190
        buf334 = buf306; del buf306  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf333, (1568, 1536), (1536, 1), 0), permute_320, out=buf334)
        del permute_320
        buf335 = empty((1536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf333, (1536, 1568), (1, 1536), 0), view_109, out=buf335)
        del view_109
        buf336 = buf308; del buf308  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf333, buf336, 19968, 121, grid=grid(19968), stream=stream0)
        buf337 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf336, buf337, 1536, 13, grid=grid(1536), stream=stream0)
        buf344 = buf316; del buf316  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf344, buf334, primals_183, mul_144, div_25, 1568, 256, grid=grid(1568), stream=stream0)
        del div_25
        del primals_183
        buf340 = reinterpret_tensor(buf319, (256, 13), (1, 256), 0); del buf319  # reuse
        buf342 = buf312; del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf334, mul_144, buf340, buf342, 3328, 121, grid=grid(3328), stream=stream0)
        del mul_144
        buf341 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf340, buf341, 256, 13, grid=grid(256), stream=stream0)
        buf343 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf342, buf343, 256, 13, grid=grid(256), stream=stream0)
        buf345 = reinterpret_tensor(buf324, (1568, 768), (768, 1), 0); del buf324  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf344, (1568, 256), (256, 1), 0), permute_324, out=buf345)
        del permute_324
        buf346 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf344, (256, 1568), (1, 256), 0), view_107, out=buf346)
        del view_107
        buf347 = reinterpret_tensor(buf342, (1, 256, 13), (3328, 1, 256), 0); del buf342  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf344, buf347, 3328, 121, grid=grid(3328), stream=stream0)
        buf348 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf347, buf348, 256, 13, grid=grid(256), stream=stream0)
        buf349 = empty((1, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf345, getitem_104, buf349, 196, 6144, grid=grid(196), stream=stream0)
        buf350 = reinterpret_tensor(buf317, (8, 768, 196), (150528, 196, 1), 0); del buf317  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf345, getitem_104, buf350, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del getitem_104
        buf351 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf350, (196, 6144), (1, 196), 0), view_105, out=buf351)
        del view_105
        buf352 = reinterpret_tensor(buf322, (6144, 196), (196, 1), 0); del buf322  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf350, (6144, 196), (196, 1), 0), permute_331, out=buf352)
        del permute_331
        buf353 = reinterpret_tensor(buf327, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf352, primals_177, buf353, 9408, 128, grid=grid(9408), stream=stream0)
        buf354 = buf328; del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf353, buf354, 1568, 6, grid=grid(1568), stream=stream0)
        buf355 = reinterpret_tensor(buf353, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf353  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf352, primals_177, mul_141, buf355, 9408, 128, grid=grid(9408), stream=stream0)
        buf356 = buf326; del buf326  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_11.run(buf355, buf356, 1568, 6, grid=grid(1568), stream=stream0)
        buf357 = buf329; del buf329  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf352, mul_141, buf357, 9984, 121, grid=grid(9984), stream=stream0)
        buf358 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf357, buf358, 768, 13, grid=grid(768), stream=stream0)
        buf359 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_14.run(buf352, buf359, 768, 1568, grid=grid(768), stream=stream0)
        buf360 = buf333; del buf333  # reuse
        buf361 = buf360; del buf360  # reuse
        # Source Nodes: [x_141], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
        triton_poi_fused_cat_gelu_gelu_backward_15.run(buf361, buf345, mm_17, primals_180, div_26, buf352, primals_177, buf354, mul_141, buf356, addmm_34, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del addmm_34
        del div_26
        del mm_17
        del mul_141
        del primals_177
        del primals_180
        buf362 = buf334; del buf334  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf361, (1568, 1536), (1536, 1), 0), permute_334, out=buf362)
        del permute_334
        buf363 = empty((1536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf361, (1536, 1568), (1, 1536), 0), view_103, out=buf363)
        del view_103
        buf364 = buf336; del buf336  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf361, buf364, 19968, 121, grid=grid(19968), stream=stream0)
        buf365 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf364, buf365, 1536, 13, grid=grid(1536), stream=stream0)
        buf372 = buf344; del buf344  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf372, buf362, primals_173, mul_136, div_27, 1568, 256, grid=grid(1568), stream=stream0)
        del div_27
        del primals_173
        buf368 = reinterpret_tensor(buf347, (256, 13), (1, 256), 0); del buf347  # reuse
        buf370 = buf340; del buf340  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf362, mul_136, buf368, buf370, 3328, 121, grid=grid(3328), stream=stream0)
        del mul_136
        buf369 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf368, buf369, 256, 13, grid=grid(256), stream=stream0)
        buf371 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf370, buf371, 256, 13, grid=grid(256), stream=stream0)
        buf373 = reinterpret_tensor(buf352, (1568, 768), (768, 1), 0); del buf352  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf372, (1568, 256), (256, 1), 0), permute_338, out=buf373)
        del permute_338
        buf374 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf372, (256, 1568), (1, 256), 0), view_101, out=buf374)
        del view_101
        buf375 = reinterpret_tensor(buf370, (1, 256, 13), (3328, 1, 256), 0); del buf370  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf372, buf375, 3328, 121, grid=grid(3328), stream=stream0)
        buf376 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf375, buf376, 256, 13, grid=grid(256), stream=stream0)
        buf377 = empty((1, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf373, getitem_98, buf377, 196, 6144, grid=grid(196), stream=stream0)
        buf378 = reinterpret_tensor(buf345, (8, 768, 196), (150528, 196, 1), 0); del buf345  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf373, getitem_98, buf378, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del getitem_98
        buf379 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf378, (196, 6144), (1, 196), 0), view_99, out=buf379)
        del view_99
        buf380 = reinterpret_tensor(buf350, (6144, 196), (196, 1), 0); del buf350  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf378, (6144, 196), (196, 1), 0), permute_345, out=buf380)
        del permute_345
        buf381 = reinterpret_tensor(buf355, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf355  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf380, primals_167, buf381, 9408, 128, grid=grid(9408), stream=stream0)
        buf382 = buf356; del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf381, buf382, 1568, 6, grid=grid(1568), stream=stream0)
        buf383 = reinterpret_tensor(buf381, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf381  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf380, primals_167, mul_133, buf383, 9408, 128, grid=grid(9408), stream=stream0)
        buf384 = buf354; del buf354  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_11.run(buf383, buf384, 1568, 6, grid=grid(1568), stream=stream0)
        buf385 = buf357; del buf357  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf380, mul_133, buf385, 9984, 121, grid=grid(9984), stream=stream0)
        buf386 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf385, buf386, 768, 13, grid=grid(768), stream=stream0)
        buf387 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_14.run(buf380, buf387, 768, 1568, grid=grid(768), stream=stream0)
        buf388 = buf361; del buf361  # reuse
        buf389 = buf388; del buf388  # reuse
        # Source Nodes: [x_133], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
        triton_poi_fused_cat_gelu_gelu_backward_15.run(buf389, buf373, mm_16, primals_170, div_28, buf380, primals_167, buf382, mul_133, buf384, addmm_32, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del addmm_32
        del div_28
        del mm_16
        del mul_133
        del primals_167
        del primals_170
        buf390 = buf362; del buf362  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf389, (1568, 1536), (1536, 1), 0), permute_348, out=buf390)
        del permute_348
        buf391 = empty((1536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf389, (1536, 1568), (1, 1536), 0), view_97, out=buf391)
        del view_97
        buf392 = buf364; del buf364  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf389, buf392, 19968, 121, grid=grid(19968), stream=stream0)
        buf393 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf392, buf393, 1536, 13, grid=grid(1536), stream=stream0)
        buf400 = buf372; del buf372  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf400, buf390, primals_163, mul_128, div_29, 1568, 256, grid=grid(1568), stream=stream0)
        del div_29
        del primals_163
        buf396 = reinterpret_tensor(buf375, (256, 13), (1, 256), 0); del buf375  # reuse
        buf398 = buf368; del buf368  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf390, mul_128, buf396, buf398, 3328, 121, grid=grid(3328), stream=stream0)
        del mul_128
        buf397 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf396, buf397, 256, 13, grid=grid(256), stream=stream0)
        buf399 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf398, buf399, 256, 13, grid=grid(256), stream=stream0)
        buf401 = reinterpret_tensor(buf380, (1568, 768), (768, 1), 0); del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf400, (1568, 256), (256, 1), 0), permute_352, out=buf401)
        del permute_352
        buf402 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf400, (256, 1568), (1, 256), 0), view_95, out=buf402)
        del view_95
        buf403 = reinterpret_tensor(buf398, (1, 256, 13), (3328, 1, 256), 0); del buf398  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf400, buf403, 3328, 121, grid=grid(3328), stream=stream0)
        buf404 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf403, buf404, 256, 13, grid=grid(256), stream=stream0)
        buf405 = empty((1, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf401, getitem_92, buf405, 196, 6144, grid=grid(196), stream=stream0)
        buf406 = reinterpret_tensor(buf373, (8, 768, 196), (150528, 196, 1), 0); del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf401, getitem_92, buf406, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del getitem_92
        buf407 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf406, (196, 6144), (1, 196), 0), view_93, out=buf407)
        del view_93
        buf408 = reinterpret_tensor(buf378, (6144, 196), (196, 1), 0); del buf378  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf406, (6144, 196), (196, 1), 0), permute_359, out=buf408)
        del permute_359
        buf409 = reinterpret_tensor(buf383, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf383  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf408, primals_157, buf409, 9408, 128, grid=grid(9408), stream=stream0)
        buf410 = buf384; del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf409, buf410, 1568, 6, grid=grid(1568), stream=stream0)
        buf411 = reinterpret_tensor(buf409, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf409  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf408, primals_157, mul_125, buf411, 9408, 128, grid=grid(9408), stream=stream0)
        buf412 = buf382; del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_11.run(buf411, buf412, 1568, 6, grid=grid(1568), stream=stream0)
        buf413 = buf385; del buf385  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf408, mul_125, buf413, 9984, 121, grid=grid(9984), stream=stream0)
        buf414 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf413, buf414, 768, 13, grid=grid(768), stream=stream0)
        buf415 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_14.run(buf408, buf415, 768, 1568, grid=grid(768), stream=stream0)
        buf416 = buf389; del buf389  # reuse
        buf417 = buf416; del buf416  # reuse
        # Source Nodes: [x_125], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
        triton_poi_fused_cat_gelu_gelu_backward_15.run(buf417, buf401, mm_15, primals_160, div_30, buf408, primals_157, buf410, mul_125, buf412, addmm_30, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del addmm_30
        del div_30
        del mm_15
        del mul_125
        del primals_157
        del primals_160
        buf418 = buf390; del buf390  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf417, (1568, 1536), (1536, 1), 0), permute_362, out=buf418)
        del permute_362
        buf419 = empty((1536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf417, (1536, 1568), (1, 1536), 0), view_91, out=buf419)
        del view_91
        buf420 = buf392; del buf392  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf417, buf420, 19968, 121, grid=grid(19968), stream=stream0)
        buf421 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf420, buf421, 1536, 13, grid=grid(1536), stream=stream0)
        buf428 = buf400; del buf400  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf428, buf418, primals_153, mul_120, div_31, 1568, 256, grid=grid(1568), stream=stream0)
        del div_31
        del primals_153
        buf424 = reinterpret_tensor(buf403, (256, 13), (1, 256), 0); del buf403  # reuse
        buf426 = buf396; del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf418, mul_120, buf424, buf426, 3328, 121, grid=grid(3328), stream=stream0)
        del mul_120
        buf425 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf424, buf425, 256, 13, grid=grid(256), stream=stream0)
        buf427 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf426, buf427, 256, 13, grid=grid(256), stream=stream0)
        buf429 = reinterpret_tensor(buf408, (1568, 768), (768, 1), 0); del buf408  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf428, (1568, 256), (256, 1), 0), permute_366, out=buf429)
        del permute_366
        buf430 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf428, (256, 1568), (1, 256), 0), view_89, out=buf430)
        del view_89
        buf431 = reinterpret_tensor(buf426, (1, 256, 13), (3328, 1, 256), 0); del buf426  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf428, buf431, 3328, 121, grid=grid(3328), stream=stream0)
        buf432 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf431, buf432, 256, 13, grid=grid(256), stream=stream0)
        buf433 = empty((1, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf429, getitem_86, buf433, 196, 6144, grid=grid(196), stream=stream0)
        buf434 = reinterpret_tensor(buf401, (8, 768, 196), (150528, 196, 1), 0); del buf401  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf429, getitem_86, buf434, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del getitem_86
        buf435 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf434, (196, 6144), (1, 196), 0), view_87, out=buf435)
        del view_87
        buf436 = reinterpret_tensor(buf406, (6144, 196), (196, 1), 0); del buf406  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf434, (6144, 196), (196, 1), 0), permute_373, out=buf436)
        del permute_373
        buf437 = reinterpret_tensor(buf411, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf411  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf436, primals_147, buf437, 9408, 128, grid=grid(9408), stream=stream0)
        buf438 = buf412; del buf412  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf437, buf438, 1568, 6, grid=grid(1568), stream=stream0)
        buf439 = reinterpret_tensor(buf437, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf437  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf436, primals_147, mul_117, buf439, 9408, 128, grid=grid(9408), stream=stream0)
        buf440 = buf410; del buf410  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_11.run(buf439, buf440, 1568, 6, grid=grid(1568), stream=stream0)
        buf441 = buf413; del buf413  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf436, mul_117, buf441, 9984, 121, grid=grid(9984), stream=stream0)
        buf442 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf441, buf442, 768, 13, grid=grid(768), stream=stream0)
        buf443 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_14.run(buf436, buf443, 768, 1568, grid=grid(768), stream=stream0)
        buf444 = buf417; del buf417  # reuse
        buf445 = buf444; del buf444  # reuse
        # Source Nodes: [x_117], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
        triton_poi_fused_cat_gelu_gelu_backward_15.run(buf445, buf429, mm_14, primals_150, div_32, buf436, primals_147, buf438, mul_117, buf440, addmm_28, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del addmm_28
        del div_32
        del mm_14
        del mul_117
        del primals_147
        del primals_150
        buf446 = buf418; del buf418  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf445, (1568, 1536), (1536, 1), 0), permute_376, out=buf446)
        del permute_376
        buf447 = empty((1536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf445, (1536, 1568), (1, 1536), 0), view_85, out=buf447)
        del view_85
        buf448 = buf420; del buf420  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf445, buf448, 19968, 121, grid=grid(19968), stream=stream0)
        buf449 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf448, buf449, 1536, 13, grid=grid(1536), stream=stream0)
        buf456 = buf428; del buf428  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf456, buf446, primals_143, mul_112, div_33, 1568, 256, grid=grid(1568), stream=stream0)
        del div_33
        del primals_143
        buf452 = reinterpret_tensor(buf431, (256, 13), (1, 256), 0); del buf431  # reuse
        buf454 = buf424; del buf424  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf446, mul_112, buf452, buf454, 3328, 121, grid=grid(3328), stream=stream0)
        del mul_112
        buf453 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf452, buf453, 256, 13, grid=grid(256), stream=stream0)
        buf455 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf454, buf455, 256, 13, grid=grid(256), stream=stream0)
        buf457 = reinterpret_tensor(buf436, (1568, 768), (768, 1), 0); del buf436  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf456, (1568, 256), (256, 1), 0), permute_380, out=buf457)
        del permute_380
        buf458 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf456, (256, 1568), (1, 256), 0), view_83, out=buf458)
        del view_83
        buf459 = reinterpret_tensor(buf454, (1, 256, 13), (3328, 1, 256), 0); del buf454  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf456, buf459, 3328, 121, grid=grid(3328), stream=stream0)
        buf460 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf459, buf460, 256, 13, grid=grid(256), stream=stream0)
        buf461 = empty((1, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf457, getitem_80, buf461, 196, 6144, grid=grid(196), stream=stream0)
        buf462 = reinterpret_tensor(buf429, (8, 768, 196), (150528, 196, 1), 0); del buf429  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf457, getitem_80, buf462, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del getitem_80
        buf463 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf462, (196, 6144), (1, 196), 0), view_81, out=buf463)
        del view_81
        buf464 = reinterpret_tensor(buf434, (6144, 196), (196, 1), 0); del buf434  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf462, (6144, 196), (196, 1), 0), permute_387, out=buf464)
        del permute_387
        buf465 = reinterpret_tensor(buf439, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf439  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf464, primals_137, buf465, 9408, 128, grid=grid(9408), stream=stream0)
        buf466 = buf440; del buf440  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf465, buf466, 1568, 6, grid=grid(1568), stream=stream0)
        buf467 = reinterpret_tensor(buf465, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf465  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf464, primals_137, mul_109, buf467, 9408, 128, grid=grid(9408), stream=stream0)
        buf468 = buf438; del buf438  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_11.run(buf467, buf468, 1568, 6, grid=grid(1568), stream=stream0)
        buf469 = buf441; del buf441  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf464, mul_109, buf469, 9984, 121, grid=grid(9984), stream=stream0)
        buf470 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf469, buf470, 768, 13, grid=grid(768), stream=stream0)
        buf471 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_14.run(buf464, buf471, 768, 1568, grid=grid(768), stream=stream0)
        buf472 = buf445; del buf445  # reuse
        buf473 = buf472; del buf472  # reuse
        # Source Nodes: [x_109], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
        triton_poi_fused_cat_gelu_gelu_backward_15.run(buf473, buf457, mm_13, primals_140, div_34, buf464, primals_137, buf466, mul_109, buf468, addmm_26, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del addmm_26
        del div_34
        del mm_13
        del mul_109
        del primals_137
        del primals_140
        buf474 = buf446; del buf446  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf473, (1568, 1536), (1536, 1), 0), permute_390, out=buf474)
        del permute_390
        buf475 = empty((1536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf473, (1536, 1568), (1, 1536), 0), view_79, out=buf475)
        del view_79
        buf476 = buf448; del buf448  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf473, buf476, 19968, 121, grid=grid(19968), stream=stream0)
        buf477 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf476, buf477, 1536, 13, grid=grid(1536), stream=stream0)
        buf484 = buf456; del buf456  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf484, buf474, primals_133, mul_104, div_35, 1568, 256, grid=grid(1568), stream=stream0)
        del div_35
        del primals_133
        buf480 = reinterpret_tensor(buf459, (256, 13), (1, 256), 0); del buf459  # reuse
        buf482 = buf452; del buf452  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf474, mul_104, buf480, buf482, 3328, 121, grid=grid(3328), stream=stream0)
        del mul_104
        buf481 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf480, buf481, 256, 13, grid=grid(256), stream=stream0)
        buf483 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf482, buf483, 256, 13, grid=grid(256), stream=stream0)
        buf485 = reinterpret_tensor(buf464, (1568, 768), (768, 1), 0); del buf464  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf484, (1568, 256), (256, 1), 0), permute_394, out=buf485)
        del permute_394
        buf486 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf484, (256, 1568), (1, 256), 0), view_77, out=buf486)
        del view_77
        buf487 = reinterpret_tensor(buf482, (1, 256, 13), (3328, 1, 256), 0); del buf482  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf484, buf487, 3328, 121, grid=grid(3328), stream=stream0)
        buf488 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf487, buf488, 256, 13, grid=grid(256), stream=stream0)
        buf489 = empty((1, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf485, getitem_74, buf489, 196, 6144, grid=grid(196), stream=stream0)
        buf490 = reinterpret_tensor(buf457, (8, 768, 196), (150528, 196, 1), 0); del buf457  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf485, getitem_74, buf490, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del getitem_74
        buf491 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf490, (196, 6144), (1, 196), 0), view_75, out=buf491)
        del view_75
        buf492 = reinterpret_tensor(buf462, (6144, 196), (196, 1), 0); del buf462  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf490, (6144, 196), (196, 1), 0), permute_401, out=buf492)
        del permute_401
        buf493 = reinterpret_tensor(buf467, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf467  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf492, primals_127, buf493, 9408, 128, grid=grid(9408), stream=stream0)
        buf494 = buf468; del buf468  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf493, buf494, 1568, 6, grid=grid(1568), stream=stream0)
        buf495 = reinterpret_tensor(buf493, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf493  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf492, primals_127, mul_101, buf495, 9408, 128, grid=grid(9408), stream=stream0)
        buf496 = buf466; del buf466  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_11.run(buf495, buf496, 1568, 6, grid=grid(1568), stream=stream0)
        buf497 = buf469; del buf469  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf492, mul_101, buf497, 9984, 121, grid=grid(9984), stream=stream0)
        buf498 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf497, buf498, 768, 13, grid=grid(768), stream=stream0)
        buf499 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_14.run(buf492, buf499, 768, 1568, grid=grid(768), stream=stream0)
        buf500 = buf473; del buf473  # reuse
        buf501 = buf500; del buf500  # reuse
        # Source Nodes: [x_101], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
        triton_poi_fused_cat_gelu_gelu_backward_15.run(buf501, buf485, mm_12, primals_130, div_36, buf492, primals_127, buf494, mul_101, buf496, addmm_24, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del addmm_24
        del div_36
        del mm_12
        del mul_101
        del primals_127
        del primals_130
        buf502 = buf474; del buf474  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf501, (1568, 1536), (1536, 1), 0), permute_404, out=buf502)
        del permute_404
        buf503 = empty((1536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf501, (1536, 1568), (1, 1536), 0), view_73, out=buf503)
        del view_73
        buf504 = buf476; del buf476  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf501, buf504, 19968, 121, grid=grid(19968), stream=stream0)
        buf505 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf504, buf505, 1536, 13, grid=grid(1536), stream=stream0)
        buf512 = buf484; del buf484  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf512, buf502, primals_123, mul_96, div_37, 1568, 256, grid=grid(1568), stream=stream0)
        del div_37
        del primals_123
        buf508 = reinterpret_tensor(buf487, (256, 13), (1, 256), 0); del buf487  # reuse
        buf510 = buf480; del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf502, mul_96, buf508, buf510, 3328, 121, grid=grid(3328), stream=stream0)
        del mul_96
        buf509 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf508, buf509, 256, 13, grid=grid(256), stream=stream0)
        buf511 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf510, buf511, 256, 13, grid=grid(256), stream=stream0)
        buf513 = reinterpret_tensor(buf492, (1568, 768), (768, 1), 0); del buf492  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf512, (1568, 256), (256, 1), 0), permute_408, out=buf513)
        del permute_408
        buf514 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf512, (256, 1568), (1, 256), 0), view_71, out=buf514)
        del view_71
        buf515 = reinterpret_tensor(buf510, (1, 256, 13), (3328, 1, 256), 0); del buf510  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf512, buf515, 3328, 121, grid=grid(3328), stream=stream0)
        buf516 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf515, buf516, 256, 13, grid=grid(256), stream=stream0)
        buf517 = empty((1, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf513, getitem_68, buf517, 196, 6144, grid=grid(196), stream=stream0)
        buf518 = reinterpret_tensor(buf485, (8, 768, 196), (150528, 196, 1), 0); del buf485  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf513, getitem_68, buf518, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del getitem_68
        buf519 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf518, (196, 6144), (1, 196), 0), view_69, out=buf519)
        del view_69
        buf520 = reinterpret_tensor(buf490, (6144, 196), (196, 1), 0); del buf490  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf518, (6144, 196), (196, 1), 0), permute_415, out=buf520)
        del permute_415
        buf521 = reinterpret_tensor(buf495, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf495  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf520, primals_117, buf521, 9408, 128, grid=grid(9408), stream=stream0)
        buf522 = buf496; del buf496  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf521, buf522, 1568, 6, grid=grid(1568), stream=stream0)
        buf523 = reinterpret_tensor(buf521, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf521  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf520, primals_117, mul_93, buf523, 9408, 128, grid=grid(9408), stream=stream0)
        buf524 = buf494; del buf494  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_11.run(buf523, buf524, 1568, 6, grid=grid(1568), stream=stream0)
        buf525 = buf497; del buf497  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf520, mul_93, buf525, 9984, 121, grid=grid(9984), stream=stream0)
        buf526 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf525, buf526, 768, 13, grid=grid(768), stream=stream0)
        buf527 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_14.run(buf520, buf527, 768, 1568, grid=grid(768), stream=stream0)
        buf528 = buf501; del buf501  # reuse
        buf529 = buf528; del buf528  # reuse
        # Source Nodes: [x_93], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
        triton_poi_fused_cat_gelu_gelu_backward_15.run(buf529, buf513, mm_11, primals_120, div_38, buf520, primals_117, buf522, mul_93, buf524, addmm_22, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del addmm_22
        del div_38
        del mm_11
        del mul_93
        del primals_117
        del primals_120
        buf530 = buf502; del buf502  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf529, (1568, 1536), (1536, 1), 0), permute_418, out=buf530)
        del permute_418
        buf531 = empty((1536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf529, (1536, 1568), (1, 1536), 0), view_67, out=buf531)
        del view_67
        buf532 = buf504; del buf504  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf529, buf532, 19968, 121, grid=grid(19968), stream=stream0)
        buf533 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf532, buf533, 1536, 13, grid=grid(1536), stream=stream0)
        buf540 = buf512; del buf512  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf540, buf530, primals_113, mul_88, div_39, 1568, 256, grid=grid(1568), stream=stream0)
        del div_39
        del primals_113
        buf536 = reinterpret_tensor(buf515, (256, 13), (1, 256), 0); del buf515  # reuse
        buf538 = buf508; del buf508  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf530, mul_88, buf536, buf538, 3328, 121, grid=grid(3328), stream=stream0)
        del mul_88
        buf537 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf536, buf537, 256, 13, grid=grid(256), stream=stream0)
        buf539 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf538, buf539, 256, 13, grid=grid(256), stream=stream0)
        buf541 = reinterpret_tensor(buf520, (1568, 768), (768, 1), 0); del buf520  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf540, (1568, 256), (256, 1), 0), permute_422, out=buf541)
        del permute_422
        buf542 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf540, (256, 1568), (1, 256), 0), view_65, out=buf542)
        del view_65
        buf543 = reinterpret_tensor(buf538, (1, 256, 13), (3328, 1, 256), 0); del buf538  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf540, buf543, 3328, 121, grid=grid(3328), stream=stream0)
        buf544 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf543, buf544, 256, 13, grid=grid(256), stream=stream0)
        buf545 = empty((1, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf541, getitem_62, buf545, 196, 6144, grid=grid(196), stream=stream0)
        buf546 = reinterpret_tensor(buf513, (8, 768, 196), (150528, 196, 1), 0); del buf513  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf541, getitem_62, buf546, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del getitem_62
        buf547 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf546, (196, 6144), (1, 196), 0), view_63, out=buf547)
        del view_63
        buf548 = reinterpret_tensor(buf518, (6144, 196), (196, 1), 0); del buf518  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf546, (6144, 196), (196, 1), 0), permute_429, out=buf548)
        del permute_429
        buf549 = reinterpret_tensor(buf523, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf523  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf548, primals_107, buf549, 9408, 128, grid=grid(9408), stream=stream0)
        buf550 = buf524; del buf524  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf549, buf550, 1568, 6, grid=grid(1568), stream=stream0)
        buf551 = reinterpret_tensor(buf549, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf549  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf548, primals_107, mul_85, buf551, 9408, 128, grid=grid(9408), stream=stream0)
        buf552 = buf522; del buf522  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_11.run(buf551, buf552, 1568, 6, grid=grid(1568), stream=stream0)
        buf553 = buf525; del buf525  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf548, mul_85, buf553, 9984, 121, grid=grid(9984), stream=stream0)
        buf554 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf553, buf554, 768, 13, grid=grid(768), stream=stream0)
        buf555 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_14.run(buf548, buf555, 768, 1568, grid=grid(768), stream=stream0)
        buf556 = buf529; del buf529  # reuse
        buf557 = buf556; del buf556  # reuse
        # Source Nodes: [x_85], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
        triton_poi_fused_cat_gelu_gelu_backward_15.run(buf557, buf541, mm_10, primals_110, div_40, buf548, primals_107, buf550, mul_85, buf552, addmm_20, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del addmm_20
        del div_40
        del mm_10
        del mul_85
        del primals_107
        del primals_110
        buf558 = buf530; del buf530  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf557, (1568, 1536), (1536, 1), 0), permute_432, out=buf558)
        del permute_432
        buf559 = empty((1536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf557, (1536, 1568), (1, 1536), 0), view_61, out=buf559)
        del view_61
        buf560 = buf532; del buf532  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf557, buf560, 19968, 121, grid=grid(19968), stream=stream0)
        buf561 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf560, buf561, 1536, 13, grid=grid(1536), stream=stream0)
        buf568 = buf540; del buf540  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf568, buf558, primals_103, mul_80, div_41, 1568, 256, grid=grid(1568), stream=stream0)
        del div_41
        del primals_103
        buf564 = reinterpret_tensor(buf543, (256, 13), (1, 256), 0); del buf543  # reuse
        buf566 = buf536; del buf536  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf558, mul_80, buf564, buf566, 3328, 121, grid=grid(3328), stream=stream0)
        del mul_80
        buf565 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf564, buf565, 256, 13, grid=grid(256), stream=stream0)
        buf567 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf566, buf567, 256, 13, grid=grid(256), stream=stream0)
        buf569 = reinterpret_tensor(buf548, (1568, 768), (768, 1), 0); del buf548  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf568, (1568, 256), (256, 1), 0), permute_436, out=buf569)
        del permute_436
        buf570 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf568, (256, 1568), (1, 256), 0), view_59, out=buf570)
        del view_59
        buf571 = reinterpret_tensor(buf566, (1, 256, 13), (3328, 1, 256), 0); del buf566  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf568, buf571, 3328, 121, grid=grid(3328), stream=stream0)
        buf572 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf571, buf572, 256, 13, grid=grid(256), stream=stream0)
        buf573 = empty((1, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf569, getitem_56, buf573, 196, 6144, grid=grid(196), stream=stream0)
        buf574 = reinterpret_tensor(buf541, (8, 768, 196), (150528, 196, 1), 0); del buf541  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf569, getitem_56, buf574, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del getitem_56
        buf575 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf574, (196, 6144), (1, 196), 0), view_57, out=buf575)
        del view_57
        buf576 = reinterpret_tensor(buf546, (6144, 196), (196, 1), 0); del buf546  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf574, (6144, 196), (196, 1), 0), permute_443, out=buf576)
        del permute_443
        buf577 = reinterpret_tensor(buf551, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf551  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf576, primals_97, buf577, 9408, 128, grid=grid(9408), stream=stream0)
        buf578 = buf552; del buf552  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf577, buf578, 1568, 6, grid=grid(1568), stream=stream0)
        buf579 = reinterpret_tensor(buf577, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf577  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf576, primals_97, mul_77, buf579, 9408, 128, grid=grid(9408), stream=stream0)
        buf580 = buf550; del buf550  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_11.run(buf579, buf580, 1568, 6, grid=grid(1568), stream=stream0)
        buf581 = buf553; del buf553  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf576, mul_77, buf581, 9984, 121, grid=grid(9984), stream=stream0)
        buf582 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf581, buf582, 768, 13, grid=grid(768), stream=stream0)
        buf583 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_14.run(buf576, buf583, 768, 1568, grid=grid(768), stream=stream0)
        buf584 = buf557; del buf557  # reuse
        buf585 = buf584; del buf584  # reuse
        # Source Nodes: [x_77], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
        triton_poi_fused_cat_gelu_gelu_backward_15.run(buf585, buf569, mm_9, primals_100, div_42, buf576, primals_97, buf578, mul_77, buf580, addmm_18, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del addmm_18
        del div_42
        del mm_9
        del mul_77
        del primals_100
        del primals_97
        buf586 = buf558; del buf558  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf585, (1568, 1536), (1536, 1), 0), permute_446, out=buf586)
        del permute_446
        buf587 = empty((1536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf585, (1536, 1568), (1, 1536), 0), view_55, out=buf587)
        del view_55
        buf588 = buf560; del buf560  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf585, buf588, 19968, 121, grid=grid(19968), stream=stream0)
        buf589 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf588, buf589, 1536, 13, grid=grid(1536), stream=stream0)
        buf596 = buf568; del buf568  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf596, buf586, primals_93, mul_72, div_43, 1568, 256, grid=grid(1568), stream=stream0)
        del div_43
        del primals_93
        buf592 = reinterpret_tensor(buf571, (256, 13), (1, 256), 0); del buf571  # reuse
        buf594 = buf564; del buf564  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf586, mul_72, buf592, buf594, 3328, 121, grid=grid(3328), stream=stream0)
        del mul_72
        buf593 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf592, buf593, 256, 13, grid=grid(256), stream=stream0)
        buf595 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf594, buf595, 256, 13, grid=grid(256), stream=stream0)
        buf597 = reinterpret_tensor(buf576, (1568, 768), (768, 1), 0); del buf576  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf596, (1568, 256), (256, 1), 0), permute_450, out=buf597)
        del permute_450
        buf598 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf596, (256, 1568), (1, 256), 0), view_53, out=buf598)
        del view_53
        buf599 = reinterpret_tensor(buf594, (1, 256, 13), (3328, 1, 256), 0); del buf594  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf596, buf599, 3328, 121, grid=grid(3328), stream=stream0)
        buf600 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf599, buf600, 256, 13, grid=grid(256), stream=stream0)
        buf601 = empty((1, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf597, getitem_50, buf601, 196, 6144, grid=grid(196), stream=stream0)
        buf602 = reinterpret_tensor(buf569, (8, 768, 196), (150528, 196, 1), 0); del buf569  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf597, getitem_50, buf602, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del getitem_50
        buf603 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf602, (196, 6144), (1, 196), 0), view_51, out=buf603)
        del view_51
        buf604 = reinterpret_tensor(buf574, (6144, 196), (196, 1), 0); del buf574  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf602, (6144, 196), (196, 1), 0), permute_457, out=buf604)
        del permute_457
        buf605 = reinterpret_tensor(buf579, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf579  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf604, primals_87, buf605, 9408, 128, grid=grid(9408), stream=stream0)
        buf606 = buf580; del buf580  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf605, buf606, 1568, 6, grid=grid(1568), stream=stream0)
        buf607 = reinterpret_tensor(buf605, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf605  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf604, primals_87, mul_69, buf607, 9408, 128, grid=grid(9408), stream=stream0)
        buf608 = buf578; del buf578  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_11.run(buf607, buf608, 1568, 6, grid=grid(1568), stream=stream0)
        buf609 = buf581; del buf581  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf604, mul_69, buf609, 9984, 121, grid=grid(9984), stream=stream0)
        buf610 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf609, buf610, 768, 13, grid=grid(768), stream=stream0)
        buf611 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_14.run(buf604, buf611, 768, 1568, grid=grid(768), stream=stream0)
        buf612 = buf585; del buf585  # reuse
        buf613 = buf612; del buf612  # reuse
        # Source Nodes: [x_69], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
        triton_poi_fused_cat_gelu_gelu_backward_15.run(buf613, buf597, mm_8, primals_90, div_44, buf604, primals_87, buf606, mul_69, buf608, addmm_16, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del addmm_16
        del div_44
        del mm_8
        del mul_69
        del primals_87
        del primals_90
        buf614 = buf586; del buf586  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf613, (1568, 1536), (1536, 1), 0), permute_460, out=buf614)
        del permute_460
        buf615 = empty((1536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf613, (1536, 1568), (1, 1536), 0), view_49, out=buf615)
        del view_49
        buf616 = buf588; del buf588  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf613, buf616, 19968, 121, grid=grid(19968), stream=stream0)
        buf617 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf616, buf617, 1536, 13, grid=grid(1536), stream=stream0)
        buf624 = buf596; del buf596  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf624, buf614, primals_83, mul_64, div_45, 1568, 256, grid=grid(1568), stream=stream0)
        del div_45
        del primals_83
        buf620 = reinterpret_tensor(buf599, (256, 13), (1, 256), 0); del buf599  # reuse
        buf622 = buf592; del buf592  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf614, mul_64, buf620, buf622, 3328, 121, grid=grid(3328), stream=stream0)
        del mul_64
        buf621 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf620, buf621, 256, 13, grid=grid(256), stream=stream0)
        buf623 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf622, buf623, 256, 13, grid=grid(256), stream=stream0)
        buf625 = reinterpret_tensor(buf604, (1568, 768), (768, 1), 0); del buf604  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf624, (1568, 256), (256, 1), 0), permute_464, out=buf625)
        del permute_464
        buf626 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf624, (256, 1568), (1, 256), 0), view_47, out=buf626)
        del view_47
        buf627 = reinterpret_tensor(buf622, (1, 256, 13), (3328, 1, 256), 0); del buf622  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf624, buf627, 3328, 121, grid=grid(3328), stream=stream0)
        buf628 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf627, buf628, 256, 13, grid=grid(256), stream=stream0)
        buf629 = empty((1, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf625, getitem_44, buf629, 196, 6144, grid=grid(196), stream=stream0)
        buf630 = reinterpret_tensor(buf597, (8, 768, 196), (150528, 196, 1), 0); del buf597  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf625, getitem_44, buf630, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del getitem_44
        buf631 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf630, (196, 6144), (1, 196), 0), view_45, out=buf631)
        del view_45
        buf632 = reinterpret_tensor(buf602, (6144, 196), (196, 1), 0); del buf602  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf630, (6144, 196), (196, 1), 0), permute_471, out=buf632)
        del permute_471
        buf633 = reinterpret_tensor(buf607, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf607  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf632, primals_77, buf633, 9408, 128, grid=grid(9408), stream=stream0)
        buf634 = buf608; del buf608  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf633, buf634, 1568, 6, grid=grid(1568), stream=stream0)
        buf635 = reinterpret_tensor(buf633, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf633  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf632, primals_77, mul_61, buf635, 9408, 128, grid=grid(9408), stream=stream0)
        buf636 = buf606; del buf606  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_11.run(buf635, buf636, 1568, 6, grid=grid(1568), stream=stream0)
        buf637 = buf609; del buf609  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf632, mul_61, buf637, 9984, 121, grid=grid(9984), stream=stream0)
        buf638 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf637, buf638, 768, 13, grid=grid(768), stream=stream0)
        buf639 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_14.run(buf632, buf639, 768, 1568, grid=grid(768), stream=stream0)
        buf640 = buf613; del buf613  # reuse
        buf641 = buf640; del buf640  # reuse
        # Source Nodes: [x_61], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
        triton_poi_fused_cat_gelu_gelu_backward_15.run(buf641, buf625, mm_7, primals_80, div_46, buf632, primals_77, buf634, mul_61, buf636, addmm_14, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del addmm_14
        del div_46
        del mm_7
        del mul_61
        del primals_77
        del primals_80
        buf642 = buf614; del buf614  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf641, (1568, 1536), (1536, 1), 0), permute_474, out=buf642)
        del permute_474
        buf643 = empty((1536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf641, (1536, 1568), (1, 1536), 0), view_43, out=buf643)
        del view_43
        buf644 = buf616; del buf616  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf641, buf644, 19968, 121, grid=grid(19968), stream=stream0)
        buf645 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf644, buf645, 1536, 13, grid=grid(1536), stream=stream0)
        buf652 = buf624; del buf624  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf652, buf642, primals_73, mul_56, div_47, 1568, 256, grid=grid(1568), stream=stream0)
        del div_47
        del primals_73
        buf648 = reinterpret_tensor(buf627, (256, 13), (1, 256), 0); del buf627  # reuse
        buf650 = buf620; del buf620  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf642, mul_56, buf648, buf650, 3328, 121, grid=grid(3328), stream=stream0)
        del mul_56
        buf649 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf648, buf649, 256, 13, grid=grid(256), stream=stream0)
        buf651 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf650, buf651, 256, 13, grid=grid(256), stream=stream0)
        buf653 = reinterpret_tensor(buf632, (1568, 768), (768, 1), 0); del buf632  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf652, (1568, 256), (256, 1), 0), permute_478, out=buf653)
        del permute_478
        buf654 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf652, (256, 1568), (1, 256), 0), view_41, out=buf654)
        del view_41
        buf655 = reinterpret_tensor(buf650, (1, 256, 13), (3328, 1, 256), 0); del buf650  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf652, buf655, 3328, 121, grid=grid(3328), stream=stream0)
        buf656 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf655, buf656, 256, 13, grid=grid(256), stream=stream0)
        buf657 = empty((1, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf653, getitem_38, buf657, 196, 6144, grid=grid(196), stream=stream0)
        buf658 = reinterpret_tensor(buf625, (8, 768, 196), (150528, 196, 1), 0); del buf625  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf653, getitem_38, buf658, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del getitem_38
        buf659 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf658, (196, 6144), (1, 196), 0), view_39, out=buf659)
        del view_39
        buf660 = reinterpret_tensor(buf630, (6144, 196), (196, 1), 0); del buf630  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf658, (6144, 196), (196, 1), 0), permute_485, out=buf660)
        del permute_485
        buf661 = reinterpret_tensor(buf635, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf635  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf660, primals_67, buf661, 9408, 128, grid=grid(9408), stream=stream0)
        buf662 = buf636; del buf636  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf661, buf662, 1568, 6, grid=grid(1568), stream=stream0)
        buf663 = reinterpret_tensor(buf661, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf661  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf660, primals_67, mul_53, buf663, 9408, 128, grid=grid(9408), stream=stream0)
        buf664 = buf634; del buf634  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_11.run(buf663, buf664, 1568, 6, grid=grid(1568), stream=stream0)
        buf665 = buf637; del buf637  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf660, mul_53, buf665, 9984, 121, grid=grid(9984), stream=stream0)
        buf666 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf665, buf666, 768, 13, grid=grid(768), stream=stream0)
        buf667 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_14.run(buf660, buf667, 768, 1568, grid=grid(768), stream=stream0)
        buf668 = buf641; del buf641  # reuse
        buf669 = buf668; del buf668  # reuse
        # Source Nodes: [x_53], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
        triton_poi_fused_cat_gelu_gelu_backward_15.run(buf669, buf653, mm_6, primals_70, div_48, buf660, primals_67, buf662, mul_53, buf664, addmm_12, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del addmm_12
        del div_48
        del mm_6
        del mul_53
        del primals_67
        del primals_70
        buf670 = buf642; del buf642  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf669, (1568, 1536), (1536, 1), 0), permute_488, out=buf670)
        del permute_488
        buf671 = empty((1536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf669, (1536, 1568), (1, 1536), 0), view_37, out=buf671)
        del view_37
        buf672 = buf644; del buf644  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf669, buf672, 19968, 121, grid=grid(19968), stream=stream0)
        buf673 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf672, buf673, 1536, 13, grid=grid(1536), stream=stream0)
        buf680 = buf652; del buf652  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf680, buf670, primals_63, mul_48, div_49, 1568, 256, grid=grid(1568), stream=stream0)
        del div_49
        del primals_63
        buf676 = reinterpret_tensor(buf655, (256, 13), (1, 256), 0); del buf655  # reuse
        buf678 = buf648; del buf648  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf670, mul_48, buf676, buf678, 3328, 121, grid=grid(3328), stream=stream0)
        del mul_48
        buf677 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf676, buf677, 256, 13, grid=grid(256), stream=stream0)
        buf679 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf678, buf679, 256, 13, grid=grid(256), stream=stream0)
        buf681 = reinterpret_tensor(buf660, (1568, 768), (768, 1), 0); del buf660  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf680, (1568, 256), (256, 1), 0), permute_492, out=buf681)
        del permute_492
        buf682 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf680, (256, 1568), (1, 256), 0), view_35, out=buf682)
        del view_35
        buf683 = reinterpret_tensor(buf678, (1, 256, 13), (3328, 1, 256), 0); del buf678  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf680, buf683, 3328, 121, grid=grid(3328), stream=stream0)
        buf684 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf683, buf684, 256, 13, grid=grid(256), stream=stream0)
        buf685 = empty((1, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf681, getitem_32, buf685, 196, 6144, grid=grid(196), stream=stream0)
        buf686 = reinterpret_tensor(buf653, (8, 768, 196), (150528, 196, 1), 0); del buf653  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf681, getitem_32, buf686, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del getitem_32
        buf687 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf686, (196, 6144), (1, 196), 0), view_33, out=buf687)
        del view_33
        buf688 = reinterpret_tensor(buf658, (6144, 196), (196, 1), 0); del buf658  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf686, (6144, 196), (196, 1), 0), permute_499, out=buf688)
        del permute_499
        buf689 = reinterpret_tensor(buf663, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf663  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf688, primals_57, buf689, 9408, 128, grid=grid(9408), stream=stream0)
        buf690 = buf664; del buf664  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf689, buf690, 1568, 6, grid=grid(1568), stream=stream0)
        buf691 = reinterpret_tensor(buf689, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf689  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf688, primals_57, mul_45, buf691, 9408, 128, grid=grid(9408), stream=stream0)
        buf692 = buf662; del buf662  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_11.run(buf691, buf692, 1568, 6, grid=grid(1568), stream=stream0)
        buf693 = buf665; del buf665  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf688, mul_45, buf693, 9984, 121, grid=grid(9984), stream=stream0)
        buf694 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf693, buf694, 768, 13, grid=grid(768), stream=stream0)
        buf695 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_14.run(buf688, buf695, 768, 1568, grid=grid(768), stream=stream0)
        buf696 = buf669; del buf669  # reuse
        buf697 = buf696; del buf696  # reuse
        # Source Nodes: [x_45], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
        triton_poi_fused_cat_gelu_gelu_backward_15.run(buf697, buf681, mm_5, primals_60, div_50, buf688, primals_57, buf690, mul_45, buf692, addmm_10, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del addmm_10
        del div_50
        del mm_5
        del mul_45
        del primals_57
        del primals_60
        buf698 = buf670; del buf670  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf697, (1568, 1536), (1536, 1), 0), permute_502, out=buf698)
        del permute_502
        buf699 = empty((1536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf697, (1536, 1568), (1, 1536), 0), view_31, out=buf699)
        del view_31
        buf700 = buf672; del buf672  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf697, buf700, 19968, 121, grid=grid(19968), stream=stream0)
        buf701 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf700, buf701, 1536, 13, grid=grid(1536), stream=stream0)
        buf708 = buf680; del buf680  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf708, buf698, primals_53, mul_40, div_51, 1568, 256, grid=grid(1568), stream=stream0)
        del div_51
        del primals_53
        buf704 = reinterpret_tensor(buf683, (256, 13), (1, 256), 0); del buf683  # reuse
        buf706 = buf676; del buf676  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf698, mul_40, buf704, buf706, 3328, 121, grid=grid(3328), stream=stream0)
        del mul_40
        buf705 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf704, buf705, 256, 13, grid=grid(256), stream=stream0)
        buf707 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf706, buf707, 256, 13, grid=grid(256), stream=stream0)
        buf709 = reinterpret_tensor(buf688, (1568, 768), (768, 1), 0); del buf688  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf708, (1568, 256), (256, 1), 0), permute_506, out=buf709)
        del permute_506
        buf710 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf708, (256, 1568), (1, 256), 0), view_29, out=buf710)
        del view_29
        buf711 = reinterpret_tensor(buf706, (1, 256, 13), (3328, 1, 256), 0); del buf706  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf708, buf711, 3328, 121, grid=grid(3328), stream=stream0)
        buf712 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf711, buf712, 256, 13, grid=grid(256), stream=stream0)
        buf713 = empty((1, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf709, getitem_26, buf713, 196, 6144, grid=grid(196), stream=stream0)
        buf714 = reinterpret_tensor(buf681, (8, 768, 196), (150528, 196, 1), 0); del buf681  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf709, getitem_26, buf714, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del getitem_26
        buf715 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf714, (196, 6144), (1, 196), 0), view_27, out=buf715)
        del view_27
        buf716 = reinterpret_tensor(buf686, (6144, 196), (196, 1), 0); del buf686  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf714, (6144, 196), (196, 1), 0), permute_513, out=buf716)
        del permute_513
        buf717 = reinterpret_tensor(buf691, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf691  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf716, primals_47, buf717, 9408, 128, grid=grid(9408), stream=stream0)
        buf718 = buf692; del buf692  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf717, buf718, 1568, 6, grid=grid(1568), stream=stream0)
        buf719 = reinterpret_tensor(buf717, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf717  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf716, primals_47, mul_37, buf719, 9408, 128, grid=grid(9408), stream=stream0)
        buf720 = buf690; del buf690  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_11.run(buf719, buf720, 1568, 6, grid=grid(1568), stream=stream0)
        buf721 = buf693; del buf693  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf716, mul_37, buf721, 9984, 121, grid=grid(9984), stream=stream0)
        buf722 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf721, buf722, 768, 13, grid=grid(768), stream=stream0)
        buf723 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_14.run(buf716, buf723, 768, 1568, grid=grid(768), stream=stream0)
        buf724 = buf697; del buf697  # reuse
        buf725 = buf724; del buf724  # reuse
        # Source Nodes: [x_37], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
        triton_poi_fused_cat_gelu_gelu_backward_15.run(buf725, buf709, mm_4, primals_50, div_52, buf716, primals_47, buf718, mul_37, buf720, addmm_8, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del addmm_8
        del div_52
        del mm_4
        del mul_37
        del primals_47
        del primals_50
        buf726 = buf698; del buf698  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf725, (1568, 1536), (1536, 1), 0), permute_516, out=buf726)
        del permute_516
        buf727 = empty((1536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf725, (1536, 1568), (1, 1536), 0), view_25, out=buf727)
        del view_25
        buf728 = buf700; del buf700  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf725, buf728, 19968, 121, grid=grid(19968), stream=stream0)
        buf729 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf728, buf729, 1536, 13, grid=grid(1536), stream=stream0)
        buf736 = buf708; del buf708  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf736, buf726, primals_43, mul_32, div_53, 1568, 256, grid=grid(1568), stream=stream0)
        del div_53
        del primals_43
        buf732 = reinterpret_tensor(buf711, (256, 13), (1, 256), 0); del buf711  # reuse
        buf734 = buf704; del buf704  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf726, mul_32, buf732, buf734, 3328, 121, grid=grid(3328), stream=stream0)
        del mul_32
        buf733 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf732, buf733, 256, 13, grid=grid(256), stream=stream0)
        buf735 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf734, buf735, 256, 13, grid=grid(256), stream=stream0)
        buf737 = reinterpret_tensor(buf716, (1568, 768), (768, 1), 0); del buf716  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf736, (1568, 256), (256, 1), 0), permute_520, out=buf737)
        del permute_520
        buf738 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf736, (256, 1568), (1, 256), 0), view_23, out=buf738)
        del view_23
        buf739 = reinterpret_tensor(buf734, (1, 256, 13), (3328, 1, 256), 0); del buf734  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf736, buf739, 3328, 121, grid=grid(3328), stream=stream0)
        buf740 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf739, buf740, 256, 13, grid=grid(256), stream=stream0)
        buf741 = empty((1, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf737, getitem_20, buf741, 196, 6144, grid=grid(196), stream=stream0)
        buf742 = reinterpret_tensor(buf709, (8, 768, 196), (150528, 196, 1), 0); del buf709  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf737, getitem_20, buf742, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del getitem_20
        buf743 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf742, (196, 6144), (1, 196), 0), view_21, out=buf743)
        del view_21
        buf744 = reinterpret_tensor(buf714, (6144, 196), (196, 1), 0); del buf714  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf742, (6144, 196), (196, 1), 0), permute_527, out=buf744)
        del permute_527
        buf745 = reinterpret_tensor(buf719, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf719  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf744, primals_37, buf745, 9408, 128, grid=grid(9408), stream=stream0)
        buf746 = buf720; del buf720  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf745, buf746, 1568, 6, grid=grid(1568), stream=stream0)
        buf747 = reinterpret_tensor(buf745, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf745  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf744, primals_37, mul_29, buf747, 9408, 128, grid=grid(9408), stream=stream0)
        buf748 = buf718; del buf718  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_11.run(buf747, buf748, 1568, 6, grid=grid(1568), stream=stream0)
        buf749 = buf721; del buf721  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf744, mul_29, buf749, 9984, 121, grid=grid(9984), stream=stream0)
        buf750 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf749, buf750, 768, 13, grid=grid(768), stream=stream0)
        buf751 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_14.run(buf744, buf751, 768, 1568, grid=grid(768), stream=stream0)
        buf752 = buf725; del buf725  # reuse
        buf753 = buf752; del buf752  # reuse
        # Source Nodes: [x_29], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
        triton_poi_fused_cat_gelu_gelu_backward_15.run(buf753, buf737, mm_3, primals_40, div_54, buf744, primals_37, buf746, mul_29, buf748, addmm_6, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del addmm_6
        del div_54
        del mm_3
        del mul_29
        del primals_37
        del primals_40
        buf754 = buf726; del buf726  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf753, (1568, 1536), (1536, 1), 0), permute_530, out=buf754)
        del permute_530
        buf755 = empty((1536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf753, (1536, 1568), (1, 1536), 0), view_19, out=buf755)
        del view_19
        buf756 = buf728; del buf728  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf753, buf756, 19968, 121, grid=grid(19968), stream=stream0)
        buf757 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf756, buf757, 1536, 13, grid=grid(1536), stream=stream0)
        buf764 = buf736; del buf736  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf764, buf754, primals_33, mul_24, div_55, 1568, 256, grid=grid(1568), stream=stream0)
        del div_55
        del primals_33
        buf760 = reinterpret_tensor(buf739, (256, 13), (1, 256), 0); del buf739  # reuse
        buf762 = buf732; del buf732  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf754, mul_24, buf760, buf762, 3328, 121, grid=grid(3328), stream=stream0)
        del mul_24
        buf761 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf760, buf761, 256, 13, grid=grid(256), stream=stream0)
        buf763 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf762, buf763, 256, 13, grid=grid(256), stream=stream0)
        buf765 = reinterpret_tensor(buf744, (1568, 768), (768, 1), 0); del buf744  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf764, (1568, 256), (256, 1), 0), permute_534, out=buf765)
        del permute_534
        buf766 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf764, (256, 1568), (1, 256), 0), view_17, out=buf766)
        del view_17
        buf767 = reinterpret_tensor(buf762, (1, 256, 13), (3328, 1, 256), 0); del buf762  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf764, buf767, 3328, 121, grid=grid(3328), stream=stream0)
        buf768 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf767, buf768, 256, 13, grid=grid(256), stream=stream0)
        buf769 = empty((1, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf765, getitem_14, buf769, 196, 6144, grid=grid(196), stream=stream0)
        buf770 = reinterpret_tensor(buf737, (8, 768, 196), (150528, 196, 1), 0); del buf737  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf765, getitem_14, buf770, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del getitem_14
        buf771 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf770, (196, 6144), (1, 196), 0), view_15, out=buf771)
        del view_15
        buf772 = reinterpret_tensor(buf742, (6144, 196), (196, 1), 0); del buf742  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf770, (6144, 196), (196, 1), 0), permute_541, out=buf772)
        del permute_541
        buf773 = reinterpret_tensor(buf747, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf747  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf772, primals_27, buf773, 9408, 128, grid=grid(9408), stream=stream0)
        buf774 = buf748; del buf748  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf773, buf774, 1568, 6, grid=grid(1568), stream=stream0)
        buf775 = reinterpret_tensor(buf773, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf773  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf772, primals_27, mul_21, buf775, 9408, 128, grid=grid(9408), stream=stream0)
        buf776 = buf746; del buf746  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_11.run(buf775, buf776, 1568, 6, grid=grid(1568), stream=stream0)
        buf777 = buf749; del buf749  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf772, mul_21, buf777, 9984, 121, grid=grid(9984), stream=stream0)
        buf778 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf777, buf778, 768, 13, grid=grid(768), stream=stream0)
        buf779 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_14.run(buf772, buf779, 768, 1568, grid=grid(768), stream=stream0)
        buf780 = buf753; del buf753  # reuse
        buf781 = buf780; del buf780  # reuse
        # Source Nodes: [x_21], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
        triton_poi_fused_cat_gelu_gelu_backward_15.run(buf781, buf765, mm_2, primals_30, div_56, buf772, primals_27, buf774, mul_21, buf776, addmm_4, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del addmm_4
        del div_56
        del mm_2
        del mul_21
        del primals_27
        del primals_30
        buf782 = buf754; del buf754  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf781, (1568, 1536), (1536, 1), 0), permute_544, out=buf782)
        del permute_544
        buf783 = empty((1536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf781, (1536, 1568), (1, 1536), 0), view_13, out=buf783)
        del view_13
        buf784 = buf756; del buf756  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf781, buf784, 19968, 121, grid=grid(19968), stream=stream0)
        buf785 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf784, buf785, 1536, 13, grid=grid(1536), stream=stream0)
        buf792 = buf764; del buf764  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf792, buf782, primals_23, mul_16, div_57, 1568, 256, grid=grid(1568), stream=stream0)
        del div_57
        del primals_23
        buf788 = reinterpret_tensor(buf767, (256, 13), (1, 256), 0); del buf767  # reuse
        buf790 = buf760; del buf760  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf782, mul_16, buf788, buf790, 3328, 121, grid=grid(3328), stream=stream0)
        del mul_16
        buf789 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf788, buf789, 256, 13, grid=grid(256), stream=stream0)
        buf791 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf790, buf791, 256, 13, grid=grid(256), stream=stream0)
        buf793 = reinterpret_tensor(buf772, (1568, 768), (768, 1), 0); del buf772  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf792, (1568, 256), (256, 1), 0), permute_548, out=buf793)
        del permute_548
        buf794 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf792, (256, 1568), (1, 256), 0), view_11, out=buf794)
        del view_11
        buf795 = reinterpret_tensor(buf790, (1, 256, 13), (3328, 1, 256), 0); del buf790  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf792, buf795, 3328, 121, grid=grid(3328), stream=stream0)
        buf796 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf795, buf796, 256, 13, grid=grid(256), stream=stream0)
        buf797 = empty((1, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf793, getitem_8, buf797, 196, 6144, grid=grid(196), stream=stream0)
        buf798 = reinterpret_tensor(buf765, (8, 768, 196), (150528, 196, 1), 0); del buf765  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf793, getitem_8, buf798, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del getitem_8
        buf799 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf798, (196, 6144), (1, 196), 0), view_9, out=buf799)
        del view_9
        buf800 = reinterpret_tensor(buf770, (6144, 196), (196, 1), 0); del buf770  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf798, (6144, 196), (196, 1), 0), permute_555, out=buf800)
        del permute_555
        buf801 = reinterpret_tensor(buf775, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf775  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf800, primals_17, buf801, 9408, 128, grid=grid(9408), stream=stream0)
        buf802 = buf776; del buf776  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf801, buf802, 1568, 6, grid=grid(1568), stream=stream0)
        buf803 = reinterpret_tensor(buf801, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf801  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf800, primals_17, mul_13, buf803, 9408, 128, grid=grid(9408), stream=stream0)
        buf804 = buf774; del buf774  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_11.run(buf803, buf804, 1568, 6, grid=grid(1568), stream=stream0)
        buf805 = buf777; del buf777  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf800, mul_13, buf805, 9984, 121, grid=grid(9984), stream=stream0)
        buf806 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf805, buf806, 768, 13, grid=grid(768), stream=stream0)
        buf807 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_14.run(buf800, buf807, 768, 1568, grid=grid(768), stream=stream0)
        buf808 = buf781; del buf781  # reuse
        buf809 = buf808; del buf808  # reuse
        # Source Nodes: [x_13], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
        triton_poi_fused_cat_gelu_gelu_backward_15.run(buf809, buf793, mm_1, primals_20, div_58, buf800, primals_17, buf802, mul_13, buf804, addmm_2, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del addmm_2
        del div_58
        del mm_1
        del mul_13
        del primals_17
        del primals_20
        buf810 = buf782; del buf782  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf809, (1568, 1536), (1536, 1), 0), permute_558, out=buf810)
        del permute_558
        buf811 = empty((1536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf809, (1536, 1568), (1, 1536), 0), view_7, out=buf811)
        del view_7
        buf812 = buf784; del buf784  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf809, buf812, 19968, 121, grid=grid(19968), stream=stream0)
        buf813 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf812, buf813, 1536, 13, grid=grid(1536), stream=stream0)
        buf820 = buf792; del buf792  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf820, buf810, primals_13, mul_8, div_59, 1568, 256, grid=grid(1568), stream=stream0)
        del div_59
        del primals_13
        buf816 = reinterpret_tensor(buf795, (256, 13), (1, 256), 0); del buf795  # reuse
        buf818 = buf788; del buf788  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf810, mul_8, buf816, buf818, 3328, 121, grid=grid(3328), stream=stream0)
        del mul_8
        buf817 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf816, buf817, 256, 13, grid=grid(256), stream=stream0)
        buf819 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf818, buf819, 256, 13, grid=grid(256), stream=stream0)
        buf821 = reinterpret_tensor(buf800, (1568, 768), (768, 1), 0); del buf800  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf820, (1568, 256), (256, 1), 0), permute_562, out=buf821)
        del permute_562
        buf822 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf820, (256, 1568), (1, 256), 0), view_5, out=buf822)
        del view_5
        buf823 = reinterpret_tensor(buf818, (1, 256, 13), (3328, 1, 256), 0); del buf818  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf820, buf823, 3328, 121, grid=grid(3328), stream=stream0)
        buf824 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf823, buf824, 256, 13, grid=grid(256), stream=stream0)
        buf825 = empty((1, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf821, getitem_2, buf825, 196, 6144, grid=grid(196), stream=stream0)
        buf826 = reinterpret_tensor(buf793, (8, 768, 196), (150528, 196, 1), 0); del buf793  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf821, getitem_2, buf826, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del getitem_2
        buf827 = empty((196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf826, (196, 6144), (1, 196), 0), view_3, out=buf827)
        del view_3
        buf828 = reinterpret_tensor(buf798, (6144, 196), (196, 1), 0); del buf798  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf826, (6144, 196), (196, 1), 0), permute_569, out=buf828)
        del buf826
        del permute_569
        buf829 = reinterpret_tensor(buf803, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf803  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_8.run(buf828, primals_7, buf829, 9408, 128, grid=grid(9408), stream=stream0)
        buf830 = buf804; del buf804  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf829, buf830, 1568, 6, grid=grid(1568), stream=stream0)
        buf831 = reinterpret_tensor(buf829, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf829  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf828, primals_7, mul_5, buf831, 9408, 128, grid=grid(9408), stream=stream0)
        buf832 = buf802; del buf802  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_11.run(buf831, buf832, 1568, 6, grid=grid(1568), stream=stream0)
        del buf831
        buf833 = buf805; del buf805  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf828, mul_5, buf833, 9984, 121, grid=grid(9984), stream=stream0)
        buf834 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_13.run(buf833, buf834, 768, 13, grid=grid(768), stream=stream0)
        del buf833
        buf835 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_14.run(buf828, buf835, 768, 1568, grid=grid(768), stream=stream0)
        buf836 = buf809; del buf809  # reuse
        buf837 = buf836; del buf836  # reuse
        # Source Nodes: [x_5], Original ATen: [aten.cat, aten.gelu, aten.gelu_backward]
        triton_poi_fused_cat_gelu_gelu_backward_15.run(buf837, buf821, mm, primals_10, div_60, buf828, primals_7, buf830, mul_5, buf832, addmm, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del addmm
        del buf821
        del buf828
        del buf830
        del buf832
        del div_60
        del mm
        del mul_5
        del primals_10
        del primals_7
        buf838 = buf810; del buf810  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf837, (1568, 1536), (1536, 1), 0), permute_572, out=buf838)
        del permute_572
        buf839 = empty((1536, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf837, (1536, 1568), (1, 1536), 0), view_1, out=buf839)
        del view_1
        buf840 = buf812; del buf812  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf837, buf840, 19968, 121, grid=grid(19968), stream=stream0)
        del buf837
        buf841 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf840, buf841, 1536, 13, grid=grid(1536), stream=stream0)
        del buf840
        buf848 = buf820; del buf820  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_18.run(buf848, buf838, primals_3, mul, div_61, 1568, 256, grid=grid(1568), stream=stream0)
        del div_61
        del primals_3
        buf844 = reinterpret_tensor(buf823, (256, 13), (1, 256), 0); del buf823  # reuse
        buf846 = buf816; del buf816  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf838, mul, buf844, buf846, 3328, 121, grid=grid(3328), stream=stream0)
        del buf838
        del mul
        buf845 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf844, buf845, 256, 13, grid=grid(256), stream=stream0)
        del buf844
        buf847 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf846, buf847, 256, 13, grid=grid(256), stream=stream0)
        buf849 = buf846; del buf846  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_20.run(buf848, buf849, 3328, 121, grid=grid(3328), stream=stream0)
        buf850 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf849, buf850, 256, 13, grid=grid(256), stream=stream0)
        del buf849
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf851 = aten.convolution_backward(reinterpret_tensor(buf848, (8, 256, 14, 14), (50176, 1, 3584, 256), 0), primals_307, primals_1, [256], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf848
        del primals_1
        del primals_307
        buf852 = buf851[1]
        return (buf852, buf850, buf845, buf847, reinterpret_tensor(buf839, (1536, 256), (256, 1), 0), reinterpret_tensor(buf841, (1536, ), (1, ), 0), buf834, buf835, reinterpret_tensor(buf827, (196, 196), (196, 1), 0), reinterpret_tensor(buf825, (196, ), (1, ), 0), reinterpret_tensor(buf822, (256, 768), (768, 1), 0), reinterpret_tensor(buf824, (256, ), (1, ), 0), buf817, buf819, reinterpret_tensor(buf811, (1536, 256), (256, 1), 0), reinterpret_tensor(buf813, (1536, ), (1, ), 0), buf806, buf807, reinterpret_tensor(buf799, (196, 196), (196, 1), 0), reinterpret_tensor(buf797, (196, ), (1, ), 0), reinterpret_tensor(buf794, (256, 768), (768, 1), 0), reinterpret_tensor(buf796, (256, ), (1, ), 0), buf789, buf791, reinterpret_tensor(buf783, (1536, 256), (256, 1), 0), reinterpret_tensor(buf785, (1536, ), (1, ), 0), buf778, buf779, reinterpret_tensor(buf771, (196, 196), (196, 1), 0), reinterpret_tensor(buf769, (196, ), (1, ), 0), reinterpret_tensor(buf766, (256, 768), (768, 1), 0), reinterpret_tensor(buf768, (256, ), (1, ), 0), buf761, buf763, reinterpret_tensor(buf755, (1536, 256), (256, 1), 0), reinterpret_tensor(buf757, (1536, ), (1, ), 0), buf750, buf751, reinterpret_tensor(buf743, (196, 196), (196, 1), 0), reinterpret_tensor(buf741, (196, ), (1, ), 0), reinterpret_tensor(buf738, (256, 768), (768, 1), 0), reinterpret_tensor(buf740, (256, ), (1, ), 0), buf733, buf735, reinterpret_tensor(buf727, (1536, 256), (256, 1), 0), reinterpret_tensor(buf729, (1536, ), (1, ), 0), buf722, buf723, reinterpret_tensor(buf715, (196, 196), (196, 1), 0), reinterpret_tensor(buf713, (196, ), (1, ), 0), reinterpret_tensor(buf710, (256, 768), (768, 1), 0), reinterpret_tensor(buf712, (256, ), (1, ), 0), buf705, buf707, reinterpret_tensor(buf699, (1536, 256), (256, 1), 0), reinterpret_tensor(buf701, (1536, ), (1, ), 0), buf694, buf695, reinterpret_tensor(buf687, (196, 196), (196, 1), 0), reinterpret_tensor(buf685, (196, ), (1, ), 0), reinterpret_tensor(buf682, (256, 768), (768, 1), 0), reinterpret_tensor(buf684, (256, ), (1, ), 0), buf677, buf679, reinterpret_tensor(buf671, (1536, 256), (256, 1), 0), reinterpret_tensor(buf673, (1536, ), (1, ), 0), buf666, buf667, reinterpret_tensor(buf659, (196, 196), (196, 1), 0), reinterpret_tensor(buf657, (196, ), (1, ), 0), reinterpret_tensor(buf654, (256, 768), (768, 1), 0), reinterpret_tensor(buf656, (256, ), (1, ), 0), buf649, buf651, reinterpret_tensor(buf643, (1536, 256), (256, 1), 0), reinterpret_tensor(buf645, (1536, ), (1, ), 0), buf638, buf639, reinterpret_tensor(buf631, (196, 196), (196, 1), 0), reinterpret_tensor(buf629, (196, ), (1, ), 0), reinterpret_tensor(buf626, (256, 768), (768, 1), 0), reinterpret_tensor(buf628, (256, ), (1, ), 0), buf621, buf623, reinterpret_tensor(buf615, (1536, 256), (256, 1), 0), reinterpret_tensor(buf617, (1536, ), (1, ), 0), buf610, buf611, reinterpret_tensor(buf603, (196, 196), (196, 1), 0), reinterpret_tensor(buf601, (196, ), (1, ), 0), reinterpret_tensor(buf598, (256, 768), (768, 1), 0), reinterpret_tensor(buf600, (256, ), (1, ), 0), buf593, buf595, reinterpret_tensor(buf587, (1536, 256), (256, 1), 0), reinterpret_tensor(buf589, (1536, ), (1, ), 0), buf582, buf583, reinterpret_tensor(buf575, (196, 196), (196, 1), 0), reinterpret_tensor(buf573, (196, ), (1, ), 0), reinterpret_tensor(buf570, (256, 768), (768, 1), 0), reinterpret_tensor(buf572, (256, ), (1, ), 0), buf565, buf567, reinterpret_tensor(buf559, (1536, 256), (256, 1), 0), reinterpret_tensor(buf561, (1536, ), (1, ), 0), buf554, buf555, reinterpret_tensor(buf547, (196, 196), (196, 1), 0), reinterpret_tensor(buf545, (196, ), (1, ), 0), reinterpret_tensor(buf542, (256, 768), (768, 1), 0), reinterpret_tensor(buf544, (256, ), (1, ), 0), buf537, buf539, reinterpret_tensor(buf531, (1536, 256), (256, 1), 0), reinterpret_tensor(buf533, (1536, ), (1, ), 0), buf526, buf527, reinterpret_tensor(buf519, (196, 196), (196, 1), 0), reinterpret_tensor(buf517, (196, ), (1, ), 0), reinterpret_tensor(buf514, (256, 768), (768, 1), 0), reinterpret_tensor(buf516, (256, ), (1, ), 0), buf509, buf511, reinterpret_tensor(buf503, (1536, 256), (256, 1), 0), reinterpret_tensor(buf505, (1536, ), (1, ), 0), buf498, buf499, reinterpret_tensor(buf491, (196, 196), (196, 1), 0), reinterpret_tensor(buf489, (196, ), (1, ), 0), reinterpret_tensor(buf486, (256, 768), (768, 1), 0), reinterpret_tensor(buf488, (256, ), (1, ), 0), buf481, buf483, reinterpret_tensor(buf475, (1536, 256), (256, 1), 0), reinterpret_tensor(buf477, (1536, ), (1, ), 0), buf470, buf471, reinterpret_tensor(buf463, (196, 196), (196, 1), 0), reinterpret_tensor(buf461, (196, ), (1, ), 0), reinterpret_tensor(buf458, (256, 768), (768, 1), 0), reinterpret_tensor(buf460, (256, ), (1, ), 0), buf453, buf455, reinterpret_tensor(buf447, (1536, 256), (256, 1), 0), reinterpret_tensor(buf449, (1536, ), (1, ), 0), buf442, buf443, reinterpret_tensor(buf435, (196, 196), (196, 1), 0), reinterpret_tensor(buf433, (196, ), (1, ), 0), reinterpret_tensor(buf430, (256, 768), (768, 1), 0), reinterpret_tensor(buf432, (256, ), (1, ), 0), buf425, buf427, reinterpret_tensor(buf419, (1536, 256), (256, 1), 0), reinterpret_tensor(buf421, (1536, ), (1, ), 0), buf414, buf415, reinterpret_tensor(buf407, (196, 196), (196, 1), 0), reinterpret_tensor(buf405, (196, ), (1, ), 0), reinterpret_tensor(buf402, (256, 768), (768, 1), 0), reinterpret_tensor(buf404, (256, ), (1, ), 0), buf397, buf399, reinterpret_tensor(buf391, (1536, 256), (256, 1), 0), reinterpret_tensor(buf393, (1536, ), (1, ), 0), buf386, buf387, reinterpret_tensor(buf379, (196, 196), (196, 1), 0), reinterpret_tensor(buf377, (196, ), (1, ), 0), reinterpret_tensor(buf374, (256, 768), (768, 1), 0), reinterpret_tensor(buf376, (256, ), (1, ), 0), buf369, buf371, reinterpret_tensor(buf363, (1536, 256), (256, 1), 0), reinterpret_tensor(buf365, (1536, ), (1, ), 0), buf358, buf359, reinterpret_tensor(buf351, (196, 196), (196, 1), 0), reinterpret_tensor(buf349, (196, ), (1, ), 0), reinterpret_tensor(buf346, (256, 768), (768, 1), 0), reinterpret_tensor(buf348, (256, ), (1, ), 0), buf341, buf343, reinterpret_tensor(buf335, (1536, 256), (256, 1), 0), reinterpret_tensor(buf337, (1536, ), (1, ), 0), buf330, buf331, reinterpret_tensor(buf323, (196, 196), (196, 1), 0), reinterpret_tensor(buf321, (196, ), (1, ), 0), reinterpret_tensor(buf318, (256, 768), (768, 1), 0), reinterpret_tensor(buf320, (256, ), (1, ), 0), buf313, buf315, reinterpret_tensor(buf307, (1536, 256), (256, 1), 0), reinterpret_tensor(buf309, (1536, ), (1, ), 0), buf302, buf303, reinterpret_tensor(buf295, (196, 196), (196, 1), 0), reinterpret_tensor(buf293, (196, ), (1, ), 0), reinterpret_tensor(buf290, (256, 768), (768, 1), 0), reinterpret_tensor(buf292, (256, ), (1, ), 0), buf285, buf287, reinterpret_tensor(buf279, (1536, 256), (256, 1), 0), reinterpret_tensor(buf281, (1536, ), (1, ), 0), buf274, buf275, reinterpret_tensor(buf267, (196, 196), (196, 1), 0), reinterpret_tensor(buf265, (196, ), (1, ), 0), reinterpret_tensor(buf262, (256, 768), (768, 1), 0), reinterpret_tensor(buf264, (256, ), (1, ), 0), buf257, buf259, reinterpret_tensor(buf251, (1536, 256), (256, 1), 0), reinterpret_tensor(buf253, (1536, ), (1, ), 0), buf246, buf247, reinterpret_tensor(buf239, (196, 196), (196, 1), 0), reinterpret_tensor(buf237, (196, ), (1, ), 0), reinterpret_tensor(buf234, (256, 768), (768, 1), 0), reinterpret_tensor(buf236, (256, ), (1, ), 0), buf229, buf231, reinterpret_tensor(buf223, (1536, 256), (256, 1), 0), reinterpret_tensor(buf225, (1536, ), (1, ), 0), buf218, buf219, reinterpret_tensor(buf211, (196, 196), (196, 1), 0), reinterpret_tensor(buf209, (196, ), (1, ), 0), reinterpret_tensor(buf206, (256, 768), (768, 1), 0), reinterpret_tensor(buf208, (256, ), (1, ), 0), buf201, buf203, reinterpret_tensor(buf195, (1536, 256), (256, 1), 0), reinterpret_tensor(buf197, (1536, ), (1, ), 0), buf190, buf191, reinterpret_tensor(buf183, (196, 196), (196, 1), 0), reinterpret_tensor(buf181, (196, ), (1, ), 0), reinterpret_tensor(buf178, (256, 768), (768, 1), 0), reinterpret_tensor(buf180, (256, ), (1, ), 0), buf173, buf175, reinterpret_tensor(buf167, (1536, 256), (256, 1), 0), reinterpret_tensor(buf169, (1536, ), (1, ), 0), buf162, buf163, reinterpret_tensor(buf155, (196, 196), (196, 1), 0), reinterpret_tensor(buf153, (196, ), (1, ), 0), reinterpret_tensor(buf150, (256, 768), (768, 1), 0), reinterpret_tensor(buf152, (256, ), (1, ), 0), buf145, buf147, reinterpret_tensor(buf139, (1536, 256), (256, 1), 0), reinterpret_tensor(buf141, (1536, ), (1, ), 0), buf134, buf135, reinterpret_tensor(buf127, (196, 196), (196, 1), 0), reinterpret_tensor(buf125, (196, ), (1, ), 0), reinterpret_tensor(buf122, (256, 768), (768, 1), 0), reinterpret_tensor(buf124, (256, ), (1, ), 0), buf117, buf119, reinterpret_tensor(buf111, (1536, 256), (256, 1), 0), reinterpret_tensor(buf113, (1536, ), (1, ), 0), buf106, buf107, reinterpret_tensor(buf99, (196, 196), (196, 1), 0), reinterpret_tensor(buf97, (196, ), (1, ), 0), reinterpret_tensor(buf94, (256, 768), (768, 1), 0), reinterpret_tensor(buf96, (256, ), (1, ), 0), buf89, buf91, reinterpret_tensor(buf83, (1536, 256), (256, 1), 0), reinterpret_tensor(buf85, (1536, ), (1, ), 0), buf78, buf79, reinterpret_tensor(buf71, (196, 196), (196, 1), 0), reinterpret_tensor(buf69, (196, ), (1, ), 0), reinterpret_tensor(buf66, (256, 768), (768, 1), 0), reinterpret_tensor(buf68, (256, ), (1, ), 0), buf61, buf63, reinterpret_tensor(buf55, (1536, 256), (256, 1), 0), reinterpret_tensor(buf57, (1536, ), (1, ), 0), buf50, buf51, reinterpret_tensor(buf43, (196, 196), (196, 1), 0), reinterpret_tensor(buf41, (196, ), (1, ), 0), reinterpret_tensor(buf38, (256, 768), (768, 1), 0), reinterpret_tensor(buf40, (256, ), (1, ), 0), buf33, buf35, reinterpret_tensor(buf27, (1536, 256), (256, 1), 0), reinterpret_tensor(buf29, (1536, ), (1, ), 0), buf22, buf23, reinterpret_tensor(buf15, (196, 196), (196, 1), 0), reinterpret_tensor(buf13, (196, ), (1, ), 0), reinterpret_tensor(buf10, (256, 768), (768, 1), 0), reinterpret_tensor(buf12, (256, ), (1, ), 0), buf7, buf8, reinterpret_tensor(buf1, (1000, 256), (256, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((256, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((196, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    mul = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_1 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_2 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    mul_5 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_3 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_5 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_8 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_7 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_2 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_8 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    mul_13 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_9 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_1 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_11 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_16 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_13 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_4 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_14 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    mul_21 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_15 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_2 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_17 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_24 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_19 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_6 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_20 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    mul_29 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_21 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_3 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_23 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_32 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_25 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_8 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_26 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    mul_37 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_27 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_4 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_29 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_40 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_31 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_32 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    mul_45 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_33 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_5 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_35 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_48 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_37 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_12 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_38 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    mul_53 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_39 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_6 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_41 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_56 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_43 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_14 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_44 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    mul_61 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_45 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_7 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_47 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_64 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_49 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_16 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_50 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    mul_69 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_51 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_8 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_53 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_72 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_55 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_18 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_56 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    mul_77 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_57 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_9 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_59 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_80 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_61 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_20 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_62 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    mul_85 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_63 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_10 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_65 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_88 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_67 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_68 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    mul_93 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_69 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_11 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_71 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_96 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_73 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_24 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_74 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    mul_101 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_75 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_12 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_77 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_104 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_79 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_26 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_80 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    mul_109 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_81 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_13 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_83 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_112 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_85 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_28 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_86 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    mul_117 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_87 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_14 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_89 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_120 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_91 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_30 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_92 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    mul_125 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_93 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_15 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_95 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_128 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_97 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_32 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_98 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    mul_133 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_99 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_16 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_101 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_136 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_103 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_104 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    mul_141 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_105 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_17 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_107 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_144 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_109 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_36 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_110 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    mul_149 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_111 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_18 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_113 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_152 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_115 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_38 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_116 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    mul_157 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_117 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_19 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_119 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_160 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_121 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_40 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_122 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    mul_165 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_123 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_20 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_125 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_168 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_127 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_42 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_128 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    mul_173 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_129 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_21 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_131 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_176 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_133 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_44 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_134 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    mul_181 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_135 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_22 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_137 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_184 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_139 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_46 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_140 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    mul_189 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_141 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_23 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_143 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_192 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_145 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_48 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_146 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    mul_197 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_147 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_24 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_149 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_200 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_151 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_50 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_152 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    mul_205 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_153 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_25 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_155 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_208 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_157 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_52 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_158 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    mul_213 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_159 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_26 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_161 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_216 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_163 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_54 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_164 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    mul_221 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_165 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_27 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_167 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_224 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_169 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_56 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_170 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    mul_229 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_171 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_28 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_173 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_232 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_175 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_58 = rand_strided((1568, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_176 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    mul_237 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_177 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_29 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    view_179 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_240 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    clone_151 = rand_strided((8, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_152 = rand_strided((1000, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_1 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_156 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_163 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_2 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_166 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_170 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_177 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_4 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_180 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_5 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_184 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_191 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_6 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_194 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_7 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_198 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_205 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_8 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_208 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_9 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_212 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_219 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_10 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_222 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_11 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_226 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_233 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_12 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_236 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_13 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_240 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_247 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_14 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_250 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_15 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_254 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_261 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_16 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_264 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_17 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_268 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_275 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_18 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_278 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_19 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_282 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_289 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_20 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_292 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_21 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_296 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_303 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_22 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_306 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_23 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_310 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_317 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_24 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_320 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_25 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_324 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_331 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_26 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_334 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_27 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_338 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_345 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_28 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_348 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_29 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_352 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_359 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_30 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_362 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_31 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_366 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_373 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_32 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_376 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_33 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_380 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_387 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_34 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_390 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_35 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_394 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_401 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_36 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_404 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_37 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_408 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_415 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_38 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_418 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_39 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_422 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_429 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_40 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_432 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_41 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_436 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_443 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_42 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_446 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_43 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_450 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_457 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_44 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_460 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_45 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_464 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_471 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_46 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_474 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_47 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_478 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_485 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_48 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_488 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_49 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_492 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_499 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_50 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_502 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_51 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_506 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_513 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_52 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_516 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_53 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_520 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_527 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_54 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_530 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_55 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_534 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_541 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_56 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_544 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_57 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_548 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_555 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_58 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_558 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_59 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_562 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_569 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_60 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_572 = rand_strided((1536, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_61 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_7, primals_10, primals_13, primals_17, primals_20, primals_23, primals_27, primals_30, primals_33, primals_37, primals_40, primals_43, primals_47, primals_50, primals_53, primals_57, primals_60, primals_63, primals_67, primals_70, primals_73, primals_77, primals_80, primals_83, primals_87, primals_90, primals_93, primals_97, primals_100, primals_103, primals_107, primals_110, primals_113, primals_117, primals_120, primals_123, primals_127, primals_130, primals_133, primals_137, primals_140, primals_143, primals_147, primals_150, primals_153, primals_157, primals_160, primals_163, primals_167, primals_170, primals_173, primals_177, primals_180, primals_183, primals_187, primals_190, primals_193, primals_197, primals_200, primals_203, primals_207, primals_210, primals_213, primals_217, primals_220, primals_223, primals_227, primals_230, primals_233, primals_237, primals_240, primals_243, primals_247, primals_250, primals_253, primals_257, primals_260, primals_263, primals_267, primals_270, primals_273, primals_277, primals_280, primals_283, primals_287, primals_290, primals_293, primals_297, primals_300, primals_303, primals_307, mul, view_1, addmm, getitem_2, mul_5, view_3, mm, view_5, mul_8, view_7, addmm_2, getitem_8, mul_13, view_9, mm_1, view_11, mul_16, view_13, addmm_4, getitem_14, mul_21, view_15, mm_2, view_17, mul_24, view_19, addmm_6, getitem_20, mul_29, view_21, mm_3, view_23, mul_32, view_25, addmm_8, getitem_26, mul_37, view_27, mm_4, view_29, mul_40, view_31, addmm_10, getitem_32, mul_45, view_33, mm_5, view_35, mul_48, view_37, addmm_12, getitem_38, mul_53, view_39, mm_6, view_41, mul_56, view_43, addmm_14, getitem_44, mul_61, view_45, mm_7, view_47, mul_64, view_49, addmm_16, getitem_50, mul_69, view_51, mm_8, view_53, mul_72, view_55, addmm_18, getitem_56, mul_77, view_57, mm_9, view_59, mul_80, view_61, addmm_20, getitem_62, mul_85, view_63, mm_10, view_65, mul_88, view_67, addmm_22, getitem_68, mul_93, view_69, mm_11, view_71, mul_96, view_73, addmm_24, getitem_74, mul_101, view_75, mm_12, view_77, mul_104, view_79, addmm_26, getitem_80, mul_109, view_81, mm_13, view_83, mul_112, view_85, addmm_28, getitem_86, mul_117, view_87, mm_14, view_89, mul_120, view_91, addmm_30, getitem_92, mul_125, view_93, mm_15, view_95, mul_128, view_97, addmm_32, getitem_98, mul_133, view_99, mm_16, view_101, mul_136, view_103, addmm_34, getitem_104, mul_141, view_105, mm_17, view_107, mul_144, view_109, addmm_36, getitem_110, mul_149, view_111, mm_18, view_113, mul_152, view_115, addmm_38, getitem_116, mul_157, view_117, mm_19, view_119, mul_160, view_121, addmm_40, getitem_122, mul_165, view_123, mm_20, view_125, mul_168, view_127, addmm_42, getitem_128, mul_173, view_129, mm_21, view_131, mul_176, view_133, addmm_44, getitem_134, mul_181, view_135, mm_22, view_137, mul_184, view_139, addmm_46, getitem_140, mul_189, view_141, mm_23, view_143, mul_192, view_145, addmm_48, getitem_146, mul_197, view_147, mm_24, view_149, mul_200, view_151, addmm_50, getitem_152, mul_205, view_153, mm_25, view_155, mul_208, view_157, addmm_52, getitem_158, mul_213, view_159, mm_26, view_161, mul_216, view_163, addmm_54, getitem_164, mul_221, view_165, mm_27, view_167, mul_224, view_169, addmm_56, getitem_170, mul_229, view_171, mm_28, view_173, mul_232, view_175, addmm_58, getitem_176, mul_237, view_177, mm_29, view_179, mul_240, clone_151, permute_152, div_1, permute_156, permute_163, div_2, permute_166, div_3, permute_170, permute_177, div_4, permute_180, div_5, permute_184, permute_191, div_6, permute_194, div_7, permute_198, permute_205, div_8, permute_208, div_9, permute_212, permute_219, div_10, permute_222, div_11, permute_226, permute_233, div_12, permute_236, div_13, permute_240, permute_247, div_14, permute_250, div_15, permute_254, permute_261, div_16, permute_264, div_17, permute_268, permute_275, div_18, permute_278, div_19, permute_282, permute_289, div_20, permute_292, div_21, permute_296, permute_303, div_22, permute_306, div_23, permute_310, permute_317, div_24, permute_320, div_25, permute_324, permute_331, div_26, permute_334, div_27, permute_338, permute_345, div_28, permute_348, div_29, permute_352, permute_359, div_30, permute_362, div_31, permute_366, permute_373, div_32, permute_376, div_33, permute_380, permute_387, div_34, permute_390, div_35, permute_394, permute_401, div_36, permute_404, div_37, permute_408, permute_415, div_38, permute_418, div_39, permute_422, permute_429, div_40, permute_432, div_41, permute_436, permute_443, div_42, permute_446, div_43, permute_450, permute_457, div_44, permute_460, div_45, permute_464, permute_471, div_46, permute_474, div_47, permute_478, permute_485, div_48, permute_488, div_49, permute_492, permute_499, div_50, permute_502, div_51, permute_506, permute_513, div_52, permute_516, div_53, permute_520, permute_527, div_54, permute_530, div_55, permute_534, permute_541, div_56, permute_544, div_57, permute_548, permute_555, div_58, permute_558, div_59, permute_562, permute_569, div_60, permute_572, div_61, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('gmlp_s16_224', benchmark_compiled_module)
