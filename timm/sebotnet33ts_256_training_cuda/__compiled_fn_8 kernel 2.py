
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


# kernel path: /tmp/torchinductor_youkaichao/ir/cirub7af37z2qccdyrc6re33povm2hazk2dhrzcxjx7ubhzjpbt4.py
# Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_div_mul_native_batch_norm_backward_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_mul_native_batch_norm_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5120
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1280
    x1 = (xindex // 1280)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1280*(r2 // 64)) + (2560*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (1280*r2) + (163840*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (1280*r2) + (163840*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 64.0
        tmp2 = tmp0 / tmp1
        tmp4 = tmp2 * tmp3
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


# kernel path: /tmp/torchinductor_youkaichao/gb/cgbmar44qm5t5l66sjy56o6oosdjwx3yh25cyz7ztkbvkmnubozz.py
# Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_div_mul_native_batch_norm_backward_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_mul_native_batch_norm_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1280*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t5/ct5r7x5hvxu3lok2kudb73zphbknx4poexi734lpbmpybl5hvyih.py
# Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_div_mul_native_batch_norm_backward_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_mul_native_batch_norm_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1280*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rh/crhnvsjpgjv47b274xu4owyg2kdpanxuqlbtbyuq2mhjqqeruj4b.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_div_mul_native_batch_norm_backward_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_mul_native_batch_norm_backward_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 1280
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y1 = (yindex // 64)
    y3 = yindex
    y0 = yindex % 64
    tmp0 = tl.load(in_ptr0 + (x2 + (1280*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (1280*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (1280*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 64.0
    tmp2 = tmp0 / tmp1
    tmp4 = tmp2 * tmp3
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
    tl.store(out_ptr0 + (y0 + (64*x2) + (81920*y1)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3y/c3ysy4igbvmcwtuw5rjpp7pa5ff2m4ty6z5akaj5frnmtmxq5b37.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1536
    x1 = (xindex // 1536)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((64*x0) + (98304*(r2 // 64)) + (196608*x1) + (r2 % 64)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (1536*r2) + (196608*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (1536*r2) + (196608*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask, tmp12, _tmp11)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4j/c4jcomycmorddudu2zh7f2x6i7fh4eqm6ajuwda6gq536n5ikwtg.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_6', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ug/cugzobnmovzwkr5r3lbla4a4t5iuxnosw423pxp2bwpozu5otd4b.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_7', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/zd/czd3epwqhcua3gffk7p7lmebtg3gshsq7elvlgsmusckzkad3ejf.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12288
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1536
    y1 = (yindex // 1536)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (1536*x2) + (98304*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (1536*x2) + (98304*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.001953125
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/46/c46jz5d3iar7htuclitcakmfvn4fqqj2dgpnf53yul46ll4qmuja.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((64*x0) + (32768*(r2 // 64)) + (65536*x1) + (r2 % 64)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + ((128*(((8*((r2 // 8) % 8)) + (r2 % 8)) % 64)) + (8192*((((8*((r2 // 8) % 8)) + (64*x0) + (32768*(r2 // 64)) + (65536*x1) + (r2 % 8)) // 8192) % 32)) + ((((8*((r2 // 8) % 8)) + (64*x0) + (r2 % 8)) // 64) % 128)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask, tmp12, _tmp11)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7y/c7yolwsctzudw25zvg76xvm2v6rglxg4rej62pvgm3nt3zah3w7a.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/de/cdei2ueuadzis7gaofxk47mu52r3ohkuivs2in6gbk4esnxeikfr.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m3/cm35ae2t3x2smp47s3ycvxj3cuzhkohmyeow4yqsciyy5rni3pui.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_mul_native_batch_norm_backward_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_native_batch_norm_backward_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (512*x2) + (32768*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + ((128*x2) + (8192*((x2 + (64*y0)) // 8192)) + (32768*y1) + (y0 % 128)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.001953125
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
    tl.store(in_out_ptr0 + (x2 + (64*y3)), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mm/cmmmbw4dedqtqrlshfviti7bsp3er2zpnamfznaxdqjcdud6usnt.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_per_fused__softmax_backward_data_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (64*x0)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qt/cqt3tgbexrp3aqsy7re5onhxshxazhxhoulya6ceyzwouzq3m5ll.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]

triton_per_fused__softmax_backward_data_mul_sum_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_mul_sum_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r4 = rindex
    x5 = xindex
    x6 = (xindex // 8)
    x0 = xindex % 8
    x1 = (xindex // 8) % 8
    x2 = (xindex // 64) % 8
    x3 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (r4 + (8*x5)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r4 + (8*x5)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x6), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (x0 + (8*r4) + (64*x6)), rmask, other=0.0)
    tmp11 = tl.load(in_ptr1 + (x0 + (8*r4) + (64*x6)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp1 * tmp3
    tmp5 = tmp2 - tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp12 = tmp10 * tmp11
    tmp13 = tmp11 * tmp3
    tmp14 = tmp12 - tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 0.08838834764831845
    tmp20 = tmp5 * tmp19
    tl.store(out_ptr2 + (r4 + (8*x5)), tmp20, rmask)
    tl.store(out_ptr0 + (x0 + (8*x2) + (64*x1) + (512*x3)), tmp9, None)
    tl.store(out_ptr1 + (x5), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/56/c56cpixf4tcoqogbngsv5yghohg7wji5cwixay2gsfo47ijgb2py.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 15
    x1 = (xindex // 15)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 16, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0 + (16*(x1 % 8))
    tmp4 = tl.full([1], 135, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = ((x0 + (16*(x1 % 8))) // 15) % 9
    tmp8 = tl.full([1], 8, tl.int64)
    tmp9 = tmp7 < tmp8
    tmp10 = tmp9 & tmp6
    tmp11 = (x0 + (16*(x1 % 8))) % 15
    tmp12 = tl.full([1], 7, tl.int64)
    tmp13 = tmp11 >= tmp12
    tmp14 = tmp13 & tmp10
    tmp15 = tl.load(in_ptr0 + ((-7) + (8*(((x0 + (16*(x1 % 8))) // 15) % 9)) + (64*(x1 // 8)) + ((x0 + (16*(x1 % 8))) % 15)), tmp14, other=0.0)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp14, tmp15, tmp16)
    tmp18 = 0.0
    tmp19 = tl.where(tmp13, tmp17, tmp18)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp10, tmp19, tmp20)
    tmp22 = tl.where(tmp9, tmp21, tmp18)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp6, tmp22, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp2, tmp24, tmp25)
    tl.store(out_ptr0 + (x2), tmp26, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lw/clwjrvwjap2rmjgqqx3cydpor6ydoijncgf7kz5rx42nhe3rtxjo.py
# Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward]

triton_poi_fused_cat_convolution_backward_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_backward_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 1536
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 8
    y1 = (yindex // 8) % 8
    y2 = (yindex // 64)
    y4 = yindex % 64
    tmp0 = x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((128*y1) + (1024*y0) + (8192*(((y0 + (8*y1) + (64*x3) + (32768*y2)) // 8192) % 32)) + (((y0 + (8*y1) + (64*x3)) // 64) % 128)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((128*y0) + (1024*y1) + (8192*(((y0 + (8*y1) + (64*x3) + (32768*y2)) // 8192) % 32)) + (((y0 + (8*y1) + (64*x3)) // 64) % 128)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((128*y0) + (1024*y1) + (8192*(((y0 + (8*y1) + (64*x3) + (32768*y2)) // 8192) % 32)) + (((y0 + (8*y1) + (64*x3)) // 64) % 128)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1, 1], 1024, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tl.load(in_ptr3 + ((-32768) + y4 + (64*x3) + (32768*y2)), tmp15 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp15, tmp16, tmp17)
    tmp19 = tmp0 >= tmp13
    tmp20 = tl.full([1, 1], 1536, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tl.load(in_ptr4 + ((128*y0) + (1024*y1) + (8192*((((-65536) + y0 + (8*y1) + (64*x3) + (32768*y2)) // 8192) % 32)) + (((y0 + (8*y1) + (64*x3)) // 64) % 128)), tmp19 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp19, tmp22, tmp23)
    tmp25 = tl.where(tmp15, tmp18, tmp24)
    tmp26 = tl.where(tmp4, tmp11, tmp25)
    tl.store(out_ptr0 + (y4 + (64*x3) + (98304*y2)), tmp26, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dx/cdxm623xbi6i3hwob5gcgo3sfkyrh6rhvnvslapcbf24b5brmbbg.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((64*x0) + (32768*(r2 // 64)) + (65536*x1) + (r2 % 64)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask, tmp12, _tmp11)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gl/cglsahhw5h6qorc4mzh7nfp4wspwzzjo6joredbxajpoflm3vdn7.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (512*x2) + (32768*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (512*x2) + (32768*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.001953125
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
    tl.store(in_out_ptr0 + (x2 + (64*y3)), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k4/ck4hicq6utooftyvqwqgtlil65rewognkahqtmx257tcozqmpbha.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_mul_native_batch_norm_backward_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_batch_norm_backward_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1536
    x1 = (xindex // 1536)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp18 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((64*x0) + (98304*(r2 // 64)) + (196608*x1) + (r2 % 64)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (1536*r2) + (196608*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + ((64*x0) + (98304*(r2 // 64)) + (196608*x1) + (r2 % 64)), rmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr3 + (x0 + (1536*r2) + (196608*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr4 + (x0 + (1536*r2) + (196608*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr6 + (x0 + (1536*r2) + (196608*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp6 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask, tmp16, _tmp15)
        tmp19 = tmp17 - tmp18
        tmp20 = tmp6 * tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask, tmp23, _tmp22)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, None)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/56/c56ggldzvnz6niydkwxklrzorcpcqbhsfaremfqxeineefddztz6.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_add_mul_native_batch_norm_backward_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: 'i32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(15, 16))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_native_batch_norm_backward_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 1536
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
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (98304*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (1536*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (64*x2) + (98304*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (1536*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (1536*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (x2 + (1536*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr10 + (x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr11 + (x2), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr12 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 * tmp5
    tmp9 = tmp7 - tmp8
    tmp11 = 0.001953125
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp23 = tmp21 - tmp22
    tmp25 = tmp24 * tmp11
    tmp27 = tmp26 * tmp26
    tmp28 = tmp25 * tmp27
    tmp29 = tmp23 * tmp28
    tmp30 = tmp6 - tmp29
    tmp31 = tmp30 - tmp19
    tl.store(out_ptr0 + (x2 + (1536*y3)), tmp20, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (1536*y3)), tmp31, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2g/c2gyqhvpl3rtcttwcmrg6rwefbpszob2vtlk7mxldra6trawmwmd.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12288
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1536
    y1 = (yindex // 1536)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1536*x2) + (98304*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u7/cu7goaixv63xfkz5p2d5qgkfy6mn6b55ecm7uavh3cd2k45lzbvr.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_mul_native_batch_norm_backward_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_native_batch_norm_backward_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 512
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
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (32768*y1)), xmask & ymask)
    tmp1 = tl.load(in_ptr1 + (x2 + (512*y3)), xmask & ymask)
    tmp3 = tl.load(in_ptr2 + (x2 + (512*y3)), xmask & ymask)
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.001953125
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dc/cdc3bdn4fcg5kr5ubmuzlwzknnjhuzlgiihw7dzn7j2xskuv6n7a.py
# Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_avg_pool2d_backward_mul_native_batch_norm_backward_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_backward_mul_native_batch_norm_backward_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 16
    x2 = (xindex // 256) % 512
    x3 = (xindex // 131072)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x2 + (512*(tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(8, 1 + (x0 // 2)))))) + (512*(tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(8, 1 + (x0 // 2))))) >= 0, 0, 8))) + (4096*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(8, 1 + (x1 // 2)))))) + (4096*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(8, 1 + (x1 // 2))))) >= 0, 0, 8))) + (32768*x3)), None, eviction_policy='evict_last')
    tmp1 = tmp0 / 4
    tmp2 = tl.math.max(0, (x1 // 2))
    tmp3 = tl.math.min(8, 1 + (x1 // 2))
    tmp4 = tmp2 < tmp3
    tmp5 = tl.math.max(0, (x0 // 2))
    tmp6 = tl.math.min(8, 1 + (x0 // 2))
    tmp7 = tmp5 < tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp1, tmp9)
    tl.store(out_ptr0 + (x6), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pp/cpp6le32zwzhz76ilo5isy6ty4huadqx5d34252t7czwv4n5pmkq.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_per_fused__softmax_backward_data_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel):
    xnumel = 8192
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
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tl.store(out_ptr0 + (x0), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sv/csvci3rdlov2fw7sxxjknssbyd4b3vqll6fdc56uyeomsowtegsp.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]

triton_per_fused__softmax_backward_data_mul_sum_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[131072, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_mul_sum_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r4 = rindex
    x5 = xindex
    x6 = (xindex // 16)
    x0 = xindex % 16
    x1 = (xindex // 16) % 16
    x2 = (xindex // 256) % 16
    x3 = (xindex // 4096)
    tmp0 = tl.load(in_ptr0 + (r4 + (16*x5)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r4 + (16*x5)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x6), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (x0 + (16*r4) + (256*x6)), rmask, other=0.0)
    tmp11 = tl.load(in_ptr1 + (x0 + (16*r4) + (256*x6)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp1 * tmp3
    tmp5 = tmp2 - tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp12 = tmp10 * tmp11
    tmp13 = tmp11 * tmp3
    tmp14 = tmp12 - tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 0.08838834764831845
    tmp20 = tmp5 * tmp19
    tl.store(out_ptr2 + (r4 + (16*x5)), tmp20, rmask)
    tl.store(out_ptr0 + (x0 + (16*x2) + (256*x1) + (4096*x3)), tmp9, None)
    tl.store(out_ptr1 + (x5), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nm/cnmuovu5ui5kfmojikhkvkvn5cj7gra7fpnl2yfyzimp52q7c3y4.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 253952
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 31
    x1 = (xindex // 31)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 32, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0 + (32*(x1 % 16))
    tmp4 = tl.full([1], 527, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = ((x0 + (32*(x1 % 16))) // 31) % 17
    tmp8 = tl.full([1], 16, tl.int64)
    tmp9 = tmp7 < tmp8
    tmp10 = tmp9 & tmp6
    tmp11 = (x0 + (32*(x1 % 16))) % 31
    tmp12 = tl.full([1], 15, tl.int64)
    tmp13 = tmp11 >= tmp12
    tmp14 = tmp13 & tmp10
    tmp15 = tl.load(in_ptr0 + ((-15) + (16*(((x0 + (32*(x1 % 16))) // 31) % 17)) + (256*(x1 // 16)) + ((x0 + (32*(x1 % 16))) % 31)), tmp14, other=0.0)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp14, tmp15, tmp16)
    tmp18 = 0.0
    tmp19 = tl.where(tmp13, tmp17, tmp18)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp10, tmp19, tmp20)
    tmp22 = tl.where(tmp9, tmp21, tmp18)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp6, tmp22, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp2, tmp24, tmp25)
    tl.store(out_ptr0 + (x2), tmp26, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/n6/cn6ykim6asbmeltm7wgo3bny2nloeubt5tyxcgzabpox7ihper35.py
# Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward]

triton_poi_fused_cat_convolution_backward_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_backward_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 1536
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16) % 16
    y2 = (yindex // 256)
    y4 = yindex % 256
    tmp0 = x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((128*y1) + (2048*y0) + (32768*(((y0 + (16*y1) + (256*x3) + (131072*y2)) // 32768) % 32)) + (((y0 + (16*y1) + (256*x3)) // 256) % 128)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((128*y0) + (2048*y1) + (32768*(((y0 + (16*y1) + (256*x3) + (131072*y2)) // 32768) % 32)) + (((y0 + (16*y1) + (256*x3)) // 256) % 128)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((128*y0) + (2048*y1) + (32768*(((y0 + (16*y1) + (256*x3) + (131072*y2)) // 32768) % 32)) + (((y0 + (16*y1) + (256*x3)) // 256) % 128)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1, 1], 1024, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tl.load(in_ptr3 + ((-131072) + y4 + (256*x3) + (131072*y2)), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp15, tmp16, tmp17)
    tmp19 = tmp0 >= tmp13
    tmp20 = tl.full([1, 1], 1536, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tl.load(in_ptr4 + ((128*y0) + (2048*y1) + (32768*((((-262144) + y0 + (16*y1) + (256*x3) + (131072*y2)) // 32768) % 32)) + (((y0 + (16*y1) + (256*x3)) // 256) % 128)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp19, tmp22, tmp23)
    tmp25 = tl.where(tmp15, tmp18, tmp24)
    tmp26 = tl.where(tmp4, tmp11, tmp25)
    tl.store(out_ptr0 + (y4 + (256*x3) + (393216*y2)), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rc/crcbirwj2hilcecypugdn2q3vgvdjmkhir4ckdna33ja4y6ee52y.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((16*(((r2 + (128*x0)) // 16) % 16)) + (256*x1) + (131072*((r2 + (128*x0)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (512*r2) + (65536*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/se/cserxbl25vd22mk3zq7rlgmtpzuqtlqagqimu6r3qchsjsyd5qhh.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (16*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yj/cyjv3rh346xat4i5i3yrel2gqdjuvzfw3yfmmghfixtylogv2mgq.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((16*(((r2 + (128*x1)) // 16) % 16)) + (256*x0) + (131072*((r2 + (128*x1)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 - tmp4
        tmp6 = tmp2 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6d/c6dwg7d77e6dfi3e673cd4p33ki5yrk37zuqaxsjxy23xmoyz7oq.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vu/cvuzw7pvb326i6egwvd6zuhgaeeq6rkcejwnoj3w7shxh3byt75l.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (512*x2) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (512*x2) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.00048828125
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
    tl.store(in_out_ptr0 + (x2 + (256*y3)), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4i/c4ixu2rhsbucjd2xzuayfpldi7ugtfeahx342xrnvnrdhiqqmhxv.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_mul_native_batch_norm_backward_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_batch_norm_backward_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (1024*r3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/at/catu2vimlvbrmy3icrocev7iga2st6ez5grcdberv5cn4fslv6s2.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_mul_native_batch_norm_backward_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_batch_norm_backward_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    tmp6 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((16*(((r2 + (128*x0)) // 16) % 16)) + (256*x1) + (262144*((r2 + (128*x0)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((16*(((r2 + (128*x0)) // 16) % 16)) + (256*x1) + (262144*((r2 + (128*x0)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x1 + (1024*r2) + (131072*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr3 + (x1 + (1024*r2) + (131072*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sf/csfekbuvqghymskzce2p2d2vaqkuskhd7e5au2hgn2fqrgr2i6ey.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_add_mul_native_batch_norm_backward_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_batch_norm_backward_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (16*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ad/cadregayzz42qmrtanda4qxhhqyc66hjkyowf7gjlph5ru3nmwff.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_add_mul_native_batch_norm_backward_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_native_batch_norm_backward_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (1024*x2) + (262144*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0 + (1024*x2) + (262144*y1)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00048828125
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gq/cgq73tla2nklaasd2nzptkwt5bxpsqatd2pcp4t6qapdbp7h5gnf.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((16*(((r2 + (128*x0)) // 16) % 16)) + (256*x1) + (65536*((r2 + (128*x0)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (256*r2) + (32768*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wr/cwrxg57qtaamgptwalapioi4zmgef5kjnes26dlfxrbwcyvnqwhg.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_38 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (16*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o7/co7yloxixslsmebwkz2pmodc64qn2xz3df74iuckvzxjmcprgbck.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((16*(((r2 + (128*x1)) // 16) % 16)) + (256*x0) + (65536*((r2 + (128*x1)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + ((64*(((16*(((r2 + (128*x1)) // 16) % 16)) + (r2 % 16)) % 256)) + (16384*((((16*(((r2 + (128*x1)) // 16) % 16)) + (256*x0) + (65536*((r2 + (128*x1)) // 256)) + (r2 % 16)) // 16384) % 32)) + ((((16*(((r2 + (128*x1)) // 16) % 16)) + (256*x0) + (r2 % 16)) // 256) % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 - tmp4
        tmp6 = tmp2 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/i4/ci4ptdhe746a3anlyyxs5hpuyf6dk4cikwbltj2er74s5p6rw4if.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wi/cwi6s4pgrhhqdh7gu67t5d4gmqigc5o7lecgfigdtxidibl4c4cy.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_mul_native_batch_norm_backward_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_native_batch_norm_backward_41', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (256*x2) + (65536*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + ((64*x2) + (16384*((x2 + (256*y0)) // 16384)) + (65536*y1) + (y0 % 64)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.00048828125
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
    tl.store(in_out_ptr0 + (x2 + (256*y3)), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ug/cugn2f2ce2mzbhxqoxyoewzqy6q5ptz46cyso7xlfb7bpg3a6v4g.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]

triton_per_fused__softmax_backward_data_mul_sum_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[131072, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_mul_sum_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r4 = rindex
    x5 = xindex
    x6 = (xindex // 16)
    x0 = xindex % 16
    x1 = (xindex // 16) % 16
    x2 = (xindex // 256) % 16
    x3 = (xindex // 4096)
    tmp0 = tl.load(in_ptr0 + (r4 + (16*x5)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r4 + (16*x5)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x6), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (x0 + (16*r4) + (256*x6)), rmask, other=0.0)
    tmp11 = tl.load(in_ptr1 + (x0 + (16*r4) + (256*x6)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp1 * tmp3
    tmp5 = tmp2 - tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp12 = tmp10 * tmp11
    tmp13 = tmp11 * tmp3
    tmp14 = tmp12 - tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 0.125
    tmp20 = tmp5 * tmp19
    tl.store(out_ptr2 + (r4 + (16*x5)), tmp20, rmask)
    tl.store(out_ptr0 + (x0 + (16*x2) + (256*x1) + (4096*x3)), tmp9, None)
    tl.store(out_ptr1 + (x5), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cw/ccwc3xsokouyrehsht26w3fc6uhhma6gep7e4oztl7jepk2xac55.py
# Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward]

triton_poi_fused_cat_convolution_backward_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_backward_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16) % 16
    y2 = (yindex // 256)
    y4 = yindex % 256
    tmp0 = x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((64*y1) + (1024*y0) + (16384*(((y0 + (16*y1) + (256*x3) + (65536*y2)) // 16384) % 32)) + (((y0 + (16*y1) + (256*x3)) // 256) % 64)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((64*y0) + (1024*y1) + (16384*(((y0 + (16*y1) + (256*x3) + (65536*y2)) // 16384) % 32)) + (((y0 + (16*y1) + (256*x3)) // 256) % 64)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((64*y0) + (1024*y1) + (16384*(((y0 + (16*y1) + (256*x3) + (65536*y2)) // 16384) % 32)) + (((y0 + (16*y1) + (256*x3)) // 256) % 64)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1, 1], 512, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tl.load(in_ptr3 + ((-65536) + y4 + (256*x3) + (65536*y2)), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp15, tmp16, tmp17)
    tmp19 = tmp0 >= tmp13
    tmp20 = tl.full([1, 1], 768, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tl.load(in_ptr4 + ((64*y0) + (1024*y1) + (16384*((((-131072) + y0 + (16*y1) + (256*x3) + (65536*y2)) // 16384) % 32)) + (((y0 + (16*y1) + (256*x3)) // 256) % 64)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp19, tmp22, tmp23)
    tmp25 = tl.where(tmp15, tmp18, tmp24)
    tmp26 = tl.where(tmp4, tmp11, tmp25)
    tl.store(out_ptr0 + (y4 + (256*x3) + (196608*y2)), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m6/cm6tpemcdkwx5v2qrpfltt3so6tjnvaedvogetfi6j3jf4dcxgdl.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((16*(((r2 + (128*x1)) // 16) % 16)) + (256*x0) + (65536*((r2 + (128*x1)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 - tmp4
        tmp6 = tmp2 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ok/cok54gaptr6gjr3vvimtw3d57g3sdnbgcwz66n4ponvbdd5bht3b.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_45', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (256*x2) + (65536*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (256*x2) + (65536*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.00048828125
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
    tl.store(in_out_ptr0 + (x2 + (256*y3)), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bn/cbn6zihvljcqcwvhu73u6rjx3zb6n5baxkmthbpuq5moxw6bcw4v.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_46', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (1024*x2) + (262144*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0 + (1024*x2) + (262144*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (256*y3)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gv/cgvyg7sqpsgzuhlm3k476xyef7vfurj3yfhtejc4n3p53qoqy254.py
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
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xo/cxojtx2vmbvjbljfgwafj4jsyru7fqjqkgwrok4gcosxcka3c5kp.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_48', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((16*(((r2 + (128*x0)) // 16) % 16)) + (256*x1) + (262144*((r2 + (128*x0)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (1024*r2) + (131072*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ys/cysp7hwil42lguucu475hk2vpho5t626geaplociksn5diymki7a.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (1024*x2) + (262144*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.00048828125
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rf/crfcdr2f76dqiez2quhs5uxt3yibcezhp66e75s5se64xqv5fljr.py
# Source Nodes: [x_186], Original ATen: [aten.mul, aten.silu, aten.sum]
# x_186 => mul_210, sigmoid_27
triton_red_fused_mul_silu_sum_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_silu_sum_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 2
    x1 = (xindex // 2) % 256
    x2 = (xindex // 512)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (128*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (256*r3) + (32768*x0) + (65536*x2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/33/c33zao5ucivppdzq4siiz4y3la52ouk3bdlwjyzl43cue3z462ry.py
# Source Nodes: [sigmoid_5, x_186], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# sigmoid_5 => sigmoid_28
# x_186 => mul_210, sigmoid_27
triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_51 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_51', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (2*x0)), rmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = 1.0
    tmp8 = tmp7 - tmp6
    tmp9 = tmp6 * tmp8
    tmp10 = tmp4 * tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/n3/cn3siijk7yrmr3hcj5vidxhq2iwo4uo3lopjo2xx5zntxid26rb7.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_52', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8
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


# kernel path: /tmp/torchinductor_youkaichao/s3/cs3zuymng25sb5dc6ey4wi3hudqahm722n7kkouf7goomcfzht2h.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_53', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/an/cang2t25y56jjv3uo57gahpv4poaincafnwpzlplawftjfdjwaru.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_54 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_54', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wq/cwqhufpqpy6oooity657xux7vvcwrql53sjpkbcf6ij3cb76aole.py
# Source Nodes: [sigmoid_5], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# sigmoid_5 => sigmoid_28
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_55', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((16*(((r2 + (128*x0)) // 16) % 16)) + (256*x1) + (65536*((r2 + (128*x0)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (256*((r2 + (128*x0)) // 256))), rmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x1 + (256*((r2 + (128*x0)) // 256))), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x1 + (256*r2) + (32768*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 256.0
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
        _tmp17 = tl.where(rmask, tmp18, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bd/cbdxbvfe46eibx4w5uuyz6aqrqnp4kgkrauuwq6psgo4nhp4func.py
# Source Nodes: [sigmoid_5], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# sigmoid_5 => sigmoid_28
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_56', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((16*(((r2 + (128*x1)) // 16) % 16)) + (256*x0) + (65536*((r2 + (128*x1)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (256*((r2 + (128*x1)) // 256))), rmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (256*((r2 + (128*x1)) // 256))), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr4 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 256.0
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
        _tmp21 = tl.where(rmask, tmp22, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/iv/civd7itttfjn4gynyuhu2ihqoj7biwyeyupei53vdv4k6md4qcoi.py
# Source Nodes: [sigmoid_5], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# sigmoid_5 => sigmoid_28
triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_57 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_57', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (65536*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (256*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (256*y1)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp5 = 256.0
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
    tmp20 = 0.00048828125
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kn/cknhi2z5q4g34f2susciko6u4ywhuodsk73u745u243oeyiouysd.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_58 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_58', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (65536*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5e/c5evb6u2fjse765wmnk3hh37bkawoh2snxyb3qew566abhcetxly.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_mul_native_batch_norm_backward_59 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_batch_norm_backward_59', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    tmp6 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp13 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((16*(((r2 + (128*x0)) // 16) % 16)) + (256*x1) + (262144*((r2 + (128*x0)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((16*(((r2 + (128*x0)) // 16) % 16)) + (256*x1) + (262144*((r2 + (128*x0)) // 256)) + (r2 % 16)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x1 + (1024*r2) + (131072*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr3 + (x1 + (1024*r2) + (131072*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr5 + (x1 + (1024*r2) + (131072*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp4 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask, tmp18, _tmp17)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fj/cfjhmnz7ls75fbb4szbkh64a677zutdb6c22x5pxok63y22dyu2y.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_add_mul_native_batch_norm_backward_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_native_batch_norm_backward_60', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (256*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (1024*x2) + (262144*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0 + (1024*x2) + (262144*y1)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr9 + (y0 + (1024*x2) + (262144*y1)), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr10 + (y0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr11 + (y0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr12 + (y0), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr13 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp7 = tmp5 - tmp6
    tmp9 = 0.00048828125
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
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp21, xmask)
    tl.store(out_ptr1 + (x2 + (256*y3)), tmp35, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ti/cti3fctpvdhhe3zeepfw5ookixb72ifoyz52luxchjj5h5khgbzh.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_61', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((32*(((r2 + (128*x0)) // 32) % 32)) + (1024*x1) + (262144*((r2 + (128*x0)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (256*r2) + (32768*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/aj/cajjltsqeaviaxqdhi3gycdclvpcflr5jrj7emlblnhhjkt6mwi5.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_62 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/th/cthqgufyx7vls4wmjw2mzxdsoe7lzsdscpprsfk5w47mjc2lkfno.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_63', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((32*(((r2 + (128*x1)) // 32) % 32)) + (1024*x0) + (262144*((r2 + (128*x1)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 - tmp4
        tmp6 = tmp2 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sa/csa7nhvz2ip56o6iqlnpwx5lo5okdnvcaaodk2clmm5rmrksp2ry.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_64 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_64', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l2/cl26eofzi74pqjfqr5bkrwuc2pzdpvm6hfnqnsg4znikzieuvcp5.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_65 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_65', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (1024*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (256*x2) + (262144*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (256*x2) + (262144*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.0001220703125
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
    tl.store(in_out_ptr0 + (x2 + (1024*y3)), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sk/cskksmx476cfukodifv744tn5dly4hnx3aucgs272xo2f4fegtux.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_mul_native_batch_norm_backward_66 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_batch_norm_backward_66', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (524288*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0) + (524288*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (512*r3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cx/ccxvunmslumvcomcw7ncnohpeuo3nxl5qzswdq276irdenk7foca.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_mul_native_batch_norm_backward_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_batch_norm_backward_67', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp6 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((32*(((r2 + (128*x0)) // 32) % 32)) + (1024*x1) + (524288*((r2 + (128*x0)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((32*(((r2 + (128*x0)) // 32) % 32)) + (1024*x1) + (524288*((r2 + (128*x0)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x1 + (512*r2) + (65536*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr3 + (x1 + (512*r2) + (65536*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/s3/cs3w5qluxxkr4wfyatwr5xja4gb5aiqqvu6okpxbiwaphzm3jgoz.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_add_mul_native_batch_norm_backward_68 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_batch_norm_backward_68', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ei/ceiyb73zykxa5u7r75nmkirk45oemeaco2bmehxbqpzfgq325d4n.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_add_mul_native_batch_norm_backward_69 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_native_batch_norm_backward_69', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2 + (1024*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (1024*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (512*x2) + (524288*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0 + (512*x2) + (524288*y1)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0001220703125
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (1024*y3)), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wl/cwlvqs7jidrywwoel55xqepjas34xxtlbmrm3hnkj73xtgg6h6hk.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_70', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((32*(((r2 + (128*x0)) // 32) % 32)) + (1024*x1) + (131072*((r2 + (128*x0)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (128*r2) + (16384*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/w2/cw2piqoszrdb7svqlkzgixsv3awgyycjl6yrsx3khn2o4pf76vu6.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_71 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_71', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zs/czs3f7lffazf3kcmr5omhz4og4m4niwolkqybgiwkrxzhzojax7f.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_72 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_72', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((32*(((r2 + (128*x1)) // 32) % 32)) + (1024*x0) + (131072*((r2 + (128*x1)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + ((32*(((32*(((r2 + (128*x1)) // 32) % 32)) + (r2 % 32)) % 1024)) + (32768*((((32*(((r2 + (128*x1)) // 32) % 32)) + (1024*x0) + (131072*((r2 + (128*x1)) // 1024)) + (r2 % 32)) // 32768) % 32)) + ((((32*(((r2 + (128*x1)) // 32) % 32)) + (1024*x0) + (r2 % 32)) // 1024) % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 - tmp4
        tmp6 = tmp2 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ef/cefrd6uz6hne7ccdot7w3xqsx7kpxj6hn6rpwcgohmkjkbwpohmd.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_73 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_73', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 64
    RBLOCK: tl.constexpr = 64
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


# kernel path: /tmp/torchinductor_youkaichao/xg/cxg4nyiy4l5difl6m2asbpuzb3ksus3i4febynztzul3gsloms52.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_mul_native_batch_norm_backward_74 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_native_batch_norm_backward_74', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (1024*y3)), xmask)
    tmp1 = tl.load(in_ptr0 + (y0 + (128*x2) + (131072*y1)), xmask)
    tmp3 = tl.load(in_ptr1 + ((32*x2) + (32768*((x2 + (1024*y0)) // 32768)) + (131072*y1) + (y0 % 32)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.0001220703125
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(in_out_ptr0 + (x2 + (1024*y3)), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/27/c27ktn334ou2ymrc62drk4reo6o3gu52ub2yqo5iv62ujx4jmdpk.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_per_fused__softmax_backward_data_75 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_75', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel):
    xnumel = 32768
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tl.store(out_ptr0 + (x0), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/v6/cv6fwaqtifsrlbidid24wuerccw4e5zve7agov6zbruzb336tazb.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]

triton_per_fused__softmax_backward_data_mul_sum_76 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1048576, 32],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_mul_sum_76', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    rnumel = 32
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r4 = rindex
    x5 = xindex
    x6 = (xindex // 32)
    x0 = xindex % 32
    x1 = (xindex // 32) % 32
    x2 = (xindex // 1024) % 32
    x3 = (xindex // 32768)
    tmp0 = tl.load(in_ptr0 + (r4 + (32*x5)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r4 + (32*x5)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x6), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (x0 + (32*r4) + (1024*x6)), rmask, other=0.0)
    tmp11 = tl.load(in_ptr1 + (x0 + (32*r4) + (1024*x6)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp1 * tmp3
    tmp5 = tmp2 - tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp12 = tmp10 * tmp11
    tmp13 = tmp11 * tmp3
    tmp14 = tmp12 - tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 0.1767766952966369
    tmp20 = tmp5 * tmp19
    tl.store(out_ptr2 + (r4 + (32*x5)), tmp20, rmask)
    tl.store(out_ptr0 + (x0 + (32*x2) + (1024*x1) + (32768*x3)), tmp9, None)
    tl.store(out_ptr1 + (x5), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cu/ccuso75jr7nqpwkhqhink7ymjk7yk4yyymhinx3y6dqan2q2u537.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_77 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_77', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2064384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 63
    x1 = (xindex // 63)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 64, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0 + (64*(x1 % 32))
    tmp4 = tl.full([1], 2079, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = ((x0 + (64*(x1 % 32))) // 63) % 33
    tmp8 = tl.full([1], 32, tl.int64)
    tmp9 = tmp7 < tmp8
    tmp10 = tmp9 & tmp6
    tmp11 = (x0 + (64*(x1 % 32))) % 63
    tmp12 = tl.full([1], 31, tl.int64)
    tmp13 = tmp11 >= tmp12
    tmp14 = tmp13 & tmp10
    tmp15 = tl.load(in_ptr0 + ((-31) + (32*(((x0 + (64*(x1 % 32))) // 63) % 33)) + (1024*(x1 // 32)) + ((x0 + (64*(x1 % 32))) % 63)), tmp14, other=0.0)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp14, tmp15, tmp16)
    tmp18 = 0.0
    tmp19 = tl.where(tmp13, tmp17, tmp18)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp10, tmp19, tmp20)
    tmp22 = tl.where(tmp9, tmp21, tmp18)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp6, tmp22, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp2, tmp24, tmp25)
    tl.store(out_ptr0 + (x2), tmp26, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bb/cbbitlx2bl5g34hfidd7scs7jyrr766sht7dgykotuwfqtoel46n.py
# Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward]

triton_poi_fused_cat_convolution_backward_78 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_backward_78', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 32
    y1 = (yindex // 32) % 32
    y2 = (yindex // 1024)
    y4 = yindex % 1024
    tmp0 = x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((32*y1) + (1024*y0) + (32768*(((y0 + (32*y1) + (1024*x3) + (131072*y2)) // 32768) % 32)) + (((y0 + (32*y1) + (1024*x3)) // 1024) % 32)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + ((32*y0) + (1024*y1) + (32768*(((y0 + (32*y1) + (1024*x3) + (131072*y2)) // 32768) % 32)) + (((y0 + (32*y1) + (1024*x3)) // 1024) % 32)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + ((32*y0) + (1024*y1) + (32768*(((y0 + (32*y1) + (1024*x3) + (131072*y2)) // 32768) % 32)) + (((y0 + (32*y1) + (1024*x3)) // 1024) % 32)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1, 1], 256, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tl.load(in_ptr3 + ((-131072) + y4 + (1024*x3) + (131072*y2)), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp15, tmp16, tmp17)
    tmp19 = tmp0 >= tmp13
    tmp20 = tl.full([1, 1], 384, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tl.load(in_ptr4 + ((32*y0) + (1024*y1) + (32768*((((-262144) + y0 + (32*y1) + (1024*x3) + (131072*y2)) // 32768) % 32)) + (((y0 + (32*y1) + (1024*x3)) // 1024) % 32)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp19, tmp22, tmp23)
    tmp25 = tl.where(tmp15, tmp18, tmp24)
    tmp26 = tl.where(tmp4, tmp11, tmp25)
    tl.store(out_ptr0 + (y4 + (1024*x3) + (393216*y2)), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uk/cukvyglx53e63zuo5jixrlzsdss6vbksb64toittyjksdate7ov2.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_79 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_79', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((32*(((r2 + (128*x1)) // 32) % 32)) + (1024*x0) + (131072*((r2 + (128*x1)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 - tmp4
        tmp6 = tmp2 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bi/cbixutf2gjf54fm54eskyunoogainj4gx73bgx7uylcqtzc6sfux.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_80 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_80', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (1024*y3)), xmask)
    tmp1 = tl.load(in_ptr0 + (y0 + (128*x2) + (131072*y1)), xmask)
    tmp3 = tl.load(in_ptr1 + (y0 + (128*x2) + (131072*y1)), xmask)
    tmp4 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.0001220703125
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(in_out_ptr0 + (x2 + (1024*y3)), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jw/cjwy74bvh2p5hc2l7q3svdkpzqprzz6wva4g52rm4od25g7776dn.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_81 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_81', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (1024*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (1024*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (512*x2) + (524288*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (1024*y3)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0 + (512*x2) + (524288*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (1024*y3)), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/la/clalkphkv75linjhrpwmhwppvaexqy6zod6tsmqvr2ydfe5fdhxs.py
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
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_82', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (524288*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p2/cp2bjldcibjwfu3vco5man4pw6uhe4tx7usjmopi4rucwkohoga5.py
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_83', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((32*(((r2 + (128*x0)) // 32) % 32)) + (1024*x1) + (524288*((r2 + (128*x0)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (512*r2) + (65536*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lq/clq3euo66ahr3tafgdbw7qjiwr6hf6hyyaywffy3xoxhldpx3rsh.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_84 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_84', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2 + (1024*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (512*x2) + (524288*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.0001220703125
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2 + (1024*y3)), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dc/cdcwwo5kl2eev4475dz7nqsqejbiakervrvs3pz4jucuwulq6hal.py
# Source Nodes: [x_106], Original ATen: [aten.mul, aten.silu, aten.sum]
# x_106 => mul_128, sigmoid_16
triton_red_fused_mul_silu_sum_85 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_silu_sum_85', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 8
    x1 = (xindex // 8) % 128
    x2 = (xindex // 1024)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (128*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (128*r3) + (16384*x0) + (131072*x2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kv/ckv2mamsdwaijitlnehpn32qtvzcdctd3wumymur7entejclrs6y.py
# Source Nodes: [sigmoid_3, x_106], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# sigmoid_3 => sigmoid_17
# x_106 => mul_128, sigmoid_16
triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_86 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_86', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (8*x0)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/7c/c7cg5jmvlzixacwrrqqvkrjdxtdpmmfk6ox7j5u724hlofstzg4n.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_87 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_87', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 8
    RBLOCK: tl.constexpr = 8
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


# kernel path: /tmp/torchinductor_youkaichao/6b/c6b6ymg6m6zothtc5nkbqer7exbfmrleilxhij5jk5mbpom5by7e.py
# Source Nodes: [], Original ATen: [aten.threshold_backward]

triton_poi_fused_threshold_backward_88 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_threshold_backward_88', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
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


# kernel path: /tmp/torchinductor_youkaichao/kv/ckvfyt66lbuchlorjl4n25h63oitcyjzf2mdl3eobln6n26suww7.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_89 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_89', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (8*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/le/cle75qsdjeezjrhgij7d6m76u4lsqa5mlsq3of3tok4swo7z5x2s.py
# Source Nodes: [sigmoid_3], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# sigmoid_3 => sigmoid_17
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_90 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_90', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((32*(((r2 + (128*x0)) // 32) % 32)) + (1024*x1) + (131072*((r2 + (128*x0)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (128*((r2 + (128*x0)) // 1024))), rmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x1 + (128*((r2 + (128*x0)) // 1024))), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x1 + (128*r2) + (16384*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 1024.0
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
        _tmp17 = tl.where(rmask, tmp18, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hy/chywurfckkoewjiogeqyxewqvzhqv5mqfli2gn7pxlrfvov2e2sn.py
# Source Nodes: [sigmoid_3], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# sigmoid_3 => sigmoid_17
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_91 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_91', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((32*(((r2 + (128*x1)) // 32) % 32)) + (1024*x0) + (131072*((r2 + (128*x1)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (128*((r2 + (128*x1)) // 1024))), rmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (128*((r2 + (128*x1)) // 1024))), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr4 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 1024.0
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
        _tmp21 = tl.where(rmask, tmp22, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/t7/ct7str4zfgo374lo5k3tm7n3rna5fe46wz2s6jwoi3ieoqjpwmno.py
# Source Nodes: [sigmoid_3], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# sigmoid_3 => sigmoid_17
triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_92 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_92', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1024*x2) + (131072*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (128*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (128*y1)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (128*y3)), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (128*y3)), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp5 = 1024.0
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
    tmp20 = 0.0001220703125
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cm/ccm5zxnbhhkbbx7b7l6vlbrvyx6c3nrkyfrm3otxypggdjoe6arz.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_93 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_93', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (128*x2) + (131072*y1)), xmask)
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (1024*y3)), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fn/cfnhycvoxyvxwf4ssyklrmweau2zekd4pame673h32wpnolfe5nw.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_mul_native_batch_norm_backward_94 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_batch_norm_backward_94', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp6 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp13 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((32*(((r2 + (128*x0)) // 32) % 32)) + (1024*x1) + (524288*((r2 + (128*x0)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((32*(((r2 + (128*x0)) // 32) % 32)) + (1024*x1) + (524288*((r2 + (128*x0)) // 1024)) + (r2 % 32)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x1 + (512*r2) + (65536*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr3 + (x1 + (512*r2) + (65536*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr5 + (x1 + (512*r2) + (65536*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp4 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask, tmp18, _tmp17)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sj/csjnq2padgjab5efmfxja24y3kzxhdiarsmjjfu6rq7twvaoghvi.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_add_mul_native_batch_norm_backward_95 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32', 17: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16, 17))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_native_batch_norm_backward_95', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2 + (1024*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (1024*y3)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (512*x2) + (524288*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0 + (512*x2) + (524288*y1)), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr9 + (y0 + (512*x2) + (524288*y1)), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr10 + (y0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr11 + (y0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr12 + (y0), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr13 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0001220703125
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
    tl.store(out_ptr0 + (x2 + (1024*y3)), tmp21, xmask)
    tl.store(out_ptr1 + (x2 + (1024*y3)), tmp35, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7e/c7eesqp4ksffo35pdwx5vzyd4l45cv22kw7shilf7g5uagl72db2.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_96 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_96', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((64*(((r2 + (128*x0)) // 64) % 64)) + (4096*x1) + (524288*((r2 + (128*x0)) // 4096)) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (128*r2) + (16384*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bl/cbl5l7d6nxgerkrj4xx4vvqgee7eracuizjptjk3mh6pzo3q2vbo.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_97 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_97', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 128
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
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gh/cghcm4rvsjnyy2z3jnykrny4eycwohwgdorpojovoeujlzna3ikk.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_98 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_98', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((64*(((r2 + (128*x1)) // 64) % 64)) + (4096*x0) + (524288*((r2 + (128*x1)) // 4096)) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 - tmp4
        tmp6 = tmp2 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ho/chozqwczw2jzaeakmniw42ylgfxbvueoqwv5o5qfu7xc33d73y27.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_99 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_99', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 256
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
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7h/c7htl5ex6gjwvitgaxbbdiyqypj6z7wavy2x2ifrjc5trmiv4n22.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_100 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_100', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (4096*y3)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (128*x2) + (524288*y1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (128*x2) + (524288*y1)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 3.0517578125e-05
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
    tl.store(in_out_ptr0 + (x2 + (4096*y3)), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ii/cii37swutf4es4vebtgynfhncfl2oqgaqrnuwhovaxw24jhijsqe.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_mul_native_batch_norm_backward_101 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_batch_norm_backward_101', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 4096
        r2 = (rindex // 4096)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (256*r3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7d/c7dhlvyceme5c2jzngmtttgkpyt4ld23lmser376xjz4u6p2qyr2.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_mul_native_batch_norm_backward_102 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_batch_norm_backward_102', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp6 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((64*(((r2 + (128*x0)) // 64) % 64)) + (4096*x1) + (1048576*((r2 + (128*x0)) // 4096)) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((64*(((r2 + (128*x0)) // 64) % 64)) + (4096*x1) + (1048576*((r2 + (128*x0)) // 4096)) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x1 + (256*r2) + (32768*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr3 + (x1 + (256*r2) + (32768*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mv/cmvasvynv45ozzk2c7d3ir26pj4fro3k2bkhi3ut6sfvyiyqzejf.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_add_mul_native_batch_norm_backward_103 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_batch_norm_backward_103', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 256
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/52/c52y5tf2ccubfdtnxyovd2gtchofv7utlta4wni25ds7uhbczqwl.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_add_mul_native_batch_norm_backward_104 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_native_batch_norm_backward_104', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + (x2 + (4096*y3)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (4096*y3)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (256*x2) + (1048576*y1)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0 + (256*x2) + (1048576*y1)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp7 = tmp5 - tmp6
    tmp9 = 3.0517578125e-05
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2 + (4096*y3)), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ti/ctija4jvmy5wdq5waogbnjjymxsp2ig5f7mw2ce3zdocddkfmutj.py
# Source Nodes: [x_55], Original ATen: [aten.mul, aten.silu, aten.sum]
# x_55 => mul_71, sigmoid_8
triton_red_fused_mul_silu_sum_105 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_silu_sum_105', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 32
    x1 = (xindex // 32) % 64
    x2 = (xindex // 2048)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (128*x4)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (64*r3) + (8192*x0) + (262144*x2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tf/ctfikh3bb463by4aloofqkg4s7khotmh2f3wtjbhtdchqthlygvm.py
# Source Nodes: [sigmoid_1, x_55], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
# sigmoid_1 => sigmoid_9
# x_55 => mul_71, sigmoid_8
triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_106 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_106', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 32
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (32*x0)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/eu/ceu5jst4wp5argik27o76jfdpxsfr3ztksvtjtjubrvwbqzwr73j.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_107 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_107', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 8
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


# kernel path: /tmp/torchinductor_youkaichao/on/conpnkeu3ic3xqmoxwqvg4xntnuxy5oe5atf7dnx7jt6p2snn5dx.py
# Source Nodes: [sigmoid_1], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# sigmoid_1 => sigmoid_9
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_108 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_108', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((64*(((r2 + (128*x0)) // 64) % 64)) + (4096*x1) + (262144*((r2 + (128*x0)) // 4096)) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (64*((r2 + (128*x0)) // 4096))), rmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x1 + (64*((r2 + (128*x0)) // 4096))), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x1 + (64*r2) + (8192*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 4096.0
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
        _tmp17 = tl.where(rmask, tmp18, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mf/cmfm2mqpvl7psmogf5fpft3nmim253m3j5dllhs2z5vpdwcltl66.py
# Source Nodes: [sigmoid_1], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# sigmoid_1 => sigmoid_9
triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_109 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_109', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 64
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
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tq/ctqnwt4fktro5nh5xeizhutxrot2ju7lrqffm6z6n4b7pbfi5476.py
# Source Nodes: [sigmoid_1], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# sigmoid_1 => sigmoid_9
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_110 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_110', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((64*(((r2 + (128*x1)) // 64) % 64)) + (4096*x0) + (262144*((r2 + (128*x1)) // 4096)) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (64*((r2 + (128*x1)) // 4096))), rmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (64*((r2 + (128*x1)) // 4096))), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (64*r2) + (8192*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr4 + (x0 + (64*r2) + (8192*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp5 = 4096.0
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
        _tmp21 = tl.where(rmask, tmp22, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/oj/coj5tm5q2zv7pq73klt34slagnzj6gb2ux25vt3h7lfm3wkvcazg.py
# Source Nodes: [sigmoid_1], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# sigmoid_1 => sigmoid_9
triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_111 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 256],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_111', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 256
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


# kernel path: /tmp/torchinductor_youkaichao/hh/chhkpohafmqbcrxsveouusq34kx7fqsom53vrpl6izywuheoteb6.py
# Source Nodes: [sigmoid_1], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
# sigmoid_1 => sigmoid_9
triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_112 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_112', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 4096
    y1 = (yindex // 4096)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (4096*x2) + (262144*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (64*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (64*y1)), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp5 = 4096.0
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
    tmp20 = 3.0517578125e-05
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/di/cdijav42u7fdn7aiq5xqa2qghllatrjulwrvbsue3vy6rrsvsk3d.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_113 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_113', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 4096
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
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (262144*y1)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + (4096*y3)), tmp4, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vn/cvnxbriqdpt5yoxbc3cl7tl3rhd6pegzkthoxpxgcuuvpj5infcg.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_114 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_114', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((64*(((r2 + (128*x0)) // 64) % 64)) + (4096*x1) + (262144*((r2 + (128*x0)) // 4096)) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (64*r2) + (8192*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vb/cvbgkmpx7t4bzcrch46zobvxroztlb5ehaoyxcczcspffyqus3tp.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_115 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_115', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((64*(((r2 + (128*x1)) // 64) % 64)) + (4096*x0) + (262144*((r2 + (128*x1)) // 4096)) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (64*r2) + (8192*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (64*r2) + (8192*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 - tmp4
        tmp6 = tmp2 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4v/c4vty4kyjy4ch667myfcpfyeeftiqjkiaeay5cgirxoza6imtkrf.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_116 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_116', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 4096
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (4096*y3)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (64*x2) + (262144*y1)), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (64*x2) + (262144*y1)), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 3.0517578125e-05
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
    tl.store(in_out_ptr0 + (x2 + (4096*y3)), tmp19, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n7/cn7bqbmjq7r6jpeb3elcp5cdavhv3grdy55yz2vyrjamb3hwrfey.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul]

triton_poi_fused_add_mul_117 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_117', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (4096*y3)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (4096*y3)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (256*x2) + (1048576*y1)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (4096*y3)), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0 + (256*x2) + (1048576*y1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (4096*y3)), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/55/c55rlowbxslge52eqhpnd3gpyayvbgdqglzcrph6kvtpy3cyvvhw.py
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
    size_hints=[256, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_118', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 4096
        r2 = (rindex // 4096)
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yf/cyf33hepsveemuwss6upv44sk2pr6hyo4ht7f2ic5yemi7amnw72.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_119 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_119', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((64*(((r2 + (128*x0)) // 64) % 64)) + (4096*x1) + (1048576*((r2 + (128*x0)) // 4096)) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (256*r2) + (32768*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x1 + (256*r2) + (32768*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp0 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mk/cmkseojrapyqrczjnqtdhnrt3a5q2hnccwu3vlcer7pygv6tnoxs.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_120 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14, 15))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_120', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + (x2 + (4096*y3)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (256*x2) + (1048576*y1)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (y0 + (256*x2) + (1048576*y1)), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (y0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (y0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr10 + (y0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr11 + (y0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 3.0517578125e-05
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
    tl.store(out_ptr0 + (x2 + (4096*y3)), tmp17, None)
    tl.store(out_ptr1 + (x2 + (4096*y3)), tmp31, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ex/cexpxoguin6tdcj2qcmup77hps2rt6573dttjsyjy2adyaw6j6q3.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_mul_native_batch_norm_backward_121 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_batch_norm_backward_121', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (64*r2) + (524288*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f6/cf6haemrrwswygj4iwu24vonldbpe7zlk7n662zympwzwm2rkbfw.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_add_mul_native_batch_norm_backward_122 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_batch_norm_backward_122', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/rk/crktfafioht2ogse6hcaco33wff4bf3omf6qk34irsmre2lcguri.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_add_mul_native_batch_norm_backward_123 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_batch_norm_backward_123', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp6 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((64*(((r2 + (128*x0)) // 64) % 64)) + (4096*x1) + (262144*((r2 + (128*x0)) // 4096)) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((64*(((r2 + (128*x0)) // 64) % 64)) + (4096*x1) + (262144*((r2 + (128*x0)) // 4096)) + (r2 % 64)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x1 + (64*r2) + (8192*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr3 + (x1 + (64*r2) + (8192*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp7 = tmp5 - tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/k2/ck2fvglavwmarrjvmnfxmi3u4njw3v3rotdhfjftko5pfckkc6rt.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_add_mul_native_batch_norm_backward_124 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_batch_norm_backward_124', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 64
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3p/c3pztd2pk6lggotfm3fupkb26nzgdgwregm6eztdupxjubupcy3d.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_add_mul_native_batch_norm_backward_125 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_native_batch_norm_backward_125', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 4096
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (4096*y3)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (4096*y3)), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (64*x2) + (262144*y1)), ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y0 + (64*x2) + (262144*y1)), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp7 = tmp5 - tmp6
    tmp9 = 3.0517578125e-05
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (4096*y3)), tmp21, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hn/chnzazqmosburwwjx5eexlnyysx5qdewr2cg3hxlim2n42rgzjtx.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_126 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_126', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (128*(x0 % 128)) + (16384*x1) + (524288*((r2 + (128*x0)) // 16384))), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (32*r2) + (4096*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4w/c4w6suhq4wromadbhrmphag773pg2e4okyvodavj2zyz2d22bhbt.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_127 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_127', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 32
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gk/cgkmff2huji4mq6wqom4o3sdwfljd3qzwyvbiy7alevepuvhn2dl.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_128 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_128', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (128*(x1 % 128)) + (16384*x0) + (524288*((r2 + (128*x1)) // 16384))), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (32*r2) + (4096*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (32*r2) + (4096*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 - tmp4
        tmp6 = tmp2 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5c/c5cedn2fmnw4kjrnf4l6zpgvi4lr5uflsxctpy57xv5g25lujnqi.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_129 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_129', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 1024
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


# kernel path: /tmp/torchinductor_youkaichao/23/c236bo5n74t4ixnxf57qbytorfpsygtxhj3iknrz45lxi7daim4g.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_130 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_130', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 32
    y1 = (yindex // 32)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (16384*y3)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (32*x2) + (524288*y1)), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (32*x2) + (524288*y1)), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 7.62939453125e-06
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
    tl.store(in_out_ptr0 + (x2 + (16384*y3)), tmp19, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ii/ciigbfzz7axfx75jyr6an6jqztveuluwalctirglbndm42mvokif.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_131 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_131', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (128*(x0 % 128)) + (16384*x1) + (393216*((r2 + (128*x0)) // 16384))), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (24*r2) + (3072*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pg/cpgd6wnzli57qrfzpwpelhu6hvetyiqvohkpwgavb5hscydogusj.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_132 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_132', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 24
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pb/cpbmygdnfolwetuyyukwgih5iitsilbwzjkyeiyfdon7pp43yxmq.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_133 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_133', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 24
    x1 = (xindex // 24)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (128*(x1 % 128)) + (16384*x0) + (393216*((r2 + (128*x1)) // 16384))), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (24*r2) + (3072*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (24*r2) + (3072*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 - tmp4
        tmp6 = tmp2 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sm/csm5wscwfyyjharjja7rk4a4lwjgfw4yxndwo5ns3cd5zchd3ntc.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_134 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_134', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 1024
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
        tmp0 = tl.load(in_ptr0 + (x0 + (24*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zb/czb3jkop3m6odz7eaiglk4vggaxxnyxab6yxazluvfkbrc2ewkex.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_135 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_135', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 24
    y1 = (yindex // 24)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (16384*y3)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (24*x2) + (393216*y1)), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (24*x2) + (393216*y1)), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 7.62939453125e-06
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
    tl.store(in_out_ptr0 + (x2 + (16384*y3)), tmp19, ymask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_61, primals_63, primals_65, primals_69, primals_71, primals_73, primals_75, primals_79, primals_81, primals_83, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_92, primals_94, primals_95, primals_96, primals_97, primals_98, primals_100, primals_102, primals_103, primals_104, primals_105, primals_107, primals_109, primals_110, primals_111, primals_112, primals_113, primals_115, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_125, primals_127, primals_128, primals_129, primals_130, primals_131, primals_133, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_263, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, mul_15, convolution_2, squeeze_7, mul_23, convolution_3, squeeze_10, mul_31, convolution_4, squeeze_13, add_24, mean, relu, convolution_6, mul_40, convolution_7, squeeze_16, convolution_8, squeeze_19, mul_55, convolution_9, squeeze_22, mul_63, convolution_10, squeeze_25, add_45, mean_1, relu_1, convolution_12, mul_72, convolution_13, squeeze_28, mul_80, convolution_14, squeeze_31, mul_88, convolution_15, squeeze_34, add_61, mean_2, relu_2, convolution_17, mul_97, convolution_18, squeeze_37, convolution_19, squeeze_40, mul_112, convolution_20, squeeze_43, mul_120, convolution_21, squeeze_46, add_82, mean_3, relu_3, convolution_23, mul_129, convolution_24, squeeze_49, mul_137, convolution_25, squeeze_52, mul_145, view_7, view_13, bmm_1, squeeze_55, mul_154, convolution_27, squeeze_58, mul_162, convolution_28, squeeze_61, mul_170, convolution_29, squeeze_64, add_116, mean_4, relu_4, convolution_31, mul_179, convolution_32, squeeze_67, convolution_33, squeeze_70, mul_194, convolution_34, squeeze_73, mul_202, convolution_35, squeeze_76, add_137, mean_5, relu_5, convolution_37, mul_211, convolution_38, squeeze_79, mul_219, convolution_39, squeeze_82, mul_227, view_31, view_37, bmm_3, squeeze_85, mul_236, convolution_41, squeeze_88, mul_244, convolution_42, squeeze_91, mul_252, view_55, view_61, view_71, avg_pool2d, squeeze_94, mul_261, convolution_44, squeeze_97, convolution_45, squeeze_100, mul_276, convolution_46, squeeze_103, mul_284, view_79, view_85, bmm_7, squeeze_106, mul_293, convolution_48, squeeze_109, mul_301, convolution_49, squeeze_112, clone_62, permute_33, mul_311, unsqueeze_154, mul_323, unsqueeze_166, mul_335, unsqueeze_178, permute_41, permute_42, alias_16, permute_46, permute_52, permute_54, permute_55, mul_350, unsqueeze_190, mul_362, unsqueeze_202, unsqueeze_214, mul_383, unsqueeze_226, permute_62, permute_63, alias_17, permute_67, permute_73, permute_75, permute_76, mul_398, unsqueeze_238, mul_410, unsqueeze_250, mul_422, unsqueeze_262, permute_83, permute_84, alias_18, permute_88, permute_94, permute_96, permute_97, mul_437, unsqueeze_274, mul_449, unsqueeze_286, unsqueeze_298, mul_477, unsqueeze_310, mul_489, unsqueeze_322, unsqueeze_334, unsqueeze_346, mul_526, unsqueeze_358, mul_538, unsqueeze_370, mul_550, unsqueeze_382, permute_110, permute_111, alias_27, permute_115, permute_121, permute_123, permute_124, mul_565, unsqueeze_394, mul_577, unsqueeze_406, unsqueeze_418, mul_605, unsqueeze_430, mul_617, unsqueeze_442, unsqueeze_454, unsqueeze_466, mul_654, unsqueeze_478, mul_666, unsqueeze_490, unsqueeze_502, mul_694, unsqueeze_514, mul_706, unsqueeze_526, unsqueeze_538, unsqueeze_550, mul_743, unsqueeze_562, mul_755, unsqueeze_574, mul_767, unsqueeze_586, mul_779, unsqueeze_598, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (24, ), (1, ))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_11, (256, ), (1, ))
    assert_size_stride(primals_13, (256, ), (1, ))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_19, (256, ), (1, ))
    assert_size_stride(primals_21, (128, ), (1, ))
    assert_size_stride(primals_23, (128, ), (1, ))
    assert_size_stride(primals_25, (512, ), (1, ))
    assert_size_stride(primals_27, (512, ), (1, ))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_31, (128, ), (1, ))
    assert_size_stride(primals_33, (512, ), (1, ))
    assert_size_stride(primals_35, (128, ), (1, ))
    assert_size_stride(primals_39, (128, ), (1, ))
    assert_size_stride(primals_41, (512, ), (1, ))
    assert_size_stride(primals_43, (256, ), (1, ))
    assert_size_stride(primals_45, (256, ), (1, ))
    assert_size_stride(primals_47, (1024, ), (1, ))
    assert_size_stride(primals_49, (1024, ), (1, ))
    assert_size_stride(primals_51, (256, ), (1, ))
    assert_size_stride(primals_53, (256, ), (1, ))
    assert_size_stride(primals_55, (1024, ), (1, ))
    assert_size_stride(primals_57, (256, ), (1, ))
    assert_size_stride(primals_61, (256, ), (1, ))
    assert_size_stride(primals_63, (1024, ), (1, ))
    assert_size_stride(primals_65, (512, ), (1, ))
    assert_size_stride(primals_69, (512, ), (1, ))
    assert_size_stride(primals_71, (1536, ), (1, ))
    assert_size_stride(primals_73, (1536, ), (1, ))
    assert_size_stride(primals_75, (512, ), (1, ))
    assert_size_stride(primals_79, (512, ), (1, ))
    assert_size_stride(primals_81, (1536, ), (1, ))
    assert_size_stride(primals_83, (1280, ), (1, ))
    assert_size_stride(primals_85, (24, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_86, (32, 24, 3, 3), (216, 1, 72, 24))
    assert_size_stride(primals_87, (64, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_88, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_89, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_90, (8, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_92, (64, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_94, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_95, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_96, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_97, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(primals_98, (8, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_100, (64, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_102, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_103, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_104, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_105, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_107, (128, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_109, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_110, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_111, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_112, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_113, (8, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_115, (128, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_117, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_118, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_119, (384, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_120, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_121, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_122, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_123, (16, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_125, (256, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_127, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_128, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_129, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_130, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(primals_131, (16, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_133, (256, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_135, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_136, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_137, (768, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_138, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_139, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_140, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_141, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_142, (1536, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_143, (512, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_144, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_145, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_146, (1280, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_263, (8, 3, 256, 256), (196608, 1, 768, 3))
    assert_size_stride(convolution, (8, 24, 128, 128), (393216, 1, 3072, 24))
    assert_size_stride(squeeze_1, (24, ), (1, ))
    assert_size_stride(mul_7, (8, 24, 128, 128), (393216, 1, 3072, 24))
    assert_size_stride(convolution_1, (8, 32, 128, 128), (524288, 1, 4096, 32))
    assert_size_stride(squeeze_4, (32, ), (1, ))
    assert_size_stride(mul_15, (8, 32, 128, 128), (524288, 1, 4096, 32))
    assert_size_stride(convolution_2, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_7, (64, ), (1, ))
    assert_size_stride(mul_23, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_3, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_10, (64, ), (1, ))
    assert_size_stride(mul_31, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_4, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_13, (64, ), (1, ))
    assert_size_stride(add_24, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(mean, (8, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(relu, (8, 8, 1, 1), (8, 1, 8, 8))
    assert_size_stride(convolution_6, (8, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(mul_40, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_7, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(squeeze_16, (256, ), (1, ))
    assert_size_stride(convolution_8, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(squeeze_19, (256, ), (1, ))
    assert_size_stride(mul_55, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(convolution_9, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_22, (64, ), (1, ))
    assert_size_stride(mul_63, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_10, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(squeeze_25, (64, ), (1, ))
    assert_size_stride(add_45, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(mean_1, (8, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(relu_1, (8, 8, 1, 1), (8, 1, 8, 8))
    assert_size_stride(convolution_12, (8, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(mul_72, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(convolution_13, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(squeeze_28, (256, ), (1, ))
    assert_size_stride(mul_80, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(convolution_14, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(squeeze_31, (128, ), (1, ))
    assert_size_stride(mul_88, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(convolution_15, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_34, (128, ), (1, ))
    assert_size_stride(add_61, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(mean_2, (8, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(relu_2, (8, 8, 1, 1), (8, 1, 8, 8))
    assert_size_stride(convolution_17, (8, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(mul_97, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_18, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(squeeze_37, (512, ), (1, ))
    assert_size_stride(convolution_19, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(squeeze_40, (512, ), (1, ))
    assert_size_stride(mul_112, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(convolution_20, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_43, (128, ), (1, ))
    assert_size_stride(mul_120, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_21, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_46, (128, ), (1, ))
    assert_size_stride(add_82, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(mean_3, (8, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(relu_3, (8, 8, 1, 1), (8, 1, 8, 8))
    assert_size_stride(convolution_23, (8, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(mul_129, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_24, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(squeeze_49, (512, ), (1, ))
    assert_size_stride(mul_137, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(convolution_25, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(squeeze_52, (128, ), (1, ))
    assert_size_stride(mul_145, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(view_7, (32768, 32), (32, 1))
    assert_size_stride(view_13, (32768, 32), (32, 1))
    assert_size_stride(bmm_1, (32, 1024, 32), (32768, 32, 1))
    assert_size_stride(squeeze_55, (128, ), (1, ))
    assert_size_stride(mul_154, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(convolution_27, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(squeeze_58, (512, ), (1, ))
    assert_size_stride(mul_162, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(convolution_28, (8, 256, 32, 32), (262144, 1, 8192, 256))
    assert_size_stride(squeeze_61, (256, ), (1, ))
    assert_size_stride(mul_170, (8, 256, 32, 32), (262144, 1, 8192, 256))
    assert_size_stride(convolution_29, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_64, (256, ), (1, ))
    assert_size_stride(add_116, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(mean_4, (8, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(relu_4, (8, 16, 1, 1), (16, 1, 16, 16))
    assert_size_stride(convolution_31, (8, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(mul_179, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_32, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(squeeze_67, (1024, ), (1, ))
    assert_size_stride(convolution_33, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(squeeze_70, (1024, ), (1, ))
    assert_size_stride(mul_194, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(convolution_34, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_73, (256, ), (1, ))
    assert_size_stride(mul_202, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_35, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_76, (256, ), (1, ))
    assert_size_stride(add_137, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(mean_5, (8, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(relu_5, (8, 16, 1, 1), (16, 1, 16, 16))
    assert_size_stride(convolution_37, (8, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(mul_211, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_38, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(squeeze_79, (1024, ), (1, ))
    assert_size_stride(mul_219, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(convolution_39, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(squeeze_82, (256, ), (1, ))
    assert_size_stride(mul_227, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(view_31, (8192, 64), (64, 1))
    assert_size_stride(view_37, (8192, 64), (64, 1))
    assert_size_stride(bmm_3, (32, 256, 64), (16384, 64, 1))
    assert_size_stride(squeeze_85, (256, ), (1, ))
    assert_size_stride(mul_236, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(convolution_41, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(squeeze_88, (1024, ), (1, ))
    assert_size_stride(mul_244, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(convolution_42, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(squeeze_91, (512, ), (1, ))
    assert_size_stride(mul_252, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(view_55, (8192, 128), (128, 1))
    assert_size_stride(view_61, (8192, 128), (128, 1))
    assert_size_stride(view_71, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(avg_pool2d, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(squeeze_94, (512, ), (1, ))
    assert_size_stride(mul_261, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(convolution_44, (8, 1536, 8, 8), (98304, 1, 12288, 1536))
    assert_size_stride(squeeze_97, (1536, ), (1, ))
    assert_size_stride(convolution_45, (8, 1536, 8, 8), (98304, 1, 12288, 1536))
    assert_size_stride(squeeze_100, (1536, ), (1, ))
    assert_size_stride(mul_276, (8, 1536, 8, 8), (98304, 1, 12288, 1536))
    assert_size_stride(convolution_46, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(squeeze_103, (512, ), (1, ))
    assert_size_stride(mul_284, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(view_79, (2048, 128), (128, 1))
    assert_size_stride(view_85, (2048, 128), (128, 1))
    assert_size_stride(bmm_7, (32, 64, 128), (8192, 128, 1))
    assert_size_stride(squeeze_106, (512, ), (1, ))
    assert_size_stride(mul_293, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(convolution_48, (8, 1536, 8, 8), (98304, 1, 12288, 1536))
    assert_size_stride(squeeze_109, (1536, ), (1, ))
    assert_size_stride(mul_301, (8, 1536, 8, 8), (98304, 1, 12288, 1536))
    assert_size_stride(convolution_49, (8, 1280, 8, 8), (81920, 1, 10240, 1280))
    assert_size_stride(squeeze_112, (1280, ), (1, ))
    assert_size_stride(clone_62, (8, 1280), (1280, 1))
    assert_size_stride(permute_33, (1000, 1280), (1280, 1))
    assert_size_stride(mul_311, (8, 1280, 8, 8), (81920, 1, 10240, 1280))
    assert_size_stride(unsqueeze_154, (1, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(mul_323, (8, 1536, 8, 8), (98304, 1, 12288, 1536))
    assert_size_stride(unsqueeze_166, (1, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(mul_335, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(unsqueeze_178, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(permute_41, (32, 64, 64), (4096, 1, 64))
    assert_size_stride(permute_42, (32, 128, 64), (8192, 64, 1))
    assert_size_stride(alias_16, (32, 64, 64), (4096, 64, 1))
    assert_size_stride(permute_46, (15, 128), (128, 1))
    assert_size_stride(permute_52, (15, 128), (128, 1))
    assert_size_stride(permute_54, (32, 128, 64), (8192, 64, 1))
    assert_size_stride(permute_55, (32, 64, 128), (8192, 1, 64))
    assert_size_stride(mul_350, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(unsqueeze_190, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(mul_362, (8, 1536, 8, 8), (98304, 1, 12288, 1536))
    assert_size_stride(unsqueeze_202, (1, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(unsqueeze_214, (1, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(mul_383, (8, 512, 8, 8), (32768, 1, 4096, 512))
    assert_size_stride(unsqueeze_226, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(permute_62, (32, 256, 256), (65536, 1, 256))
    assert_size_stride(permute_63, (32, 128, 256), (32768, 256, 1))
    assert_size_stride(alias_17, (32, 256, 256), (65536, 256, 1))
    assert_size_stride(permute_67, (31, 128), (128, 1))
    assert_size_stride(permute_73, (31, 128), (128, 1))
    assert_size_stride(permute_75, (32, 128, 256), (32768, 256, 1))
    assert_size_stride(permute_76, (32, 256, 128), (32768, 1, 256))
    assert_size_stride(mul_398, (8, 512, 16, 16), (131072, 1, 8192, 512))
    assert_size_stride(unsqueeze_238, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(mul_410, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(unsqueeze_250, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(mul_422, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(unsqueeze_262, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(permute_83, (32, 256, 256), (65536, 1, 256))
    assert_size_stride(permute_84, (32, 64, 256), (16384, 256, 1))
    assert_size_stride(alias_18, (32, 256, 256), (65536, 256, 1))
    assert_size_stride(permute_88, (31, 64), (64, 1))
    assert_size_stride(permute_94, (31, 64), (64, 1))
    assert_size_stride(permute_96, (32, 64, 256), (16384, 256, 1))
    assert_size_stride(permute_97, (32, 256, 64), (16384, 1, 256))
    assert_size_stride(mul_437, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(unsqueeze_274, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(mul_449, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(unsqueeze_286, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_298, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(mul_477, (8, 256, 16, 16), (65536, 1, 4096, 256))
    assert_size_stride(unsqueeze_310, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(mul_489, (8, 1024, 16, 16), (262144, 1, 16384, 1024))
    assert_size_stride(unsqueeze_322, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_334, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(unsqueeze_346, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(mul_526, (8, 256, 32, 32), (262144, 1, 8192, 256))
    assert_size_stride(unsqueeze_358, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(mul_538, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(unsqueeze_370, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(mul_550, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(unsqueeze_382, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(permute_110, (32, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_111, (32, 32, 1024), (32768, 1024, 1))
    assert_size_stride(alias_27, (32, 1024, 1024), (1048576, 1024, 1))
    assert_size_stride(permute_115, (63, 32), (32, 1))
    assert_size_stride(permute_121, (63, 32), (32, 1))
    assert_size_stride(permute_123, (32, 32, 1024), (32768, 1024, 1))
    assert_size_stride(permute_124, (32, 1024, 32), (32768, 1, 1024))
    assert_size_stride(mul_565, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(unsqueeze_394, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(mul_577, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(unsqueeze_406, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_418, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(mul_605, (8, 128, 32, 32), (131072, 1, 4096, 128))
    assert_size_stride(unsqueeze_430, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(mul_617, (8, 512, 32, 32), (524288, 1, 16384, 512))
    assert_size_stride(unsqueeze_442, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_454, (1, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(unsqueeze_466, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(mul_654, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(unsqueeze_478, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(mul_666, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(unsqueeze_490, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_502, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(mul_694, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(unsqueeze_514, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(mul_706, (8, 256, 64, 64), (1048576, 1, 16384, 256))
    assert_size_stride(unsqueeze_526, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_538, (1, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(unsqueeze_550, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(mul_743, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(unsqueeze_562, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(mul_755, (8, 64, 64, 64), (262144, 1, 4096, 64))
    assert_size_stride(unsqueeze_574, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(mul_767, (8, 32, 128, 128), (524288, 1, 4096, 32))
    assert_size_stride(unsqueeze_586, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(mul_779, (8, 24, 128, 128), (393216, 1, 3072, 24))
    assert_size_stride(unsqueeze_598, (1, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_33, out=buf0)
        del permute_33
        buf1 = empty((1000, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_62, out=buf1)
        del clone_62
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf3 = empty_strided((1280, 4), (1, 1280), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((1280, 4), (1, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_div_mul_native_batch_norm_backward_1.run(buf0, mul_311, convolution_49, unsqueeze_154, buf3, buf5, 5120, 128, grid=grid(5120), stream=stream0)
        buf4 = empty((1280, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_div_mul_native_batch_norm_backward_2.run(buf3, buf4, 1280, 4, grid=grid(1280), stream=stream0)
        del buf3
        buf6 = empty((1280, ), device='cuda', dtype=torch.float32)
        buf7 = empty((1280, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_div_mul_native_batch_norm_backward_3.run(buf5, squeeze_112, buf6, buf7, 1280, 4, grid=grid(1280), stream=stream0)
        del buf5
        buf8 = empty((8, 1280, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_div_mul_native_batch_norm_backward_4.run(buf0, mul_311, convolution_49, unsqueeze_154, buf6, squeeze_112, buf4, primals_83, buf8, 512, 1280, grid=grid(512, 1280), stream=stream0)
        del buf0
        del buf6
        del convolution_49
        del mul_311
        del primals_83
        del squeeze_112
        del unsqueeze_154
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward]
        buf9 = aten.convolution_backward(buf8, mul_301, primals_146, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf8
        del mul_301
        del primals_146
        buf10 = buf9[0]
        buf11 = buf9[1]
        del buf9
        buf12 = empty_strided((1536, 4), (1, 1536), device='cuda', dtype=torch.float32)
        buf14 = empty_strided((1536, 4), (1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_5.run(buf10, mul_323, convolution_48, unsqueeze_166, buf12, buf14, 6144, 128, grid=grid(6144), stream=stream0)
        buf13 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_6.run(buf12, buf13, 1536, 4, grid=grid(1536), stream=stream0)
        buf15 = empty((1536, ), device='cuda', dtype=torch.float32)
        buf16 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_7.run(buf14, squeeze_109, buf15, buf16, 1536, 4, grid=grid(1536), stream=stream0)
        buf17 = empty((8, 1536, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_8.run(buf10, mul_323, convolution_48, unsqueeze_166, buf15, squeeze_109, buf13, primals_81, buf17, 12288, 64, grid=grid(12288, 64), stream=stream0)
        del convolution_48
        del primals_81
        del squeeze_109
        del unsqueeze_166
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf18 = aten.convolution_backward(buf17, mul_293, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_293
        del primals_145
        buf19 = buf18[0]
        buf20 = buf18[1]
        del buf18
        buf21 = empty_strided((512, 4), (1, 512), device='cuda', dtype=torch.float32)
        buf23 = empty_strided((512, 4), (1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_9.run(buf19, mul_335, bmm_7, unsqueeze_178, buf21, buf23, 2048, 128, grid=grid(2048), stream=stream0)
        buf22 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_10.run(buf21, buf22, 512, 4, grid=grid(512), stream=stream0)
        buf24 = empty((512, ), device='cuda', dtype=torch.float32)
        buf25 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_11.run(buf23, squeeze_106, buf24, buf25, 512, 4, grid=grid(512), stream=stream0)
        buf26 = buf19; del buf19  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_mul_native_batch_norm_backward_12.run(buf26, mul_335, bmm_7, unsqueeze_178, buf24, squeeze_106, buf22, primals_79, 4096, 64, grid=grid(4096, 64), stream=stream0)
        del bmm_7
        del mul_335
        del primals_79
        del squeeze_106
        del unsqueeze_178
        buf27 = empty((32, 64, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_41, reinterpret_tensor(buf26, (32, 64, 128), (8192, 1, 64), 0), out=buf27)
        del permute_41
        buf28 = empty((32, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf26, (32, 64, 128), (8192, 1, 64), 0), permute_42, out=buf28)
        del permute_42
        buf29 = reinterpret_tensor(buf23, (32, 64, 1), (64, 1, 2048), 0); del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_13.run(buf28, alias_16, buf29, 2048, 64, grid=grid(2048), stream=stream0)
        buf30 = empty((32, 8, 1, 8, 8), device='cuda', dtype=torch.float32)
        buf34 = empty((32, 8, 1, 8, 8), device='cuda', dtype=torch.float32)
        buf38 = empty((32, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_14.run(buf28, alias_16, buf29, buf30, buf34, buf38, 16384, 8, grid=grid(16384), stream=stream0)
        del alias_16
        buf31 = empty((2048, 15), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf30, buf31, 30720, grid=grid(30720), stream=stream0)
        buf32 = empty((15, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf31, (15, 2048), (1, 15), 0), view_85, out=buf32)
        del view_85
        buf33 = reinterpret_tensor(buf26, (2048, 128), (128, 1), 0); del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf31, permute_46, out=buf33)
        del permute_46
        buf35 = buf31; del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf34, buf35, 30720, grid=grid(30720), stream=stream0)
        buf36 = empty((15, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf35, (15, 2048), (1, 15), 0), view_79, out=buf36)
        del view_79
        buf37 = empty((2048, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf35, permute_52, out=buf37)
        del buf35
        del permute_52
        buf39 = empty((32, 128, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_54, buf38, out=buf39)
        del permute_54
        buf40 = empty((32, 64, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf38, permute_55, out=buf40)
        del permute_55
        buf41 = buf17; del buf17  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward]
        triton_poi_fused_cat_convolution_backward_16.run(buf33, buf37, buf40, buf39, buf27, buf41, 512, 1536, grid=grid(512, 1536), stream=stream0)
        del buf27
        del buf33
        del buf37
        del buf39
        del buf40
        # Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward]
        buf42 = aten.convolution_backward(buf41, mul_284, primals_144, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_284
        del primals_144
        buf43 = buf42[0]
        buf44 = buf42[1]
        del buf42
        buf45 = reinterpret_tensor(buf29, (512, 4), (1, 512), 0); del buf29  # reuse
        buf47 = buf21; del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_17.run(buf43, mul_350, convolution_46, unsqueeze_190, buf45, buf47, 2048, 128, grid=grid(2048), stream=stream0)
        buf46 = buf24; del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_10.run(buf45, buf46, 512, 4, grid=grid(512), stream=stream0)
        buf48 = empty((512, ), device='cuda', dtype=torch.float32)
        buf49 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_11.run(buf47, squeeze_103, buf48, buf49, 512, 4, grid=grid(512), stream=stream0)
        buf50 = buf43; del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_18.run(buf50, mul_350, convolution_46, unsqueeze_190, buf48, squeeze_103, buf46, primals_75, 4096, 64, grid=grid(4096, 64), stream=stream0)
        del convolution_46
        del mul_350
        del primals_75
        del squeeze_103
        del unsqueeze_190
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf51 = aten.convolution_backward(buf50, mul_276, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_276
        del primals_143
        buf52 = buf51[0]
        buf53 = buf51[1]
        del buf51
        buf54 = buf14; del buf14  # reuse
        buf56 = buf12; del buf12  # reuse
        buf64 = empty_strided((1536, 4), (1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_mul_native_batch_norm_backward_19.run(buf10, mul_323, buf52, mul_362, convolution_45, unsqueeze_202, convolution_44, unsqueeze_214, buf54, buf56, buf64, 6144, 128, grid=grid(6144), stream=stream0)
        buf55 = buf15; del buf15  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_6.run(buf54, buf55, 1536, 4, grid=grid(1536), stream=stream0)
        del buf54
        buf57 = empty((1536, ), device='cuda', dtype=torch.float32)
        buf59 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_7.run(buf56, squeeze_100, buf57, buf59, 1536, 4, grid=grid(1536), stream=stream0)
        del buf56
        buf65 = empty((1536, ), device='cuda', dtype=torch.float32)
        buf67 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_7.run(buf64, squeeze_97, buf65, buf67, 1536, 4, grid=grid(1536), stream=stream0)
        del buf64
        buf58 = reinterpret_tensor(buf41, (8, 1536, 8, 8), (98304, 1, 12288, 1536), 0); del buf41  # reuse
        buf66 = empty_strided((8, 1536, 8, 8), (98304, 1, 12288, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_mul_native_batch_norm_backward_20.run(buf10, mul_323, buf52, mul_362, convolution_45, unsqueeze_202, buf57, squeeze_100, buf55, convolution_44, unsqueeze_214, buf65, squeeze_97, buf58, buf66, 512, 1536, grid=grid(512, 1536), stream=stream0)
        del buf10
        del buf57
        del buf65
        del convolution_44
        del convolution_45
        del mul_323
        del mul_362
        del unsqueeze_202
        del unsqueeze_214
        buf60 = buf52; del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_21.run(buf58, squeeze_100, primals_73, buf60, 12288, 64, grid=grid(12288, 64), stream=stream0)
        del buf58
        del primals_73
        del squeeze_100
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf61 = aten.convolution_backward(buf60, mul_244, primals_142, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_142
        buf62 = buf61[0]
        buf63 = buf61[1]
        del buf61
        buf68 = buf60; del buf60  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_21.run(buf66, squeeze_97, primals_71, buf68, 12288, 64, grid=grid(12288, 64), stream=stream0)
        del buf66
        del primals_71
        del squeeze_97
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf69 = aten.convolution_backward(buf68, mul_261, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf68
        del mul_261
        del primals_141
        buf70 = buf69[0]
        buf71 = buf69[1]
        del buf69
        buf72 = buf47; del buf47  # reuse
        buf74 = buf45; del buf45  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_17.run(buf70, mul_383, avg_pool2d, unsqueeze_226, buf72, buf74, 2048, 128, grid=grid(2048), stream=stream0)
        buf73 = buf48; del buf48  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_10.run(buf72, buf73, 512, 4, grid=grid(512), stream=stream0)
        del buf72
        buf75 = empty((512, ), device='cuda', dtype=torch.float32)
        buf76 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_11.run(buf74, squeeze_94, buf75, buf76, 512, 4, grid=grid(512), stream=stream0)
        buf77 = reinterpret_tensor(buf50, (8, 512, 8, 8), (32768, 1, 4096, 512), 0); del buf50  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_mul_native_batch_norm_backward_22.run(buf70, mul_383, avg_pool2d, unsqueeze_226, buf75, squeeze_94, buf73, primals_69, buf77, 512, 512, grid=grid(512, 512), stream=stream0)
        del avg_pool2d
        del buf70
        del mul_383
        del primals_69
        del squeeze_94
        del unsqueeze_226
        buf78 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.avg_pool2d_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_avg_pool2d_backward_mul_native_batch_norm_backward_23.run(buf77, buf78, 1048576, grid=grid(1048576), stream=stream0)
        del buf77
        buf79 = empty((32, 256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_62, reinterpret_tensor(buf78, (32, 256, 128), (32768, 1, 256), 0), out=buf79)
        del permute_62
        buf80 = empty((32, 256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf78, (32, 256, 128), (32768, 1, 256), 0), permute_63, out=buf80)
        del permute_63
        buf81 = empty_strided((32, 256, 1), (256, 1, 8192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_24.run(buf80, alias_17, buf81, 8192, 256, grid=grid(8192), stream=stream0)
        buf82 = reinterpret_tensor(buf38, (32, 16, 1, 16, 16), (4096, 256, 256, 16, 1), 0); del buf38  # reuse
        buf86 = reinterpret_tensor(buf28, (32, 16, 1, 16, 16), (4096, 256, 256, 16, 1), 0); del buf28  # reuse
        buf90 = empty((32, 256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_25.run(buf80, alias_17, buf81, buf82, buf86, buf90, 131072, 16, grid=grid(131072), stream=stream0)
        del alias_17
        buf83 = empty((8192, 31), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_26.run(buf82, buf83, 253952, grid=grid(253952), stream=stream0)
        buf84 = empty((31, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf83, (31, 8192), (1, 31), 0), view_61, out=buf84)
        del view_61
        buf85 = reinterpret_tensor(buf78, (8192, 128), (128, 1), 0); del buf78  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf83, permute_67, out=buf85)
        del permute_67
        buf87 = buf83; del buf83  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_26.run(buf86, buf87, 253952, grid=grid(253952), stream=stream0)
        buf88 = empty((31, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf87, (31, 8192), (1, 31), 0), view_55, out=buf88)
        del view_55
        buf89 = empty((8192, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf87, permute_73, out=buf89)
        del permute_73
        buf91 = empty((32, 128, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_75, buf90, out=buf91)
        del permute_75
        buf92 = empty((32, 256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf90, permute_76, out=buf92)
        del permute_76
        buf93 = empty((8, 1536, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward]
        triton_poi_fused_cat_convolution_backward_27.run(buf85, buf89, buf92, buf91, buf79, buf93, 2048, 1536, grid=grid(2048, 1536), stream=stream0)
        del buf79
        del buf85
        # Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward]
        buf94 = aten.convolution_backward(buf93, mul_252, primals_140, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_252
        del primals_140
        buf95 = buf94[0]
        buf96 = buf94[1]
        del buf94
        buf97 = reinterpret_tensor(buf81, (512, 16), (16, 1), 0); del buf81  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_28.run(buf95, mul_398, buf97, 8192, 128, grid=grid(8192), stream=stream0)
        buf98 = buf75; del buf75  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_29.run(buf97, buf98, 512, 16, grid=grid(512), stream=stream0)
        buf99 = reinterpret_tensor(buf97, (512, 16), (1, 512), 0); del buf97  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_30.run(buf95, mul_398, convolution_42, unsqueeze_238, buf99, 8192, 128, grid=grid(8192), stream=stream0)
        buf100 = empty((512, ), device='cuda', dtype=torch.float32)
        buf101 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_31.run(buf99, squeeze_91, buf100, buf101, 512, 16, grid=grid(512), stream=stream0)
        buf102 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_32.run(buf102, mul_398, convolution_42, unsqueeze_238, buf100, squeeze_91, buf98, primals_65, 4096, 256, grid=grid(4096, 256), stream=stream0)
        del convolution_42
        del mul_398
        del primals_65
        del squeeze_91
        del unsqueeze_238
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf103 = aten.convolution_backward(buf102, mul_244, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_244
        del primals_139
        buf104 = buf103[0]
        buf105 = buf103[1]
        del buf103
        buf106 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_mul_native_batch_norm_backward_33.run(buf62, buf104, mul_410, buf106, 1024, 2048, grid=grid(1024), stream=stream0)
        buf107 = reinterpret_tensor(buf34, (1024, 16), (16, 1), 0); del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_mul_native_batch_norm_backward_34.run(buf62, buf104, mul_410, convolution_41, unsqueeze_250, buf107, 16384, 128, grid=grid(16384), stream=stream0)
        buf108 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf110 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_mul_native_batch_norm_backward_35.run(buf107, squeeze_88, buf108, buf110, 1024, 16, grid=grid(1024), stream=stream0)
        buf109 = reinterpret_tensor(buf90, (8, 1024, 16, 16), (262144, 256, 16, 1), 0); del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_mul_native_batch_norm_backward_36.run(buf62, buf104, mul_410, convolution_41, unsqueeze_250, buf108, squeeze_88, buf106, primals_63, buf109, 8192, 256, grid=grid(8192, 256), stream=stream0)
        del convolution_41
        del primals_63
        del squeeze_88
        del unsqueeze_250
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf111 = aten.convolution_backward(buf109, mul_236, primals_138, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_236
        del primals_138
        buf112 = buf111[0]
        buf113 = buf111[1]
        del buf111
        buf114 = empty((256, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_37.run(buf112, mul_422, buf114, 4096, 128, grid=grid(4096), stream=stream0)
        buf115 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_38.run(buf114, buf115, 256, 16, grid=grid(256), stream=stream0)
        buf116 = reinterpret_tensor(buf114, (256, 16), (1, 256), 0); del buf114  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_39.run(buf112, mul_422, bmm_3, unsqueeze_262, buf116, 4096, 128, grid=grid(4096), stream=stream0)
        buf117 = empty((256, ), device='cuda', dtype=torch.float32)
        buf118 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_40.run(buf116, squeeze_85, buf117, buf118, 256, 16, grid=grid(256), stream=stream0)
        buf119 = buf112; del buf112  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_mul_native_batch_norm_backward_41.run(buf119, mul_422, bmm_3, unsqueeze_262, buf117, squeeze_85, buf115, primals_61, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del bmm_3
        del mul_422
        del primals_61
        del squeeze_85
        del unsqueeze_262
        buf120 = empty((32, 256, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_83, reinterpret_tensor(buf119, (32, 256, 64), (16384, 1, 256), 0), out=buf120)
        del permute_83
        buf121 = reinterpret_tensor(buf109, (32, 256, 256), (65536, 256, 1), 0); del buf109  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf119, (32, 256, 64), (16384, 1, 256), 0), permute_84, out=buf121)
        del permute_84
        buf122 = reinterpret_tensor(buf99, (32, 256, 1), (256, 1, 8192), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_24.run(buf121, alias_18, buf122, 8192, 256, grid=grid(8192), stream=stream0)
        buf123 = buf86; del buf86  # reuse
        buf127 = buf82; del buf82  # reuse
        buf131 = buf80; del buf80  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_42.run(buf121, alias_18, buf122, buf123, buf127, buf131, 131072, 16, grid=grid(131072), stream=stream0)
        del alias_18
        del buf121
        buf124 = buf87; del buf87  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_26.run(buf123, buf124, 253952, grid=grid(253952), stream=stream0)
        del buf123
        buf125 = empty((31, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf124, (31, 8192), (1, 31), 0), view_37, out=buf125)
        del view_37
        buf126 = reinterpret_tensor(buf119, (8192, 64), (64, 1), 0); del buf119  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf124, permute_88, out=buf126)
        del permute_88
        buf128 = buf124; del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_26.run(buf127, buf128, 253952, grid=grid(253952), stream=stream0)
        del buf127
        buf129 = empty((31, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf128, (31, 8192), (1, 31), 0), view_31, out=buf129)
        del view_31
        buf130 = empty((8192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf128, permute_94, out=buf130)
        del buf128
        del permute_94
        buf132 = empty((32, 64, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_96, buf131, out=buf132)
        del permute_96
        buf133 = empty((32, 256, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf131, permute_97, out=buf133)
        del buf131
        del permute_97
        buf134 = empty((8, 768, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward]
        triton_poi_fused_cat_convolution_backward_43.run(buf126, buf130, buf133, buf132, buf120, buf134, 2048, 768, grid=grid(2048, 768), stream=stream0)
        del buf120
        del buf126
        del buf130
        del buf132
        del buf133
        # Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward]
        buf135 = aten.convolution_backward(buf134, mul_227, primals_137, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf134
        del mul_227
        del primals_137
        buf136 = buf135[0]
        buf137 = buf135[1]
        del buf135
        buf138 = reinterpret_tensor(buf116, (256, 16), (16, 1), 0); del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_37.run(buf136, mul_437, buf138, 4096, 128, grid=grid(4096), stream=stream0)
        buf139 = buf117; del buf117  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_38.run(buf138, buf139, 256, 16, grid=grid(256), stream=stream0)
        buf140 = reinterpret_tensor(buf138, (256, 16), (1, 256), 0); del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_44.run(buf136, mul_437, convolution_39, unsqueeze_274, buf140, 4096, 128, grid=grid(4096), stream=stream0)
        buf141 = empty((256, ), device='cuda', dtype=torch.float32)
        buf142 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_40.run(buf140, squeeze_82, buf141, buf142, 256, 16, grid=grid(256), stream=stream0)
        buf143 = buf136; del buf136  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_45.run(buf143, mul_437, convolution_39, unsqueeze_274, buf141, squeeze_82, buf139, primals_57, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del convolution_39
        del mul_437
        del primals_57
        del squeeze_82
        del unsqueeze_274
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf144 = aten.convolution_backward(buf143, mul_219, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_219
        del primals_136
        buf145 = buf144[0]
        buf146 = buf144[1]
        del buf144
        buf147 = buf104; del buf104  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_46.run(buf147, buf62, mul_410, buf145, mul_449, 8192, 256, grid=grid(8192, 256), stream=stream0)
        del mul_410
        del mul_449
        buf148 = buf108; del buf108  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_47.run(buf147, buf148, 1024, 2048, grid=grid(1024), stream=stream0)
        buf149 = buf107; del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_48.run(buf147, convolution_38, unsqueeze_286, buf149, 16384, 128, grid=grid(16384), stream=stream0)
        buf150 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf151 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_mul_native_batch_norm_backward_35.run(buf149, squeeze_79, buf150, buf151, 1024, 16, grid=grid(1024), stream=stream0)
        buf152 = buf62; del buf62  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_49.run(buf147, convolution_38, unsqueeze_286, buf150, squeeze_79, buf148, primals_55, buf152, 8192, 256, grid=grid(8192, 256), stream=stream0)
        del convolution_38
        del primals_55
        del squeeze_79
        del unsqueeze_286
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf153 = aten.convolution_backward(buf152, mul_211, primals_135, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_211
        del primals_135
        buf154 = buf153[0]
        buf155 = buf153[1]
        del buf153
        buf156 = reinterpret_tensor(buf140, (8, 256, 1, 1, 2), (512, 2, 4096, 4096, 1), 0); del buf140  # reuse
        # Source Nodes: [x_186], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_50.run(buf154, add_137, buf156, 4096, 128, grid=grid(4096), stream=stream0)
        buf157 = reinterpret_tensor(buf74, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf74  # reuse
        buf158 = reinterpret_tensor(buf157, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf157  # reuse
        # Source Nodes: [sigmoid_5, x_186], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_51.run(buf158, buf156, convolution_37, 2048, 2, grid=grid(2048), stream=stream0)
        buf159 = buf141; del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_52.run(buf158, buf159, 256, 8, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf160 = aten.convolution_backward(buf158, relu_5, primals_133, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf158
        del primals_133
        buf161 = buf160[0]
        buf162 = buf160[1]
        del buf160
        buf163 = buf161; del buf161  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_53.run(buf163, relu_5, 128, grid=grid(128), stream=stream0)
        del relu_5
        buf164 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_54.run(buf163, buf164, 16, 8, grid=grid(16), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf165 = aten.convolution_backward(buf163, mean_5, primals_131, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_5
        del primals_131
        buf166 = buf165[0]
        buf167 = buf165[1]
        del buf165
        buf168 = reinterpret_tensor(buf156, (256, 16), (16, 1), 0); del buf156  # reuse
        # Source Nodes: [sigmoid_5], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_55.run(buf154, convolution_37, buf166, add_137, buf168, 4096, 128, grid=grid(4096), stream=stream0)
        buf169 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_5], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_mul_native_batch_norm_backward_38.run(buf168, buf169, 256, 16, grid=grid(256), stream=stream0)
        buf170 = reinterpret_tensor(buf168, (256, 16), (1, 256), 0); del buf168  # reuse
        # Source Nodes: [sigmoid_5], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_56.run(buf154, convolution_37, buf166, add_137, convolution_35, unsqueeze_298, buf170, 4096, 128, grid=grid(4096), stream=stream0)
        buf171 = empty((256, ), device='cuda', dtype=torch.float32)
        buf173 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_5], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_mul_native_batch_norm_backward_40.run(buf170, squeeze_76, buf171, buf173, 256, 16, grid=grid(256), stream=stream0)
        buf172 = reinterpret_tensor(buf143, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf143  # reuse
        # Source Nodes: [sigmoid_5], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_57.run(buf154, convolution_37, buf166, add_137, convolution_35, unsqueeze_298, buf171, squeeze_76, buf169, buf172, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del add_137
        del convolution_35
        del convolution_37
        del unsqueeze_298
        buf174 = buf154; del buf154  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_58.run(buf172, squeeze_76, primals_53, buf174, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del buf172
        del primals_53
        del squeeze_76
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf175 = aten.convolution_backward(buf174, mul_202, primals_130, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf174
        del mul_202
        del primals_130
        buf176 = buf175[0]
        buf177 = buf175[1]
        del buf175
        buf178 = reinterpret_tensor(buf170, (256, 16), (16, 1), 0); del buf170  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_37.run(buf176, mul_477, buf178, 4096, 128, grid=grid(4096), stream=stream0)
        buf179 = buf171; del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_38.run(buf178, buf179, 256, 16, grid=grid(256), stream=stream0)
        buf180 = reinterpret_tensor(buf178, (256, 16), (1, 256), 0); del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_44.run(buf176, mul_477, convolution_34, unsqueeze_310, buf180, 4096, 128, grid=grid(4096), stream=stream0)
        buf181 = empty((256, ), device='cuda', dtype=torch.float32)
        buf182 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_40.run(buf180, squeeze_73, buf181, buf182, 256, 16, grid=grid(256), stream=stream0)
        buf183 = buf176; del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_45.run(buf183, mul_477, convolution_34, unsqueeze_310, buf181, squeeze_73, buf179, primals_51, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del convolution_34
        del mul_477
        del primals_51
        del squeeze_73
        del unsqueeze_310
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf184 = aten.convolution_backward(buf183, mul_194, primals_129, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_194
        del primals_129
        buf185 = buf184[0]
        buf186 = buf184[1]
        del buf184
        buf187 = buf150; del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_mul_native_batch_norm_backward_33.run(buf147, buf185, mul_489, buf187, 1024, 2048, grid=grid(1024), stream=stream0)
        buf188 = buf149; del buf149  # reuse
        buf195 = reinterpret_tensor(buf30, (1024, 16), (16, 1), 0); del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_mul_native_batch_norm_backward_59.run(buf147, buf185, mul_489, convolution_33, unsqueeze_322, convolution_32, unsqueeze_334, buf188, buf195, 16384, 128, grid=grid(16384), stream=stream0)
        buf189 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf191 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_mul_native_batch_norm_backward_35.run(buf188, squeeze_70, buf189, buf191, 1024, 16, grid=grid(1024), stream=stream0)
        del buf188
        buf196 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf198 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_mul_native_batch_norm_backward_35.run(buf195, squeeze_67, buf196, buf198, 1024, 16, grid=grid(1024), stream=stream0)
        buf190 = buf152; del buf152  # reuse
        buf197 = buf145; del buf145  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_mul_native_batch_norm_backward_60.run(buf147, buf185, mul_489, convolution_33, unsqueeze_322, buf189, squeeze_70, buf187, primals_49, convolution_32, unsqueeze_334, buf196, squeeze_67, primals_47, buf190, buf197, 8192, 256, grid=grid(8192, 256), stream=stream0)
        del buf147
        del buf185
        del buf189
        del convolution_32
        del convolution_33
        del mul_489
        del primals_47
        del primals_49
        del squeeze_67
        del squeeze_70
        del unsqueeze_322
        del unsqueeze_334
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf192 = aten.convolution_backward(buf190, mul_162, primals_128, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf190
        del primals_128
        buf193 = buf192[0]
        buf194 = buf192[1]
        del buf192
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf199 = aten.convolution_backward(buf197, mul_179, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf197
        del mul_179
        del primals_127
        buf200 = buf199[0]
        buf201 = buf199[1]
        del buf199
        buf202 = reinterpret_tensor(buf180, (8, 256, 1, 1, 2), (512, 2, 4096, 4096, 1), 0); del buf180  # reuse
        # Source Nodes: [x_158], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_50.run(buf200, add_116, buf202, 4096, 128, grid=grid(4096), stream=stream0)
        buf203 = reinterpret_tensor(buf166, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf166  # reuse
        buf204 = reinterpret_tensor(buf203, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf203  # reuse
        # Source Nodes: [sigmoid_4, x_158], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_51.run(buf204, buf202, convolution_31, 2048, 2, grid=grid(2048), stream=stream0)
        buf205 = buf181; del buf181  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_52.run(buf204, buf205, 256, 8, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf206 = aten.convolution_backward(buf204, relu_4, primals_125, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf204
        del primals_125
        buf207 = buf206[0]
        buf208 = buf206[1]
        del buf206
        buf209 = buf207; del buf207  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_53.run(buf209, relu_4, 128, grid=grid(128), stream=stream0)
        del relu_4
        buf210 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_54.run(buf209, buf210, 16, 8, grid=grid(16), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf211 = aten.convolution_backward(buf209, mean_4, primals_123, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_4
        del primals_123
        buf212 = buf211[0]
        buf213 = buf211[1]
        del buf211
        buf214 = reinterpret_tensor(buf202, (256, 16), (16, 1), 0); del buf202  # reuse
        # Source Nodes: [sigmoid_4], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_55.run(buf200, convolution_31, buf212, add_116, buf214, 4096, 128, grid=grid(4096), stream=stream0)
        buf215 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_4], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_mul_native_batch_norm_backward_38.run(buf214, buf215, 256, 16, grid=grid(256), stream=stream0)
        buf216 = reinterpret_tensor(buf214, (256, 16), (1, 256), 0); del buf214  # reuse
        # Source Nodes: [sigmoid_4], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_56.run(buf200, convolution_31, buf212, add_116, convolution_29, unsqueeze_346, buf216, 4096, 128, grid=grid(4096), stream=stream0)
        buf217 = empty((256, ), device='cuda', dtype=torch.float32)
        buf219 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_4], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_mul_native_batch_norm_backward_40.run(buf216, squeeze_64, buf217, buf219, 256, 16, grid=grid(256), stream=stream0)
        del buf216
        buf218 = reinterpret_tensor(buf183, (8, 256, 16, 16), (65536, 1, 4096, 256), 0); del buf183  # reuse
        # Source Nodes: [sigmoid_4], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_57.run(buf200, convolution_31, buf212, add_116, convolution_29, unsqueeze_346, buf217, squeeze_64, buf215, buf218, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del add_116
        del buf212
        del convolution_29
        del convolution_31
        del unsqueeze_346
        buf220 = buf200; del buf200  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_58.run(buf218, squeeze_64, primals_45, buf220, 2048, 256, grid=grid(2048, 256), stream=stream0)
        del buf218
        del primals_45
        del squeeze_64
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf221 = aten.convolution_backward(buf220, mul_170, primals_122, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf220
        del mul_170
        del primals_122
        buf222 = buf221[0]
        buf223 = buf221[1]
        del buf221
        buf224 = reinterpret_tensor(buf195, (256, 64), (64, 1), 0); del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_61.run(buf222, mul_526, buf224, 16384, 128, grid=grid(16384), stream=stream0)
        buf225 = buf217; del buf217  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_62.run(buf224, buf225, 256, 64, grid=grid(256), stream=stream0)
        buf226 = reinterpret_tensor(buf224, (256, 64), (1, 256), 0); del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_63.run(buf222, mul_526, convolution_28, unsqueeze_358, buf226, 16384, 128, grid=grid(16384), stream=stream0)
        buf227 = empty((256, ), device='cuda', dtype=torch.float32)
        buf228 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_64.run(buf226, squeeze_61, buf227, buf228, 256, 64, grid=grid(256), stream=stream0)
        buf229 = buf222; del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_65.run(buf229, mul_526, convolution_28, unsqueeze_358, buf227, squeeze_61, buf225, primals_43, 2048, 1024, grid=grid(2048, 1024), stream=stream0)
        del convolution_28
        del mul_526
        del primals_43
        del squeeze_61
        del unsqueeze_358
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf230 = aten.convolution_backward(buf229, mul_162, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_162
        del primals_121
        buf231 = buf230[0]
        buf232 = buf230[1]
        del buf230
        buf233 = buf100; del buf100  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_mul_native_batch_norm_backward_66.run(buf193, buf231, mul_538, buf233, 512, 8192, grid=grid(512), stream=stream0)
        buf234 = empty((512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_mul_native_batch_norm_backward_67.run(buf193, buf231, mul_538, convolution_27, unsqueeze_370, buf234, 32768, 128, grid=grid(32768), stream=stream0)
        buf235 = empty((512, ), device='cuda', dtype=torch.float32)
        buf237 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_mul_native_batch_norm_backward_68.run(buf234, squeeze_58, buf235, buf237, 512, 64, grid=grid(512), stream=stream0)
        buf236 = empty((8, 512, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_mul_native_batch_norm_backward_69.run(buf193, buf231, mul_538, convolution_27, unsqueeze_370, buf235, squeeze_58, buf233, primals_41, buf236, 4096, 1024, grid=grid(4096, 1024), stream=stream0)
        del convolution_27
        del primals_41
        del squeeze_58
        del unsqueeze_370
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf238 = aten.convolution_backward(buf236, mul_154, primals_120, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf236
        del mul_154
        del primals_120
        buf239 = buf238[0]
        buf240 = buf238[1]
        del buf238
        buf241 = reinterpret_tensor(buf122, (128, 64), (64, 1), 0); del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_70.run(buf239, mul_550, buf241, 8192, 128, grid=grid(8192), stream=stream0)
        buf242 = reinterpret_tensor(buf209, (128, ), (1, ), 0); del buf209  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_71.run(buf241, buf242, 128, 64, grid=grid(128), stream=stream0)
        buf243 = reinterpret_tensor(buf241, (128, 64), (1, 128), 0); del buf241  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_72.run(buf239, mul_550, bmm_1, unsqueeze_382, buf243, 8192, 128, grid=grid(8192), stream=stream0)
        buf244 = reinterpret_tensor(buf163, (128, ), (1, ), 0); del buf163  # reuse
        buf245 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_73.run(buf243, squeeze_55, buf244, buf245, 128, 64, grid=grid(128), stream=stream0)
        buf246 = buf239; del buf239  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_mul_native_batch_norm_backward_74.run(buf246, mul_550, bmm_1, unsqueeze_382, buf244, squeeze_55, buf242, primals_39, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        del bmm_1
        del mul_550
        del primals_39
        del squeeze_55
        del unsqueeze_382
        buf247 = reinterpret_tensor(buf102, (32, 1024, 32), (32768, 32, 1), 0); del buf102  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_110, reinterpret_tensor(buf246, (32, 1024, 32), (32768, 1, 1024), 0), out=buf247)
        del permute_110
        buf248 = empty((32, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf246, (32, 1024, 32), (32768, 1, 1024), 0), permute_111, out=buf248)
        del permute_111
        buf249 = reinterpret_tensor(buf234, (32, 1024, 1), (1024, 1, 32768), 0); del buf234  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_75.run(buf248, alias_27, buf249, 32768, 1024, grid=grid(32768), stream=stream0)
        buf250 = reinterpret_tensor(buf246, (32, 32, 1, 32, 32), (32768, 1024, 1024, 32, 1), 0); del buf246  # reuse
        buf254 = reinterpret_tensor(buf92, (32, 32, 1, 32, 32), (32768, 1024, 1024, 32, 1), 0); del buf92  # reuse
        buf258 = empty((32, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul, aten.sum]
        triton_per_fused__softmax_backward_data_mul_sum_76.run(buf248, alias_27, buf249, buf250, buf254, buf258, 1048576, 32, grid=grid(1048576), stream=stream0)
        del alias_27
        del buf248
        buf251 = empty((32768, 63), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_77.run(buf250, buf251, 2064384, grid=grid(2064384), stream=stream0)
        buf252 = empty((63, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf251, (63, 32768), (1, 63), 0), view_13, out=buf252)
        del view_13
        buf253 = reinterpret_tensor(buf250, (32768, 32), (32, 1), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf251, permute_115, out=buf253)
        del permute_115
        buf255 = buf251; del buf251  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_77.run(buf254, buf255, 2064384, grid=grid(2064384), stream=stream0)
        buf256 = empty((63, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf255, (63, 32768), (1, 63), 0), view_7, out=buf256)
        del view_7
        buf257 = reinterpret_tensor(buf254, (32768, 32), (32, 1), 0); del buf254  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf255, permute_121, out=buf257)
        del buf255
        del permute_121
        buf259 = reinterpret_tensor(buf91, (32, 32, 1024), (32768, 1024, 1), 0); del buf91  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_123, buf258, out=buf259)
        del permute_123
        buf260 = reinterpret_tensor(buf89, (32, 1024, 32), (32768, 32, 1), 0); del buf89  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf258, permute_124, out=buf260)
        del buf258
        del permute_124
        buf261 = reinterpret_tensor(buf93, (8, 384, 32, 32), (393216, 1024, 32, 1), 0); del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward]
        triton_poi_fused_cat_convolution_backward_78.run(buf253, buf257, buf260, buf259, buf247, buf261, 8192, 384, grid=grid(8192, 384), stream=stream0)
        del buf247
        del buf253
        del buf257
        del buf259
        del buf260
        # Source Nodes: [], Original ATen: [aten.cat, aten.convolution_backward]
        buf262 = aten.convolution_backward(buf261, mul_145, primals_119, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf261
        del mul_145
        del primals_119
        buf263 = buf262[0]
        buf264 = buf262[1]
        del buf262
        buf265 = reinterpret_tensor(buf243, (128, 64), (64, 1), 0); del buf243  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_70.run(buf263, mul_565, buf265, 8192, 128, grid=grid(8192), stream=stream0)
        buf266 = buf244; del buf244  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_71.run(buf265, buf266, 128, 64, grid=grid(128), stream=stream0)
        buf267 = reinterpret_tensor(buf265, (128, 64), (1, 128), 0); del buf265  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_79.run(buf263, mul_565, convolution_25, unsqueeze_394, buf267, 8192, 128, grid=grid(8192), stream=stream0)
        buf268 = empty((128, ), device='cuda', dtype=torch.float32)
        buf269 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_73.run(buf267, squeeze_52, buf268, buf269, 128, 64, grid=grid(128), stream=stream0)
        buf270 = buf263; del buf263  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_80.run(buf270, mul_565, convolution_25, unsqueeze_394, buf268, squeeze_52, buf266, primals_35, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        del convolution_25
        del mul_565
        del primals_35
        del squeeze_52
        del unsqueeze_394
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf271 = aten.convolution_backward(buf270, mul_137, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_137
        del primals_118
        buf272 = buf271[0]
        buf273 = buf271[1]
        del buf271
        buf274 = buf193; del buf193  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_81.run(buf274, buf231, mul_538, buf272, mul_577, 4096, 1024, grid=grid(4096, 1024), stream=stream0)
        del mul_538
        del mul_577
        buf275 = buf235; del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_82.run(buf274, buf275, 512, 8192, grid=grid(512), stream=stream0)
        buf276 = reinterpret_tensor(buf249, (512, 64), (64, 1), 0); del buf249  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_83.run(buf274, convolution_24, unsqueeze_406, buf276, 32768, 128, grid=grid(32768), stream=stream0)
        buf277 = empty((512, ), device='cuda', dtype=torch.float32)
        buf278 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_mul_native_batch_norm_backward_68.run(buf276, squeeze_49, buf277, buf278, 512, 64, grid=grid(512), stream=stream0)
        buf279 = buf272; del buf272  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_84.run(buf274, convolution_24, unsqueeze_406, buf277, squeeze_49, buf275, primals_33, buf279, 4096, 1024, grid=grid(4096, 1024), stream=stream0)
        del convolution_24
        del primals_33
        del squeeze_49
        del unsqueeze_406
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf280 = aten.convolution_backward(buf279, mul_129, primals_117, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_129
        del primals_117
        buf281 = buf280[0]
        buf282 = buf280[1]
        del buf280
        buf283 = reinterpret_tensor(buf267, (8, 128, 1, 1, 8), (1024, 8, 8192, 8192, 1), 0); del buf267  # reuse
        # Source Nodes: [x_106], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_85.run(buf281, add_82, buf283, 8192, 128, grid=grid(8192), stream=stream0)
        buf284 = reinterpret_tensor(buf196, (8, 128, 1, 1), (128, 1, 1024, 1024), 0); del buf196  # reuse
        buf285 = reinterpret_tensor(buf284, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf284  # reuse
        # Source Nodes: [sigmoid_3, x_106], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_86.run(buf285, buf283, convolution_23, 1024, 8, grid=grid(1024), stream=stream0)
        buf286 = buf268; del buf268  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_87.run(buf285, buf286, 128, 8, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf287 = aten.convolution_backward(buf285, relu_3, primals_115, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf285
        del primals_115
        buf288 = buf287[0]
        buf289 = buf287[1]
        del buf287
        buf290 = buf288; del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_88.run(buf290, relu_3, 64, grid=grid(64), stream=stream0)
        del relu_3
        buf291 = empty((8, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_89.run(buf290, buf291, 8, 8, grid=grid(8), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf292 = aten.convolution_backward(buf290, mean_3, primals_113, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_3
        del primals_113
        buf293 = buf292[0]
        buf294 = buf292[1]
        del buf292
        buf295 = reinterpret_tensor(buf283, (128, 64), (64, 1), 0); del buf283  # reuse
        # Source Nodes: [sigmoid_3], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_90.run(buf281, convolution_23, buf293, add_82, buf295, 8192, 128, grid=grid(8192), stream=stream0)
        buf296 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_3], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_mul_native_batch_norm_backward_71.run(buf295, buf296, 128, 64, grid=grid(128), stream=stream0)
        buf297 = reinterpret_tensor(buf295, (128, 64), (1, 128), 0); del buf295  # reuse
        # Source Nodes: [sigmoid_3], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_91.run(buf281, convolution_23, buf293, add_82, convolution_21, unsqueeze_418, buf297, 8192, 128, grid=grid(8192), stream=stream0)
        buf298 = empty((128, ), device='cuda', dtype=torch.float32)
        buf300 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_3], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_mul_native_batch_norm_backward_73.run(buf297, squeeze_46, buf298, buf300, 128, 64, grid=grid(128), stream=stream0)
        buf299 = reinterpret_tensor(buf270, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf270  # reuse
        # Source Nodes: [sigmoid_3], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_92.run(buf281, convolution_23, buf293, add_82, convolution_21, unsqueeze_418, buf298, squeeze_46, buf296, buf299, 8192, 128, grid=grid(8192, 128), stream=stream0)
        del add_82
        del convolution_21
        del convolution_23
        del unsqueeze_418
        buf301 = buf281; del buf281  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_93.run(buf299, squeeze_46, primals_31, buf301, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        del buf299
        del primals_31
        del squeeze_46
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf302 = aten.convolution_backward(buf301, mul_120, primals_112, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf301
        del mul_120
        del primals_112
        buf303 = buf302[0]
        buf304 = buf302[1]
        del buf302
        buf305 = reinterpret_tensor(buf297, (128, 64), (64, 1), 0); del buf297  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_70.run(buf303, mul_605, buf305, 8192, 128, grid=grid(8192), stream=stream0)
        buf306 = buf298; del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_71.run(buf305, buf306, 128, 64, grid=grid(128), stream=stream0)
        buf307 = reinterpret_tensor(buf305, (128, 64), (1, 128), 0); del buf305  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_79.run(buf303, mul_605, convolution_20, unsqueeze_430, buf307, 8192, 128, grid=grid(8192), stream=stream0)
        buf308 = empty((128, ), device='cuda', dtype=torch.float32)
        buf309 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_73.run(buf307, squeeze_43, buf308, buf309, 128, 64, grid=grid(128), stream=stream0)
        buf310 = buf303; del buf303  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_80.run(buf310, mul_605, convolution_20, unsqueeze_430, buf308, squeeze_43, buf306, primals_29, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        del convolution_20
        del mul_605
        del primals_29
        del squeeze_43
        del unsqueeze_430
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf311 = aten.convolution_backward(buf310, mul_112, primals_111, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_112
        del primals_111
        buf312 = buf311[0]
        buf313 = buf311[1]
        del buf311
        buf314 = buf277; del buf277  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_mul_native_batch_norm_backward_66.run(buf274, buf312, mul_617, buf314, 512, 8192, grid=grid(512), stream=stream0)
        buf315 = buf276; del buf276  # reuse
        buf322 = empty((512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_mul_native_batch_norm_backward_94.run(buf274, buf312, mul_617, convolution_19, unsqueeze_442, convolution_18, unsqueeze_454, buf315, buf322, 32768, 128, grid=grid(32768), stream=stream0)
        buf316 = empty((512, ), device='cuda', dtype=torch.float32)
        buf318 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_mul_native_batch_norm_backward_68.run(buf315, squeeze_40, buf316, buf318, 512, 64, grid=grid(512), stream=stream0)
        del buf315
        buf323 = empty((512, ), device='cuda', dtype=torch.float32)
        buf325 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_mul_native_batch_norm_backward_68.run(buf322, squeeze_37, buf323, buf325, 512, 64, grid=grid(512), stream=stream0)
        buf317 = buf279; del buf279  # reuse
        buf324 = buf231; del buf231  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_mul_native_batch_norm_backward_95.run(buf274, buf312, mul_617, convolution_19, unsqueeze_442, buf316, squeeze_40, buf314, primals_27, convolution_18, unsqueeze_454, buf323, squeeze_37, primals_25, buf317, buf324, 4096, 1024, grid=grid(4096, 1024), stream=stream0)
        del buf274
        del buf312
        del buf316
        del convolution_18
        del convolution_19
        del mul_617
        del primals_25
        del primals_27
        del squeeze_37
        del squeeze_40
        del unsqueeze_442
        del unsqueeze_454
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf319 = aten.convolution_backward(buf317, mul_80, primals_110, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf317
        del primals_110
        buf320 = buf319[0]
        buf321 = buf319[1]
        del buf319
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf326 = aten.convolution_backward(buf324, mul_97, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf324
        del mul_97
        del primals_109
        buf327 = buf326[0]
        buf328 = buf326[1]
        del buf326
        buf329 = reinterpret_tensor(buf307, (8, 128, 1, 1, 8), (1024, 8, 8192, 8192, 1), 0); del buf307  # reuse
        # Source Nodes: [x_78], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_85.run(buf327, add_61, buf329, 8192, 128, grid=grid(8192), stream=stream0)
        buf330 = reinterpret_tensor(buf293, (8, 128, 1, 1), (128, 1, 1024, 1024), 0); del buf293  # reuse
        buf331 = reinterpret_tensor(buf330, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf330  # reuse
        # Source Nodes: [sigmoid_2, x_78], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_86.run(buf331, buf329, convolution_17, 1024, 8, grid=grid(1024), stream=stream0)
        buf332 = buf308; del buf308  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_87.run(buf331, buf332, 128, 8, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf333 = aten.convolution_backward(buf331, relu_2, primals_107, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf331
        del primals_107
        buf334 = buf333[0]
        buf335 = buf333[1]
        del buf333
        buf336 = buf334; del buf334  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_88.run(buf336, relu_2, 64, grid=grid(64), stream=stream0)
        del relu_2
        buf337 = empty((8, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_89.run(buf336, buf337, 8, 8, grid=grid(8), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf338 = aten.convolution_backward(buf336, mean_2, primals_105, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_2
        del primals_105
        buf339 = buf338[0]
        buf340 = buf338[1]
        del buf338
        buf341 = reinterpret_tensor(buf329, (128, 64), (64, 1), 0); del buf329  # reuse
        # Source Nodes: [sigmoid_2], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_90.run(buf327, convolution_17, buf339, add_61, buf341, 8192, 128, grid=grid(8192), stream=stream0)
        buf342 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_2], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_mul_native_batch_norm_backward_71.run(buf341, buf342, 128, 64, grid=grid(128), stream=stream0)
        buf343 = reinterpret_tensor(buf341, (128, 64), (1, 128), 0); del buf341  # reuse
        # Source Nodes: [sigmoid_2], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_91.run(buf327, convolution_17, buf339, add_61, convolution_15, unsqueeze_466, buf343, 8192, 128, grid=grid(8192), stream=stream0)
        buf344 = empty((128, ), device='cuda', dtype=torch.float32)
        buf346 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_2], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_mul_native_batch_norm_backward_73.run(buf343, squeeze_34, buf344, buf346, 128, 64, grid=grid(128), stream=stream0)
        del buf343
        buf345 = reinterpret_tensor(buf310, (8, 128, 32, 32), (131072, 1, 4096, 128), 0); del buf310  # reuse
        # Source Nodes: [sigmoid_2], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_92.run(buf327, convolution_17, buf339, add_61, convolution_15, unsqueeze_466, buf344, squeeze_34, buf342, buf345, 8192, 128, grid=grid(8192, 128), stream=stream0)
        del add_61
        del buf339
        del convolution_15
        del convolution_17
        del unsqueeze_466
        buf347 = buf327; del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_93.run(buf345, squeeze_34, primals_23, buf347, 1024, 1024, grid=grid(1024, 1024), stream=stream0)
        del buf345
        del primals_23
        del squeeze_34
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf348 = aten.convolution_backward(buf347, mul_88, primals_104, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf347
        del mul_88
        del primals_104
        buf349 = buf348[0]
        buf350 = buf348[1]
        del buf348
        buf351 = reinterpret_tensor(buf322, (128, 256), (256, 1), 0); del buf322  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_96.run(buf349, mul_654, buf351, 32768, 128, grid=grid(32768), stream=stream0)
        buf352 = buf344; del buf344  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_97.run(buf351, buf352, 128, 256, grid=grid(128), stream=stream0)
        buf353 = reinterpret_tensor(buf351, (128, 256), (1, 128), 0); del buf351  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_98.run(buf349, mul_654, convolution_14, unsqueeze_478, buf353, 32768, 128, grid=grid(32768), stream=stream0)
        buf354 = empty((128, ), device='cuda', dtype=torch.float32)
        buf355 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_99.run(buf353, squeeze_31, buf354, buf355, 128, 256, grid=grid(128), stream=stream0)
        buf356 = buf349; del buf349  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_100.run(buf356, mul_654, convolution_14, unsqueeze_478, buf354, squeeze_31, buf352, primals_21, 1024, 4096, grid=grid(1024, 4096), stream=stream0)
        del buf354
        del convolution_14
        del mul_654
        del primals_21
        del squeeze_31
        del unsqueeze_478
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf357 = aten.convolution_backward(buf356, mul_80, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf356
        del mul_80
        del primals_103
        buf358 = buf357[0]
        buf359 = buf357[1]
        del buf357
        buf360 = buf227; del buf227  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_mul_native_batch_norm_backward_101.run(buf320, buf358, mul_666, buf360, 256, 32768, grid=grid(256), stream=stream0)
        buf361 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_mul_native_batch_norm_backward_102.run(buf320, buf358, mul_666, convolution_13, unsqueeze_490, buf361, 65536, 128, grid=grid(65536), stream=stream0)
        buf362 = empty((256, ), device='cuda', dtype=torch.float32)
        buf364 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_mul_native_batch_norm_backward_103.run(buf361, squeeze_28, buf362, buf364, 256, 256, grid=grid(256), stream=stream0)
        buf363 = empty((8, 256, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_mul_native_batch_norm_backward_104.run(buf320, buf358, mul_666, convolution_13, unsqueeze_490, buf362, squeeze_28, buf360, primals_19, buf363, 2048, 4096, grid=grid(2048, 4096), stream=stream0)
        del convolution_13
        del primals_19
        del squeeze_28
        del unsqueeze_490
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf365 = aten.convolution_backward(buf363, mul_72, primals_102, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf363
        del mul_72
        del primals_102
        buf366 = buf365[0]
        buf367 = buf365[1]
        del buf365
        buf368 = reinterpret_tensor(buf226, (8, 64, 1, 1, 32), (2048, 32, 16384, 16384, 1), 0); del buf226  # reuse
        # Source Nodes: [x_55], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_105.run(buf366, add_45, buf368, 16384, 128, grid=grid(16384), stream=stream0)
        buf369 = reinterpret_tensor(buf323, (8, 64, 1, 1), (64, 1, 512, 512), 0); del buf323  # reuse
        buf370 = reinterpret_tensor(buf369, (8, 64, 1, 1), (64, 1, 1, 1), 0); del buf369  # reuse
        # Source Nodes: [sigmoid_1, x_55], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_106.run(buf370, buf368, convolution_12, 512, 32, grid=grid(512), stream=stream0)
        buf371 = reinterpret_tensor(buf336, (64, ), (1, ), 0); del buf336  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_107.run(buf370, buf371, 64, 8, grid=grid(64), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf372 = aten.convolution_backward(buf370, relu_1, primals_100, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf370
        del primals_100
        buf373 = buf372[0]
        buf374 = buf372[1]
        del buf372
        buf375 = buf373; del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_88.run(buf375, relu_1, 64, grid=grid(64), stream=stream0)
        del relu_1
        buf376 = empty((8, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_89.run(buf375, buf376, 8, 8, grid=grid(8), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf377 = aten.convolution_backward(buf375, mean_1, primals_98, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_1
        del primals_98
        buf378 = buf377[0]
        buf379 = buf377[1]
        del buf377
        buf380 = reinterpret_tensor(buf368, (64, 256), (256, 1), 0); del buf368  # reuse
        # Source Nodes: [sigmoid_1], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_108.run(buf366, convolution_12, buf378, add_45, buf380, 16384, 128, grid=grid(16384), stream=stream0)
        buf381 = reinterpret_tensor(buf375, (64, ), (1, ), 0); del buf375  # reuse
        # Source Nodes: [sigmoid_1], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_109.run(buf380, buf381, 64, 256, grid=grid(64), stream=stream0)
        buf382 = reinterpret_tensor(buf380, (64, 256), (1, 64), 0); del buf380  # reuse
        # Source Nodes: [sigmoid_1], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_110.run(buf366, convolution_12, buf378, add_45, convolution_10, unsqueeze_502, buf382, 16384, 128, grid=grid(16384), stream=stream0)
        buf383 = reinterpret_tensor(buf290, (64, ), (1, ), 0); del buf290  # reuse
        buf385 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_1], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_111.run(buf382, squeeze_25, buf383, buf385, 64, 256, grid=grid(64), stream=stream0)
        buf384 = reinterpret_tensor(buf229, (8, 64, 64, 64), (262144, 1, 4096, 64), 0); del buf229  # reuse
        # Source Nodes: [sigmoid_1], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_112.run(buf366, convolution_12, buf378, add_45, convolution_10, unsqueeze_502, buf383, squeeze_25, buf381, buf384, 32768, 64, grid=grid(32768, 64), stream=stream0)
        del add_45
        del convolution_10
        del convolution_12
        del unsqueeze_502
        buf386 = buf366; del buf366  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_113.run(buf384, squeeze_25, primals_17, buf386, 512, 4096, grid=grid(512, 4096), stream=stream0)
        del buf384
        del primals_17
        del squeeze_25
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf387 = aten.convolution_backward(buf386, mul_63, primals_97, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf386
        del mul_63
        del primals_97
        buf388 = buf387[0]
        buf389 = buf387[1]
        del buf387
        buf390 = reinterpret_tensor(buf382, (64, 256), (256, 1), 0); del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_114.run(buf388, mul_694, buf390, 16384, 128, grid=grid(16384), stream=stream0)
        buf391 = buf383; del buf383  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_109.run(buf390, buf391, 64, 256, grid=grid(64), stream=stream0)
        buf392 = reinterpret_tensor(buf390, (64, 256), (1, 64), 0); del buf390  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_115.run(buf388, mul_694, convolution_9, unsqueeze_514, buf392, 16384, 128, grid=grid(16384), stream=stream0)
        buf393 = empty((64, ), device='cuda', dtype=torch.float32)
        buf394 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_111.run(buf392, squeeze_22, buf393, buf394, 64, 256, grid=grid(64), stream=stream0)
        buf395 = buf388; del buf388  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_116.run(buf395, mul_694, convolution_9, unsqueeze_514, buf393, squeeze_22, buf391, primals_15, 512, 4096, grid=grid(512, 4096), stream=stream0)
        del convolution_9
        del mul_694
        del primals_15
        del squeeze_22
        del unsqueeze_514
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf396 = aten.convolution_backward(buf395, mul_55, primals_96, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mul_55
        del primals_96
        buf397 = buf396[0]
        buf398 = buf396[1]
        del buf396
        buf399 = buf320; del buf320  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul]
        triton_poi_fused_add_mul_117.run(buf399, buf358, mul_666, buf397, mul_706, 2048, 4096, grid=grid(2048, 4096), stream=stream0)
        del mul_666
        del mul_706
        buf400 = buf362; del buf362  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_118.run(buf399, buf400, 256, 32768, grid=grid(256), stream=stream0)
        buf401 = buf361; del buf361  # reuse
        buf408 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_119.run(buf399, convolution_8, unsqueeze_526, convolution_7, unsqueeze_538, buf401, buf408, 65536, 128, grid=grid(65536), stream=stream0)
        buf402 = empty((256, ), device='cuda', dtype=torch.float32)
        buf403 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_mul_native_batch_norm_backward_103.run(buf401, squeeze_19, buf402, buf403, 256, 256, grid=grid(256), stream=stream0)
        del buf401
        buf409 = empty((256, ), device='cuda', dtype=torch.float32)
        buf410 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_add_mul_native_batch_norm_backward_103.run(buf408, squeeze_16, buf409, buf410, 256, 256, grid=grid(256), stream=stream0)
        del buf408
        buf404 = buf397; del buf397  # reuse
        buf411 = buf358; del buf358  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_120.run(buf399, convolution_8, unsqueeze_526, buf402, squeeze_19, buf400, primals_13, convolution_7, unsqueeze_538, buf409, squeeze_16, primals_11, buf404, buf411, 2048, 4096, grid=grid(2048, 4096), stream=stream0)
        del buf399
        del buf402
        del convolution_7
        del convolution_8
        del primals_11
        del primals_13
        del squeeze_16
        del squeeze_19
        del unsqueeze_526
        del unsqueeze_538
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf405 = aten.convolution_backward(buf404, mul_23, primals_95, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf404
        del primals_95
        buf406 = buf405[0]
        buf407 = buf405[1]
        del buf405
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf412 = aten.convolution_backward(buf411, mul_40, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf411
        del mul_40
        del primals_94
        buf413 = buf412[0]
        buf414 = buf412[1]
        del buf412
        buf415 = reinterpret_tensor(buf392, (8, 64, 1, 1, 32), (2048, 32, 16384, 16384, 1), 0); del buf392  # reuse
        # Source Nodes: [x_27], Original ATen: [aten.mul, aten.silu, aten.sum]
        triton_red_fused_mul_silu_sum_105.run(buf413, add_24, buf415, 16384, 128, grid=grid(16384), stream=stream0)
        buf416 = reinterpret_tensor(buf378, (8, 64, 1, 1), (64, 1, 512, 512), 0); del buf378  # reuse
        buf417 = reinterpret_tensor(buf416, (8, 64, 1, 1), (64, 1, 1, 1), 0); del buf416  # reuse
        # Source Nodes: [sigmoid, x_27], Original ATen: [aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.silu, aten.sum]
        triton_per_fused_mul_sigmoid_sigmoid_backward_silu_sum_106.run(buf417, buf415, convolution_6, 512, 32, grid=grid(512), stream=stream0)
        buf418 = buf393; del buf393  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_107.run(buf417, buf418, 64, 8, grid=grid(64), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf419 = aten.convolution_backward(buf417, relu, primals_92, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf417
        del primals_92
        buf420 = buf419[0]
        buf421 = buf419[1]
        del buf419
        buf422 = buf420; del buf420  # reuse
        # Source Nodes: [], Original ATen: [aten.threshold_backward]
        triton_poi_fused_threshold_backward_88.run(buf422, relu, 64, grid=grid(64), stream=stream0)
        del relu
        buf423 = empty((8, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_89.run(buf422, buf423, 8, 8, grid=grid(8), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf424 = aten.convolution_backward(buf422, mean, primals_90, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean
        del primals_90
        buf425 = buf424[0]
        buf426 = buf424[1]
        del buf424
        buf427 = reinterpret_tensor(buf415, (64, 256), (256, 1), 0); del buf415  # reuse
        # Source Nodes: [sigmoid], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_108.run(buf413, convolution_6, buf425, add_24, buf427, 16384, 128, grid=grid(16384), stream=stream0)
        buf428 = reinterpret_tensor(buf422, (64, ), (1, ), 0); del buf422  # reuse
        # Source Nodes: [sigmoid], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_109.run(buf427, buf428, 64, 256, grid=grid(64), stream=stream0)
        buf429 = reinterpret_tensor(buf427, (64, 256), (1, 64), 0); del buf427  # reuse
        # Source Nodes: [sigmoid], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_110.run(buf413, convolution_6, buf425, add_24, convolution_4, unsqueeze_550, buf429, 16384, 128, grid=grid(16384), stream=stream0)
        buf430 = empty((64, ), device='cuda', dtype=torch.float32)
        buf432 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_111.run(buf429, squeeze_13, buf430, buf432, 64, 256, grid=grid(64), stream=stream0)
        buf431 = reinterpret_tensor(buf395, (8, 64, 64, 64), (262144, 1, 4096, 64), 0); del buf395  # reuse
        # Source Nodes: [sigmoid], Original ATen: [aten.add, aten.div, aten.fill, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.sub]
        triton_poi_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_112.run(buf413, convolution_6, buf425, add_24, convolution_4, unsqueeze_550, buf430, squeeze_13, buf428, buf431, 32768, 64, grid=grid(32768, 64), stream=stream0)
        del add_24
        del buf425
        del convolution_4
        del convolution_6
        del unsqueeze_550
        buf433 = buf413; del buf413  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_113.run(buf431, squeeze_13, primals_9, buf433, 512, 4096, grid=grid(512, 4096), stream=stream0)
        del buf431
        del primals_9
        del squeeze_13
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf434 = aten.convolution_backward(buf433, mul_31, primals_89, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf433
        del mul_31
        del primals_89
        buf435 = buf434[0]
        buf436 = buf434[1]
        del buf434
        buf437 = reinterpret_tensor(buf429, (64, 256), (256, 1), 0); del buf429  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_114.run(buf435, mul_743, buf437, 16384, 128, grid=grid(16384), stream=stream0)
        buf438 = buf430; del buf430  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_109.run(buf437, buf438, 64, 256, grid=grid(64), stream=stream0)
        buf439 = reinterpret_tensor(buf437, (64, 256), (1, 64), 0); del buf437  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_115.run(buf435, mul_743, convolution_3, unsqueeze_562, buf439, 16384, 128, grid=grid(16384), stream=stream0)
        buf440 = empty((64, ), device='cuda', dtype=torch.float32)
        buf441 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_div_fill_mul_native_batch_norm_backward_sigmoid_sub_111.run(buf439, squeeze_10, buf440, buf441, 64, 256, grid=grid(64), stream=stream0)
        buf442 = buf435; del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_116.run(buf442, mul_743, convolution_3, unsqueeze_562, buf440, squeeze_10, buf438, primals_7, 512, 4096, grid=grid(512, 4096), stream=stream0)
        del convolution_3
        del mul_743
        del primals_7
        del squeeze_10
        del unsqueeze_562
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf443 = aten.convolution_backward(buf442, mul_23, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf442
        del mul_23
        del primals_88
        buf444 = buf443[0]
        buf445 = buf443[1]
        del buf443
        buf446 = reinterpret_tensor(buf409, (64, 4), (1, 64), 0); del buf409  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_mul_native_batch_norm_backward_121.run(buf406, buf444, mul_755, buf446, 256, 8192, grid=grid(256), stream=stream0)
        buf447 = buf440; del buf440  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_mul_native_batch_norm_backward_122.run(buf446, buf447, 64, 4, grid=grid(64), stream=stream0)
        del buf446
        buf448 = reinterpret_tensor(buf439, (64, 256), (256, 1), 0); del buf439  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_add_mul_native_batch_norm_backward_123.run(buf406, buf444, mul_755, convolution_2, unsqueeze_574, buf448, 16384, 128, grid=grid(16384), stream=stream0)
        buf449 = empty((64, ), device='cuda', dtype=torch.float32)
        buf451 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_mul_native_batch_norm_backward_124.run(buf448, squeeze_7, buf449, buf451, 64, 256, grid=grid(64), stream=stream0)
        del buf448
        buf450 = buf406; del buf406  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_add_mul_native_batch_norm_backward_125.run(buf450, buf444, mul_755, convolution_2, unsqueeze_574, buf449, squeeze_7, buf447, primals_5, 512, 4096, grid=grid(512, 4096), stream=stream0)
        del buf444
        del buf449
        del convolution_2
        del mul_755
        del primals_5
        del squeeze_7
        del unsqueeze_574
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf452 = aten.convolution_backward(buf450, mul_15, primals_87, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf450
        del mul_15
        del primals_87
        buf453 = buf452[0]
        buf454 = buf452[1]
        del buf452
        buf455 = reinterpret_tensor(buf353, (32, 1024), (1024, 1), 0); del buf353  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_126.run(buf453, mul_767, buf455, 32768, 128, grid=grid(32768), stream=stream0)
        buf456 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_127.run(buf455, buf456, 32, 1024, grid=grid(32), stream=stream0)
        buf457 = reinterpret_tensor(buf455, (32, 1024), (1, 32), 0); del buf455  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_128.run(buf453, mul_767, convolution_1, unsqueeze_586, buf457, 32768, 128, grid=grid(32768), stream=stream0)
        buf458 = empty((32, ), device='cuda', dtype=torch.float32)
        buf459 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_129.run(buf457, squeeze_4, buf458, buf459, 32, 1024, grid=grid(32), stream=stream0)
        del buf457
        buf460 = buf453; del buf453  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_130.run(buf460, mul_767, convolution_1, unsqueeze_586, buf458, squeeze_4, buf456, primals_3, 256, 16384, grid=grid(256, 16384), stream=stream0)
        del buf458
        del convolution_1
        del mul_767
        del primals_3
        del squeeze_4
        del unsqueeze_586
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf461 = aten.convolution_backward(buf460, mul_7, primals_86, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf460
        del mul_7
        del primals_86
        buf462 = buf461[0]
        buf463 = buf461[1]
        del buf461
        buf464 = empty((24, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_131.run(buf462, mul_779, buf464, 24576, 128, grid=grid(24576), stream=stream0)
        buf465 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_132.run(buf464, buf465, 24, 1024, grid=grid(24), stream=stream0)
        buf466 = reinterpret_tensor(buf464, (24, 1024), (1, 24), 0); del buf464  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_133.run(buf462, mul_779, convolution, unsqueeze_598, buf466, 24576, 128, grid=grid(24576), stream=stream0)
        buf467 = empty((24, ), device='cuda', dtype=torch.float32)
        buf468 = empty((24, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_134.run(buf466, squeeze_1, buf467, buf468, 24, 1024, grid=grid(24), stream=stream0)
        del buf466
        buf469 = buf462; del buf462  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_135.run(buf469, mul_779, convolution, unsqueeze_598, buf467, squeeze_1, buf465, primals_1, 192, 16384, grid=grid(192, 16384), stream=stream0)
        del buf467
        del convolution
        del mul_779
        del primals_1
        del squeeze_1
        del unsqueeze_598
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf470 = aten.convolution_backward(buf469, primals_263, primals_85, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf469
        del primals_263
        del primals_85
        buf471 = buf470[1]
        return (buf468, buf465, buf459, buf456, buf451, buf447, buf441, buf438, buf432, buf428, buf410, buf400, buf403, buf400, buf394, buf391, buf385, buf381, buf364, buf360, buf355, buf352, buf346, buf342, buf325, buf314, buf318, buf314, buf309, buf306, buf300, buf296, buf278, buf275, buf269, buf266, reinterpret_tensor(buf256, (63, 32), (32, 1), 0), reinterpret_tensor(buf252, (63, 32), (32, 1), 0), buf245, buf242, buf237, buf233, buf228, buf225, buf219, buf215, buf198, buf187, buf191, buf187, buf182, buf179, buf173, buf169, buf151, buf148, buf142, buf139, reinterpret_tensor(buf129, (31, 64), (64, 1), 0), reinterpret_tensor(buf125, (31, 64), (64, 1), 0), buf118, buf115, buf110, buf106, buf101, buf98, reinterpret_tensor(buf88, (31, 128), (128, 1), 0), reinterpret_tensor(buf84, (31, 128), (128, 1), 0), buf76, buf73, buf67, buf55, buf59, buf55, buf49, buf46, reinterpret_tensor(buf36, (15, 128), (128, 1), 0), reinterpret_tensor(buf32, (15, 128), (128, 1), 0), buf25, buf22, buf16, buf13, buf7, buf4, buf471, buf463, buf454, buf445, buf436, buf426, buf423, buf421, buf418, buf414, buf407, buf398, buf389, buf379, buf376, buf374, buf371, buf367, buf359, buf350, buf340, buf337, buf335, buf332, buf328, buf321, buf313, buf304, buf294, buf291, buf289, buf286, buf282, buf273, buf264, buf240, buf232, buf223, buf213, buf210, buf208, buf205, buf201, buf194, buf186, buf177, buf167, buf164, buf162, buf159, buf155, buf146, buf137, buf113, buf105, buf96, buf71, buf63, buf53, buf44, buf20, buf11, reinterpret_tensor(buf1, (1000, 1280), (1280, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((24, 3, 3, 3), (27, 1, 9, 3), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((32, 24, 3, 3), (216, 1, 72, 24), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((8, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((64, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((8, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((64, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((128, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((8, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((128, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((384, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((16, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((256, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((16, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((256, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((768, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((1536, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((512, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((1280, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((8, 3, 256, 256), (196608, 1, 768, 3), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 24, 128, 128), (393216, 1, 3072, 24), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_7 = rand_strided((8, 24, 128, 128), (393216, 1, 3072, 24), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_15 = rand_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_23 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_31 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_24 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    mean = rand_strided((8, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((8, 8, 1, 1), (8, 1, 8, 8), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    mul_40 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_55 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_63 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_45 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    mean_1 = rand_strided((8, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((8, 8, 1, 1), (8, 1, 8, 8), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    mul_72 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_80 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_88 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_61 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    mean_2 = rand_strided((8, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((8, 8, 1, 1), (8, 1, 8, 8), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((8, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    mul_97 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_112 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_120 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_82 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    mean_3 = rand_strided((8, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((8, 8, 1, 1), (8, 1, 8, 8), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((8, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    mul_129 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_137 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_145 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    view_7 = rand_strided((32768, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    view_13 = rand_strided((32768, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    bmm_1 = rand_strided((32, 1024, 32), (32768, 32, 1), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_154 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_162 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_170 = rand_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_116 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    mean_4 = rand_strided((8, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((8, 16, 1, 1), (16, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((8, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    mul_179 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_70 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_194 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_202 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_137 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    mean_5 = rand_strided((8, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((8, 16, 1, 1), (16, 1, 16, 16), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((8, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    mul_211 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_219 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    squeeze_82 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_227 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    view_31 = rand_strided((8192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    view_37 = rand_strided((8192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    bmm_3 = rand_strided((32, 256, 64), (16384, 64, 1), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_236 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cuda:0', dtype=torch.float32)
    squeeze_88 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_244 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_252 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cuda:0', dtype=torch.float32)
    view_55 = rand_strided((8192, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_61 = rand_strided((8192, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_71 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cuda:0', dtype=torch.float32)
    avg_pool2d = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.float32)
    squeeze_94 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_261 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((8, 1536, 8, 8), (98304, 1, 12288, 1536), device='cuda:0', dtype=torch.float32)
    squeeze_97 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((8, 1536, 8, 8), (98304, 1, 12288, 1536), device='cuda:0', dtype=torch.float32)
    squeeze_100 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_276 = rand_strided((8, 1536, 8, 8), (98304, 1, 12288, 1536), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.float32)
    squeeze_103 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_284 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.float32)
    view_79 = rand_strided((2048, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_85 = rand_strided((2048, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    bmm_7 = rand_strided((32, 64, 128), (8192, 128, 1), device='cuda:0', dtype=torch.float32)
    squeeze_106 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_293 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.float32)
    convolution_48 = rand_strided((8, 1536, 8, 8), (98304, 1, 12288, 1536), device='cuda:0', dtype=torch.float32)
    squeeze_109 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_301 = rand_strided((8, 1536, 8, 8), (98304, 1, 12288, 1536), device='cuda:0', dtype=torch.float32)
    convolution_49 = rand_strided((8, 1280, 8, 8), (81920, 1, 10240, 1280), device='cuda:0', dtype=torch.float32)
    squeeze_112 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_62 = rand_strided((8, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_33 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    mul_311 = rand_strided((8, 1280, 8, 8), (81920, 1, 10240, 1280), device='cuda:0', dtype=torch.float32)
    unsqueeze_154 = rand_strided((1, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_323 = rand_strided((8, 1536, 8, 8), (98304, 1, 12288, 1536), device='cuda:0', dtype=torch.float32)
    unsqueeze_166 = rand_strided((1, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_335 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.float32)
    unsqueeze_178 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_41 = rand_strided((32, 64, 64), (4096, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_42 = rand_strided((32, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    alias_16 = rand_strided((32, 64, 64), (4096, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_46 = rand_strided((15, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_52 = rand_strided((15, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_54 = rand_strided((32, 128, 64), (8192, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_55 = rand_strided((32, 64, 128), (8192, 1, 64), device='cuda:0', dtype=torch.float32)
    mul_350 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.float32)
    unsqueeze_190 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_362 = rand_strided((8, 1536, 8, 8), (98304, 1, 12288, 1536), device='cuda:0', dtype=torch.float32)
    unsqueeze_202 = rand_strided((1, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_214 = rand_strided((1, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_383 = rand_strided((8, 512, 8, 8), (32768, 1, 4096, 512), device='cuda:0', dtype=torch.float32)
    unsqueeze_226 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_62 = rand_strided((32, 256, 256), (65536, 1, 256), device='cuda:0', dtype=torch.float32)
    permute_63 = rand_strided((32, 128, 256), (32768, 256, 1), device='cuda:0', dtype=torch.float32)
    alias_17 = rand_strided((32, 256, 256), (65536, 256, 1), device='cuda:0', dtype=torch.float32)
    permute_67 = rand_strided((31, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_73 = rand_strided((31, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_75 = rand_strided((32, 128, 256), (32768, 256, 1), device='cuda:0', dtype=torch.float32)
    permute_76 = rand_strided((32, 256, 128), (32768, 1, 256), device='cuda:0', dtype=torch.float32)
    mul_398 = rand_strided((8, 512, 16, 16), (131072, 1, 8192, 512), device='cuda:0', dtype=torch.float32)
    unsqueeze_238 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_410 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cuda:0', dtype=torch.float32)
    unsqueeze_250 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_422 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    unsqueeze_262 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_83 = rand_strided((32, 256, 256), (65536, 1, 256), device='cuda:0', dtype=torch.float32)
    permute_84 = rand_strided((32, 64, 256), (16384, 256, 1), device='cuda:0', dtype=torch.float32)
    alias_18 = rand_strided((32, 256, 256), (65536, 256, 1), device='cuda:0', dtype=torch.float32)
    permute_88 = rand_strided((31, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    permute_94 = rand_strided((31, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    permute_96 = rand_strided((32, 64, 256), (16384, 256, 1), device='cuda:0', dtype=torch.float32)
    permute_97 = rand_strided((32, 256, 64), (16384, 1, 256), device='cuda:0', dtype=torch.float32)
    mul_437 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    unsqueeze_274 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_449 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cuda:0', dtype=torch.float32)
    unsqueeze_286 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_298 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_477 = rand_strided((8, 256, 16, 16), (65536, 1, 4096, 256), device='cuda:0', dtype=torch.float32)
    unsqueeze_310 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_489 = rand_strided((8, 1024, 16, 16), (262144, 1, 16384, 1024), device='cuda:0', dtype=torch.float32)
    unsqueeze_322 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_334 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_346 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_526 = rand_strided((8, 256, 32, 32), (262144, 1, 8192, 256), device='cuda:0', dtype=torch.float32)
    unsqueeze_358 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_538 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cuda:0', dtype=torch.float32)
    unsqueeze_370 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_550 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    unsqueeze_382 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_110 = rand_strided((32, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_111 = rand_strided((32, 32, 1024), (32768, 1024, 1), device='cuda:0', dtype=torch.float32)
    alias_27 = rand_strided((32, 1024, 1024), (1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_115 = rand_strided((63, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    permute_121 = rand_strided((63, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    permute_123 = rand_strided((32, 32, 1024), (32768, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_124 = rand_strided((32, 1024, 32), (32768, 1, 1024), device='cuda:0', dtype=torch.float32)
    mul_565 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    unsqueeze_394 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_577 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cuda:0', dtype=torch.float32)
    unsqueeze_406 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_418 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_605 = rand_strided((8, 128, 32, 32), (131072, 1, 4096, 128), device='cuda:0', dtype=torch.float32)
    unsqueeze_430 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_617 = rand_strided((8, 512, 32, 32), (524288, 1, 16384, 512), device='cuda:0', dtype=torch.float32)
    unsqueeze_442 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_454 = rand_strided((1, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_466 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_654 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cuda:0', dtype=torch.float32)
    unsqueeze_478 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_666 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cuda:0', dtype=torch.float32)
    unsqueeze_490 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_502 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_694 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    unsqueeze_514 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_706 = rand_strided((8, 256, 64, 64), (1048576, 1, 16384, 256), device='cuda:0', dtype=torch.float32)
    unsqueeze_526 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_538 = rand_strided((1, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_550 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_743 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    unsqueeze_562 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_755 = rand_strided((8, 64, 64, 64), (262144, 1, 4096, 64), device='cuda:0', dtype=torch.float32)
    unsqueeze_574 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_767 = rand_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cuda:0', dtype=torch.float32)
    unsqueeze_586 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_779 = rand_strided((8, 24, 128, 128), (393216, 1, 3072, 24), device='cuda:0', dtype=torch.float32)
    unsqueeze_598 = rand_strided((1, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_61, primals_63, primals_65, primals_69, primals_71, primals_73, primals_75, primals_79, primals_81, primals_83, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_92, primals_94, primals_95, primals_96, primals_97, primals_98, primals_100, primals_102, primals_103, primals_104, primals_105, primals_107, primals_109, primals_110, primals_111, primals_112, primals_113, primals_115, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_125, primals_127, primals_128, primals_129, primals_130, primals_131, primals_133, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_263, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, mul_15, convolution_2, squeeze_7, mul_23, convolution_3, squeeze_10, mul_31, convolution_4, squeeze_13, add_24, mean, relu, convolution_6, mul_40, convolution_7, squeeze_16, convolution_8, squeeze_19, mul_55, convolution_9, squeeze_22, mul_63, convolution_10, squeeze_25, add_45, mean_1, relu_1, convolution_12, mul_72, convolution_13, squeeze_28, mul_80, convolution_14, squeeze_31, mul_88, convolution_15, squeeze_34, add_61, mean_2, relu_2, convolution_17, mul_97, convolution_18, squeeze_37, convolution_19, squeeze_40, mul_112, convolution_20, squeeze_43, mul_120, convolution_21, squeeze_46, add_82, mean_3, relu_3, convolution_23, mul_129, convolution_24, squeeze_49, mul_137, convolution_25, squeeze_52, mul_145, view_7, view_13, bmm_1, squeeze_55, mul_154, convolution_27, squeeze_58, mul_162, convolution_28, squeeze_61, mul_170, convolution_29, squeeze_64, add_116, mean_4, relu_4, convolution_31, mul_179, convolution_32, squeeze_67, convolution_33, squeeze_70, mul_194, convolution_34, squeeze_73, mul_202, convolution_35, squeeze_76, add_137, mean_5, relu_5, convolution_37, mul_211, convolution_38, squeeze_79, mul_219, convolution_39, squeeze_82, mul_227, view_31, view_37, bmm_3, squeeze_85, mul_236, convolution_41, squeeze_88, mul_244, convolution_42, squeeze_91, mul_252, view_55, view_61, view_71, avg_pool2d, squeeze_94, mul_261, convolution_44, squeeze_97, convolution_45, squeeze_100, mul_276, convolution_46, squeeze_103, mul_284, view_79, view_85, bmm_7, squeeze_106, mul_293, convolution_48, squeeze_109, mul_301, convolution_49, squeeze_112, clone_62, permute_33, mul_311, unsqueeze_154, mul_323, unsqueeze_166, mul_335, unsqueeze_178, permute_41, permute_42, alias_16, permute_46, permute_52, permute_54, permute_55, mul_350, unsqueeze_190, mul_362, unsqueeze_202, unsqueeze_214, mul_383, unsqueeze_226, permute_62, permute_63, alias_17, permute_67, permute_73, permute_75, permute_76, mul_398, unsqueeze_238, mul_410, unsqueeze_250, mul_422, unsqueeze_262, permute_83, permute_84, alias_18, permute_88, permute_94, permute_96, permute_97, mul_437, unsqueeze_274, mul_449, unsqueeze_286, unsqueeze_298, mul_477, unsqueeze_310, mul_489, unsqueeze_322, unsqueeze_334, unsqueeze_346, mul_526, unsqueeze_358, mul_538, unsqueeze_370, mul_550, unsqueeze_382, permute_110, permute_111, alias_27, permute_115, permute_121, permute_123, permute_124, mul_565, unsqueeze_394, mul_577, unsqueeze_406, unsqueeze_418, mul_605, unsqueeze_430, mul_617, unsqueeze_442, unsqueeze_454, unsqueeze_466, mul_654, unsqueeze_478, mul_666, unsqueeze_490, unsqueeze_502, mul_694, unsqueeze_514, mul_706, unsqueeze_526, unsqueeze_538, unsqueeze_550, mul_743, unsqueeze_562, mul_755, unsqueeze_574, mul_767, unsqueeze_586, mul_779, unsqueeze_598, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('sebotnet33ts_256', benchmark_compiled_module)
