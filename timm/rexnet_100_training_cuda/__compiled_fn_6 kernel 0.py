
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


# kernel path: /tmp/torchinductor_youkaichao/af/cafofysdi24imukzirxx7gaj23eab7l6plf6gwtufaikrontxatg.py
# Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.native_batch_norm_backward]

triton_red_fused_div_mul_native_batch_norm_backward_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_mul_native_batch_norm_backward_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5120
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x0 + (1280*(r2 // 49)) + (2560*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (1280*r2) + (125440*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (1280*r2) + (125440*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 49.0
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

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/eh/cehrsnopw367omjfjk7kuornxzw6ixtbveizykuwiflwsrfhp2zm.py
# Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.native_batch_norm_backward]

triton_per_fused_div_mul_native_batch_norm_backward_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_mul_native_batch_norm_backward_1', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/55/c55vlqguxwrd6bwldngmpyypfdyu2r5s4qqviiuld4gkjmc2a6wa.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_mul_native_batch_norm_backward_2', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/o7/co7btzntqvf2rxwfzn35ens2722pwwisameo42kymi2q7tt46dpx.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_div_mul_native_batch_norm_backward_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_div_mul_native_batch_norm_backward_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 1280
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y1 = (yindex // 49)
    y3 = yindex
    y0 = yindex % 49
    tmp0 = tl.load(in_ptr0 + (x2 + (1280*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2 + (1280*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + (1280*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 49.0
    tmp2 = tmp0 / tmp1
    tmp4 = tmp2 * tmp3
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
    tl.store(out_ptr0 + (y0 + (49*x2) + (62720*y1)), tmp21, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m3/cm3zkpisefbsb4f67i4ccxq7fyo2qrbpz664bhad3t2ovstyowjx.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]

triton_per_fused_add_native_batch_norm_backward_slice_backward_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_slice_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 185
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex % 49
    r2 = (rindex // 49)
    tmp0 = x0
    tmp1 = tl.full([1], 174, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (r1 + (49*x0) + (9065*r2)), rmask & tmp2 & xmask, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = tmp0 < tmp1
    tmp9 = tl.load(in_ptr0 + (r1 + (49*x0) + (9065*r2)), rmask & tmp8 & xmask, other=0.0)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = tl.where(tmp8, tmp11, tmp6)
    tmp13 = tmp7 + tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tl.store(out_ptr0 + (x0), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wf/cwff4sadszf4srtzsism7h5cet4aqpsvcwmfsddwgsjwd6qmarpc.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]

triton_red_fused_add_native_batch_norm_backward_slice_backward_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_slice_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 740
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 185
    x1 = (xindex // 185)
    tmp15 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp14 = tl.load(in_ptr1 + (x0 + (185*r2) + (18130*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 174, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.load(in_ptr0 + ((49*x0) + (9065*(r2 // 49)) + (18130*x1) + (r2 % 49)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tmp0 < tmp1
        tmp9 = tl.load(in_ptr0 + ((49*x0) + (9065*(r2 // 49)) + (18130*x1) + (r2 % 49)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
        tmp11 = tl.where(tmp8, tmp9, tmp10)
        tmp12 = tl.where(tmp8, tmp11, tmp6)
        tmp13 = tmp7 + tmp12
        tmp16 = tmp14 - tmp15
        tmp17 = tmp13 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j5/cj5zdhk55wo65zqnfssbhd4z37wxjfpmrixjo6uo4i7c4fekultq.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]

triton_per_fused_add_native_batch_norm_backward_slice_backward_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_slice_backward_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 185
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (185*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sk/csktsfqyzaf2irtbaox7mlof4yohfeuzhokbi6dojekhwgv5jm3c.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1480
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 185
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 185)
    tmp14 = tl.load(in_ptr1 + (y0 + (185*x2) + (9065*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 174, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x2 + (49*y3)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = tmp0 < tmp1
    tmp9 = tl.load(in_ptr0 + (x2 + (49*y3)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = tl.where(tmp8, tmp11, tmp6)
    tmp13 = tmp7 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = 0.002551020408163265
    tmp19 = tmp17 * tmp18
    tmp21 = tmp20 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp16 * tmp22
    tmp24 = tmp13 - tmp23
    tmp26 = tmp25 * tmp18
    tmp27 = tmp24 - tmp26
    tmp29 = tmp20 * tmp28
    tmp30 = tmp27 * tmp29
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp30, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wi/cwipawuvrbindjuoitgrb3e42nea7bh7kiwkajowdqiclvgdwgrf.py
# Source Nodes: [sigmoid_12, x_319], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_12 => sigmoid_28
# x_319 => mul_448
triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_8', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8352
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1044
    x1 = (xindex // 1044)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1044*r2) + (51156*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tmp6 = 6.0
    tmp7 = tmp3 >= tmp6
    tmp8 = tmp5 | tmp7
    tmp10 = tl.where(tmp8, tmp4, tmp9)
    tmp11 = tmp10 * tmp0
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = 1.0
    tmp17 = tmp16 - tmp2
    tmp18 = tmp2 * tmp17
    tmp19 = tmp15 * tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wq/cwqia5iingkt4thfhao2ooeja6vhh5y5vxppn5rg7rildll5rnco.py
# Source Nodes: [getattr_l__mod___features___15___se_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
# getattr_l__mod___features___15___se_bn => var_mean_59
triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 87
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (87*r1)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr1 + (x0 + (87*r1)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr2 + (x0 + (87*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 8, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tmp21 = tl.where(tmp19, tmp18, tmp20)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp0 - tmp10
    tmp27 = tmp21 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None]
    tmp32 = 8.0
    tmp33 = tmp16 / tmp32
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tmp37 = tmp31 * tmp36
    tl.store(out_ptr4 + (x0), tmp37, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp25, xmask)
    tl.store(out_ptr2 + (x0), tmp31, xmask)
    tl.store(out_ptr3 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ej/cejfwkisstvpk7ln4kiybcet5pllbrvtuwxvlqwcwpqic7zirpzx.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 696
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 87
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr1 + (x2), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.125
    tmp10 = tmp8 * tmp9
    tmp12 = 8.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = tl.math.rsqrt(tmp15)
    tmp17 = tmp16 * tmp16
    tmp18 = tmp10 * tmp17
    tmp19 = tmp7 * tmp18
    tmp20 = tmp4 - tmp19
    tmp22 = tmp21 * tmp9
    tmp23 = tmp20 - tmp22
    tmp25 = tmp16 * tmp24
    tmp26 = tmp23 * tmp25
    tl.store(in_out_ptr0 + (x2), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rl/crld6shvnjwlmynnjuauatal3gciev3chz7xj5mcc3aof2wemqll.py
# Source Nodes: [sigmoid_12, x_319], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_12 => sigmoid_28
# x_319 => mul_448
triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4176
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1044
    x1 = (xindex // 1044)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp20 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1044*r2) + (102312*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (1044*(r2 // 49)) + (2088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr2 + ((49*x0) + (51156*(r2 // 49)) + (102312*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr3 + (x0 + (1044*(r2 // 49)) + (2088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr4 + (x0 + (1044*r2) + (102312*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = 6.0
        tmp7 = tmp3 >= tmp6
        tmp8 = tmp5 | tmp7
        tmp10 = tl.where(tmp8, tmp4, tmp9)
        tmp11 = tmp10 * tmp2
        tmp13 = 49.0
        tmp14 = tmp12 / tmp13
        tmp15 = tmp11 + tmp14
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


# kernel path: /tmp/torchinductor_youkaichao/um/cumy6znf6ymrtcpzc27bdtycwf7nx7aevouijnadrlbc2rkna25z.py
# Source Nodes: [sigmoid_12, x_319], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_12 => sigmoid_28
# x_319 => mul_448
triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1044
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1044*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3g/c3gwks6lslm6vvco77b2v4ak52dizkk67bcemexdixziptbhqo4l.py
# Source Nodes: [sigmoid_12, x_319], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_12 => sigmoid_28
# x_319 => mul_448
triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1044
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1044*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g4/cg4yvfyzy7jtr45jveghlcwdgmidvqskh7wptfottkawigpnjtog.py
# Source Nodes: [sigmoid_12, x_319], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_12 => sigmoid_28
# x_319 => mul_448
triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 1044
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 49)
    y0 = yindex % 49
    tmp0 = tl.load(in_ptr0 + (x2 + (1044*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (1044*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (y0 + (49*x2) + (51156*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2 + (1044*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (1044*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tmp6 = 6.0
    tmp7 = tmp3 >= tmp6
    tmp8 = tmp5 | tmp7
    tmp10 = tl.where(tmp8, tmp4, tmp9)
    tmp11 = tmp10 * tmp2
    tmp13 = 49.0
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 0.002551020408163265
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tmp31 = tmp22 * tmp30
    tmp32 = tmp29 * tmp31
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (1044*y3)), tmp32, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m2/cm2gbgin7elyb4iaw67s5c6d5rnkpgqwfeygzjk5htgike3lqhz7.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4176
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1044
    x1 = (xindex // 1044)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((49*x0) + (51156*(r2 // 49)) + (102312*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (1044*r2) + (102312*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (1044*r2) + (102312*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7g/c7gmyykgtmzpdtteffethfua2sjo5rfwegfhbqq2rrry5vggydai.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8352
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1044
    y1 = (yindex // 1044)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (1044*x2) + (51156*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (1044*x2) + (51156*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (49*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gw/cgw4gpz4uxbadffqj7uyhxmg32u6vzzhuzqbe6peb7v277zcspxq.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]

triton_per_fused_add_native_batch_norm_backward_slice_backward_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_slice_backward_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 174
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex % 49
    r2 = (rindex // 49)
    r3 = rindex
    tmp22 = tl.load(in_ptr2 + (x0 + (174*r3)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 162, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (r1 + (49*x0) + (9065*r2)), rmask & tmp2 & xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (r1 + (49*x0) + (8526*r2)), rmask & tmp2 & xmask, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tmp0 < tmp1
    tmp11 = tl.load(in_ptr0 + (r1 + (49*x0) + (9065*r2)), rmask & tmp10 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr1 + (r1 + (49*x0) + (8526*r2)), rmask & tmp10 & xmask, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp10, tmp15, tmp8)
    tmp17 = tmp9 + tmp16
    tmp18 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp20 = tl.where(rmask & xmask, tmp18, 0)
    tmp21 = triton_helpers.promote_to_tensor(tl.sum(tmp20, 0))
    tmp24 = tmp22 - tmp23
    tmp25 = tmp17 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp31 = tmp29 * tmp30
    tl.store(out_ptr2 + (x0), tmp31, xmask)
    tl.store(out_ptr0 + (x0), tmp21, xmask)
    tl.store(out_ptr1 + (x0), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cm/ccmr362furd5w37vbimb7ghygyb3jc3k62oc2elntcnvwqeufue5.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1392
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 174
    x2 = xindex
    y1 = (yindex // 174)
    y3 = yindex
    tmp18 = tl.load(in_ptr2 + (y0 + (174*x2) + (8526*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 162, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x2 + (49*y0) + (9065*y1)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr1 + (x2 + (49*y3)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tmp0 < tmp1
    tmp11 = tl.load(in_ptr0 + (x2 + (49*y0) + (9065*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr1 + (x2 + (49*y3)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp10, tmp15, tmp8)
    tmp17 = tmp9 + tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = 0.002551020408163265
    tmp23 = tmp21 * tmp22
    tmp25 = tmp24 * tmp24
    tmp26 = tmp23 * tmp25
    tmp27 = tmp20 * tmp26
    tmp28 = tmp17 - tmp27
    tmp30 = tmp29 * tmp22
    tmp31 = tmp28 - tmp30
    tmp33 = tmp24 * tmp32
    tmp34 = tmp31 * tmp33
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp34, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gs/cgs72qi2cxyg64hehvkm4sphilkxxe6pgcgrsqtcw3sib25kq6cn.py
# Source Nodes: [sigmoid_11, x_298], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_11 => sigmoid_26
# x_298 => mul_418
triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_19', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 7776
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 972
    x1 = (xindex // 972)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (972*r2) + (47628*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tmp6 = 6.0
    tmp7 = tmp3 >= tmp6
    tmp8 = tmp5 | tmp7
    tmp10 = tl.where(tmp8, tmp4, tmp9)
    tmp11 = tmp10 * tmp0
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = 1.0
    tmp17 = tmp16 - tmp2
    tmp18 = tmp2 * tmp17
    tmp19 = tmp15 * tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v2/cv2s5sqtxx6tzxd7xpbvg6pfpvstp22dk3hsvdwrnynmvtpepbag.py
# Source Nodes: [getattr_l__mod___features___14___se_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
# getattr_l__mod___features___14___se_bn => var_mean_55
triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 81
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (81*r1)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr1 + (x0 + (81*r1)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr2 + (x0 + (81*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 8, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tmp21 = tl.where(tmp19, tmp18, tmp20)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp0 - tmp10
    tmp27 = tmp21 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None]
    tmp32 = 8.0
    tmp33 = tmp16 / tmp32
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tmp37 = tmp31 * tmp36
    tl.store(out_ptr4 + (x0), tmp37, xmask)
    tl.store(out_ptr0 + (x0), tmp16, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
    tl.store(out_ptr2 + (x0), tmp25, xmask)
    tl.store(out_ptr3 + (x0), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wh/cwhvh5s2wo6sd7dsa6sva6kpfzjhd7q6awljllpvd2k4rv6mdwqn.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 81
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr1 + (x2), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.125
    tmp10 = tmp8 * tmp9
    tmp12 = 8.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = tl.math.rsqrt(tmp15)
    tmp17 = tmp16 * tmp16
    tmp18 = tmp10 * tmp17
    tmp19 = tmp7 * tmp18
    tmp20 = tmp4 - tmp19
    tmp22 = tmp21 * tmp9
    tmp23 = tmp20 - tmp22
    tmp25 = tmp16 * tmp24
    tmp26 = tmp23 * tmp25
    tl.store(in_out_ptr0 + (x2), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7z/c7zzuy42oywqzutmajcfa55od5dtdpeikdnqskbr6bdwpkivztfn.py
# Source Nodes: [sigmoid_11, x_298], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_11 => sigmoid_26
# x_298 => mul_418
triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3888
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 972
    x1 = (xindex // 972)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp20 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (972*r2) + (95256*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (972*(r2 // 49)) + (1944*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr2 + ((49*x0) + (47628*(r2 // 49)) + (95256*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr3 + (x0 + (972*(r2 // 49)) + (1944*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr4 + (x0 + (972*r2) + (95256*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = 6.0
        tmp7 = tmp3 >= tmp6
        tmp8 = tmp5 | tmp7
        tmp10 = tl.where(tmp8, tmp4, tmp9)
        tmp11 = tmp10 * tmp2
        tmp13 = 49.0
        tmp14 = tmp12 / tmp13
        tmp15 = tmp11 + tmp14
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


# kernel path: /tmp/torchinductor_youkaichao/p6/cp6utsscs2k7j26swfpqaqqjtwr5k2ica4fijasn4zzx2cgufyzx.py
# Source Nodes: [sigmoid_11, x_298], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_11 => sigmoid_26
# x_298 => mul_418
triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 972
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (972*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lx/clxakqfbe7ozl4ifkgb6665w7q46y6lcwwkl4cl5dobg55rinnaf.py
# Source Nodes: [sigmoid_11, x_298], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_11 => sigmoid_26
# x_298 => mul_418
triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 972
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (972*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jp/cjphg7hrp37om45tqhhawrjkwdas4fsmw2rmxtilvb2rhjlb6bat.py
# Source Nodes: [sigmoid_11, x_298], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_11 => sigmoid_26
# x_298 => mul_418
triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_25', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 972
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 49)
    y0 = yindex % 49
    tmp0 = tl.load(in_ptr0 + (x2 + (972*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (972*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (y0 + (49*x2) + (47628*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2 + (972*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (972*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tmp6 = 6.0
    tmp7 = tmp3 >= tmp6
    tmp8 = tmp5 | tmp7
    tmp10 = tl.where(tmp8, tmp4, tmp9)
    tmp11 = tmp10 * tmp2
    tmp13 = 49.0
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 0.002551020408163265
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tmp31 = tmp22 * tmp30
    tmp32 = tmp29 * tmp31
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (972*y3)), tmp32, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wg/cwgavuiodvwt34okotrwp5mgdd6t35ggazacgaxn2f6zdrnc4vkm.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3888
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 972
    x1 = (xindex // 972)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((49*x0) + (47628*(r2 // 49)) + (95256*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (972*r2) + (95256*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (972*r2) + (95256*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ep/cepbnj5zm3kotrhds5e247jbgsglamicwhot6yssc23u34jspzxo.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_27', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7776
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 972
    y1 = (yindex // 972)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (972*x2) + (47628*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (972*x2) + (47628*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (49*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ia/ciaq3zpbnt7rhnanos3ds5lnvi7mhutjqattt725gj63arvsernw.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]

triton_per_fused_add_native_batch_norm_backward_slice_backward_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_slice_backward_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 162
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex % 49
    r2 = (rindex // 49)
    r3 = rindex
    tmp26 = tl.load(in_ptr3 + (x0 + (162*r3)), rmask & xmask, other=0.0)
    tmp27 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 151, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (r1 + (49*x0) + (9065*r2)), rmask & tmp2 & xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (r1 + (49*x0) + (8526*r2)), rmask & tmp2 & xmask, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (r1 + (49*x0) + (7938*r2)), rmask & tmp2 & xmask, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = 0.0
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = tmp0 < tmp1
    tmp13 = tl.load(in_ptr0 + (r1 + (49*x0) + (9065*r2)), rmask & tmp12 & xmask, other=0.0)
    tmp14 = tl.load(in_ptr1 + (r1 + (49*x0) + (8526*r2)), rmask & tmp12 & xmask, other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.load(in_ptr2 + (r1 + (49*x0) + (7938*r2)), rmask & tmp12 & xmask, other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp12, tmp17, tmp18)
    tmp20 = tl.where(tmp12, tmp19, tmp10)
    tmp21 = tmp11 + tmp20
    tmp22 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp28 = tmp26 - tmp27
    tmp29 = tmp21 * tmp28
    tmp30 = tl.broadcast_to(tmp29, [RBLOCK])
    tmp32 = tl.where(rmask & xmask, tmp30, 0)
    tmp33 = triton_helpers.promote_to_tensor(tl.sum(tmp32, 0))
    tmp35 = tmp33 * tmp34
    tl.store(out_ptr2 + (x0), tmp35, xmask)
    tl.store(out_ptr0 + (x0), tmp25, xmask)
    tl.store(out_ptr1 + (x0), tmp33, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3n/c3n2i4meuacu5coczwfgfs76utwgnx7b4f5zbr7bvyaqaza3d4l2.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]

triton_poi_fused_add_native_batch_norm_backward_slice_backward_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_slice_backward_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1296
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 162
    x2 = xindex
    y1 = (yindex // 162)
    y3 = yindex
    tmp22 = tl.load(in_ptr3 + (y0 + (162*x2) + (7938*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 151, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x2 + (49*y0) + (9065*y1)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr1 + (x2 + (49*y0) + (8526*y1)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (x2 + (49*y3)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = 0.0
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = tmp0 < tmp1
    tmp13 = tl.load(in_ptr0 + (x2 + (49*y0) + (9065*y1)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr1 + (x2 + (49*y0) + (8526*y1)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.load(in_ptr2 + (x2 + (49*y3)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp12, tmp17, tmp18)
    tmp20 = tl.where(tmp12, tmp19, tmp10)
    tmp21 = tmp11 + tmp20
    tmp24 = tmp22 - tmp23
    tmp26 = 0.002551020408163265
    tmp27 = tmp25 * tmp26
    tmp29 = tmp28 * tmp28
    tmp30 = tmp27 * tmp29
    tmp31 = tmp24 * tmp30
    tmp32 = tmp21 - tmp31
    tmp34 = tmp33 * tmp26
    tmp35 = tmp32 - tmp34
    tmp37 = tmp28 * tmp36
    tmp38 = tmp35 * tmp37
    tl.store(out_ptr0 + (x2 + (49*y3)), tmp38, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a2/ca2albady7dddiqlgpspi26oqjmd3qislg3apjlqgmmkc2qui4cp.py
# Source Nodes: [sigmoid_10, x_277], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_10 => sigmoid_24
# x_277 => mul_388
triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_30', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 7248
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 906
    x1 = (xindex // 906)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (906*r2) + (44394*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tmp6 = 6.0
    tmp7 = tmp3 >= tmp6
    tmp8 = tmp5 | tmp7
    tmp10 = tl.where(tmp8, tmp4, tmp9)
    tmp11 = tmp10 * tmp0
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = 1.0
    tmp17 = tmp16 - tmp2
    tmp18 = tmp2 * tmp17
    tmp19 = tmp15 * tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/im/cimy7cjfotydjuc6bi5ag5cblhyr7mn24km77ajoytmfcsomupla.py
# Source Nodes: [getattr_l__mod___features___13___se_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
# getattr_l__mod___features___13___se_bn => var_mean_51
triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 75
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (75*r1)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr1 + (x0 + (75*r1)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr2 + (x0 + (75*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 8, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tmp21 = tl.where(tmp19, tmp18, tmp20)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp0 - tmp10
    tmp27 = tmp21 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None]
    tmp32 = 8.0
    tmp33 = tmp16 / tmp32
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tmp37 = tmp31 * tmp36
    tl.store(out_ptr4 + (x0), tmp37, xmask)
    tl.store(out_ptr0 + (x0), tmp16, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
    tl.store(out_ptr2 + (x0), tmp25, xmask)
    tl.store(out_ptr3 + (x0), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oy/coyohns7p67jgunamcdxe3w4arttziyuijq7gukbgvbbvxt37va7.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 75
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr1 + (x2), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.125
    tmp10 = tmp8 * tmp9
    tmp12 = 8.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = tl.math.rsqrt(tmp15)
    tmp17 = tmp16 * tmp16
    tmp18 = tmp10 * tmp17
    tmp19 = tmp7 * tmp18
    tmp20 = tmp4 - tmp19
    tmp22 = tmp21 * tmp9
    tmp23 = tmp20 - tmp22
    tmp25 = tmp16 * tmp24
    tmp26 = tmp23 * tmp25
    tl.store(in_out_ptr0 + (x2), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u2/cu2d5hoiizkw5vkxokrmcalokqzjir7s6pupvd5772hm4thfbfl3.py
# Source Nodes: [sigmoid_10, x_277], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_10 => sigmoid_24
# x_277 => mul_388
triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3624
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 906
    x1 = (xindex // 906)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp20 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (906*r2) + (88788*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (906*(r2 // 49)) + (1812*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr2 + ((49*x0) + (44394*(r2 // 49)) + (88788*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr3 + (x0 + (906*(r2 // 49)) + (1812*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr4 + (x0 + (906*r2) + (88788*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = 6.0
        tmp7 = tmp3 >= tmp6
        tmp8 = tmp5 | tmp7
        tmp10 = tl.where(tmp8, tmp4, tmp9)
        tmp11 = tmp10 * tmp2
        tmp13 = 49.0
        tmp14 = tmp12 / tmp13
        tmp15 = tmp11 + tmp14
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


# kernel path: /tmp/torchinductor_youkaichao/re/cretb23womssfukhppex6knuybswee5xcp5ydc2b56rtz4v6xono.py
# Source Nodes: [sigmoid_10, x_277], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_10 => sigmoid_24
# x_277 => mul_388
triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 906
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (906*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ik/cikaqgixy4oigbkg3nzarzukc4rimivh6tfnuuuunr56pxceiqip.py
# Source Nodes: [sigmoid_10, x_277], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_10 => sigmoid_24
# x_277 => mul_388
triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 906
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (906*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m3/cm3pdfjuaukefvizchamp3ucqflxubwstuksgj2q7y7y4imrk67z.py
# Source Nodes: [sigmoid_10, x_277], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_10 => sigmoid_24
# x_277 => mul_388
triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_36', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 906
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 49)
    y0 = yindex % 49
    tmp0 = tl.load(in_ptr0 + (x2 + (906*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (906*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (y0 + (49*x2) + (44394*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2 + (906*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (906*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tmp6 = 6.0
    tmp7 = tmp3 >= tmp6
    tmp8 = tmp5 | tmp7
    tmp10 = tl.where(tmp8, tmp4, tmp9)
    tmp11 = tmp10 * tmp2
    tmp13 = 49.0
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 0.002551020408163265
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tmp31 = tmp22 * tmp30
    tmp32 = tmp29 * tmp31
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (906*y3)), tmp32, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ag/cagwme3rarjzopgqesezbml5ezfbf7h3gxwelzkd6iqhf63ao6bl.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3624
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 906
    x1 = (xindex // 906)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((49*x0) + (44394*(r2 // 49)) + (88788*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (906*r2) + (88788*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (906*r2) + (88788*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w7/cw7z2ux3gmoy67zroczetcxac43oc7dynvyhsdn5trf5dtnikfyr.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_38', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7248
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 906
    y1 = (yindex // 906)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (906*x2) + (44394*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (906*x2) + (44394*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (49*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2j/c2jnsxomcoud67yisai3qdzsch4hy4t4tazfijniup55zoxoglci.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]

triton_per_fused_add_native_batch_norm_backward_slice_backward_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_slice_backward_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 151
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex % 49
    r2 = (rindex // 49)
    r3 = rindex
    tmp30 = tl.load(in_ptr4 + (x0 + (151*r3)), rmask & xmask, other=0.0)
    tmp31 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 140, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (r1 + (49*x0) + (9065*r2)), rmask & tmp2 & xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (r1 + (49*x0) + (8526*r2)), rmask & tmp2 & xmask, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (r1 + (49*x0) + (7938*r2)), rmask & tmp2 & xmask, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr3 + (r1 + (49*x0) + (7399*r2)), rmask & tmp2 & xmask, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = 0.0
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = tmp0 < tmp1
    tmp15 = tl.load(in_ptr0 + (r1 + (49*x0) + (9065*r2)), rmask & tmp14 & xmask, other=0.0)
    tmp16 = tl.load(in_ptr1 + (r1 + (49*x0) + (8526*r2)), rmask & tmp14 & xmask, other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.load(in_ptr2 + (r1 + (49*x0) + (7938*r2)), rmask & tmp14 & xmask, other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.load(in_ptr3 + (r1 + (49*x0) + (7399*r2)), rmask & tmp14 & xmask, other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp14, tmp21, tmp22)
    tmp24 = tl.where(tmp14, tmp23, tmp12)
    tmp25 = tmp13 + tmp24
    tmp26 = tl.broadcast_to(tmp25, [RBLOCK])
    tmp28 = tl.where(rmask & xmask, tmp26, 0)
    tmp29 = triton_helpers.promote_to_tensor(tl.sum(tmp28, 0))
    tmp32 = tmp30 - tmp31
    tmp33 = tmp25 * tmp32
    tmp34 = tl.broadcast_to(tmp33, [RBLOCK])
    tmp36 = tl.where(rmask & xmask, tmp34, 0)
    tmp37 = triton_helpers.promote_to_tensor(tl.sum(tmp36, 0))
    tmp39 = tmp37 * tmp38
    tl.store(out_ptr2 + (x0), tmp39, xmask)
    tl.store(out_ptr0 + (x0), tmp29, xmask)
    tl.store(out_ptr1 + (x0), tmp37, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rl/crlefuf4bpjav3gwgbjcrrfbx7gqpk6hodfop4dizzdawvuxantf.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_40', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1208
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 151
    x2 = xindex
    y1 = (yindex // 151)
    y3 = yindex
    tmp26 = tl.load(in_ptr4 + (y0 + (151*x2) + (7399*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr9 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 140, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x2 + (49*y0) + (9065*y1)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr1 + (x2 + (49*y0) + (8526*y1)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (x2 + (49*y0) + (7938*y1)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr3 + (x2 + (49*y3)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = 0.0
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = tmp0 < tmp1
    tmp15 = tl.load(in_ptr0 + (x2 + (49*y0) + (9065*y1)), tmp14 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr1 + (x2 + (49*y0) + (8526*y1)), tmp14 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.load(in_ptr2 + (x2 + (49*y0) + (7938*y1)), tmp14 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.load(in_ptr3 + (x2 + (49*y3)), tmp14 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp14, tmp21, tmp22)
    tmp24 = tl.where(tmp14, tmp23, tmp12)
    tmp25 = tmp13 + tmp24
    tmp28 = tmp26 - tmp27
    tmp30 = 0.002551020408163265
    tmp31 = tmp29 * tmp30
    tmp33 = tmp32 * tmp32
    tmp34 = tmp31 * tmp33
    tmp35 = tmp28 * tmp34
    tmp36 = tmp25 - tmp35
    tmp38 = tmp37 * tmp30
    tmp39 = tmp36 - tmp38
    tmp41 = tmp32 * tmp40
    tmp42 = tmp39 * tmp41
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (49*y3)), tmp42, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wb/cwboeobladavfg7vpnsb3hkxpd65rb3p4qde6tnpmbjexrokwbar.py
# Source Nodes: [sigmoid_9, x_256], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_9 => sigmoid_22
# x_256 => mul_358
triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_41', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6720
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 840
    x1 = (xindex // 840)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (840*r2) + (41160*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tmp6 = 6.0
    tmp7 = tmp3 >= tmp6
    tmp8 = tmp5 | tmp7
    tmp10 = tl.where(tmp8, tmp4, tmp9)
    tmp11 = tmp10 * tmp0
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = 1.0
    tmp17 = tmp16 - tmp2
    tmp18 = tmp2 * tmp17
    tmp19 = tmp15 * tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rb/crbdputp2nxxkbx5enzpvrwrwjawwll5hylvmr2w5yte7jxz75ya.py
# Source Nodes: [getattr_l__mod___features___12___se_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
# getattr_l__mod___features___12___se_bn => var_mean_47
triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 70
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (70*r1)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr1 + (x0 + (70*r1)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr2 + (x0 + (70*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 8, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tmp21 = tl.where(tmp19, tmp18, tmp20)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp0 - tmp10
    tmp27 = tmp21 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None]
    tmp32 = 8.0
    tmp33 = tmp16 / tmp32
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tmp37 = tmp31 * tmp36
    tl.store(out_ptr4 + (x0), tmp37, xmask)
    tl.store(out_ptr0 + (x0), tmp16, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
    tl.store(out_ptr2 + (x0), tmp25, xmask)
    tl.store(out_ptr3 + (x0), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mu/cmuedhvshx4hrlcqycwm767q6qkzrjao7heeusmhvk5inzv2pojx.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_43', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 70
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr1 + (x2), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.125
    tmp10 = tmp8 * tmp9
    tmp12 = 8.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = tl.math.rsqrt(tmp15)
    tmp17 = tmp16 * tmp16
    tmp18 = tmp10 * tmp17
    tmp19 = tmp7 * tmp18
    tmp20 = tmp4 - tmp19
    tmp22 = tmp21 * tmp9
    tmp23 = tmp20 - tmp22
    tmp25 = tmp16 * tmp24
    tmp26 = tmp23 * tmp25
    tl.store(in_out_ptr0 + (x2), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4a/c4app4sgd7bw2oae2yfzwppaijo2clfzvzenen62prcs4dkzhq5y.py
# Source Nodes: [sigmoid_9, x_256], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_9 => sigmoid_22
# x_256 => mul_358
triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3360
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 840
    x1 = (xindex // 840)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp20 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (840*r2) + (82320*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (840*(r2 // 49)) + (1680*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr2 + ((49*x0) + (41160*(r2 // 49)) + (82320*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr3 + (x0 + (840*(r2 // 49)) + (1680*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr4 + (x0 + (840*r2) + (82320*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = 6.0
        tmp7 = tmp3 >= tmp6
        tmp8 = tmp5 | tmp7
        tmp10 = tl.where(tmp8, tmp4, tmp9)
        tmp11 = tmp10 * tmp2
        tmp13 = 49.0
        tmp14 = tmp12 / tmp13
        tmp15 = tmp11 + tmp14
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


# kernel path: /tmp/torchinductor_youkaichao/aq/caqrcen5g4fktkdcukzscshvrjnh36mhouix7vgbzflpaez2awaq.py
# Source Nodes: [sigmoid_9, x_256], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_9 => sigmoid_22
# x_256 => mul_358
triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 840
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (840*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h6/ch6dw5qyzr7qbkmfxs3zuqhsnw6ei6nlchx4z2mwfopdck6dx3pq.py
# Source Nodes: [sigmoid_9, x_256], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_9 => sigmoid_22
# x_256 => mul_358
triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 840
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (840*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4n/c4nm5dbxrjrp2ak2hfcne3rb5an5xbpoontngv5h4lioxxkwpnfe.py
# Source Nodes: [sigmoid_9, x_256], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_9 => sigmoid_22
# x_256 => mul_358
triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_47', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 840
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 49)
    y0 = yindex % 49
    tmp0 = tl.load(in_ptr0 + (x2 + (840*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (840*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (y0 + (49*x2) + (41160*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2 + (840*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (840*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tmp6 = 6.0
    tmp7 = tmp3 >= tmp6
    tmp8 = tmp5 | tmp7
    tmp10 = tl.where(tmp8, tmp4, tmp9)
    tmp11 = tmp10 * tmp2
    tmp13 = 49.0
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 0.002551020408163265
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tmp31 = tmp22 * tmp30
    tmp32 = tmp29 * tmp31
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (840*y3)), tmp32, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uz/cuzyytete6es2cbgfyc4ahd33qh4n7uvhtvdj7jcew3wzut64n6z.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_48', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3360
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 840
    x1 = (xindex // 840)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((49*x0) + (41160*(r2 // 49)) + (82320*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (840*r2) + (82320*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (840*r2) + (82320*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sq/csqebuw7yrq27h5isbadyceazr7jkfou4pm6eh2r2trmmml6rc5f.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_49', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6720
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 840
    y1 = (yindex // 840)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (840*x2) + (41160*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (840*x2) + (41160*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (49*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/43/c4357xhfpdgys4n5jlafdbt6kz767i6lfm36jmt4joume4l3sxok.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_50', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 54880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 6860
    x1 = (xindex // 6860)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (9065*x1)), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (8526*x1)), xmask)
    tmp3 = tl.load(in_ptr2 + (x0 + (7938*x1)), xmask)
    tmp5 = tl.load(in_ptr3 + (x0 + (7399*x1)), xmask)
    tmp7 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vv/cvvzc5vgvpfoyhiperqfniowd5lzhl6nb5fzvsikj6xlumx5vsdu.py
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
    size_hints=[256, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 140
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (6860*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cf/ccf3l7adt3yzubrcb7eqb4ihidvupmwc3jy5fp3uqtnvbvb6aci2.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_52 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_52', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 560
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 140
    x1 = (xindex // 140)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((49*x0) + (6860*(r2 // 49)) + (13720*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (140*r2) + (13720*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t4/ct4lm3urofff7mww6h7nylkqxt5bcvpsmob2strfufmhclel6dds.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 140
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (140*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w2/cw2gqzbf3zc5xbdpngocv3im7yxo3lxav3k7ejrr2hnp5cllmxl3.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_54 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_54', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1120
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 140
    y1 = (yindex // 140)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (140*x2) + (6860*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/ml/cml3wn5l6mznhgi7cblufyvfyetukmt7b35zaqknzvrimwmlek3r.py
# Source Nodes: [sigmoid_8, x_236], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_8 => sigmoid_20
# x_236 => mul_328
triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_55', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 768
    x1 = (xindex // 768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (37632*x1)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (r2 + (49*x3)), rmask, other=0.0)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tmp6 = 6.0
    tmp7 = tmp3 >= tmp6
    tmp8 = tmp5 | tmp7
    tmp10 = tl.where(tmp8, tmp4, tmp9)
    tmp11 = tmp10 * tmp0
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = 1.0
    tmp17 = tmp16 - tmp2
    tmp18 = tmp2 * tmp17
    tmp19 = tmp15 * tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2d/c2dv2jp5ibplzeucemq42vdq2sbbkgpg66soel5gq4ltpccowrq5.py
# Source Nodes: [getattr_l__mod___features___11___se_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
# getattr_l__mod___features___11___se_bn => var_mean_43
triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_56', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp17 = tl.load(in_ptr1 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr2 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 8, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tmp21 = tl.where(tmp19, tmp18, tmp20)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp0 - tmp10
    tmp27 = tmp21 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None]
    tmp32 = 8.0
    tmp33 = tmp16 / tmp32
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tmp37 = tmp31 * tmp36
    tl.store(out_ptr4 + (x0), tmp37, xmask)
    tl.store(out_ptr0 + (x0), tmp16, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
    tl.store(out_ptr2 + (x0), tmp25, xmask)
    tl.store(out_ptr3 + (x0), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2a/c2adxjnrbzemezkot3farm6hw3emyujs53dlt7r5pnkvrnizkufz.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_57 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_57', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
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
    tmp21 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.125
    tmp10 = tmp8 * tmp9
    tmp12 = 8.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = tl.math.rsqrt(tmp15)
    tmp17 = tmp16 * tmp16
    tmp18 = tmp10 * tmp17
    tmp19 = tmp7 * tmp18
    tmp20 = tmp4 - tmp19
    tmp22 = tmp21 * tmp9
    tmp23 = tmp20 - tmp22
    tmp25 = tmp16 * tmp24
    tmp26 = tmp23 * tmp25
    tl.store(in_out_ptr0 + (x2), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4i/c4iyacgaacu64i3jndhjamarhp67cnzv2tk3z2sizrq2aqkeshb4.py
# Source Nodes: [sigmoid_8, x_236], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_8 => sigmoid_20
# x_236 => mul_328
triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_58', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp20 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (75264*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (768*(r2 // 49)) + (1536*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr2 + ((49*x0) + (37632*(r2 // 49)) + (75264*x1) + (r2 % 49)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr3 + (x0 + (768*(r2 // 49)) + (1536*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr4 + (x0 + (768*r2) + (75264*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = 6.0
        tmp7 = tmp3 >= tmp6
        tmp8 = tmp5 | tmp7
        tmp10 = tl.where(tmp8, tmp4, tmp9)
        tmp11 = tmp10 * tmp2
        tmp13 = 49.0
        tmp14 = tmp12 / tmp13
        tmp15 = tmp11 + tmp14
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


# kernel path: /tmp/torchinductor_youkaichao/ro/croc2gjgnj7g4iuvcrmk6zi5vbzguxqnz3ci46ouqnaryqshqjzz.py
# Source Nodes: [sigmoid_8, x_236], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_8 => sigmoid_20
# x_236 => mul_328
triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_59', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g2/cg2ncahm6l7egoxaedmouwhjkdafohgmjnevxj2wtbsi2ca2gecj.py
# Source Nodes: [sigmoid_8, x_236], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_8 => sigmoid_20
# x_236 => mul_328
triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_60', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/up/cup3jpmisliaz4intdhnlvtyrh65kyzu52hhnulpix7rxli442td.py
# Source Nodes: [sigmoid_8, x_236], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_8 => sigmoid_20
# x_236 => mul_328
triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_61 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_61', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 49)
    y0 = yindex % 49
    tmp0 = tl.load(in_ptr0 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (y0 + (49*x2) + (37632*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2 + (768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tmp6 = 6.0
    tmp7 = tmp3 >= tmp6
    tmp8 = tmp5 | tmp7
    tmp10 = tl.where(tmp8, tmp4, tmp9)
    tmp11 = tmp10 * tmp2
    tmp13 = 49.0
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 0.002551020408163265
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tmp31 = tmp22 * tmp30
    tmp32 = tmp29 * tmp31
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (768*y3)), tmp32, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wq/cwq4dgm2rul27qunjgkcvl43zgci3qpzomoepl6xz77sex5ew73w.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_62', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/zz/czziwxkjjbemfsojg7rigkcbakh7owdl7qzg72q45gbybxdz2rrc.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_63', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/sd/csduvevs5qjxqm4gungskhbpbsrepf3zdyeu3foqqz4zj3smb5ks.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_64', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9984
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 768)
    x0 = xindex % 768
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x0) + (150528*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (768*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.load(in_ptr2 + (x0 + (768*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/of/cofv4gzt4yn6acn6aifdlmy6fvlzx4sxs7unwvfjge3asdb52ltk.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_65', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jn/cjninw6h7wkk6uypkym3i2m6xjst7m52tjtu5ldk2fcrlfnkxgiz.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_66 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_66', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (768*x2) + (150528*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (768*x2) + (150528*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b2/cb2tcppap5wpuhmifvsgibdyg7aup54yx4zmg5pz4kbp5fpjonmt.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]

triton_red_fused_add_native_batch_norm_backward_slice_backward_67 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_slice_backward_67', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 117, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.load(in_ptr0 + (r1 + (196*x0) + (25088*r2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tmp0 < tmp1
        tmp9 = tl.load(in_ptr0 + (r1 + (196*x0) + (25088*r2)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
        tmp11 = tl.where(tmp8, tmp9, tmp10)
        tmp12 = tl.where(tmp8, tmp11, tmp6)
        tmp13 = tmp7 + tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i6/ci6j4clrvy2dqfttfmjk2gxkkjw5gwn4apooslbet5kvnv4vn573.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]

triton_red_fused_add_native_batch_norm_backward_slice_backward_68 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_slice_backward_68', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1664
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.broadcast_to(x1, [XBLOCK, RBLOCK])
        tmp4 = tl.full([1, 1], 117, tl.int64)
        tmp5 = tmp3 >= tmp4
        tmp6 = tmp5 & tmp2
        tmp7 = tl.load(in_ptr0 + ((196*x1) + (25088*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
        tmp9 = tl.where(tmp6, tmp7, tmp8)
        tmp10 = 0.0
        tmp11 = tl.where(tmp5, tmp9, tmp10)
        tmp12 = tmp3 < tmp4
        tmp13 = tmp12 & tmp2
        tmp14 = tl.load(in_ptr0 + ((196*x1) + (25088*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
        tmp16 = tl.where(tmp13, tmp14, tmp15)
        tmp17 = tl.where(tmp12, tmp16, tmp10)
        tmp18 = tmp11 + tmp17
        tmp19 = tl.load(in_ptr1 + (x1 + (128*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.load(in_ptr2 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/gy/cgyj5hckz5y7dx6nymxrk64f4qk3ewdd5eun2ecnfdjjw342sd5g.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]

triton_per_fused_add_native_batch_norm_backward_slice_backward_69 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_slice_backward_69', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
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


# kernel path: /tmp/torchinductor_youkaichao/br/cbr4tepxputqhjrn3sqnjrlw6el3lfkg355buq2hwawxqq7w7uar.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_70 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_70', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 128
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 128)
    tmp14 = tl.load(in_ptr1 + (y0 + (128*x2) + (25088*y1)), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (y0), None, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 117, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x2 + (196*y3)), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = tmp0 < tmp1
    tmp9 = tl.load(in_ptr0 + (x2 + (196*y3)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = tl.where(tmp8, tmp11, tmp6)
    tmp13 = tmp7 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = 0.0006377551020408163
    tmp19 = tmp17 * tmp18
    tmp21 = tmp20 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp16 * tmp22
    tmp24 = tmp13 - tmp23
    tmp26 = tmp25 * tmp18
    tmp27 = tmp24 - tmp26
    tmp29 = tmp20 * tmp28
    tmp30 = tmp27 * tmp29
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sl/cslraqizd4oorwdoh7j7kss7qj5747osxva6gvr274j5empyo4cb.py
# Source Nodes: [sigmoid_7, x_215], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sum]
# sigmoid_7 => sigmoid_18
# x_215 => mul_298
triton_red_fused_hardtanh_backward_mul_sigmoid_sum_71 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_mul_sigmoid_sum_71', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 11232
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2) % 702
    x2 = (xindex // 1404)
    x4 = (xindex // 2)
    tmp1 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    x5 = xindex
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (702*r3) + (68796*x0) + (137592*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr2 + (r3 + (98*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = 6.0
        tmp7 = tmp3 >= tmp6
        tmp8 = tmp5 | tmp7
        tmp10 = tl.where(tmp8, tmp4, tmp9)
        tmp11 = tmp10 * tmp0
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ec/cec7eca7myeyqkbsyp5lvbwypbkrvtw4xmbw43by5jnirwppea6l.py
# Source Nodes: [sigmoid_7, x_215], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_7 => sigmoid_18
# x_215 => mul_298
triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_72 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_72', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5616
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


# kernel path: /tmp/torchinductor_youkaichao/yu/cyuee6cgzphm2vwqs2he5qz76vsog2htoikk5fzdwzd2qgzqp2jt.py
# Source Nodes: [getattr_l__mod___features___10___se_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
# getattr_l__mod___features___10___se_bn => var_mean_39
triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_73 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_73', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 58
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (58*r1)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr1 + (x0 + (58*r1)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr2 + (x0 + (58*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 8, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tmp21 = tl.where(tmp19, tmp18, tmp20)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp0 - tmp10
    tmp27 = tmp21 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None]
    tmp32 = 8.0
    tmp33 = tmp16 / tmp32
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tmp37 = tmp31 * tmp36
    tl.store(out_ptr4 + (x0), tmp37, xmask)
    tl.store(out_ptr0 + (x0), tmp16, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
    tl.store(out_ptr2 + (x0), tmp25, xmask)
    tl.store(out_ptr3 + (x0), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rm/crmglpjjlpyllf3me6blompmmtiahdqybnrzg4nvlgcpefxrtqqj.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_74 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_74', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 58
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr1 + (x2), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.125
    tmp10 = tmp8 * tmp9
    tmp12 = 8.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = tl.math.rsqrt(tmp15)
    tmp17 = tmp16 * tmp16
    tmp18 = tmp10 * tmp17
    tmp19 = tmp7 * tmp18
    tmp20 = tmp4 - tmp19
    tmp22 = tmp21 * tmp9
    tmp23 = tmp20 - tmp22
    tmp25 = tmp16 * tmp24
    tmp26 = tmp23 * tmp25
    tl.store(in_out_ptr0 + (x2), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ps/cpsel2agrxikuffhuldzmxgqbzzyyju47s6aeugttdpydsyk7hyj.py
# Source Nodes: [sigmoid_7, x_215], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_7 => sigmoid_18
# x_215 => mul_298
triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_75 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_75', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9126
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
        tmp3 = tl.load(in_ptr0 + (x1 + (702*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (702*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = 0.0
        tmp8 = tmp6 <= tmp7
        tmp9 = 6.0
        tmp10 = tmp6 >= tmp9
        tmp11 = tmp8 | tmp10
        tmp12 = tl.load(in_ptr2 + ((196*x1) + (137592*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.where(tmp11, tmp7, tmp12)
        tmp14 = tmp13 * tmp5
        tmp15 = tl.load(in_ptr3 + (x1 + (702*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = 196.0
        tmp17 = tmp15 / tmp16
        tmp18 = tmp14 + tmp17
        tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
        tmp20 = tl.where(tmp2, tmp18, tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nc/cncmmswf2cjsm6yldryzh2y6xhvzln44xt6khyv6rgq375hmkjq2.py
# Source Nodes: [sigmoid_7, x_215], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_7 => sigmoid_18
# x_215 => mul_298
triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_76 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_76', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 702
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


# kernel path: /tmp/torchinductor_youkaichao/jc/cjc5je6b3mgi7btbkvgake2dhhhj6adjljxmtlxpigdhgpufberl.py
# Source Nodes: [sigmoid_7, x_215], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_7 => sigmoid_18
# x_215 => mul_298
triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_77 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_77', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9126
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 702)
    x0 = xindex % 702
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (702*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (702*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = 0.0
        tmp8 = tmp6 <= tmp7
        tmp9 = 6.0
        tmp10 = tmp6 >= tmp9
        tmp11 = tmp8 | tmp10
        tmp12 = tl.load(in_ptr2 + ((196*x0) + (137592*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.where(tmp11, tmp7, tmp12)
        tmp14 = tmp13 * tmp5
        tmp15 = tl.load(in_ptr3 + (x0 + (702*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = 196.0
        tmp17 = tmp15 / tmp16
        tmp18 = tmp14 + tmp17
        tmp19 = tl.load(in_ptr4 + (x0 + (702*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/7q/c7qt7bup5sbvxcghpogqfzxlkxbgusoqla5wpg6d65mwucgcihte.py
# Source Nodes: [sigmoid_7, x_215], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_7 => sigmoid_18
# x_215 => mul_298
triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_78 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_78', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 702
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (702*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5q/c5qzcc4nqrc75i474vxlxqucw32pkk6rfxpcncktwog3ufgfkgc5.py
# Source Nodes: [sigmoid_7, x_215], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_7 => sigmoid_18
# x_215 => mul_298
triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_79 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_79', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 702
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 196)
    y0 = yindex % 196
    tmp0 = tl.load(in_ptr0 + (x2 + (702*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (702*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (y0 + (196*x2) + (137592*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2 + (702*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (702*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tmp6 = 6.0
    tmp7 = tmp3 >= tmp6
    tmp8 = tmp5 | tmp7
    tmp10 = tl.where(tmp8, tmp4, tmp9)
    tmp11 = tmp10 * tmp2
    tmp13 = 196.0
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 0.0006377551020408163
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tmp31 = tmp22 * tmp30
    tmp32 = tmp29 * tmp31
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (702*y3)), tmp32, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fw/cfwhxg6yydmu6ecvlmvietee73ppuxcqpuodyznxbhcd63mm3prs.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_80 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_80', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9126
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
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (137592*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (702*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fn/cfnecalki5qi7pfmzgotwgyaucyozgbfz4x7ume6zushhmf35y4b.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_81 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_81', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9126
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 702)
    x0 = xindex % 702
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x0) + (137592*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (702*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.load(in_ptr2 + (x0 + (702*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/eb/cebwkywswwq3z2cdm33ns4tyyluo54li24k6jhnkc4ysbexcj6ma.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_82 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_82', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5616
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 702
    y1 = (yindex // 702)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (702*x2) + (137592*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (702*x2) + (137592*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/dz/cdzmvnlgahpxugu2d2j47oe36jnryajihl4jw5zd472w6xua2pzl.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]

triton_red_fused_add_native_batch_norm_backward_slice_backward_83 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_slice_backward_83', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 117
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp22 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        r3 = rindex
        tmp21 = tl.load(in_ptr2 + (x0 + (117*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 106, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.load(in_ptr0 + (r1 + (196*x0) + (25088*r2)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (196*x0) + (22932*r2)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = 0.0
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tmp0 < tmp1
        tmp11 = tl.load(in_ptr0 + (r1 + (196*x0) + (25088*r2)), rmask & tmp10 & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr1 + (r1 + (196*x0) + (22932*r2)), rmask & tmp10 & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tmp11 + tmp12
        tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
        tmp15 = tl.where(tmp10, tmp13, tmp14)
        tmp16 = tl.where(tmp10, tmp15, tmp8)
        tmp17 = tmp9 + tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
        tmp23 = tmp21 - tmp22
        tmp24 = tmp17 * tmp23
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp19, xmask)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp26, xmask)
    tmp28 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tmp26 * tmp28
    tl.store(out_ptr2 + (x0), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jm/cjmulw63wzxig7uwp75zb7irt5zwwxe3qs3zj33jbin2pztatus4.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_84 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_84', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 936
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 117
    x2 = xindex
    y1 = (yindex // 117)
    y3 = yindex
    tmp18 = tl.load(in_ptr2 + (y0 + (117*x2) + (22932*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 106, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x2 + (196*y0) + (25088*y1)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr1 + (x2 + (196*y3)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tmp0 < tmp1
    tmp11 = tl.load(in_ptr0 + (x2 + (196*y0) + (25088*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr1 + (x2 + (196*y3)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp10, tmp15, tmp8)
    tmp17 = tmp9 + tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = 0.0006377551020408163
    tmp23 = tmp21 * tmp22
    tmp25 = tmp24 * tmp24
    tmp26 = tmp23 * tmp25
    tmp27 = tmp20 * tmp26
    tmp28 = tmp17 - tmp27
    tmp30 = tmp29 * tmp22
    tmp31 = tmp28 - tmp30
    tmp33 = tmp24 * tmp32
    tmp34 = tmp31 * tmp33
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp34, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/45/c45h6xg4yyzydakzbzgzx5kl4cz2ogsblobtjbpefn5adqhviv5b.py
# Source Nodes: [sigmoid_6, x_194], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sum]
# sigmoid_6 => sigmoid_16
# x_194 => mul_268
triton_red_fused_hardtanh_backward_mul_sigmoid_sum_85 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_mul_sigmoid_sum_85', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 10176
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2) % 636
    x2 = (xindex // 1272)
    x4 = (xindex // 2)
    tmp1 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    x5 = xindex
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (636*r3) + (62328*x0) + (124656*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr2 + (r3 + (98*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = 6.0
        tmp7 = tmp3 >= tmp6
        tmp8 = tmp5 | tmp7
        tmp10 = tl.where(tmp8, tmp4, tmp9)
        tmp11 = tmp10 * tmp0
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xl/cxlljlxejfvqb2yqi34jghmt7vxxelldj63u65enta4bhu4wbi42.py
# Source Nodes: [sigmoid_6, x_194], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_6 => sigmoid_16
# x_194 => mul_268
triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_86 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_86', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5088
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


# kernel path: /tmp/torchinductor_youkaichao/ny/cnytx3g7dsntwiydweif6fj2k7sqavt5adsu7d64c5c2ixlrncmv.py
# Source Nodes: [getattr_l__mod___features___9___se_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
# getattr_l__mod___features___9___se_bn => var_mean_35
triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_87 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_87', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 53
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (53*r1)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr1 + (x0 + (53*r1)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr2 + (x0 + (53*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 8, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tmp21 = tl.where(tmp19, tmp18, tmp20)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp0 - tmp10
    tmp27 = tmp21 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None]
    tmp32 = 8.0
    tmp33 = tmp16 / tmp32
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tmp37 = tmp31 * tmp36
    tl.store(out_ptr4 + (x0), tmp37, xmask)
    tl.store(out_ptr0 + (x0), tmp16, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
    tl.store(out_ptr2 + (x0), tmp25, xmask)
    tl.store(out_ptr3 + (x0), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l5/cl5cvfhyx43zpvekhc6i4edg6t3cg7jih6sxy65l27gak3ke4wqz.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_88 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_88', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 53
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr1 + (x2), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.125
    tmp10 = tmp8 * tmp9
    tmp12 = 8.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = tl.math.rsqrt(tmp15)
    tmp17 = tmp16 * tmp16
    tmp18 = tmp10 * tmp17
    tmp19 = tmp7 * tmp18
    tmp20 = tmp4 - tmp19
    tmp22 = tmp21 * tmp9
    tmp23 = tmp20 - tmp22
    tmp25 = tmp16 * tmp24
    tmp26 = tmp23 * tmp25
    tl.store(in_out_ptr0 + (x2), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/75/c75q3cxw4ywyededfkoksvp5xzpjuiwdzm7d5nrlfqdjslebn4ul.py
# Source Nodes: [sigmoid_6, x_194], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_6 => sigmoid_16
# x_194 => mul_268
triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_89 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_89', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8268
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
        tmp3 = tl.load(in_ptr0 + (x1 + (636*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (636*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = 0.0
        tmp8 = tmp6 <= tmp7
        tmp9 = 6.0
        tmp10 = tmp6 >= tmp9
        tmp11 = tmp8 | tmp10
        tmp12 = tl.load(in_ptr2 + ((196*x1) + (124656*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.where(tmp11, tmp7, tmp12)
        tmp14 = tmp13 * tmp5
        tmp15 = tl.load(in_ptr3 + (x1 + (636*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = 196.0
        tmp17 = tmp15 / tmp16
        tmp18 = tmp14 + tmp17
        tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
        tmp20 = tl.where(tmp2, tmp18, tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/se/cse6xtbpjdhsk64ryhdeyybp3fyk52fdmb4vvtpeblk6ck36m72g.py
# Source Nodes: [sigmoid_6, x_194], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_6 => sigmoid_16
# x_194 => mul_268
triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_90 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_90', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 636
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


# kernel path: /tmp/torchinductor_youkaichao/zo/czonl2mnwez7wswnykciztqitnxk2ikgpuotjxw6v6xwb2om2gbq.py
# Source Nodes: [sigmoid_6, x_194], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_6 => sigmoid_16
# x_194 => mul_268
triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_91 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_91', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8268
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 636)
    x0 = xindex % 636
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (636*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (636*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = 0.0
        tmp8 = tmp6 <= tmp7
        tmp9 = 6.0
        tmp10 = tmp6 >= tmp9
        tmp11 = tmp8 | tmp10
        tmp12 = tl.load(in_ptr2 + ((196*x0) + (124656*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.where(tmp11, tmp7, tmp12)
        tmp14 = tmp13 * tmp5
        tmp15 = tl.load(in_ptr3 + (x0 + (636*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = 196.0
        tmp17 = tmp15 / tmp16
        tmp18 = tmp14 + tmp17
        tmp19 = tl.load(in_ptr4 + (x0 + (636*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/fi/cfiq52qdsbt7hyau5yboflauity5r3tzfbrg75yz2bgthzikubyk.py
# Source Nodes: [sigmoid_6, x_194], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_6 => sigmoid_16
# x_194 => mul_268
triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_92 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_92', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 636
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (636*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7y/c7yuw5abb3afi7vntx5wbzv7bbd5sm6i37xlqvkr6myyytxi3ht2.py
# Source Nodes: [sigmoid_6, x_194], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_6 => sigmoid_16
# x_194 => mul_268
triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_93 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_93', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 636
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 196)
    y0 = yindex % 196
    tmp0 = tl.load(in_ptr0 + (x2 + (636*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (636*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (y0 + (196*x2) + (124656*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2 + (636*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (636*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tmp6 = 6.0
    tmp7 = tmp3 >= tmp6
    tmp8 = tmp5 | tmp7
    tmp10 = tl.where(tmp8, tmp4, tmp9)
    tmp11 = tmp10 * tmp2
    tmp13 = 196.0
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 0.0006377551020408163
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tmp31 = tmp22 * tmp30
    tmp32 = tmp29 * tmp31
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (636*y3)), tmp32, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fo/cfovgge45g3twwfoh7vymycvkuj34naha4wtgdibgc3oq44acpva.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_94 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_94', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8268
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
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (124656*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (636*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ql/cql6x3buf2fuchfmeum4okfn67vmhoygyjlrf3mrkxcbrz6zyxhh.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_95 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_95', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8268
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 636)
    x0 = xindex % 636
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x0) + (124656*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (636*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.load(in_ptr2 + (x0 + (636*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/o7/co725gpwane4wfugwhfdfyq7xaxq4hzkafifsegt2tchbas2llp3.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_96 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_96', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5088
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 636
    y1 = (yindex // 636)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (636*x2) + (124656*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (636*x2) + (124656*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/r2/cr2xjs6kkqp2c2ym536e4xgbkr5ymoatmw4wuvz47rbdlm5xcoqo.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]

triton_red_fused_add_native_batch_norm_backward_slice_backward_97 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_slice_backward_97', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 106
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp26 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp30 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        r3 = rindex
        tmp25 = tl.load(in_ptr3 + (x0 + (106*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 95, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.load(in_ptr0 + (r1 + (196*x0) + (25088*r2)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (196*x0) + (22932*r2)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.load(in_ptr2 + (r1 + (196*x0) + (20776*r2)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = 0.0
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tmp0 < tmp1
        tmp13 = tl.load(in_ptr0 + (r1 + (196*x0) + (25088*r2)), rmask & tmp12 & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr1 + (r1 + (196*x0) + (22932*r2)), rmask & tmp12 & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tmp13 + tmp14
        tmp16 = tl.load(in_ptr2 + (r1 + (196*x0) + (20776*r2)), rmask & tmp12 & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tmp15 + tmp16
        tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
        tmp19 = tl.where(tmp12, tmp17, tmp18)
        tmp20 = tl.where(tmp12, tmp19, tmp10)
        tmp21 = tmp11 + tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
        tmp27 = tmp25 - tmp26
        tmp28 = tmp21 * tmp27
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
        tmp31 = _tmp30 + tmp29
        _tmp30 = tl.where(rmask & xmask, tmp31, _tmp30)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp23, xmask)
    tmp30 = tl.sum(_tmp30, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp30, xmask)
    tmp32 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp33 = tmp30 * tmp32
    tl.store(out_ptr2 + (x0), tmp33, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wh/cwhuloiwvnb2hsi43femmrjed4stpxsbqnqj4a5j4obhkvgvc552.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]

triton_poi_fused_add_native_batch_norm_backward_slice_backward_98 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_slice_backward_98', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 848
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 106
    x2 = xindex
    y1 = (yindex // 106)
    y3 = yindex
    tmp22 = tl.load(in_ptr3 + (y0 + (106*x2) + (20776*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 95, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x2 + (196*y0) + (25088*y1)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr1 + (x2 + (196*y0) + (22932*y1)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (x2 + (196*y3)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = 0.0
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = tmp0 < tmp1
    tmp13 = tl.load(in_ptr0 + (x2 + (196*y0) + (25088*y1)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr1 + (x2 + (196*y0) + (22932*y1)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.load(in_ptr2 + (x2 + (196*y3)), tmp12 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp12, tmp17, tmp18)
    tmp20 = tl.where(tmp12, tmp19, tmp10)
    tmp21 = tmp11 + tmp20
    tmp24 = tmp22 - tmp23
    tmp26 = 0.0006377551020408163
    tmp27 = tmp25 * tmp26
    tmp29 = tmp28 * tmp28
    tmp30 = tmp27 * tmp29
    tmp31 = tmp24 * tmp30
    tmp32 = tmp21 - tmp31
    tmp34 = tmp33 * tmp26
    tmp35 = tmp32 - tmp34
    tmp37 = tmp28 * tmp36
    tmp38 = tmp35 * tmp37
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp38, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rt/crt2v273quddugj2dgravtc65gftpvq3hrtav7wnafnkb7t2k77g.py
# Source Nodes: [sigmoid_5, x_173], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sum]
# sigmoid_5 => sigmoid_14
# x_173 => mul_238
triton_red_fused_hardtanh_backward_mul_sigmoid_sum_99 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_mul_sigmoid_sum_99', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9120
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2) % 570
    x2 = (xindex // 1140)
    x4 = (xindex // 2)
    tmp1 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    x5 = xindex
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (570*r3) + (55860*x0) + (111720*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr2 + (r3 + (98*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = 6.0
        tmp7 = tmp3 >= tmp6
        tmp8 = tmp5 | tmp7
        tmp10 = tl.where(tmp8, tmp4, tmp9)
        tmp11 = tmp10 * tmp0
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ct/ccttcqlbz4cok2dh7v2oysovfjnffciu6bj2nqt3c6axmo6l275n.py
# Source Nodes: [sigmoid_5, x_173], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_5 => sigmoid_14
# x_173 => mul_238
triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_100 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_100', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4560
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


# kernel path: /tmp/torchinductor_youkaichao/ct/cctoctai2ufvcvlnb2m6hgpaaffob6hijgm7rjq3gvxf7yw52rn7.py
# Source Nodes: [getattr_l__mod___features___8___se_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
# getattr_l__mod___features___8___se_bn => var_mean_31
triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_101 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_101', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 47
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (47*r1)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr1 + (x0 + (47*r1)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr2 + (x0 + (47*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 8, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tmp21 = tl.where(tmp19, tmp18, tmp20)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp0 - tmp10
    tmp27 = tmp21 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None]
    tmp32 = 8.0
    tmp33 = tmp16 / tmp32
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tmp37 = tmp31 * tmp36
    tl.store(out_ptr4 + (x0), tmp37, xmask)
    tl.store(out_ptr0 + (x0), tmp16, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
    tl.store(out_ptr2 + (x0), tmp25, xmask)
    tl.store(out_ptr3 + (x0), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3u/c3ut2ldxamvcrxzy6uahqvammc24yubwynafktg2qs2kh6b4woum.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_102 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_102', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 47
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr1 + (x2), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.125
    tmp10 = tmp8 * tmp9
    tmp12 = 8.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = tl.math.rsqrt(tmp15)
    tmp17 = tmp16 * tmp16
    tmp18 = tmp10 * tmp17
    tmp19 = tmp7 * tmp18
    tmp20 = tmp4 - tmp19
    tmp22 = tmp21 * tmp9
    tmp23 = tmp20 - tmp22
    tmp25 = tmp16 * tmp24
    tmp26 = tmp23 * tmp25
    tl.store(in_out_ptr0 + (x2), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s2/cs2n6ooztzre2jsf5mjgayofq35ocxxfqtciwkul3rhslzdvh2na.py
# Source Nodes: [sigmoid_5, x_173], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_5 => sigmoid_14
# x_173 => mul_238
triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_103 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_103', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7410
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
        tmp3 = tl.load(in_ptr0 + (x1 + (570*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (570*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = 0.0
        tmp8 = tmp6 <= tmp7
        tmp9 = 6.0
        tmp10 = tmp6 >= tmp9
        tmp11 = tmp8 | tmp10
        tmp12 = tl.load(in_ptr2 + ((196*x1) + (111720*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.where(tmp11, tmp7, tmp12)
        tmp14 = tmp13 * tmp5
        tmp15 = tl.load(in_ptr3 + (x1 + (570*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = 196.0
        tmp17 = tmp15 / tmp16
        tmp18 = tmp14 + tmp17
        tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
        tmp20 = tl.where(tmp2, tmp18, tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xq/cxqh276vu4ixxn6k6gwdrcurupmvjkd3rjkvu3zvbftrfgvc26gv.py
# Source Nodes: [sigmoid_5, x_173], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_5 => sigmoid_14
# x_173 => mul_238
triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_104 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_104', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 570
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


# kernel path: /tmp/torchinductor_youkaichao/nc/cncr2goani7pi2xahmisqfubeetmif4g2phf7zy7b74iydkmv3xm.py
# Source Nodes: [sigmoid_5, x_173], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_5 => sigmoid_14
# x_173 => mul_238
triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_105 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_105', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7410
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 570)
    x0 = xindex % 570
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (570*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (570*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = 0.0
        tmp8 = tmp6 <= tmp7
        tmp9 = 6.0
        tmp10 = tmp6 >= tmp9
        tmp11 = tmp8 | tmp10
        tmp12 = tl.load(in_ptr2 + ((196*x0) + (111720*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.where(tmp11, tmp7, tmp12)
        tmp14 = tmp13 * tmp5
        tmp15 = tl.load(in_ptr3 + (x0 + (570*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = 196.0
        tmp17 = tmp15 / tmp16
        tmp18 = tmp14 + tmp17
        tmp19 = tl.load(in_ptr4 + (x0 + (570*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/rh/crhmocd5ifwwhl47kfygwdi3lf22yicjy5p3vcfm3boflsyfwa45.py
# Source Nodes: [sigmoid_5, x_173], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_5 => sigmoid_14
# x_173 => mul_238
triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_106 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_106', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 570
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (570*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/us/cus24retmab2w326unr6mobrqcxvfyajesgzkxyjqwsnrkztkehy.py
# Source Nodes: [sigmoid_5, x_173], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_5 => sigmoid_14
# x_173 => mul_238
triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_107 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_107', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 570
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 196)
    y0 = yindex % 196
    tmp0 = tl.load(in_ptr0 + (x2 + (570*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (570*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (y0 + (196*x2) + (111720*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2 + (570*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (570*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tmp6 = 6.0
    tmp7 = tmp3 >= tmp6
    tmp8 = tmp5 | tmp7
    tmp10 = tl.where(tmp8, tmp4, tmp9)
    tmp11 = tmp10 * tmp2
    tmp13 = 196.0
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 0.0006377551020408163
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tmp31 = tmp22 * tmp30
    tmp32 = tmp29 * tmp31
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (570*y3)), tmp32, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/is/cisya3ddniumeibe5oeig2h7yhnfdymcict26ztjrwlkrvy4b3bw.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_108 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_108', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7410
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
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (111720*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (570*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e2/ce233alrhdadr6heu6rapqjojncqoxzil3dgoajvbku7tegk7sw4.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_109 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_109', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7410
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 570)
    x0 = xindex % 570
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x0) + (111720*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (570*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.load(in_ptr2 + (x0 + (570*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/qi/cqinsrb234xxjapz2zskfvutgwhc7k2dwy3woajqrpana4w3vjby.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_110 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_110', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4560
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 570
    y1 = (yindex // 570)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (570*x2) + (111720*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (570*x2) + (111720*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/cf/ccfohnaf5tz4ih5acobpznsdabjnesicwkjk24qbpnwhta3i3iww.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]

triton_red_fused_add_native_batch_norm_backward_slice_backward_111 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_slice_backward_111', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 95
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp30 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp34 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        r3 = rindex
        tmp29 = tl.load(in_ptr4 + (x0 + (95*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 84, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.load(in_ptr0 + (r1 + (196*x0) + (25088*r2)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r1 + (196*x0) + (22932*r2)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.load(in_ptr2 + (r1 + (196*x0) + (20776*r2)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.load(in_ptr3 + (r1 + (196*x0) + (18620*r2)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tmp7 + tmp8
        tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = 0.0
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tmp0 < tmp1
        tmp15 = tl.load(in_ptr0 + (r1 + (196*x0) + (25088*r2)), rmask & tmp14 & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr1 + (r1 + (196*x0) + (22932*r2)), rmask & tmp14 & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tmp15 + tmp16
        tmp18 = tl.load(in_ptr2 + (r1 + (196*x0) + (20776*r2)), rmask & tmp14 & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tmp17 + tmp18
        tmp20 = tl.load(in_ptr3 + (r1 + (196*x0) + (18620*r2)), rmask & tmp14 & xmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tmp19 + tmp20
        tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
        tmp23 = tl.where(tmp14, tmp21, tmp22)
        tmp24 = tl.where(tmp14, tmp23, tmp12)
        tmp25 = tmp13 + tmp24
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(rmask & xmask, tmp28, _tmp27)
        tmp31 = tmp29 - tmp30
        tmp32 = tmp25 * tmp31
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
        tmp35 = _tmp34 + tmp33
        _tmp34 = tl.where(rmask & xmask, tmp35, _tmp34)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp27, xmask)
    tmp34 = tl.sum(_tmp34, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp34, xmask)
    tmp36 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp37 = tmp34 * tmp36
    tl.store(out_ptr2 + (x0), tmp37, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a6/ca66k2eb2pp5mqq6vwvmcertuwjlagan6v7qs2m75ghroe5hgkip.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_112 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_112', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 760
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 95
    x2 = xindex
    y1 = (yindex // 95)
    y3 = yindex
    tmp26 = tl.load(in_ptr4 + (y0 + (95*x2) + (18620*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr7 + (y0), ymask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr8 + (y0), ymask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr9 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 84, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x2 + (196*y0) + (25088*y1)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr1 + (x2 + (196*y0) + (22932*y1)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr2 + (x2 + (196*y0) + (20776*y1)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr3 + (x2 + (196*y3)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = 0.0
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = tmp0 < tmp1
    tmp15 = tl.load(in_ptr0 + (x2 + (196*y0) + (25088*y1)), tmp14 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr1 + (x2 + (196*y0) + (22932*y1)), tmp14 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.load(in_ptr2 + (x2 + (196*y0) + (20776*y1)), tmp14 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.load(in_ptr3 + (x2 + (196*y3)), tmp14 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp14, tmp21, tmp22)
    tmp24 = tl.where(tmp14, tmp23, tmp12)
    tmp25 = tmp13 + tmp24
    tmp28 = tmp26 - tmp27
    tmp30 = 0.0006377551020408163
    tmp31 = tmp29 * tmp30
    tmp33 = tmp32 * tmp32
    tmp34 = tmp31 * tmp33
    tmp35 = tmp28 * tmp34
    tmp36 = tmp25 - tmp35
    tmp38 = tmp37 * tmp30
    tmp39 = tmp36 - tmp38
    tmp41 = tmp32 * tmp40
    tmp42 = tmp39 * tmp41
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (196*y3)), tmp42, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l4/cl4linc7crjm6ome6pnc6zk4ew6aysmmn7uulb4onucoy67qohiq.py
# Source Nodes: [sigmoid_4, x_152], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sum]
# sigmoid_4 => sigmoid_12
# x_152 => mul_208
triton_red_fused_hardtanh_backward_mul_sigmoid_sum_113 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_mul_sigmoid_sum_113', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8064
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2) % 504
    x2 = (xindex // 1008)
    x4 = (xindex // 2)
    tmp1 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    x5 = xindex
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (504*r3) + (49392*x0) + (98784*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr2 + (r3 + (98*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = 6.0
        tmp7 = tmp3 >= tmp6
        tmp8 = tmp5 | tmp7
        tmp10 = tl.where(tmp8, tmp4, tmp9)
        tmp11 = tmp10 * tmp0
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/he/che4aog5t7x6zlexpqzbu3jz5cz4bripjm6ltejmsv7vvibcq27s.py
# Source Nodes: [sigmoid_4, x_152], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_4 => sigmoid_12
# x_152 => mul_208
triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_114 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_114', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4032
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


# kernel path: /tmp/torchinductor_youkaichao/wh/cwhix2eujmpkxhpo4izu47xhp24dlqgh6nqnmiwl57xw4yuauhr6.py
# Source Nodes: [getattr_l__mod___features___7___se_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
# getattr_l__mod___features___7___se_bn => var_mean_27
triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_115 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_115', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 42
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (42*r1)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr1 + (x0 + (42*r1)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr2 + (x0 + (42*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 8, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tmp21 = tl.where(tmp19, tmp18, tmp20)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp0 - tmp10
    tmp27 = tmp21 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None]
    tmp32 = 8.0
    tmp33 = tmp16 / tmp32
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tmp37 = tmp31 * tmp36
    tl.store(out_ptr4 + (x0), tmp37, xmask)
    tl.store(out_ptr0 + (x0), tmp16, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
    tl.store(out_ptr2 + (x0), tmp25, xmask)
    tl.store(out_ptr3 + (x0), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pk/cpk5l326dt47drqj2htni5jxzl367dhiawjdxiezhbpbph3qjoow.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_116 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_116', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 42
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr1 + (x2), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.125
    tmp10 = tmp8 * tmp9
    tmp12 = 8.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = tl.math.rsqrt(tmp15)
    tmp17 = tmp16 * tmp16
    tmp18 = tmp10 * tmp17
    tmp19 = tmp7 * tmp18
    tmp20 = tmp4 - tmp19
    tmp22 = tmp21 * tmp9
    tmp23 = tmp20 - tmp22
    tmp25 = tmp16 * tmp24
    tmp26 = tmp23 * tmp25
    tl.store(in_out_ptr0 + (x2), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hv/chv5hj4vciyjnkd3liajspilcf5p6tk74hm5necmjta55o3b67ju.py
# Source Nodes: [sigmoid_4, x_152], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_4 => sigmoid_12
# x_152 => mul_208
triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_117 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_117', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6552
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
        tmp3 = tl.load(in_ptr0 + (x1 + (504*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (504*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = 0.0
        tmp8 = tmp6 <= tmp7
        tmp9 = 6.0
        tmp10 = tmp6 >= tmp9
        tmp11 = tmp8 | tmp10
        tmp12 = tl.load(in_ptr2 + ((196*x1) + (98784*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.where(tmp11, tmp7, tmp12)
        tmp14 = tmp13 * tmp5
        tmp15 = tl.load(in_ptr3 + (x1 + (504*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = 196.0
        tmp17 = tmp15 / tmp16
        tmp18 = tmp14 + tmp17
        tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
        tmp20 = tl.where(tmp2, tmp18, tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/no/cnotd2n7b3z2nfq5obmpe6yvoerrtpmqjjzfi3x4bq6g7mdpocky.py
# Source Nodes: [sigmoid_4, x_152], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_4 => sigmoid_12
# x_152 => mul_208
triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_118 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_118', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 504
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


# kernel path: /tmp/torchinductor_youkaichao/2d/c2dds3bjit6kvvslssig5jxjunhddwxlnjpwjosnzfwemuifyp37.py
# Source Nodes: [sigmoid_4, x_152], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_4 => sigmoid_12
# x_152 => mul_208
triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_119 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_119', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6552
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 504)
    x0 = xindex % 504
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (504*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (504*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = 0.0
        tmp8 = tmp6 <= tmp7
        tmp9 = 6.0
        tmp10 = tmp6 >= tmp9
        tmp11 = tmp8 | tmp10
        tmp12 = tl.load(in_ptr2 + ((196*x0) + (98784*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.where(tmp11, tmp7, tmp12)
        tmp14 = tmp13 * tmp5
        tmp15 = tl.load(in_ptr3 + (x0 + (504*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = 196.0
        tmp17 = tmp15 / tmp16
        tmp18 = tmp14 + tmp17
        tmp19 = tl.load(in_ptr4 + (x0 + (504*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/qn/cqngobkypkqcgvhmgbd5vonmsd4dbeyk6f73otbupglcairnpfo5.py
# Source Nodes: [sigmoid_4, x_152], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_4 => sigmoid_12
# x_152 => mul_208
triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_120 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_120', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 504
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (504*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vo/cvol6t7s6fvhu5kskdqvib3vw4kknz6fs565t7lreqemz65ofsb3.py
# Source Nodes: [sigmoid_4, x_152], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_4 => sigmoid_12
# x_152 => mul_208
triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_121 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_121', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 504
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 196)
    y0 = yindex % 196
    tmp0 = tl.load(in_ptr0 + (x2 + (504*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (504*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (y0 + (196*x2) + (98784*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2 + (504*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (504*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tmp6 = 6.0
    tmp7 = tmp3 >= tmp6
    tmp8 = tmp5 | tmp7
    tmp10 = tl.where(tmp8, tmp4, tmp9)
    tmp11 = tmp10 * tmp2
    tmp13 = 196.0
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 0.0006377551020408163
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tmp31 = tmp22 * tmp30
    tmp32 = tmp29 * tmp31
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (504*y3)), tmp32, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vt/cvtxmud5u4iivtb4u6bkesxchso67kmvfrwe33oorropgr5y3fsh.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_122 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_122', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6552
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
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (98784*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (504*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sc/cscnaqxy7heeeylydvmz74wzj3r5ljubnoiiyy37esp6bov6qv5b.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_123 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_123', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6552
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 504)
    x0 = xindex % 504
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x0) + (98784*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (504*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.load(in_ptr2 + (x0 + (504*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/hr/chrvqiphlt4tt4j7mrengx3i22e6367qdv624hedlenjlekzcs67.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_124 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_124', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4032
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 504
    y1 = (yindex // 504)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (504*x2) + (98784*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (504*x2) + (98784*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/ub/cubcfe5nncaihgmfhni33imn2k4e2abhmtxr563oianadvvcfo22.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_125 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_125', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131712
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16464
    x1 = (xindex // 16464)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (25088*x1)), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (22932*x1)), xmask)
    tmp3 = tl.load(in_ptr2 + (x0 + (20776*x1)), xmask)
    tmp5 = tl.load(in_ptr3 + (x0 + (18620*x1)), xmask)
    tmp7 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/db/cdbksqygldzlji2hdctkuejggnz6cdfsebsi4twnn4hx67qazs4x.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]

triton_red_fused_add_native_batch_norm_backward_slice_backward_126 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_slice_backward_126', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 84
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 72, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.load(in_ptr0 + (r1 + (196*x0) + (16464*r2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tmp0 < tmp1
        tmp9 = tl.load(in_ptr0 + (r1 + (196*x0) + (16464*r2)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
        tmp11 = tl.where(tmp8, tmp9, tmp10)
        tmp12 = tl.where(tmp8, tmp11, tmp6)
        tmp13 = tmp7 + tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gj/cgj2uymaic5fv5bqfmzzhqgs3k4r3farrwqs6peixkb5exnxdvyk.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]

triton_red_fused_add_native_batch_norm_backward_slice_backward_127 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_slice_backward_127', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1092
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.broadcast_to(x1, [XBLOCK, RBLOCK])
        tmp4 = tl.full([1, 1], 72, tl.int64)
        tmp5 = tmp3 >= tmp4
        tmp6 = tmp5 & tmp2
        tmp7 = tl.load(in_ptr0 + ((196*x1) + (16464*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
        tmp9 = tl.where(tmp6, tmp7, tmp8)
        tmp10 = 0.0
        tmp11 = tl.where(tmp5, tmp9, tmp10)
        tmp12 = tmp3 < tmp4
        tmp13 = tmp12 & tmp2
        tmp14 = tl.load(in_ptr0 + ((196*x1) + (16464*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp13 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
        tmp16 = tl.where(tmp13, tmp14, tmp15)
        tmp17 = tl.where(tmp12, tmp16, tmp10)
        tmp18 = tmp11 + tmp17
        tmp19 = tl.load(in_ptr1 + (x1 + (84*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.load(in_ptr2 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/kv/ckvytzbp5zcmgmaie7wyilvllut5hurbnpqabfr46kezksday4fk.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]

triton_per_fused_add_native_batch_norm_backward_slice_backward_128 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_slice_backward_128', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 84
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


# kernel path: /tmp/torchinductor_youkaichao/kt/cktjhyb2l2rfy5il7yyw2vnr4ni3lfaod4pubeugzjelp5ouiygq.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_129 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_129', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 672
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 84
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 84)
    tmp14 = tl.load(in_ptr1 + (y0 + (84*x2) + (16464*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 72, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x2 + (196*y3)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = tmp0 < tmp1
    tmp9 = tl.load(in_ptr0 + (x2 + (196*y3)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = tl.where(tmp8, tmp11, tmp6)
    tmp13 = tmp7 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = 0.0006377551020408163
    tmp19 = tmp17 * tmp18
    tmp21 = tmp20 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp16 * tmp22
    tmp24 = tmp13 - tmp23
    tmp26 = tmp25 * tmp18
    tmp27 = tmp24 - tmp26
    tmp29 = tmp20 * tmp28
    tmp30 = tmp27 * tmp29
    tl.store(out_ptr0 + (x2 + (196*y3)), tmp30, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wt/cwtrd6gbfypmqgzlek6kf47dczqcnnfby7ykp6m5mzh2ne7a4ux7.py
# Source Nodes: [sigmoid_3, x_131], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sum]
# sigmoid_3 => sigmoid_10
# x_131 => mul_178
triton_red_fused_hardtanh_backward_mul_sigmoid_sum_130 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_mul_sigmoid_sum_130', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6912
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2) % 432
    x2 = (xindex // 864)
    x4 = (xindex // 2)
    tmp1 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    x5 = xindex
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (432*r3) + (42336*x0) + (84672*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr2 + (r3 + (98*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = 6.0
        tmp7 = tmp3 >= tmp6
        tmp8 = tmp5 | tmp7
        tmp10 = tl.where(tmp8, tmp4, tmp9)
        tmp11 = tmp10 * tmp0
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x6/cx6owvjt635lslb4tfyicigktiz5zoj4bvzgqxbf73iknciidmdb.py
# Source Nodes: [sigmoid_3, x_131], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_3 => sigmoid_10
# x_131 => mul_178
triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_131 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_131', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3456
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


# kernel path: /tmp/torchinductor_youkaichao/sy/csyx6gpxl6j6h2qjqfx7cn3cukfk2hvuixu5o2l64p5fv4u4jr5f.py
# Source Nodes: [getattr_l__mod___features___6___se_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
# getattr_l__mod___features___6___se_bn => var_mean_23
triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_132 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_132', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 36
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (36*r1)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr1 + (x0 + (36*r1)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr2 + (x0 + (36*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 8, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tmp21 = tl.where(tmp19, tmp18, tmp20)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp0 - tmp10
    tmp27 = tmp21 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None]
    tmp32 = 8.0
    tmp33 = tmp16 / tmp32
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tmp37 = tmp31 * tmp36
    tl.store(out_ptr4 + (x0), tmp37, xmask)
    tl.store(out_ptr0 + (x0), tmp16, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
    tl.store(out_ptr2 + (x0), tmp25, xmask)
    tl.store(out_ptr3 + (x0), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/we/cwe5kdfcqwgndoiqtlwfkgkvle3bvoyavtmvq4ej5uw72vqgtomp.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_133 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_133', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 36
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr1 + (x2), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.125
    tmp10 = tmp8 * tmp9
    tmp12 = 8.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = tl.math.rsqrt(tmp15)
    tmp17 = tmp16 * tmp16
    tmp18 = tmp10 * tmp17
    tmp19 = tmp7 * tmp18
    tmp20 = tmp4 - tmp19
    tmp22 = tmp21 * tmp9
    tmp23 = tmp20 - tmp22
    tmp25 = tmp16 * tmp24
    tmp26 = tmp23 * tmp25
    tl.store(in_out_ptr0 + (x2), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/to/cto7fpwy5vveu6t5aorrjt2ccmudgse5b2op2i4eqnd4y7o2dzkz.py
# Source Nodes: [sigmoid_3, x_131], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_3 => sigmoid_10
# x_131 => mul_178
triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_134 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_134', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5616
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
        tmp3 = tl.load(in_ptr0 + (x1 + (432*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (432*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = 0.0
        tmp8 = tmp6 <= tmp7
        tmp9 = 6.0
        tmp10 = tmp6 >= tmp9
        tmp11 = tmp8 | tmp10
        tmp12 = tl.load(in_ptr2 + ((196*x1) + (84672*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.where(tmp11, tmp7, tmp12)
        tmp14 = tmp13 * tmp5
        tmp15 = tl.load(in_ptr3 + (x1 + (432*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = 196.0
        tmp17 = tmp15 / tmp16
        tmp18 = tmp14 + tmp17
        tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
        tmp20 = tl.where(tmp2, tmp18, tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h5/ch5ggngk3d7qyyt5makpbzq5pjzfsrfr2jvzgzi5gmflhxym3des.py
# Source Nodes: [sigmoid_3, x_131], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_3 => sigmoid_10
# x_131 => mul_178
triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_135 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_135', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 432
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


# kernel path: /tmp/torchinductor_youkaichao/2y/c2yvbisf6ynyxvenfqsmyxsakq67hyh2i3nyrthkgujkdirqr7zt.py
# Source Nodes: [sigmoid_3, x_131], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_3 => sigmoid_10
# x_131 => mul_178
triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_136 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_136', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5616
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 432)
    x0 = xindex % 432
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (432*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (432*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = 0.0
        tmp8 = tmp6 <= tmp7
        tmp9 = 6.0
        tmp10 = tmp6 >= tmp9
        tmp11 = tmp8 | tmp10
        tmp12 = tl.load(in_ptr2 + ((196*x0) + (84672*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.where(tmp11, tmp7, tmp12)
        tmp14 = tmp13 * tmp5
        tmp15 = tl.load(in_ptr3 + (x0 + (432*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = 196.0
        tmp17 = tmp15 / tmp16
        tmp18 = tmp14 + tmp17
        tmp19 = tl.load(in_ptr4 + (x0 + (432*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/nr/cnrtcnvo4p7wdhpotsvxfwbuzgspym3snrrujpekxha7bhhe5fx6.py
# Source Nodes: [sigmoid_3, x_131], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_3 => sigmoid_10
# x_131 => mul_178
triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_137 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_137', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 432
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (432*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7g/c7gqxhbwvi6b37gshhqnkuftzbi4rh7hwz4swju36vpegc5flev6.py
# Source Nodes: [sigmoid_3, x_131], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_3 => sigmoid_10
# x_131 => mul_178
triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_138 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_138', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 432
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 196)
    y0 = yindex % 196
    tmp0 = tl.load(in_ptr0 + (x2 + (432*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (432*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (y0 + (196*x2) + (84672*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2 + (432*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (432*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tmp6 = 6.0
    tmp7 = tmp3 >= tmp6
    tmp8 = tmp5 | tmp7
    tmp10 = tl.where(tmp8, tmp4, tmp9)
    tmp11 = tmp10 * tmp2
    tmp13 = 196.0
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 0.0006377551020408163
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tmp31 = tmp22 * tmp30
    tmp32 = tmp29 * tmp31
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (432*y3)), tmp32, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jf/cjfohzwj5e2ulo2f4s32g67ofl3nc3d2ikwu6szaw32cqqmpi5ka.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_139 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_139', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5616
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
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (84672*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (432*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bb/cbbcdbbgcn53upyxp76mb66xkz2ppsoqszymhp2ayiqdic3szrgu.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_140 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_140', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5616
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 432)
    x0 = xindex % 432
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((196*x0) + (84672*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (432*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.load(in_ptr2 + (x0 + (432*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/w4/cw4e365o2vckenp34lx4o5jsyamivbasdwnzv6veucmq2cq2ewcu.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_141 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_141', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3456
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 432
    y1 = (yindex // 432)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (432*x2) + (84672*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (432*x2) + (84672*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/tc/ctcyxtyofdsft4wo2laobmtwepnbtpc4cznsx5jwzcpv3ztrglkd.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_142 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_142', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 72
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 196
        r2 = (rindex // 196)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (16464*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (196*x0) + (14112*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (72*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp11, xmask)
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tmp11 * tmp13
    tl.store(out_ptr2 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mx/cmxo4j5zcdgyz3ylz4qj3kmrsn4zxq4gilqe24ntoohklws4sb7t.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_143 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_143', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 576
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 72
    y1 = (yindex // 72)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y0) + (16464*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (72*x2) + (14112*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
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


# kernel path: /tmp/torchinductor_youkaichao/vs/cvso3dgf2iaxjld4z7w6ovfaktj4nnbckssnwgzws4y7w5yngi2j.py
# Source Nodes: [sigmoid_2, x_111], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sum]
# sigmoid_2 => sigmoid_8
# x_111 => mul_148
triton_red_fused_hardtanh_backward_mul_sigmoid_sum_144 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_mul_sigmoid_sum_144', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5856
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2) % 366
    x2 = (xindex // 732)
    x4 = (xindex // 2)
    tmp1 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    x5 = xindex
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (366*r3) + (35868*x0) + (71736*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr2 + (r3 + (98*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = 6.0
        tmp7 = tmp3 >= tmp6
        tmp8 = tmp5 | tmp7
        tmp10 = tl.where(tmp8, tmp4, tmp9)
        tmp11 = tmp10 * tmp0
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xv/cxvprc3ijqc6q4skaqu4bqjjordk4qune5lo2cgt46mmiuaywngt.py
# Source Nodes: [sigmoid_2, x_111], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_2 => sigmoid_8
# x_111 => mul_148
triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_145 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_145', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2928
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


# kernel path: /tmp/torchinductor_youkaichao/q5/cq5i7c3ymwhlhhxbou3wac65p6dnrtrpxqa5fx47xkmzuinrfupx.py
# Source Nodes: [getattr_l__mod___features___5___se_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
# getattr_l__mod___features___5___se_bn => var_mean_19
triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_146 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_146', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 30
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (30*r1)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr1 + (x0 + (30*r1)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr2 + (x0 + (30*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 8, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tmp21 = tl.where(tmp19, tmp18, tmp20)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp0 - tmp10
    tmp27 = tmp21 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None]
    tmp32 = 8.0
    tmp33 = tmp16 / tmp32
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tmp37 = tmp31 * tmp36
    tl.store(out_ptr4 + (x0), tmp37, xmask)
    tl.store(out_ptr0 + (x0), tmp16, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
    tl.store(out_ptr2 + (x0), tmp25, xmask)
    tl.store(out_ptr3 + (x0), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xt/cxtlaswuetgthu7elsgendaiw2cwg3sktorrsbyod5sgmoczvs6e.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_147 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_147', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 30
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr1 + (x2), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.125
    tmp10 = tmp8 * tmp9
    tmp12 = 8.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = tl.math.rsqrt(tmp15)
    tmp17 = tmp16 * tmp16
    tmp18 = tmp10 * tmp17
    tmp19 = tmp7 * tmp18
    tmp20 = tmp4 - tmp19
    tmp22 = tmp21 * tmp9
    tmp23 = tmp20 - tmp22
    tmp25 = tmp16 * tmp24
    tmp26 = tmp23 * tmp25
    tl.store(in_out_ptr0 + (x2), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yp/cypntji6p7n2zxilbefpq3kjqax47onpq2dn2yqgerk5dfy2qixq.py
# Source Nodes: [sigmoid_2, x_111], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_2 => sigmoid_8
# x_111 => mul_148
triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_148 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_148', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4758
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
        tmp3 = tl.load(in_ptr0 + (x1 + (366*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (366*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = 0.0
        tmp8 = tmp6 <= tmp7
        tmp9 = 6.0
        tmp10 = tmp6 >= tmp9
        tmp11 = tmp8 | tmp10
        tmp12 = tl.load(in_ptr2 + ((196*x1) + (71736*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.where(tmp11, tmp7, tmp12)
        tmp14 = tmp13 * tmp5
        tmp15 = tl.load(in_ptr3 + (x1 + (366*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = 196.0
        tmp17 = tmp15 / tmp16
        tmp18 = tmp14 + tmp17
        tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
        tmp20 = tl.where(tmp2, tmp18, tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vf/cvfwelbbhmk5yrxwzupgk2476t5konvllrsyaesgb5vu555i7wgm.py
# Source Nodes: [sigmoid_2, x_111], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_2 => sigmoid_8
# x_111 => mul_148
triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_149 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_149', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 366
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


# kernel path: /tmp/torchinductor_youkaichao/hn/chn66xb3xq5thhiclxajj5yylkquypypiykwz4bvmjqvv4pbhqqa.py
# Source Nodes: [sigmoid_2, x_111], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_2 => sigmoid_8
# x_111 => mul_148
triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_150 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_150', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4758
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 366)
    x0 = xindex % 366
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (366*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (366*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.sigmoid(tmp4)
        tmp6 = tmp3 * tmp5
        tmp7 = 0.0
        tmp8 = tmp6 <= tmp7
        tmp9 = 6.0
        tmp10 = tmp6 >= tmp9
        tmp11 = tmp8 | tmp10
        tmp12 = tl.load(in_ptr2 + ((196*x0) + (71736*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.where(tmp11, tmp7, tmp12)
        tmp14 = tmp13 * tmp5
        tmp15 = tl.load(in_ptr3 + (x0 + (366*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = 196.0
        tmp17 = tmp15 / tmp16
        tmp18 = tmp14 + tmp17
        tmp19 = tl.load(in_ptr4 + (x0 + (366*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/pv/cpvqplsm5es5pdomwhnavwvrxpqgkppfppxwglp27qe3hwqqk7lf.py
# Source Nodes: [sigmoid_2, x_111], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_2 => sigmoid_8
# x_111 => mul_148
triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_151 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_151', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 366
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (366*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o7/co7ektxxw7yrplnxnzv3edqokuiayzy5br6mc6ko4m37sl5qnxh6.py
# Source Nodes: [sigmoid_2, x_111], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_2 => sigmoid_8
# x_111 => mul_148
triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_152 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_152', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 366
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 196)
    y0 = yindex % 196
    tmp0 = tl.load(in_ptr0 + (x2 + (366*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (366*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (y0 + (196*x2) + (71736*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2 + (366*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (366*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tmp6 = 6.0
    tmp7 = tmp3 >= tmp6
    tmp8 = tmp5 | tmp7
    tmp10 = tl.where(tmp8, tmp4, tmp9)
    tmp11 = tmp10 * tmp2
    tmp13 = 196.0
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 0.0006377551020408163
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tmp31 = tmp22 * tmp30
    tmp32 = tmp29 * tmp31
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (366*y3)), tmp32, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nr/cnrsq7apq63z3yru7hfbtxwbbgkb4cmgsltppiz2yynxmhexrb7y.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_153 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_153', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 17934
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
        tmp0 = tl.load(in_ptr0 + ((784*x1) + (286944*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (366*r2) + (46848*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/se/cse6slphgxdakb77mlcbyqvaafxerj2hdaabqstkgdclacqqp4v6.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_154 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_154', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 366
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


# kernel path: /tmp/torchinductor_youkaichao/ey/ceyky5pj7crvennw53lcxzpqwmswhbx6gpws7is2wj7lf44ojzcd.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_155 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_155', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 17934
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 366
    x1 = (xindex // 366)
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((784*x0) + (286944*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (366*r2) + (46848*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (366*r2) + (46848*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 - tmp4
        tmp6 = tmp2 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ps/cpsccrsmxyy725j2rtktii5o63to64mrjl5jtzsx7yapbvagxbuu.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_156 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_156', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 366
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (366*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ac/cacfxqkpy5kw3dw37vgnulzood7umvxr7wfsg52eagoaswgnbsmi.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_157 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_157', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2928
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 366
    y1 = (yindex // 366)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (366*x2) + (286944*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (366*x2) + (286944*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/77/c77q3htgfqa7ru5sodqhoxelhzplxths3xtgd3dl2ikvhnoxh23t.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]

triton_red_fused_add_native_batch_norm_backward_slice_backward_158 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_slice_backward_158', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 61
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 50, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.load(in_ptr0 + (r1 + (784*x0) + (47824*r2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tmp0 < tmp1
        tmp9 = tl.load(in_ptr0 + (r1 + (784*x0) + (47824*r2)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
        tmp11 = tl.where(tmp8, tmp9, tmp10)
        tmp12 = tl.where(tmp8, tmp11, tmp6)
        tmp13 = tmp7 + tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/db/cdbmrcd5jkhgqdvxvt6vxwfzftedmc7sbznugi24xnctxtkbmo4z.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]

triton_red_fused_add_native_batch_norm_backward_slice_backward_159 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_slice_backward_159', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2989
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 49)
    x0 = xindex % 49
    tmp15 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp14 = tl.load(in_ptr1 + (x1 + (61*r2) + (7808*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x1
        tmp1 = tl.full([1, 1], 50, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.load(in_ptr0 + ((784*x1) + (47824*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tmp0 < tmp1
        tmp9 = tl.load(in_ptr0 + ((784*x1) + (47824*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
        tmp11 = tl.where(tmp8, tmp9, tmp10)
        tmp12 = tl.where(tmp8, tmp11, tmp6)
        tmp13 = tmp7 + tmp12
        tmp16 = tmp14 - tmp15
        tmp17 = tmp13 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dy/cdynidjpvvqwzhqzixlo667bmyphy7eej5bl5zkjm65ukxuphn4c.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]

triton_per_fused_add_native_batch_norm_backward_slice_backward_160 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_slice_backward_160', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 61
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


# kernel path: /tmp/torchinductor_youkaichao/ti/ctiut7f4wmms57wn4uiko547jxdi7fddte6egcqzxbusx4knmcsb.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_161 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_161', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 488
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 61
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 61)
    tmp14 = tl.load(in_ptr1 + (y0 + (61*x2) + (47824*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 50, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x2 + (784*y3)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = tmp0 < tmp1
    tmp9 = tl.load(in_ptr0 + (x2 + (784*y3)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = tl.where(tmp8, tmp11, tmp6)
    tmp13 = tmp7 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = 0.00015943877551020407
    tmp19 = tmp17 * tmp18
    tmp21 = tmp20 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp16 * tmp22
    tmp24 = tmp13 - tmp23
    tmp26 = tmp25 * tmp18
    tmp27 = tmp24 - tmp26
    tmp29 = tmp20 * tmp28
    tmp30 = tmp27 * tmp29
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp30, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b3/cb3vb5shrkn3tlslnldj2jjuvsebr2fm72arx5ikqoez5rvfbeca.py
# Source Nodes: [sigmoid_1, x_90], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sum]
# sigmoid_1 => sigmoid_6
# x_90 => mul_118
triton_red_fused_hardtanh_backward_mul_sigmoid_sum_162 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_mul_sigmoid_sum_162', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16800
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7) % 300
    x2 = (xindex // 2100)
    x4 = (xindex // 7)
    tmp1 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    x5 = xindex
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (300*r3) + (33600*x0) + (235200*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr2 + (r3 + (112*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = 6.0
        tmp7 = tmp3 >= tmp6
        tmp8 = tmp5 | tmp7
        tmp10 = tl.where(tmp8, tmp4, tmp9)
        tmp11 = tmp10 * tmp0
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tv/ctvamr2yfah3q7znrhqs2cdr6z5v2gde3n2sswhzulocit236mox.py
# Source Nodes: [sigmoid_1, x_90], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid_1 => sigmoid_6
# x_90 => mul_118
triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_163 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 8],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_163', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2400
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


# kernel path: /tmp/torchinductor_youkaichao/gt/cgtjzftiuvhxlloeg6bu6wshi7pq6f467kedhhkuunkeqidszqbr.py
# Source Nodes: [getattr_l__mod___features___4___se_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
# getattr_l__mod___features___4___se_bn => var_mean_15
triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_164 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_164', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (25*r1)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr1 + (x0 + (25*r1)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr2 + (x0 + (25*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 8, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tmp21 = tl.where(tmp19, tmp18, tmp20)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp0 - tmp10
    tmp27 = tmp21 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None]
    tmp32 = 8.0
    tmp33 = tmp16 / tmp32
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tmp37 = tmp31 * tmp36
    tl.store(out_ptr4 + (x0), tmp37, xmask)
    tl.store(out_ptr0 + (x0), tmp16, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
    tl.store(out_ptr2 + (x0), tmp25, xmask)
    tl.store(out_ptr3 + (x0), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ck/cckcczh3yubgcwk4ceapcqosscymj2ny7726rtkqya67hq5okmx3.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_165 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_165', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 25
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr1 + (x2), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.125
    tmp10 = tmp8 * tmp9
    tmp12 = 8.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = tl.math.rsqrt(tmp15)
    tmp17 = tmp16 * tmp16
    tmp18 = tmp10 * tmp17
    tmp19 = tmp7 * tmp18
    tmp20 = tmp4 - tmp19
    tmp22 = tmp21 * tmp9
    tmp23 = tmp20 - tmp22
    tmp25 = tmp16 * tmp24
    tmp26 = tmp23 * tmp25
    tl.store(in_out_ptr0 + (x2), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/os/cossz2m7nfwhrkkz65bxxnmkid5b4jzmqjft5axkjy5dbjvb7v4b.py
# Source Nodes: [sigmoid_1, x_90], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_1 => sigmoid_6
# x_90 => mul_118
triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_166 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_166', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 14700
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
        tmp0 = tl.load(in_ptr0 + (x1 + (300*r2) + (38400*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (300*((r2 + (128*x0)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr2 + ((784*x1) + (235200*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr3 + (x1 + (300*((r2 + (128*x0)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = 6.0
        tmp7 = tmp3 >= tmp6
        tmp8 = tmp5 | tmp7
        tmp10 = tl.where(tmp8, tmp4, tmp9)
        tmp11 = tmp10 * tmp2
        tmp13 = 784.0
        tmp14 = tmp12 / tmp13
        tmp15 = tmp11 + tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/md/cmd5qpl7vgppms2neihwt335nb4apq7jipel4b56ze4bvbfz4rdl.py
# Source Nodes: [sigmoid_1, x_90], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_1 => sigmoid_6
# x_90 => mul_118
triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_167 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_167', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 300
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


# kernel path: /tmp/torchinductor_youkaichao/c6/cc63yhdhpwxhylbzrkflkpgfjmkinqwphsvginmmgikucxh2wcba.py
# Source Nodes: [sigmoid_1, x_90], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_1 => sigmoid_6
# x_90 => mul_118
triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_168 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_168', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 14700
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 300
    x1 = (xindex // 300)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (300*r2) + (38400*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (300*((r2 + (128*x1)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr2 + ((784*x0) + (235200*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr3 + (x0 + (300*((r2 + (128*x1)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr4 + (x0 + (300*r2) + (38400*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = 6.0
        tmp7 = tmp3 >= tmp6
        tmp8 = tmp5 | tmp7
        tmp10 = tl.where(tmp8, tmp4, tmp9)
        tmp11 = tmp10 * tmp2
        tmp13 = 784.0
        tmp14 = tmp12 / tmp13
        tmp15 = tmp11 + tmp14
        tmp18 = tmp16 - tmp17
        tmp19 = tmp15 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask & xmask, tmp22, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m5/cm56imjpbdu2gxjjlmmihmfzjv6iexxz6v45yeqxln5koe7f6u43.py
# Source Nodes: [sigmoid_1, x_90], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_1 => sigmoid_6
# x_90 => mul_118
triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_169 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_169', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 300
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (300*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mp/cmpl5h27zn6pwfl7tttsltn3uekvxvf4jrobuhmmmz4tz4zim64n.py
# Source Nodes: [sigmoid_1, x_90], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid_1 => sigmoid_6
# x_90 => mul_118
triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_170 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_170', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 300
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 784)
    y0 = yindex % 784
    tmp0 = tl.load(in_ptr0 + (x2 + (300*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (300*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (y0 + (784*x2) + (235200*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2 + (300*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (300*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tmp6 = 6.0
    tmp7 = tmp3 >= tmp6
    tmp8 = tmp5 | tmp7
    tmp10 = tl.where(tmp8, tmp4, tmp9)
    tmp11 = tmp10 * tmp2
    tmp13 = 784.0
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 0.00015943877551020407
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tmp31 = tmp22 * tmp30
    tmp32 = tmp29 * tmp31
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (300*y3)), tmp32, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pd/cpdzej7bzjwpblrhy4yzwhq72ohdu4t4enwdc27nghxsj2p6kunj.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_171 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_171', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 14700
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
        tmp0 = tl.load(in_ptr0 + ((784*x1) + (235200*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (300*r2) + (38400*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ee/ceebzzxiuchyrrcjyayu6tlne6vv34qlt5slgff5ff6ibm5qtank.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_172 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_172', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 14700
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 300
    x1 = (xindex // 300)
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((784*x0) + (235200*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (300*r2) + (38400*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (300*r2) + (38400*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 - tmp4
        tmp6 = tmp2 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/if/cifvss6seyfgqefe45jriuvw5rxdrn7wlol6uwqvdq3arnkfoivw.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_173 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_173', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2400
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 300
    y1 = (yindex // 300)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (300*x2) + (235200*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (300*x2) + (235200*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/5g/c5gsogmcynyxqsefehmuka2etdiwvem27emcn3ltluxwkhz5tqbu.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_174 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_174', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (47824*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (784*x0) + (39200*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (50*r3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp11, xmask)
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tmp11 * tmp13
    tl.store(out_ptr2 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jd/cjdlkbu3xosb66rrrioynsg7cblquz6cz7tzmig5zris5cebhp2b.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_175 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_175', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 400
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 50
    y1 = (yindex // 50)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y0) + (47824*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (50*x2) + (39200*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
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


# kernel path: /tmp/torchinductor_youkaichao/l3/cl3dtzmszg2rwyhzo4rekvzu2bvdl5zcotnb6gbb6e7ntarogbmu.py
# Source Nodes: [sigmoid, x_70], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sum]
# sigmoid => sigmoid_4
# x_70 => mul_88
triton_red_fused_hardtanh_backward_mul_sigmoid_sum_176 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_mul_sigmoid_sum_176', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12768
    rnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x1 = (xindex // 7) % 228
    x2 = (xindex // 1596)
    x4 = (xindex // 7)
    tmp1 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    x5 = xindex
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (228*r3) + (25536*x0) + (178752*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr2 + (r3 + (112*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = 6.0
        tmp7 = tmp3 >= tmp6
        tmp8 = tmp5 | tmp7
        tmp10 = tl.where(tmp8, tmp4, tmp9)
        tmp11 = tmp10 * tmp0
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o3/co32tbjxj64n5lrxul7qneya2iamq2cexu66gpwj7kmslxqx25fz.py
# Source Nodes: [sigmoid, x_70], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
# sigmoid => sigmoid_4
# x_70 => mul_88
triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_177 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_177', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1824
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


# kernel path: /tmp/torchinductor_youkaichao/ze/czecycuhszuho6oboxqcnuceoj3k4y25huxkrhk5fbmi3267svy6.py
# Source Nodes: [getattr_l__mod___features___3___se_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
# getattr_l__mod___features___3___se_bn => var_mean_11
triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_178 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_178', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 19
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (19*r1)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr1 + (x0 + (19*r1)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr2 + (x0 + (19*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 8, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tmp21 = tl.where(tmp19, tmp18, tmp20)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp0 - tmp10
    tmp27 = tmp21 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None]
    tmp32 = 8.0
    tmp33 = tmp16 / tmp32
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tmp37 = tmp31 * tmp36
    tl.store(out_ptr4 + (x0), tmp37, xmask)
    tl.store(out_ptr0 + (x0), tmp16, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
    tl.store(out_ptr2 + (x0), tmp25, xmask)
    tl.store(out_ptr3 + (x0), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q4/cq4q6zxtmqci5iyphhn3m3iy2blzgd5igdi7wwup4z3cjzv4rpnx.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_179 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_179', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/kc/ckcuainlqwq53xbfrmu5ifuon3meyf37ig7y2x2lttvbzqtn62ls.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_180 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_180', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1044
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1044*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ji/cjin43v5gkk7y32pjwkdnbwnu3leepbn7sbxc3rtopt45qwnkdgb.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_181 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_181', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 87
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (87*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7b/c7bfi54slynpeiihfg7x35hvsrsrh7sugnfnmk2majtrjol7v26u.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_182 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_182', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 972
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (972*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kd/ckd6todit2j6vmergvmnjxjqemx76ckeqexpl4u2xub7tm7hcecz.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_183 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_183', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 81
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (81*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m2/cm2oivrlayeo3kjjrwfudxozgowxbizg3b7ghhcdjkc436gcjloh.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_184 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_184', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 906
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (906*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/37/c37kt4llh7qxm7t5k2qaip2pxalq5mrhcg4aiupo3h55ac4iua7l.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_185 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_185', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 75
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (75*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ea/ceack5ldczum6zf7uh65pc6zvzrjepprzvriqn77tdffawp6pzhu.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_186 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_186', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 840
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (840*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ap/capoypkihijmy3rysofgoskjdyv3jvb6kvuupcwlqwicnrdaq3i5.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_187 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_187', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 70
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (70*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lt/cltbacsct4ze46c7omrhjx6qbx7l6ye3ezatynsofluxqtk4lzp6.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_188 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_188', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bo/cboteikg2jzfhwktphiuztgthlh7wlvhf5hxqdd7grrvhna7ygtm.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_189 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_189', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ba/cba3r6rsuaaccaody4xqxyna62yy262x4r6svk277sxq6nzi6gli.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_190 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_190', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 702
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (702*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/56/c564nqqn7rgfbpo4m2gflexaewgyufdqryd2oiwvwu5gwj5n2x4a.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_191 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_191', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 58
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (58*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sq/csqodebwlea3yedmungezmz5e66mfeo2uqdqssliay23gsrtytai.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_192 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_192', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 636
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (636*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ty/ctyfh55wav7ybexblglqfcltwbexu3dlq5wwdodregmmpcmcmroa.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_193 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_193', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 53
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (53*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/55/c55gyzmenx4caqsxhhaqeoowjhgxmrkr7fwayjeainz3crnyc2rs.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_194 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_194', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 570
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (570*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/la/clafxsjct4c2pektnhngdf3r5kwx7vl4s2f6ejysx4empcyf7uvv.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_195 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_195', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 47
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (47*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rz/crztonv5mld7xfjvc6brvyyybltez4ohwnqp7kb4lszqzlf4jey7.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_196 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_196', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 504
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (504*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eg/cegvkqdgd3j5zpmrzqt5z7rgofzlk4j3vgsxvrythl5d6w5adlo6.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_197 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_197', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 42
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (42*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7d/c7ddc57j4q2c7zjppbdkh6ni5znaohbznxwpuh5zcsefzkxse4yi.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_198 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_198', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 432
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (432*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s4/cs4rkifdtd756ptjpvzxydbrp7vocwv2yizkq7dgru7p7wpfxjta.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_199 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_199', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 36
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (36*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wx/cwxwdlyo7cb4xgaziud7b32sf2lyjispepgc7iqqxvxfazat53i3.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_200 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_200', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 366
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (366*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tn/ctnnop3qg2qjndhwqcvbow2ngpnvtkl4se7thb7y42kq6l4ykq4b.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_201 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_201', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 30
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (30*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u2/cu2es3ns2pelqxxa7hryl7wxrlqbvrszcdw52ehbzdutrcb5ttd2.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_202 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_202', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 300
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (300*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/es/cesrtnfxdjkenm7frrmwkcfgaq3is6fhnjj45ipsdhw473oghetc.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_203 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_203', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (25*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4l/c4ly6jfvi4tkwyiygz2olnxxeb3z3s5ghzp6mfwg74juqnytusc6.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_204 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_204', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 228
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (228*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rb/crbr75ftr2raozgnyh6o52nxico3hvjknk5hfbuunxy26uxfd6r4.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_205 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_205', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 19
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr1 + (x2), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 0.125
    tmp10 = tmp8 * tmp9
    tmp12 = 8.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = tl.math.rsqrt(tmp15)
    tmp17 = tmp16 * tmp16
    tmp18 = tmp10 * tmp17
    tmp19 = tmp7 * tmp18
    tmp20 = tmp4 - tmp19
    tmp22 = tmp21 * tmp9
    tmp23 = tmp20 - tmp22
    tmp25 = tmp16 * tmp24
    tmp26 = tmp23 * tmp25
    tl.store(in_out_ptr0 + (x2), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ou/couz2h3rdo7shdzok6z45knvbp2zv5nq7fatogvgnfrhfjdhjyho.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_206 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_206', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 19
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (19*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sh/cshaf7zks5zulyhze5kz3bkmy5r6ugehrqmu566k2gz5o262aylg.py
# Source Nodes: [sigmoid, x_70], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid => sigmoid_4
# x_70 => mul_88
triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_207 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_207', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 11172
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
        tmp0 = tl.load(in_ptr0 + (x1 + (228*r2) + (29184*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (228*((r2 + (128*x0)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr2 + ((784*x1) + (178752*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr3 + (x1 + (228*((r2 + (128*x0)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = 6.0
        tmp7 = tmp3 >= tmp6
        tmp8 = tmp5 | tmp7
        tmp10 = tl.where(tmp8, tmp4, tmp9)
        tmp11 = tmp10 * tmp2
        tmp13 = 784.0
        tmp14 = tmp12 / tmp13
        tmp15 = tmp11 + tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zs/czsr6zzhm4qaaudeaar5wg33vr755cquzav57ebfmz6w4f3z4sns.py
# Source Nodes: [sigmoid, x_70], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid => sigmoid_4
# x_70 => mul_88
triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_208 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_208', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 228
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


# kernel path: /tmp/torchinductor_youkaichao/tr/ctrthp47tgkkxqprvidaqt7bb6d5kxswha6bgfrzdpvzqqumw2qs.py
# Source Nodes: [sigmoid, x_70], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid => sigmoid_4
# x_70 => mul_88
triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_209 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_209', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 11172
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 228
    x1 = (xindex // 228)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (228*r2) + (29184*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (228*((r2 + (128*x1)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr2 + ((784*x0) + (178752*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr3 + (x0 + (228*((r2 + (128*x1)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr4 + (x0 + (228*r2) + (29184*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.sigmoid(tmp1)
        tmp3 = tmp0 * tmp2
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = 6.0
        tmp7 = tmp3 >= tmp6
        tmp8 = tmp5 | tmp7
        tmp10 = tl.where(tmp8, tmp4, tmp9)
        tmp11 = tmp10 * tmp2
        tmp13 = 784.0
        tmp14 = tmp12 / tmp13
        tmp15 = tmp11 + tmp14
        tmp18 = tmp16 - tmp17
        tmp19 = tmp15 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask & xmask, tmp22, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/54/c54sn63ufmrpthua63zalxo3iprvzph7hgq626yipwnbxyuf4cy7.py
# Source Nodes: [sigmoid, x_70], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid => sigmoid_4
# x_70 => mul_88
triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_210 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_210', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 228
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (228*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ut/cutspvoh4ocumuwj5iczvmbtmy7krwhabex5g6kw3ile7lqtyxxo.py
# Source Nodes: [sigmoid, x_70], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
# sigmoid => sigmoid_4
# x_70 => mul_88
triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_211 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_211', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 228
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 784)
    y0 = yindex % 784
    tmp0 = tl.load(in_ptr0 + (x2 + (228*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (228*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (y0 + (784*x2) + (178752*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2 + (228*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2 + (228*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tmp6 = 6.0
    tmp7 = tmp3 >= tmp6
    tmp8 = tmp5 | tmp7
    tmp10 = tl.where(tmp8, tmp4, tmp9)
    tmp11 = tmp10 * tmp2
    tmp13 = 784.0
    tmp14 = tmp12 / tmp13
    tmp15 = tmp11 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 0.00015943877551020407
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tmp31 = tmp22 * tmp30
    tmp32 = tmp29 * tmp31
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (228*y3)), tmp32, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6t/c6ty47izrpvogabnnnno3aej5hsxs4ipgbq3v3oud46texw3ycjq.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_212 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_212', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 44688
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
        tmp0 = tl.load(in_ptr0 + ((3136*x1) + (715008*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (228*r2) + (29184*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r6/cr6ukoxfyvsp6twaxfhs54vd76g747hvflhunuqj773z7kmz7cew.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_213 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_213', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 228
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


# kernel path: /tmp/torchinductor_youkaichao/sf/csfqv2lwzq76lcwcmp6tmfuzcj45laohl56fd2s43myu4zbxj5t3.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_214 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_214', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 44688
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 228
    x1 = (xindex // 228)
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (715008*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (228*r2) + (29184*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (228*r2) + (29184*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 - tmp4
        tmp6 = tmp2 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/23/c23ac23ezuzwtzrq3i5j7cdyi6aejgf7l5ynb3hsc7pu4twxvuom.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_215 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_215', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 228
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
        tmp0 = tl.load(in_ptr0 + (x0 + (228*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fa/cfakeojllddodvus7jp4yc54gbdqv5tps3jtdm2y4maeqxlxfp4l.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_216 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_216', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1824
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 228
    y1 = (yindex // 228)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (228*x2) + (715008*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (228*x2) + (715008*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/5s/c5sr4e2ximoqcvloay67p76l7hg43py2udwvsdtx2kzw6zt3kece.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]

triton_red_fused_add_native_batch_norm_backward_slice_backward_217 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_slice_backward_217', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 152
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 38
    x1 = (xindex // 38)
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = x0
        tmp1 = tl.full([1, 1], 27, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.load(in_ptr0 + ((3136*x0) + (119168*(r2 // 3136)) + (238336*x1) + (r2 % 3136)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tmp0 < tmp1
        tmp9 = tl.load(in_ptr0 + ((3136*x0) + (119168*(r2 // 3136)) + (238336*x1) + (r2 % 3136)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
        tmp11 = tl.where(tmp8, tmp9, tmp10)
        tmp12 = tl.where(tmp8, tmp11, tmp6)
        tmp13 = tmp7 + tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t4/ct4zdi4t26euvvln445cxxwi3fzo6xtrne45nqzcb5nigzevclpe.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]

triton_per_fused_add_native_batch_norm_backward_slice_backward_218 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_slice_backward_218', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 38
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (38*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dp/cdpe3ld7dmj7e76tcgyr4kp55fbbjfn7cklljmkclywypiihbdk4.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]

triton_red_fused_add_native_batch_norm_backward_slice_backward_219 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_slice_backward_219', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7448
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 196)
    x0 = xindex % 196
    tmp15 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp14 = tl.load(in_ptr1 + (x1 + (38*r2) + (4864*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x1
        tmp1 = tl.full([1, 1], 27, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.load(in_ptr0 + ((3136*x1) + (119168*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tmp0 < tmp1
        tmp9 = tl.load(in_ptr0 + ((3136*x1) + (119168*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
        tmp11 = tl.where(tmp8, tmp9, tmp10)
        tmp12 = tl.where(tmp8, tmp11, tmp6)
        tmp13 = tmp7 + tmp12
        tmp16 = tmp14 - tmp15
        tmp17 = tmp13 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eq/ceqje4ggmfpjsd2ayhxnw266yssrmaxz4kfhwwkl363wbv7z7lty.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]

triton_per_fused_add_native_batch_norm_backward_slice_backward_220 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_slice_backward_220', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 38
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


# kernel path: /tmp/torchinductor_youkaichao/7z/c7zxkustvdoy5zweaejfxfcfymoflsh7jrh7ddlnycks44t6l6hw.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_221 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_221', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 304
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 38
    x2 = xindex
    y3 = yindex
    y1 = (yindex // 38)
    tmp14 = tl.load(in_ptr1 + (y0 + (38*x2) + (119168*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 27, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x2 + (3136*y3)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = tmp0 < tmp1
    tmp9 = tl.load(in_ptr0 + (x2 + (3136*y3)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = tl.where(tmp8, tmp11, tmp6)
    tmp13 = tmp7 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = 3.985969387755102e-05
    tmp19 = tmp17 * tmp18
    tmp21 = tmp20 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp16 * tmp22
    tmp24 = tmp13 - tmp23
    tmp26 = tmp25 * tmp18
    tmp27 = tmp24 - tmp26
    tmp29 = tmp20 * tmp28
    tmp30 = tmp27 * tmp29
    tl.store(out_ptr0 + (x2 + (3136*y3)), tmp30, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ee/ceev3a4wdxy2enrtyikzbpgcbwq27i35tmb4x2ycknk3wehwmavb.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_222 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_222', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 31752
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (162*r2) + (20736*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((3136*x1) + (508032*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cm/ccm4igfxfinywer3fsz2vazo3erotm2fulbhfndky5zhchuxguw4.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_223 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_223', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 162
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


# kernel path: /tmp/torchinductor_youkaichao/nl/cnlioxy6ecbljrndhd6gtsjxgfdduqqrf675q2ohqfey2ykcxnsy.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_224 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_224', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 31752
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 162
    x1 = (xindex // 162)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (162*r2) + (20736*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((3136*x0) + (508032*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (162*r2) + (20736*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp3 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nw/cnwteaoyqtkcqwoxevoali772pilce3yfid6hgk7ent3uww7ze5h.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_225 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_225', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 162
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
        tmp0 = tl.load(in_ptr0 + (x0 + (162*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zk/czkqt4mbzcb7nlmcvrfymrlqwavlqqtv5elotbo2kdzuue4nvddx.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_226 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_226', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 162
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
    tmp0 = tl.load(in_ptr0 + (x2 + (162*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (3136*x2) + (508032*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (162*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (162*y3)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cx/ccxjljv2scfkh5gj3zmqqxbe4seyej4hbank7rtj4nqpoaaqsnzn.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_227 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_227', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 31752
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
        tmp0 = tl.load(in_ptr0 + ((3136*x1) + (508032*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (162*r2) + (20736*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zf/czfitj476aw36vpfpf5as2vimkoaru5zybtmpnpiytsj3y6k3iis.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_228 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_228', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 31752
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 162
    x1 = (xindex // 162)
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (508032*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (162*r2) + (20736*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (162*r2) + (20736*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 - tmp4
        tmp6 = tmp2 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nq/cnq7eeqrm6qurz3wulgbcoz7lqkh5cqaoinlbxfkmyhtvrfpzu4j.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_229 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_229', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1296
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 162
    y1 = (yindex // 162)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (162*x2) + (508032*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (162*x2) + (508032*y1)), xmask & ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/gz/cgzy3pkiuarbx7wskzet6xlyedesu4cpaflvamwmav7oxtfaasvj.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_red_fused_add_native_batch_norm_backward_230 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_batch_norm_backward_230', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 108
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 27
    x1 = (xindex // 27)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (119168*(r2 // 3136)) + (238336*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((3136*x0) + (84672*(r2 // 3136)) + (169344*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (27*r2) + (169344*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ky/ckyilj3xlini5zsigua3pje27aopfnfmvcnd7hvrsaydr5kwkj5g.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_231 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_231', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 27
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (27*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hu/chuwjuitnyn3ikx4vy4o2tiz7xgnzxwwzi64h74bjpa2lhuk32lw.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_per_fused_add_native_batch_norm_backward_232 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_batch_norm_backward_232', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 27
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (27*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uj/cujgmjmrll5ifqxpq5bx2jxlremrrjmvdxziicjdg4tyvysqhuaa.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_add_convolution_backward_native_batch_norm_backward_233 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_backward_native_batch_norm_backward_233', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 216
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 27
    y1 = (yindex // 27)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y0) + (119168*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (27*x2) + (84672*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
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


# kernel path: /tmp/torchinductor_youkaichao/g2/cg2xecp4d3mt6jlx7qxuxcu3y6miocv5pr65l3gjorsbxydfoydd.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_234 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_234', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18816
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (96*r2) + (12288*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((3136*x1) + (301056*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ku/ckuaiuk2nusmzsiefllmjf32e6lb5o5trxjwrmqhkvlnoeglh75d.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_235 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_235', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pk/cpk4aqftxxc3t22fnh7jrpvsh2o7bhppziebjxlliemsnplc5jdu.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_236 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_236', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18816
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 96
    x1 = (xindex // 96)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (96*r2) + (12288*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((3136*x0) + (301056*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (96*r2) + (12288*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp3 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lc/clcifjar3lgze3gb3t46lc4xyl6vsczob4yv5pokyoxs2t57rgi4.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_237 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_237', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
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


# kernel path: /tmp/torchinductor_youkaichao/yq/cyqiqtingcjp3u3cqgnlun3j3zr3fnn5ytvhammfpkhmnj37cxyj.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_238 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_238', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (3136*x2) + (301056*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (96*y3)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o3/co3kmukp5kckaut6qz4c63ydt42c3ckjmiaorndivehx3ekzk52r.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_239 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_239', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 75264
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((12544*x1) + (1204224*((r2 + (128*x0)) // 12544)) + ((r2 + (128*x0)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (96*r2) + (12288*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eu/ceuffwz3gzvqpy3q6bboxiz7vwbtb6dhj4af37hihj6xcombdazm.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_per_fused_mul_native_batch_norm_backward_240 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_native_batch_norm_backward_240', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 96
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


# kernel path: /tmp/torchinductor_youkaichao/u5/cu5ishrejgcupkeyjci27r657jvvv72odlymihbbbrl57hwxx7di.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_241 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_241', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 75264
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 96
    x1 = (xindex // 96)
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((12544*x0) + (1204224*((r2 + (128*x1)) // 12544)) + ((r2 + (128*x1)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (96*r2) + (12288*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (96*r2) + (12288*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 - tmp4
        tmp6 = tmp2 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vy/cvye77ccbgdfjkle7wlgkltay22drbtxbgvlvnimqokiiserahkq.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_242 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_242', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
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


# kernel path: /tmp/torchinductor_youkaichao/pg/cpgwbp73efqxuf2ht6373bqpkdwctn7menidljwoxggqnem4xtmx.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_243 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_243', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (96*x2) + (1204224*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (96*x2) + (1204224*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 9.964923469387754e-06
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
    tl.store(in_out_ptr0 + (x2 + (12544*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qz/cqz3jjvvfmwxuujeqso7dl2rlhitwjbp2bqsd77pdwemvmqbiz6r.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_244 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_244', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 208
    rnumel = 7720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7720*x0)
        tmp1 = tl.full([1, 1], 100352, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((12544*x1) + (200704*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ky/cky65b6qugabzv32lszrssvo53umlcztnzehvh7ykv677bz6azfj.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_245 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_245', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
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


# kernel path: /tmp/torchinductor_youkaichao/pd/cpd4wgcihb3ugghm62c3yuafzfnqnq26sahabvx7whcvcdvfld6l.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_246 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_246', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((12544*x1) + (200704*((r2 + (128*x0)) // 12544)) + ((r2 + (128*x0)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (16*r2) + (2048*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n2/cn272l766sdk5upvulc5aoyu6ctpq5dsuta6xkxj273ucbzkv52l.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_247 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_247', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 16
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


# kernel path: /tmp/torchinductor_youkaichao/27/c27tecdpkvgyixeov5344sjx3naokuzqumdxb2y53krkjvjtoojf.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_248 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_248', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 16
    y1 = (yindex // 16)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (16*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 9.964923469387754e-06
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
    tl.store(in_out_ptr0 + (x2 + (12544*y3)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yp/cypfg36wbu3tiaj3hsbzymqu66dhitxefkmu4xubl3exutmbybgq.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_249 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_249', 'mutated_arg_names': []}
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
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (32*r2) + (4096*x0)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((12544*x1) + (401408*((r2 + (128*x0)) // 12544)) + ((r2 + (128*x0)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/76/c76gmncehzhmupv7dgalprdfwsz625ojz5lrlihv5tidvqmh56cm.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_per_fused_hardtanh_backward_native_batch_norm_backward_250 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardtanh_backward_native_batch_norm_backward_250', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/7o/c7ovx6dbplfot23a5cmf2h7trloutqzssbyzqfu57bdcwx3olzg7.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_251 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_251', 'mutated_arg_names': []}
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
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r2) + (4096*x1)), rmask & xmask, eviction_policy='evict_last').to(tl.int1)
        tmp1 = tl.load(in_ptr1 + ((12544*x0) + (401408*((r2 + (128*x1)) // 12544)) + ((r2 + (128*x1)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (x0 + (32*r2) + (4096*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = 0.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp3 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hs/chsaqb4ndsawxjnn6tjebkg27f3pxrvjywx2m2zun3hqzf4jk4fc.py
# Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_red_fused_hardtanh_backward_native_batch_norm_backward_252 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardtanh_backward_native_batch_norm_backward_252', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/64/c64zwetof5y77f7xkkxemymxyavcyl4c5iptvuwnnj6mkevyhy6m.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_253 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_253', 'mutated_arg_names': []},
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
    tmp0 = tl.load(in_ptr0 + (x2 + (32*y3)), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (y0 + (12544*x2) + (401408*y1)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2 + (32*y3)), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (32*y3)), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xn/cxno25o7rvmoup43vvsxfzhnzlabkk2mnguzy6ntexcjo42bin5h.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_254 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_254', 'mutated_arg_names': []}
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
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((12544*x1) + (401408*((r2 + (128*x0)) // 12544)) + ((r2 + (128*x0)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x1 + (32*r2) + (4096*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7p/c7p27mjb436ul7g6wtxwyhcvrywlzwosw7ywqmi3lpbvslspgzcs.py
# Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]

triton_red_fused_mul_native_batch_norm_backward_255 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_native_batch_norm_backward_255', 'mutated_arg_names': []}
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
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((12544*x0) + (401408*((r2 + (128*x1)) // 12544)) + ((r2 + (128*x1)) % 12544)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (32*r2) + (4096*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (32*r2) + (4096*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp5 = tmp3 - tmp4
        tmp6 = tmp2 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ay/caymidr6dsx6dgo2s67lxg2hpn3ikiauds7etzconq4ejyfo4bi6.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_256 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_256', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 12544
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0 + (32*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (32*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 9.964923469387754e-06
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
    tl.store(in_out_ptr0 + (x2 + (12544*y3)), tmp19, xmask & ymask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_112, primals_114, primals_116, primals_117, primals_118, primals_119, primals_121, primals_123, primals_125, primals_126, primals_127, primals_128, primals_130, primals_132, primals_134, primals_135, primals_136, primals_137, primals_139, primals_141, primals_143, primals_144, primals_145, primals_146, primals_148, primals_150, primals_152, primals_153, primals_154, primals_155, primals_157, primals_159, primals_161, primals_162, primals_163, primals_164, primals_166, primals_168, primals_170, primals_171, primals_172, primals_173, primals_175, primals_177, primals_179, primals_180, primals_181, primals_182, primals_184, primals_186, primals_188, primals_189, primals_190, primals_191, primals_193, primals_195, primals_197, primals_198, primals_199, primals_200, primals_202, primals_204, primals_206, primals_207, primals_208, primals_209, primals_211, primals_213, primals_215, primals_216, primals_217, primals_218, primals_220, primals_222, primals_224, primals_225, primals_414, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, clamp_max, convolution_2, squeeze_7, add_14, convolution_3, squeeze_10, mul_29, convolution_4, squeeze_13, clamp_max_1, convolution_5, squeeze_16, add_29, convolution_6, squeeze_19, mul_51, convolution_7, squeeze_22, clamp_max_2, convolution_8, squeeze_25, cat, convolution_9, squeeze_28, mul_73, convolution_10, squeeze_31, add_55, mean, convolution_11, relu, convolution_12, clamp_max_3, convolution_13, squeeze_37, add_65, convolution_14, squeeze_40, mul_103, convolution_15, squeeze_43, add_75, mean_1, convolution_16, relu_1, convolution_17, clamp_max_4, convolution_18, squeeze_49, cat_1, convolution_19, squeeze_52, mul_133, convolution_20, squeeze_55, add_96, mean_2, convolution_21, relu_2, convolution_22, clamp_max_5, convolution_23, squeeze_61, add_106, convolution_24, squeeze_64, mul_163, convolution_25, squeeze_67, add_116, mean_3, convolution_26, relu_3, convolution_27, clamp_max_6, convolution_28, squeeze_73, cat_2, convolution_29, squeeze_76, mul_193, convolution_30, squeeze_79, add_137, mean_4, convolution_31, relu_4, convolution_32, clamp_max_7, convolution_33, squeeze_85, cat_3, convolution_34, squeeze_88, mul_223, convolution_35, squeeze_91, add_158, mean_5, convolution_36, relu_5, convolution_37, clamp_max_8, convolution_38, squeeze_97, cat_4, convolution_39, squeeze_100, mul_253, convolution_40, squeeze_103, add_179, mean_6, convolution_41, relu_6, convolution_42, clamp_max_9, convolution_43, squeeze_109, cat_5, convolution_44, squeeze_112, mul_283, convolution_45, squeeze_115, add_200, mean_7, convolution_46, relu_7, convolution_47, clamp_max_10, convolution_48, squeeze_121, cat_6, convolution_49, squeeze_124, mul_313, convolution_50, squeeze_127, add_221, mean_8, convolution_51, relu_8, convolution_52, clamp_max_11, convolution_53, squeeze_133, add_231, convolution_54, squeeze_136, mul_343, convolution_55, squeeze_139, add_241, mean_9, convolution_56, relu_9, convolution_57, clamp_max_12, convolution_58, squeeze_145, cat_7, convolution_59, squeeze_148, mul_373, convolution_60, squeeze_151, add_262, mean_10, convolution_61, relu_10, convolution_62, clamp_max_13, convolution_63, squeeze_157, cat_8, convolution_64, squeeze_160, mul_403, convolution_65, squeeze_163, add_283, mean_11, convolution_66, relu_11, convolution_67, clamp_max_14, convolution_68, squeeze_169, cat_9, convolution_69, squeeze_172, mul_433, convolution_70, squeeze_175, add_304, mean_12, convolution_71, relu_12, convolution_72, clamp_max_15, convolution_73, squeeze_181, cat_10, convolution_74, squeeze_184, clone_17, permute_1, mul_465, unsqueeze_250, unsqueeze_262, unsqueeze_286, mul_508, unsqueeze_298, unsqueeze_310, unsqueeze_334, mul_551, unsqueeze_346, unsqueeze_358, unsqueeze_382, mul_594, unsqueeze_394, unsqueeze_406, unsqueeze_430, mul_637, unsqueeze_442, unsqueeze_454, unsqueeze_478, mul_680, unsqueeze_490, unsqueeze_502, unsqueeze_526, mul_723, unsqueeze_538, unsqueeze_550, unsqueeze_574, mul_766, unsqueeze_586, unsqueeze_598, unsqueeze_622, mul_809, unsqueeze_634, unsqueeze_646, unsqueeze_670, mul_852, unsqueeze_682, unsqueeze_694, unsqueeze_718, mul_895, unsqueeze_730, unsqueeze_742, unsqueeze_766, mul_938, unsqueeze_778, unsqueeze_790, unsqueeze_814, mul_981, unsqueeze_826, unsqueeze_838, unsqueeze_862, mul_1024, unsqueeze_874, unsqueeze_886, bitwise_or_13, unsqueeze_898, mul_1054, unsqueeze_910, unsqueeze_922, bitwise_or_14, unsqueeze_934, mul_1084, unsqueeze_946, unsqueeze_958, bitwise_or_15, unsqueeze_970, mul_1114, unsqueeze_982, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, ), (1, ))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_7, (96, ), (1, ))
    assert_size_stride(primals_9, (96, ), (1, ))
    assert_size_stride(primals_11, (27, ), (1, ))
    assert_size_stride(primals_13, (162, ), (1, ))
    assert_size_stride(primals_15, (162, ), (1, ))
    assert_size_stride(primals_17, (38, ), (1, ))
    assert_size_stride(primals_19, (228, ), (1, ))
    assert_size_stride(primals_21, (228, ), (1, ))
    assert_size_stride(primals_23, (50, ), (1, ))
    assert_size_stride(primals_25, (300, ), (1, ))
    assert_size_stride(primals_27, (300, ), (1, ))
    assert_size_stride(primals_29, (61, ), (1, ))
    assert_size_stride(primals_31, (366, ), (1, ))
    assert_size_stride(primals_33, (366, ), (1, ))
    assert_size_stride(primals_35, (72, ), (1, ))
    assert_size_stride(primals_37, (432, ), (1, ))
    assert_size_stride(primals_39, (432, ), (1, ))
    assert_size_stride(primals_41, (84, ), (1, ))
    assert_size_stride(primals_43, (504, ), (1, ))
    assert_size_stride(primals_45, (504, ), (1, ))
    assert_size_stride(primals_47, (95, ), (1, ))
    assert_size_stride(primals_49, (570, ), (1, ))
    assert_size_stride(primals_51, (570, ), (1, ))
    assert_size_stride(primals_53, (106, ), (1, ))
    assert_size_stride(primals_55, (636, ), (1, ))
    assert_size_stride(primals_57, (636, ), (1, ))
    assert_size_stride(primals_59, (117, ), (1, ))
    assert_size_stride(primals_61, (702, ), (1, ))
    assert_size_stride(primals_63, (702, ), (1, ))
    assert_size_stride(primals_65, (128, ), (1, ))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_69, (768, ), (1, ))
    assert_size_stride(primals_71, (140, ), (1, ))
    assert_size_stride(primals_73, (840, ), (1, ))
    assert_size_stride(primals_75, (840, ), (1, ))
    assert_size_stride(primals_77, (151, ), (1, ))
    assert_size_stride(primals_79, (906, ), (1, ))
    assert_size_stride(primals_81, (906, ), (1, ))
    assert_size_stride(primals_83, (162, ), (1, ))
    assert_size_stride(primals_85, (972, ), (1, ))
    assert_size_stride(primals_87, (972, ), (1, ))
    assert_size_stride(primals_89, (174, ), (1, ))
    assert_size_stride(primals_91, (1044, ), (1, ))
    assert_size_stride(primals_93, (1044, ), (1, ))
    assert_size_stride(primals_95, (185, ), (1, ))
    assert_size_stride(primals_97, (1280, ), (1, ))
    assert_size_stride(primals_99, (32, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_100, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_101, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_102, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_103, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_104, (27, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_105, (162, 27, 1, 1), (27, 1, 1, 1))
    assert_size_stride(primals_106, (162, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_107, (38, 162, 1, 1), (162, 1, 1, 1))
    assert_size_stride(primals_108, (228, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(primals_109, (228, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_110, (19, 228, 1, 1), (228, 1, 1, 1))
    assert_size_stride(primals_112, (19, ), (1, ))
    assert_size_stride(primals_114, (228, 19, 1, 1), (19, 1, 1, 1))
    assert_size_stride(primals_116, (50, 228, 1, 1), (228, 1, 1, 1))
    assert_size_stride(primals_117, (300, 50, 1, 1), (50, 1, 1, 1))
    assert_size_stride(primals_118, (300, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_119, (25, 300, 1, 1), (300, 1, 1, 1))
    assert_size_stride(primals_121, (25, ), (1, ))
    assert_size_stride(primals_123, (300, 25, 1, 1), (25, 1, 1, 1))
    assert_size_stride(primals_125, (61, 300, 1, 1), (300, 1, 1, 1))
    assert_size_stride(primals_126, (366, 61, 1, 1), (61, 1, 1, 1))
    assert_size_stride(primals_127, (366, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_128, (30, 366, 1, 1), (366, 1, 1, 1))
    assert_size_stride(primals_130, (30, ), (1, ))
    assert_size_stride(primals_132, (366, 30, 1, 1), (30, 1, 1, 1))
    assert_size_stride(primals_134, (72, 366, 1, 1), (366, 1, 1, 1))
    assert_size_stride(primals_135, (432, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_136, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_137, (36, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(primals_139, (36, ), (1, ))
    assert_size_stride(primals_141, (432, 36, 1, 1), (36, 1, 1, 1))
    assert_size_stride(primals_143, (84, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(primals_144, (504, 84, 1, 1), (84, 1, 1, 1))
    assert_size_stride(primals_145, (504, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_146, (42, 504, 1, 1), (504, 1, 1, 1))
    assert_size_stride(primals_148, (42, ), (1, ))
    assert_size_stride(primals_150, (504, 42, 1, 1), (42, 1, 1, 1))
    assert_size_stride(primals_152, (95, 504, 1, 1), (504, 1, 1, 1))
    assert_size_stride(primals_153, (570, 95, 1, 1), (95, 1, 1, 1))
    assert_size_stride(primals_154, (570, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_155, (47, 570, 1, 1), (570, 1, 1, 1))
    assert_size_stride(primals_157, (47, ), (1, ))
    assert_size_stride(primals_159, (570, 47, 1, 1), (47, 1, 1, 1))
    assert_size_stride(primals_161, (106, 570, 1, 1), (570, 1, 1, 1))
    assert_size_stride(primals_162, (636, 106, 1, 1), (106, 1, 1, 1))
    assert_size_stride(primals_163, (636, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_164, (53, 636, 1, 1), (636, 1, 1, 1))
    assert_size_stride(primals_166, (53, ), (1, ))
    assert_size_stride(primals_168, (636, 53, 1, 1), (53, 1, 1, 1))
    assert_size_stride(primals_170, (117, 636, 1, 1), (636, 1, 1, 1))
    assert_size_stride(primals_171, (702, 117, 1, 1), (117, 1, 1, 1))
    assert_size_stride(primals_172, (702, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_173, (58, 702, 1, 1), (702, 1, 1, 1))
    assert_size_stride(primals_175, (58, ), (1, ))
    assert_size_stride(primals_177, (702, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_179, (128, 702, 1, 1), (702, 1, 1, 1))
    assert_size_stride(primals_180, (768, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_181, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_182, (64, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_184, (64, ), (1, ))
    assert_size_stride(primals_186, (768, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_188, (140, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_189, (840, 140, 1, 1), (140, 1, 1, 1))
    assert_size_stride(primals_190, (840, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_191, (70, 840, 1, 1), (840, 1, 1, 1))
    assert_size_stride(primals_193, (70, ), (1, ))
    assert_size_stride(primals_195, (840, 70, 1, 1), (70, 1, 1, 1))
    assert_size_stride(primals_197, (151, 840, 1, 1), (840, 1, 1, 1))
    assert_size_stride(primals_198, (906, 151, 1, 1), (151, 1, 1, 1))
    assert_size_stride(primals_199, (906, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_200, (75, 906, 1, 1), (906, 1, 1, 1))
    assert_size_stride(primals_202, (75, ), (1, ))
    assert_size_stride(primals_204, (906, 75, 1, 1), (75, 1, 1, 1))
    assert_size_stride(primals_206, (162, 906, 1, 1), (906, 1, 1, 1))
    assert_size_stride(primals_207, (972, 162, 1, 1), (162, 1, 1, 1))
    assert_size_stride(primals_208, (972, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_209, (81, 972, 1, 1), (972, 1, 1, 1))
    assert_size_stride(primals_211, (81, ), (1, ))
    assert_size_stride(primals_213, (972, 81, 1, 1), (81, 1, 1, 1))
    assert_size_stride(primals_215, (174, 972, 1, 1), (972, 1, 1, 1))
    assert_size_stride(primals_216, (1044, 174, 1, 1), (174, 1, 1, 1))
    assert_size_stride(primals_217, (1044, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_218, (87, 1044, 1, 1), (1044, 1, 1, 1))
    assert_size_stride(primals_220, (87, ), (1, ))
    assert_size_stride(primals_222, (1044, 87, 1, 1), (87, 1, 1, 1))
    assert_size_stride(primals_224, (185, 1044, 1, 1), (1044, 1, 1, 1))
    assert_size_stride(primals_225, (1280, 185, 1, 1), (185, 1, 1, 1))
    assert_size_stride(primals_414, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(squeeze_1, (32, ), (1, ))
    assert_size_stride(mul_7, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(convolution_1, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(squeeze_4, (32, ), (1, ))
    assert_size_stride(clamp_max, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(convolution_2, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(squeeze_7, (16, ), (1, ))
    assert_size_stride(add_14, (8, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_3, (8, 96, 112, 112), (1204224, 1, 10752, 96))
    assert_size_stride(squeeze_10, (96, ), (1, ))
    assert_size_stride(mul_29, (8, 96, 112, 112), (1204224, 1, 10752, 96))
    assert_size_stride(convolution_4, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(squeeze_13, (96, ), (1, ))
    assert_size_stride(clamp_max_1, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(convolution_5, (8, 27, 56, 56), (84672, 1, 1512, 27))
    assert_size_stride(squeeze_16, (27, ), (1, ))
    assert_size_stride(add_29, (8, 27, 56, 56), (84672, 1, 1512, 27))
    assert_size_stride(convolution_6, (8, 162, 56, 56), (508032, 1, 9072, 162))
    assert_size_stride(squeeze_19, (162, ), (1, ))
    assert_size_stride(mul_51, (8, 162, 56, 56), (508032, 1, 9072, 162))
    assert_size_stride(convolution_7, (8, 162, 56, 56), (508032, 1, 9072, 162))
    assert_size_stride(squeeze_22, (162, ), (1, ))
    assert_size_stride(clamp_max_2, (8, 162, 56, 56), (508032, 1, 9072, 162))
    assert_size_stride(convolution_8, (8, 38, 56, 56), (119168, 1, 2128, 38))
    assert_size_stride(squeeze_25, (38, ), (1, ))
    assert_size_stride(cat, (8, 38, 56, 56), (119168, 1, 2128, 38))
    assert_size_stride(convolution_9, (8, 228, 56, 56), (715008, 1, 12768, 228))
    assert_size_stride(squeeze_28, (228, ), (1, ))
    assert_size_stride(mul_73, (8, 228, 56, 56), (715008, 1, 12768, 228))
    assert_size_stride(convolution_10, (8, 228, 28, 28), (178752, 1, 6384, 228))
    assert_size_stride(squeeze_31, (228, ), (1, ))
    assert_size_stride(add_55, (8, 228, 28, 28), (178752, 1, 6384, 228))
    assert_size_stride(mean, (8, 228, 1, 1), (228, 1, 228, 228))
    assert_size_stride(convolution_11, (8, 19, 1, 1), (19, 1, 19, 19))
    assert_size_stride(relu, (8, 19, 1, 1), (19, 1, 19, 19))
    assert_size_stride(convolution_12, (8, 228, 1, 1), (228, 1, 228, 228))
    assert_size_stride(clamp_max_3, (8, 228, 28, 28), (178752, 1, 6384, 228))
    assert_size_stride(convolution_13, (8, 50, 28, 28), (39200, 1, 1400, 50))
    assert_size_stride(squeeze_37, (50, ), (1, ))
    assert_size_stride(add_65, (8, 50, 28, 28), (39200, 1, 1400, 50))
    assert_size_stride(convolution_14, (8, 300, 28, 28), (235200, 1, 8400, 300))
    assert_size_stride(squeeze_40, (300, ), (1, ))
    assert_size_stride(mul_103, (8, 300, 28, 28), (235200, 1, 8400, 300))
    assert_size_stride(convolution_15, (8, 300, 28, 28), (235200, 1, 8400, 300))
    assert_size_stride(squeeze_43, (300, ), (1, ))
    assert_size_stride(add_75, (8, 300, 28, 28), (235200, 1, 8400, 300))
    assert_size_stride(mean_1, (8, 300, 1, 1), (300, 1, 300, 300))
    assert_size_stride(convolution_16, (8, 25, 1, 1), (25, 1, 25, 25))
    assert_size_stride(relu_1, (8, 25, 1, 1), (25, 1, 25, 25))
    assert_size_stride(convolution_17, (8, 300, 1, 1), (300, 1, 300, 300))
    assert_size_stride(clamp_max_4, (8, 300, 28, 28), (235200, 1, 8400, 300))
    assert_size_stride(convolution_18, (8, 61, 28, 28), (47824, 1, 1708, 61))
    assert_size_stride(squeeze_49, (61, ), (1, ))
    assert_size_stride(cat_1, (8, 61, 28, 28), (47824, 1, 1708, 61))
    assert_size_stride(convolution_19, (8, 366, 28, 28), (286944, 1, 10248, 366))
    assert_size_stride(squeeze_52, (366, ), (1, ))
    assert_size_stride(mul_133, (8, 366, 28, 28), (286944, 1, 10248, 366))
    assert_size_stride(convolution_20, (8, 366, 14, 14), (71736, 1, 5124, 366))
    assert_size_stride(squeeze_55, (366, ), (1, ))
    assert_size_stride(add_96, (8, 366, 14, 14), (71736, 1, 5124, 366))
    assert_size_stride(mean_2, (8, 366, 1, 1), (366, 1, 366, 366))
    assert_size_stride(convolution_21, (8, 30, 1, 1), (30, 1, 30, 30))
    assert_size_stride(relu_2, (8, 30, 1, 1), (30, 1, 30, 30))
    assert_size_stride(convolution_22, (8, 366, 1, 1), (366, 1, 366, 366))
    assert_size_stride(clamp_max_5, (8, 366, 14, 14), (71736, 1, 5124, 366))
    assert_size_stride(convolution_23, (8, 72, 14, 14), (14112, 1, 1008, 72))
    assert_size_stride(squeeze_61, (72, ), (1, ))
    assert_size_stride(add_106, (8, 72, 14, 14), (14112, 1, 1008, 72))
    assert_size_stride(convolution_24, (8, 432, 14, 14), (84672, 1, 6048, 432))
    assert_size_stride(squeeze_64, (432, ), (1, ))
    assert_size_stride(mul_163, (8, 432, 14, 14), (84672, 1, 6048, 432))
    assert_size_stride(convolution_25, (8, 432, 14, 14), (84672, 1, 6048, 432))
    assert_size_stride(squeeze_67, (432, ), (1, ))
    assert_size_stride(add_116, (8, 432, 14, 14), (84672, 1, 6048, 432))
    assert_size_stride(mean_3, (8, 432, 1, 1), (432, 1, 432, 432))
    assert_size_stride(convolution_26, (8, 36, 1, 1), (36, 1, 36, 36))
    assert_size_stride(relu_3, (8, 36, 1, 1), (36, 1, 36, 36))
    assert_size_stride(convolution_27, (8, 432, 1, 1), (432, 1, 432, 432))
    assert_size_stride(clamp_max_6, (8, 432, 14, 14), (84672, 1, 6048, 432))
    assert_size_stride(convolution_28, (8, 84, 14, 14), (16464, 1, 1176, 84))
    assert_size_stride(squeeze_73, (84, ), (1, ))
    assert_size_stride(cat_2, (8, 84, 14, 14), (16464, 1, 1176, 84))
    assert_size_stride(convolution_29, (8, 504, 14, 14), (98784, 1, 7056, 504))
    assert_size_stride(squeeze_76, (504, ), (1, ))
    assert_size_stride(mul_193, (8, 504, 14, 14), (98784, 1, 7056, 504))
    assert_size_stride(convolution_30, (8, 504, 14, 14), (98784, 1, 7056, 504))
    assert_size_stride(squeeze_79, (504, ), (1, ))
    assert_size_stride(add_137, (8, 504, 14, 14), (98784, 1, 7056, 504))
    assert_size_stride(mean_4, (8, 504, 1, 1), (504, 1, 504, 504))
    assert_size_stride(convolution_31, (8, 42, 1, 1), (42, 1, 42, 42))
    assert_size_stride(relu_4, (8, 42, 1, 1), (42, 1, 42, 42))
    assert_size_stride(convolution_32, (8, 504, 1, 1), (504, 1, 504, 504))
    assert_size_stride(clamp_max_7, (8, 504, 14, 14), (98784, 1, 7056, 504))
    assert_size_stride(convolution_33, (8, 95, 14, 14), (18620, 1, 1330, 95))
    assert_size_stride(squeeze_85, (95, ), (1, ))
    assert_size_stride(cat_3, (8, 95, 14, 14), (18620, 1, 1330, 95))
    assert_size_stride(convolution_34, (8, 570, 14, 14), (111720, 1, 7980, 570))
    assert_size_stride(squeeze_88, (570, ), (1, ))
    assert_size_stride(mul_223, (8, 570, 14, 14), (111720, 1, 7980, 570))
    assert_size_stride(convolution_35, (8, 570, 14, 14), (111720, 1, 7980, 570))
    assert_size_stride(squeeze_91, (570, ), (1, ))
    assert_size_stride(add_158, (8, 570, 14, 14), (111720, 1, 7980, 570))
    assert_size_stride(mean_5, (8, 570, 1, 1), (570, 1, 570, 570))
    assert_size_stride(convolution_36, (8, 47, 1, 1), (47, 1, 47, 47))
    assert_size_stride(relu_5, (8, 47, 1, 1), (47, 1, 47, 47))
    assert_size_stride(convolution_37, (8, 570, 1, 1), (570, 1, 570, 570))
    assert_size_stride(clamp_max_8, (8, 570, 14, 14), (111720, 1, 7980, 570))
    assert_size_stride(convolution_38, (8, 106, 14, 14), (20776, 1, 1484, 106))
    assert_size_stride(squeeze_97, (106, ), (1, ))
    assert_size_stride(cat_4, (8, 106, 14, 14), (20776, 1, 1484, 106))
    assert_size_stride(convolution_39, (8, 636, 14, 14), (124656, 1, 8904, 636))
    assert_size_stride(squeeze_100, (636, ), (1, ))
    assert_size_stride(mul_253, (8, 636, 14, 14), (124656, 1, 8904, 636))
    assert_size_stride(convolution_40, (8, 636, 14, 14), (124656, 1, 8904, 636))
    assert_size_stride(squeeze_103, (636, ), (1, ))
    assert_size_stride(add_179, (8, 636, 14, 14), (124656, 1, 8904, 636))
    assert_size_stride(mean_6, (8, 636, 1, 1), (636, 1, 636, 636))
    assert_size_stride(convolution_41, (8, 53, 1, 1), (53, 1, 53, 53))
    assert_size_stride(relu_6, (8, 53, 1, 1), (53, 1, 53, 53))
    assert_size_stride(convolution_42, (8, 636, 1, 1), (636, 1, 636, 636))
    assert_size_stride(clamp_max_9, (8, 636, 14, 14), (124656, 1, 8904, 636))
    assert_size_stride(convolution_43, (8, 117, 14, 14), (22932, 1, 1638, 117))
    assert_size_stride(squeeze_109, (117, ), (1, ))
    assert_size_stride(cat_5, (8, 117, 14, 14), (22932, 1, 1638, 117))
    assert_size_stride(convolution_44, (8, 702, 14, 14), (137592, 1, 9828, 702))
    assert_size_stride(squeeze_112, (702, ), (1, ))
    assert_size_stride(mul_283, (8, 702, 14, 14), (137592, 1, 9828, 702))
    assert_size_stride(convolution_45, (8, 702, 14, 14), (137592, 1, 9828, 702))
    assert_size_stride(squeeze_115, (702, ), (1, ))
    assert_size_stride(add_200, (8, 702, 14, 14), (137592, 1, 9828, 702))
    assert_size_stride(mean_7, (8, 702, 1, 1), (702, 1, 702, 702))
    assert_size_stride(convolution_46, (8, 58, 1, 1), (58, 1, 58, 58))
    assert_size_stride(relu_7, (8, 58, 1, 1), (58, 1, 58, 58))
    assert_size_stride(convolution_47, (8, 702, 1, 1), (702, 1, 702, 702))
    assert_size_stride(clamp_max_10, (8, 702, 14, 14), (137592, 1, 9828, 702))
    assert_size_stride(convolution_48, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(squeeze_121, (128, ), (1, ))
    assert_size_stride(cat_6, (8, 128, 14, 14), (25088, 1, 1792, 128))
    assert_size_stride(convolution_49, (8, 768, 14, 14), (150528, 1, 10752, 768))
    assert_size_stride(squeeze_124, (768, ), (1, ))
    assert_size_stride(mul_313, (8, 768, 14, 14), (150528, 1, 10752, 768))
    assert_size_stride(convolution_50, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(squeeze_127, (768, ), (1, ))
    assert_size_stride(add_221, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(mean_8, (8, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(convolution_51, (8, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(relu_8, (8, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(convolution_52, (8, 768, 1, 1), (768, 1, 768, 768))
    assert_size_stride(clamp_max_11, (8, 768, 7, 7), (37632, 1, 5376, 768))
    assert_size_stride(convolution_53, (8, 140, 7, 7), (6860, 1, 980, 140))
    assert_size_stride(squeeze_133, (140, ), (1, ))
    assert_size_stride(add_231, (8, 140, 7, 7), (6860, 1, 980, 140))
    assert_size_stride(convolution_54, (8, 840, 7, 7), (41160, 1, 5880, 840))
    assert_size_stride(squeeze_136, (840, ), (1, ))
    assert_size_stride(mul_343, (8, 840, 7, 7), (41160, 1, 5880, 840))
    assert_size_stride(convolution_55, (8, 840, 7, 7), (41160, 1, 5880, 840))
    assert_size_stride(squeeze_139, (840, ), (1, ))
    assert_size_stride(add_241, (8, 840, 7, 7), (41160, 1, 5880, 840))
    assert_size_stride(mean_9, (8, 840, 1, 1), (840, 1, 840, 840))
    assert_size_stride(convolution_56, (8, 70, 1, 1), (70, 1, 70, 70))
    assert_size_stride(relu_9, (8, 70, 1, 1), (70, 1, 70, 70))
    assert_size_stride(convolution_57, (8, 840, 1, 1), (840, 1, 840, 840))
    assert_size_stride(clamp_max_12, (8, 840, 7, 7), (41160, 1, 5880, 840))
    assert_size_stride(convolution_58, (8, 151, 7, 7), (7399, 1, 1057, 151))
    assert_size_stride(squeeze_145, (151, ), (1, ))
    assert_size_stride(cat_7, (8, 151, 7, 7), (7399, 1, 1057, 151))
    assert_size_stride(convolution_59, (8, 906, 7, 7), (44394, 1, 6342, 906))
    assert_size_stride(squeeze_148, (906, ), (1, ))
    assert_size_stride(mul_373, (8, 906, 7, 7), (44394, 1, 6342, 906))
    assert_size_stride(convolution_60, (8, 906, 7, 7), (44394, 1, 6342, 906))
    assert_size_stride(squeeze_151, (906, ), (1, ))
    assert_size_stride(add_262, (8, 906, 7, 7), (44394, 1, 6342, 906))
    assert_size_stride(mean_10, (8, 906, 1, 1), (906, 1, 906, 906))
    assert_size_stride(convolution_61, (8, 75, 1, 1), (75, 1, 75, 75))
    assert_size_stride(relu_10, (8, 75, 1, 1), (75, 1, 75, 75))
    assert_size_stride(convolution_62, (8, 906, 1, 1), (906, 1, 906, 906))
    assert_size_stride(clamp_max_13, (8, 906, 7, 7), (44394, 1, 6342, 906))
    assert_size_stride(convolution_63, (8, 162, 7, 7), (7938, 1, 1134, 162))
    assert_size_stride(squeeze_157, (162, ), (1, ))
    assert_size_stride(cat_8, (8, 162, 7, 7), (7938, 1, 1134, 162))
    assert_size_stride(convolution_64, (8, 972, 7, 7), (47628, 1, 6804, 972))
    assert_size_stride(squeeze_160, (972, ), (1, ))
    assert_size_stride(mul_403, (8, 972, 7, 7), (47628, 1, 6804, 972))
    assert_size_stride(convolution_65, (8, 972, 7, 7), (47628, 1, 6804, 972))
    assert_size_stride(squeeze_163, (972, ), (1, ))
    assert_size_stride(add_283, (8, 972, 7, 7), (47628, 1, 6804, 972))
    assert_size_stride(mean_11, (8, 972, 1, 1), (972, 1, 972, 972))
    assert_size_stride(convolution_66, (8, 81, 1, 1), (81, 1, 81, 81))
    assert_size_stride(relu_11, (8, 81, 1, 1), (81, 1, 81, 81))
    assert_size_stride(convolution_67, (8, 972, 1, 1), (972, 1, 972, 972))
    assert_size_stride(clamp_max_14, (8, 972, 7, 7), (47628, 1, 6804, 972))
    assert_size_stride(convolution_68, (8, 174, 7, 7), (8526, 1, 1218, 174))
    assert_size_stride(squeeze_169, (174, ), (1, ))
    assert_size_stride(cat_9, (8, 174, 7, 7), (8526, 1, 1218, 174))
    assert_size_stride(convolution_69, (8, 1044, 7, 7), (51156, 1, 7308, 1044))
    assert_size_stride(squeeze_172, (1044, ), (1, ))
    assert_size_stride(mul_433, (8, 1044, 7, 7), (51156, 1, 7308, 1044))
    assert_size_stride(convolution_70, (8, 1044, 7, 7), (51156, 1, 7308, 1044))
    assert_size_stride(squeeze_175, (1044, ), (1, ))
    assert_size_stride(add_304, (8, 1044, 7, 7), (51156, 1, 7308, 1044))
    assert_size_stride(mean_12, (8, 1044, 1, 1), (1044, 1, 1044, 1044))
    assert_size_stride(convolution_71, (8, 87, 1, 1), (87, 1, 87, 87))
    assert_size_stride(relu_12, (8, 87, 1, 1), (87, 1, 87, 87))
    assert_size_stride(convolution_72, (8, 1044, 1, 1), (1044, 1, 1044, 1044))
    assert_size_stride(clamp_max_15, (8, 1044, 7, 7), (51156, 1, 7308, 1044))
    assert_size_stride(convolution_73, (8, 185, 7, 7), (9065, 1, 1295, 185))
    assert_size_stride(squeeze_181, (185, ), (1, ))
    assert_size_stride(cat_10, (8, 185, 7, 7), (9065, 1, 1295, 185))
    assert_size_stride(convolution_74, (8, 1280, 7, 7), (62720, 1, 8960, 1280))
    assert_size_stride(squeeze_184, (1280, ), (1, ))
    assert_size_stride(clone_17, (8, 1280), (1280, 1))
    assert_size_stride(permute_1, (1000, 1280), (1280, 1))
    assert_size_stride(mul_465, (8, 1280, 7, 7), (62720, 1, 8960, 1280))
    assert_size_stride(unsqueeze_250, (1, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(unsqueeze_262, (1, 185, 1, 1), (185, 1, 1, 1))
    assert_size_stride(unsqueeze_286, (1, 1044, 1, 1), (1044, 1, 1, 1))
    assert_size_stride(mul_508, (8, 1044, 7, 7), (51156, 1, 7308, 1044))
    assert_size_stride(unsqueeze_298, (1, 1044, 1, 1), (1044, 1, 1, 1))
    assert_size_stride(unsqueeze_310, (1, 174, 1, 1), (174, 1, 1, 1))
    assert_size_stride(unsqueeze_334, (1, 972, 1, 1), (972, 1, 1, 1))
    assert_size_stride(mul_551, (8, 972, 7, 7), (47628, 1, 6804, 972))
    assert_size_stride(unsqueeze_346, (1, 972, 1, 1), (972, 1, 1, 1))
    assert_size_stride(unsqueeze_358, (1, 162, 1, 1), (162, 1, 1, 1))
    assert_size_stride(unsqueeze_382, (1, 906, 1, 1), (906, 1, 1, 1))
    assert_size_stride(mul_594, (8, 906, 7, 7), (44394, 1, 6342, 906))
    assert_size_stride(unsqueeze_394, (1, 906, 1, 1), (906, 1, 1, 1))
    assert_size_stride(unsqueeze_406, (1, 151, 1, 1), (151, 1, 1, 1))
    assert_size_stride(unsqueeze_430, (1, 840, 1, 1), (840, 1, 1, 1))
    assert_size_stride(mul_637, (8, 840, 7, 7), (41160, 1, 5880, 840))
    assert_size_stride(unsqueeze_442, (1, 840, 1, 1), (840, 1, 1, 1))
    assert_size_stride(unsqueeze_454, (1, 140, 1, 1), (140, 1, 1, 1))
    assert_size_stride(unsqueeze_478, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(mul_680, (8, 768, 14, 14), (150528, 1, 10752, 768))
    assert_size_stride(unsqueeze_490, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_502, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_526, (1, 702, 1, 1), (702, 1, 1, 1))
    assert_size_stride(mul_723, (8, 702, 14, 14), (137592, 1, 9828, 702))
    assert_size_stride(unsqueeze_538, (1, 702, 1, 1), (702, 1, 1, 1))
    assert_size_stride(unsqueeze_550, (1, 117, 1, 1), (117, 1, 1, 1))
    assert_size_stride(unsqueeze_574, (1, 636, 1, 1), (636, 1, 1, 1))
    assert_size_stride(mul_766, (8, 636, 14, 14), (124656, 1, 8904, 636))
    assert_size_stride(unsqueeze_586, (1, 636, 1, 1), (636, 1, 1, 1))
    assert_size_stride(unsqueeze_598, (1, 106, 1, 1), (106, 1, 1, 1))
    assert_size_stride(unsqueeze_622, (1, 570, 1, 1), (570, 1, 1, 1))
    assert_size_stride(mul_809, (8, 570, 14, 14), (111720, 1, 7980, 570))
    assert_size_stride(unsqueeze_634, (1, 570, 1, 1), (570, 1, 1, 1))
    assert_size_stride(unsqueeze_646, (1, 95, 1, 1), (95, 1, 1, 1))
    assert_size_stride(unsqueeze_670, (1, 504, 1, 1), (504, 1, 1, 1))
    assert_size_stride(mul_852, (8, 504, 14, 14), (98784, 1, 7056, 504))
    assert_size_stride(unsqueeze_682, (1, 504, 1, 1), (504, 1, 1, 1))
    assert_size_stride(unsqueeze_694, (1, 84, 1, 1), (84, 1, 1, 1))
    assert_size_stride(unsqueeze_718, (1, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(mul_895, (8, 432, 14, 14), (84672, 1, 6048, 432))
    assert_size_stride(unsqueeze_730, (1, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(unsqueeze_742, (1, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(unsqueeze_766, (1, 366, 1, 1), (366, 1, 1, 1))
    assert_size_stride(mul_938, (8, 366, 28, 28), (286944, 1, 10248, 366))
    assert_size_stride(unsqueeze_778, (1, 366, 1, 1), (366, 1, 1, 1))
    assert_size_stride(unsqueeze_790, (1, 61, 1, 1), (61, 1, 1, 1))
    assert_size_stride(unsqueeze_814, (1, 300, 1, 1), (300, 1, 1, 1))
    assert_size_stride(mul_981, (8, 300, 28, 28), (235200, 1, 8400, 300))
    assert_size_stride(unsqueeze_826, (1, 300, 1, 1), (300, 1, 1, 1))
    assert_size_stride(unsqueeze_838, (1, 50, 1, 1), (50, 1, 1, 1))
    assert_size_stride(unsqueeze_862, (1, 228, 1, 1), (228, 1, 1, 1))
    assert_size_stride(mul_1024, (8, 228, 56, 56), (715008, 1, 12768, 228))
    assert_size_stride(unsqueeze_874, (1, 228, 1, 1), (228, 1, 1, 1))
    assert_size_stride(unsqueeze_886, (1, 38, 1, 1), (38, 1, 1, 1))
    assert_size_stride(bitwise_or_13, (8, 162, 56, 56), (508032, 1, 9072, 162))
    assert_size_stride(unsqueeze_898, (1, 162, 1, 1), (162, 1, 1, 1))
    assert_size_stride(mul_1054, (8, 162, 56, 56), (508032, 1, 9072, 162))
    assert_size_stride(unsqueeze_910, (1, 162, 1, 1), (162, 1, 1, 1))
    assert_size_stride(unsqueeze_922, (1, 27, 1, 1), (27, 1, 1, 1))
    assert_size_stride(bitwise_or_14, (8, 96, 56, 56), (301056, 1, 5376, 96))
    assert_size_stride(unsqueeze_934, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(mul_1084, (8, 96, 112, 112), (1204224, 1, 10752, 96))
    assert_size_stride(unsqueeze_946, (1, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(unsqueeze_958, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(bitwise_or_15, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(unsqueeze_970, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(mul_1114, (8, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(unsqueeze_982, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf39 = empty((8, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_1, out=buf39)
        del permute_1
        buf42 = empty_strided((1280, 4), (1, 1280), device='cuda', dtype=torch.float32)
        buf44 = empty_strided((1280, 4), (1, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.native_batch_norm_backward]
        stream0 = get_cuda_stream(0)
        triton_red_fused_div_mul_native_batch_norm_backward_0.run(buf39, mul_465, convolution_74, unsqueeze_250, buf42, buf44, 5120, 98, grid=grid(5120), stream=stream0)
        buf43 = empty((1280, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_div_mul_native_batch_norm_backward_1.run(buf42, buf43, 1280, 4, grid=grid(1280), stream=stream0)
        del buf42
        buf45 = empty((1280, ), device='cuda', dtype=torch.float32)
        buf46 = empty((1280, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_div_mul_native_batch_norm_backward_2.run(buf44, squeeze_184, buf45, buf46, 1280, 4, grid=grid(1280), stream=stream0)
        del buf44
        buf47 = empty((8, 1280, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_div_mul_native_batch_norm_backward_3.run(buf39, mul_465, convolution_74, unsqueeze_250, buf45, squeeze_184, buf43, primals_97, buf47, 392, 1280, grid=grid(392, 1280), stream=stream0)
        del buf39
        del buf45
        del convolution_74
        del mul_465
        del primals_97
        del squeeze_184
        del unsqueeze_250
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward]
        buf48 = aten.convolution_backward(buf47, cat_10, primals_225, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf47
        del cat_10
        del primals_225
        buf49 = buf48[0]
        buf51 = empty((185, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]
        triton_per_fused_add_native_batch_norm_backward_slice_backward_4.run(buf49, buf51, 185, 392, grid=grid(185), stream=stream0)
        buf52 = empty_strided((185, 4), (1, 185), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]
        triton_red_fused_add_native_batch_norm_backward_slice_backward_5.run(buf49, convolution_73, unsqueeze_262, buf52, 740, 98, grid=grid(740), stream=stream0)
        buf53 = empty((185, ), device='cuda', dtype=torch.float32)
        buf54 = empty((185, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]
        triton_per_fused_add_native_batch_norm_backward_slice_backward_6.run(buf52, squeeze_181, buf53, buf54, 185, 4, grid=grid(185), stream=stream0)
        del buf52
        buf55 = empty((8, 185, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_7.run(buf49, convolution_73, unsqueeze_262, buf53, squeeze_181, buf51, primals_95, buf55, 1480, 49, grid=grid(1480, 49), stream=stream0)
        del buf53
        del convolution_73
        del primals_95
        del squeeze_181
        del unsqueeze_262
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]
        buf56 = aten.convolution_backward(buf55, clamp_max_15, primals_224, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf55
        del clamp_max_15
        del primals_224
        buf57 = buf56[0]
        buf59 = empty_strided((8, 1044, 1, 1), (1044, 1, 8352, 8352), device='cuda', dtype=torch.float32)
        buf60 = reinterpret_tensor(buf59, (8, 1044, 1, 1), (1044, 1, 1, 1), 0); del buf59  # reuse
        # Source Nodes: [sigmoid_12, x_319], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_8.run(buf60, add_304, convolution_72, buf57, 8352, 49, grid=grid(8352), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf62 = aten.convolution_backward(buf60, relu_12, primals_222, [1044], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_222
        buf63 = buf62[0]
        buf36 = empty((1, 87, 1, 1), device='cuda', dtype=torch.float32)
        buf65 = empty((87, ), device='cuda', dtype=torch.float32)
        buf66 = empty((87, ), device='cuda', dtype=torch.float32)
        buf37 = empty_strided((1, 87, 1, 1), (87, 1, 87, 87), device='cuda', dtype=torch.float32)
        buf68 = empty((87, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___15___se_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_9.run(convolution_71, relu_12, buf63, buf36, buf65, buf66, buf37, buf68, 87, 8, grid=grid(87), stream=stream0)
        buf67 = buf63; del buf63  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_10.run(buf67, relu_12, convolution_71, buf36, buf66, buf37, buf65, primals_220, 696, grid=grid(696), stream=stream0)
        del buf36
        del buf37
        del convolution_71
        del primals_220
        del relu_12
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf70 = aten.convolution_backward(buf67, mean_12, primals_218, [87], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_12
        del primals_218
        buf71 = buf70[0]
        buf73 = empty_strided((1044, 4), (1, 1044), device='cuda', dtype=torch.float32)
        buf75 = empty_strided((1044, 4), (1, 1044), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_12, x_319], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_11.run(add_304, convolution_72, buf57, buf71, convolution_70, unsqueeze_286, buf73, buf75, 4176, 98, grid=grid(4176), stream=stream0)
        buf74 = empty((1044, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_12, x_319], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_12.run(buf73, buf74, 1044, 4, grid=grid(1044), stream=stream0)
        buf76 = empty((1044, ), device='cuda', dtype=torch.float32)
        buf78 = empty((1044, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_12, x_319], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_13.run(buf75, squeeze_175, buf76, buf78, 1044, 4, grid=grid(1044), stream=stream0)
        buf77 = empty_strided((8, 1044, 7, 7), (51156, 1, 7308, 1044), device='cuda', dtype=torch.float32)
        buf79 = buf77; del buf77  # reuse
        # Source Nodes: [sigmoid_12, x_319], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_14.run(buf79, add_304, convolution_72, buf57, buf71, convolution_70, unsqueeze_286, buf76, squeeze_175, buf74, primals_93, 392, 1044, grid=grid(392, 1044), stream=stream0)
        del add_304
        del buf57
        del buf71
        del convolution_70
        del convolution_72
        del primals_93
        del squeeze_175
        del unsqueeze_286
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf80 = aten.convolution_backward(buf79, mul_433, primals_217, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1044, [True, True, False])
        del buf79
        del mul_433
        del primals_217
        buf81 = buf80[0]
        buf83 = buf75; del buf75  # reuse
        buf85 = buf73; del buf73  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_15.run(buf81, mul_508, convolution_69, unsqueeze_298, buf83, buf85, 4176, 98, grid=grid(4176), stream=stream0)
        buf84 = buf76; del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_12.run(buf83, buf84, 1044, 4, grid=grid(1044), stream=stream0)
        del buf83
        buf86 = empty((1044, ), device='cuda', dtype=torch.float32)
        buf87 = empty((1044, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_13.run(buf85, squeeze_172, buf86, buf87, 1044, 4, grid=grid(1044), stream=stream0)
        del buf85
        buf88 = buf81; del buf81  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_16.run(buf88, mul_508, convolution_69, unsqueeze_298, buf86, squeeze_172, buf84, primals_91, 8352, 49, grid=grid(8352, 49), stream=stream0)
        del convolution_69
        del mul_508
        del primals_91
        del squeeze_172
        del unsqueeze_298
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf89 = aten.convolution_backward(buf88, cat_9, primals_216, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf88
        del cat_9
        del primals_216
        buf90 = buf89[0]
        buf92 = empty((174, ), device='cuda', dtype=torch.float32)
        buf93 = empty((174, ), device='cuda', dtype=torch.float32)
        buf94 = empty((174, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]
        triton_per_fused_add_native_batch_norm_backward_slice_backward_17.run(buf49, buf90, convolution_68, unsqueeze_310, squeeze_169, buf92, buf93, buf94, 174, 392, grid=grid(174), stream=stream0)
        buf95 = empty((8, 174, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_18.run(buf49, buf90, convolution_68, unsqueeze_310, buf93, squeeze_169, buf92, primals_89, buf95, 1392, 49, grid=grid(1392, 49), stream=stream0)
        del buf93
        del convolution_68
        del primals_89
        del squeeze_169
        del unsqueeze_310
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]
        buf96 = aten.convolution_backward(buf95, clamp_max_14, primals_215, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf95
        del clamp_max_14
        del primals_215
        buf97 = buf96[0]
        buf99 = empty_strided((8, 972, 1, 1), (972, 1, 7776, 7776), device='cuda', dtype=torch.float32)
        buf100 = reinterpret_tensor(buf99, (8, 972, 1, 1), (972, 1, 1, 1), 0); del buf99  # reuse
        # Source Nodes: [sigmoid_11, x_298], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_19.run(buf100, add_283, convolution_67, buf97, 7776, 49, grid=grid(7776), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf102 = aten.convolution_backward(buf100, relu_11, primals_213, [972], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_213
        buf103 = buf102[0]
        buf34 = empty_strided((1, 81, 1, 1), (81, 1, 81, 81), device='cuda', dtype=torch.float32)
        buf33 = empty((1, 81, 1, 1), device='cuda', dtype=torch.float32)
        buf105 = empty((81, ), device='cuda', dtype=torch.float32)
        buf106 = empty((81, ), device='cuda', dtype=torch.float32)
        buf108 = empty((81, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___14___se_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_20.run(convolution_66, relu_11, buf103, buf34, buf33, buf105, buf106, buf108, 81, 8, grid=grid(81), stream=stream0)
        buf107 = buf103; del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_21.run(buf107, relu_11, convolution_66, buf33, buf106, buf34, buf105, primals_211, 648, grid=grid(648), stream=stream0)
        del buf106
        del buf33
        del convolution_66
        del primals_211
        del relu_11
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf110 = aten.convolution_backward(buf107, mean_11, primals_209, [81], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_11
        del primals_209
        buf111 = buf110[0]
        buf113 = empty_strided((972, 4), (1, 972), device='cuda', dtype=torch.float32)
        buf115 = empty_strided((972, 4), (1, 972), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_11, x_298], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_22.run(add_283, convolution_67, buf97, buf111, convolution_65, unsqueeze_334, buf113, buf115, 3888, 98, grid=grid(3888), stream=stream0)
        buf114 = empty((972, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_11, x_298], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_23.run(buf113, buf114, 972, 4, grid=grid(972), stream=stream0)
        buf116 = empty((972, ), device='cuda', dtype=torch.float32)
        buf118 = empty((972, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_11, x_298], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_24.run(buf115, squeeze_163, buf116, buf118, 972, 4, grid=grid(972), stream=stream0)
        buf117 = empty_strided((8, 972, 7, 7), (47628, 1, 6804, 972), device='cuda', dtype=torch.float32)
        buf119 = buf117; del buf117  # reuse
        # Source Nodes: [sigmoid_11, x_298], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_25.run(buf119, add_283, convolution_67, buf97, buf111, convolution_65, unsqueeze_334, buf116, squeeze_163, buf114, primals_87, 392, 972, grid=grid(392, 972), stream=stream0)
        del add_283
        del buf111
        del buf97
        del convolution_65
        del convolution_67
        del primals_87
        del squeeze_163
        del unsqueeze_334
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf120 = aten.convolution_backward(buf119, mul_403, primals_208, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 972, [True, True, False])
        del buf119
        del mul_403
        del primals_208
        buf121 = buf120[0]
        buf123 = buf115; del buf115  # reuse
        buf125 = buf113; del buf113  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_26.run(buf121, mul_551, convolution_64, unsqueeze_346, buf123, buf125, 3888, 98, grid=grid(3888), stream=stream0)
        buf124 = buf116; del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_23.run(buf123, buf124, 972, 4, grid=grid(972), stream=stream0)
        del buf123
        buf126 = empty((972, ), device='cuda', dtype=torch.float32)
        buf127 = empty((972, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_24.run(buf125, squeeze_160, buf126, buf127, 972, 4, grid=grid(972), stream=stream0)
        del buf125
        buf128 = buf121; del buf121  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_27.run(buf128, mul_551, convolution_64, unsqueeze_346, buf126, squeeze_160, buf124, primals_85, 7776, 49, grid=grid(7776, 49), stream=stream0)
        del convolution_64
        del mul_551
        del primals_85
        del squeeze_160
        del unsqueeze_346
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf129 = aten.convolution_backward(buf128, cat_8, primals_207, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf128
        del cat_8
        del primals_207
        buf130 = buf129[0]
        buf132 = empty((162, ), device='cuda', dtype=torch.float32)
        buf133 = empty((162, ), device='cuda', dtype=torch.float32)
        buf135 = empty((162, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]
        triton_per_fused_add_native_batch_norm_backward_slice_backward_28.run(buf49, buf90, buf130, convolution_63, unsqueeze_358, squeeze_157, buf132, buf133, buf135, 162, 392, grid=grid(162), stream=stream0)
        buf134 = empty((8, 162, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]
        triton_poi_fused_add_native_batch_norm_backward_slice_backward_29.run(buf49, buf90, buf130, convolution_63, unsqueeze_358, buf133, squeeze_157, buf132, primals_83, buf134, 1296, 49, grid=grid(1296, 49), stream=stream0)
        del convolution_63
        del primals_83
        del squeeze_157
        del unsqueeze_358
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf136 = aten.convolution_backward(buf134, clamp_max_13, primals_206, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf134
        del clamp_max_13
        del primals_206
        buf137 = buf136[0]
        buf139 = empty_strided((8, 906, 1, 1), (906, 1, 7248, 7248), device='cuda', dtype=torch.float32)
        buf140 = reinterpret_tensor(buf139, (8, 906, 1, 1), (906, 1, 1, 1), 0); del buf139  # reuse
        # Source Nodes: [sigmoid_10, x_277], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_30.run(buf140, add_262, convolution_62, buf137, 7248, 49, grid=grid(7248), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf142 = aten.convolution_backward(buf140, relu_10, primals_204, [906], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_204
        buf143 = buf142[0]
        buf31 = empty_strided((1, 75, 1, 1), (75, 1, 75, 75), device='cuda', dtype=torch.float32)
        buf30 = empty((1, 75, 1, 1), device='cuda', dtype=torch.float32)
        buf145 = empty((75, ), device='cuda', dtype=torch.float32)
        buf146 = empty((75, ), device='cuda', dtype=torch.float32)
        buf148 = empty((75, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___13___se_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_31.run(convolution_61, relu_10, buf143, buf31, buf30, buf145, buf146, buf148, 75, 8, grid=grid(75), stream=stream0)
        buf147 = buf143; del buf143  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_32.run(buf147, relu_10, convolution_61, buf30, buf146, buf31, buf145, primals_202, 600, grid=grid(600), stream=stream0)
        del buf146
        del buf30
        del convolution_61
        del primals_202
        del relu_10
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf150 = aten.convolution_backward(buf147, mean_10, primals_200, [75], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_10
        del primals_200
        buf151 = buf150[0]
        buf153 = empty_strided((906, 4), (1, 906), device='cuda', dtype=torch.float32)
        buf155 = empty_strided((906, 4), (1, 906), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_10, x_277], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_33.run(add_262, convolution_62, buf137, buf151, convolution_60, unsqueeze_382, buf153, buf155, 3624, 98, grid=grid(3624), stream=stream0)
        buf154 = empty((906, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_10, x_277], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_34.run(buf153, buf154, 906, 4, grid=grid(906), stream=stream0)
        buf156 = empty((906, ), device='cuda', dtype=torch.float32)
        buf158 = empty((906, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_10, x_277], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_35.run(buf155, squeeze_151, buf156, buf158, 906, 4, grid=grid(906), stream=stream0)
        buf157 = empty_strided((8, 906, 7, 7), (44394, 1, 6342, 906), device='cuda', dtype=torch.float32)
        buf159 = buf157; del buf157  # reuse
        # Source Nodes: [sigmoid_10, x_277], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_36.run(buf159, add_262, convolution_62, buf137, buf151, convolution_60, unsqueeze_382, buf156, squeeze_151, buf154, primals_81, 392, 906, grid=grid(392, 906), stream=stream0)
        del add_262
        del buf137
        del buf151
        del convolution_60
        del convolution_62
        del primals_81
        del squeeze_151
        del unsqueeze_382
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf160 = aten.convolution_backward(buf159, mul_373, primals_199, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 906, [True, True, False])
        del buf159
        del mul_373
        del primals_199
        buf161 = buf160[0]
        buf163 = buf155; del buf155  # reuse
        buf165 = buf153; del buf153  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_37.run(buf161, mul_594, convolution_59, unsqueeze_394, buf163, buf165, 3624, 98, grid=grid(3624), stream=stream0)
        buf164 = buf156; del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_34.run(buf163, buf164, 906, 4, grid=grid(906), stream=stream0)
        del buf163
        buf166 = empty((906, ), device='cuda', dtype=torch.float32)
        buf167 = empty((906, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_35.run(buf165, squeeze_148, buf166, buf167, 906, 4, grid=grid(906), stream=stream0)
        del buf165
        buf168 = buf161; del buf161  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_38.run(buf168, mul_594, convolution_59, unsqueeze_394, buf166, squeeze_148, buf164, primals_79, 7248, 49, grid=grid(7248, 49), stream=stream0)
        del convolution_59
        del mul_594
        del primals_79
        del squeeze_148
        del unsqueeze_394
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf169 = aten.convolution_backward(buf168, cat_7, primals_198, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf168
        del cat_7
        del primals_198
        buf170 = buf169[0]
        buf172 = empty((151, ), device='cuda', dtype=torch.float32)
        buf173 = empty((151, ), device='cuda', dtype=torch.float32)
        buf175 = empty((151, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]
        triton_per_fused_add_native_batch_norm_backward_slice_backward_39.run(buf49, buf90, buf130, buf170, convolution_58, unsqueeze_406, squeeze_145, buf172, buf173, buf175, 151, 392, grid=grid(151), stream=stream0)
        buf174 = empty((8, 151, 7, 7), device='cuda', dtype=torch.float32)
        buf176 = buf174; del buf174  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_40.run(buf176, buf49, buf90, buf130, buf170, convolution_58, unsqueeze_406, buf173, squeeze_145, buf172, primals_77, 1208, 49, grid=grid(1208, 49), stream=stream0)
        del buf173
        del convolution_58
        del primals_77
        del squeeze_145
        del unsqueeze_406
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf177 = aten.convolution_backward(buf176, clamp_max_12, primals_197, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf176
        del clamp_max_12
        del primals_197
        buf178 = buf177[0]
        buf180 = empty_strided((8, 840, 1, 1), (840, 1, 6720, 6720), device='cuda', dtype=torch.float32)
        buf181 = reinterpret_tensor(buf180, (8, 840, 1, 1), (840, 1, 1, 1), 0); del buf180  # reuse
        # Source Nodes: [sigmoid_9, x_256], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_41.run(buf181, add_241, convolution_57, buf178, 6720, 49, grid=grid(6720), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf183 = aten.convolution_backward(buf181, relu_9, primals_195, [840], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_195
        buf184 = buf183[0]
        buf28 = empty_strided((1, 70, 1, 1), (70, 1, 70, 70), device='cuda', dtype=torch.float32)
        buf27 = empty((1, 70, 1, 1), device='cuda', dtype=torch.float32)
        buf186 = empty((70, ), device='cuda', dtype=torch.float32)
        buf187 = empty((70, ), device='cuda', dtype=torch.float32)
        buf189 = empty((70, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___12___se_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_42.run(convolution_56, relu_9, buf184, buf28, buf27, buf186, buf187, buf189, 70, 8, grid=grid(70), stream=stream0)
        buf188 = buf184; del buf184  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_43.run(buf188, relu_9, convolution_56, buf27, buf187, buf28, buf186, primals_193, 560, grid=grid(560), stream=stream0)
        del buf187
        del buf27
        del convolution_56
        del primals_193
        del relu_9
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf191 = aten.convolution_backward(buf188, mean_9, primals_191, [70], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_9
        del primals_191
        buf192 = buf191[0]
        buf194 = empty_strided((840, 4), (1, 840), device='cuda', dtype=torch.float32)
        buf196 = empty_strided((840, 4), (1, 840), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_9, x_256], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_44.run(add_241, convolution_57, buf178, buf192, convolution_55, unsqueeze_430, buf194, buf196, 3360, 98, grid=grid(3360), stream=stream0)
        buf195 = empty((840, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_9, x_256], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_45.run(buf194, buf195, 840, 4, grid=grid(840), stream=stream0)
        buf197 = empty((840, ), device='cuda', dtype=torch.float32)
        buf199 = empty((840, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_9, x_256], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_46.run(buf196, squeeze_139, buf197, buf199, 840, 4, grid=grid(840), stream=stream0)
        buf198 = empty_strided((8, 840, 7, 7), (41160, 1, 5880, 840), device='cuda', dtype=torch.float32)
        buf200 = buf198; del buf198  # reuse
        # Source Nodes: [sigmoid_9, x_256], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_47.run(buf200, add_241, convolution_57, buf178, buf192, convolution_55, unsqueeze_430, buf197, squeeze_139, buf195, primals_75, 392, 840, grid=grid(392, 840), stream=stream0)
        del add_241
        del buf178
        del buf192
        del convolution_55
        del convolution_57
        del primals_75
        del squeeze_139
        del unsqueeze_430
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf201 = aten.convolution_backward(buf200, mul_343, primals_190, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 840, [True, True, False])
        del buf200
        del mul_343
        del primals_190
        buf202 = buf201[0]
        buf204 = buf196; del buf196  # reuse
        buf206 = buf194; del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_48.run(buf202, mul_637, convolution_54, unsqueeze_442, buf204, buf206, 3360, 98, grid=grid(3360), stream=stream0)
        buf205 = buf197; del buf197  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_45.run(buf204, buf205, 840, 4, grid=grid(840), stream=stream0)
        del buf204
        buf207 = empty((840, ), device='cuda', dtype=torch.float32)
        buf208 = empty((840, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_46.run(buf206, squeeze_136, buf207, buf208, 840, 4, grid=grid(840), stream=stream0)
        del buf206
        buf209 = buf202; del buf202  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_49.run(buf209, mul_637, convolution_54, unsqueeze_442, buf207, squeeze_136, buf205, primals_73, 6720, 49, grid=grid(6720, 49), stream=stream0)
        del convolution_54
        del mul_637
        del primals_73
        del squeeze_136
        del unsqueeze_442
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf210 = aten.convolution_backward(buf209, add_231, primals_189, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_231
        del buf209
        del primals_189
        buf211 = buf210[0]
        buf213 = buf211; del buf211  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_50.run(buf213, buf49, buf90, buf130, buf170, 54880, grid=grid(54880), stream=stream0)
        del buf130
        del buf170
        del buf49
        del buf90
        buf214 = empty((140, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_51.run(buf213, buf214, 140, 392, grid=grid(140), stream=stream0)
        buf215 = empty_strided((140, 4), (1, 140), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_52.run(buf213, convolution_53, unsqueeze_454, buf215, 560, 98, grid=grid(560), stream=stream0)
        buf216 = empty((140, ), device='cuda', dtype=torch.float32)
        buf217 = empty((140, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_53.run(buf215, squeeze_133, buf216, buf217, 140, 4, grid=grid(140), stream=stream0)
        del buf215
        buf218 = buf213; del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_54.run(buf218, convolution_53, unsqueeze_454, buf216, squeeze_133, buf214, primals_71, 1120, 49, grid=grid(1120, 49), stream=stream0)
        del buf216
        del convolution_53
        del primals_71
        del squeeze_133
        del unsqueeze_454
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf219 = aten.convolution_backward(buf218, clamp_max_11, primals_188, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf218
        del clamp_max_11
        del primals_188
        buf220 = buf219[0]
        buf222 = empty_strided((8, 768, 1, 1), (768, 1, 6144, 6144), device='cuda', dtype=torch.float32)
        buf223 = reinterpret_tensor(buf222, (8, 768, 1, 1), (768, 1, 1, 1), 0); del buf222  # reuse
        # Source Nodes: [sigmoid_8, x_236], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_55.run(buf223, add_221, convolution_52, buf220, 6144, 49, grid=grid(6144), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf225 = aten.convolution_backward(buf223, relu_8, primals_186, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_186
        buf226 = buf225[0]
        buf25 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf24 = empty((1, 64, 1, 1), device='cuda', dtype=torch.float32)
        buf228 = empty((64, ), device='cuda', dtype=torch.float32)
        buf229 = empty((64, ), device='cuda', dtype=torch.float32)
        buf231 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___11___se_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_56.run(convolution_51, relu_8, buf226, buf25, buf24, buf228, buf229, buf231, 64, 8, grid=grid(64), stream=stream0)
        buf230 = buf226; del buf226  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_57.run(buf230, relu_8, convolution_51, buf24, buf229, buf25, buf228, primals_184, 512, grid=grid(512), stream=stream0)
        del buf229
        del buf24
        del convolution_51
        del primals_184
        del relu_8
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf233 = aten.convolution_backward(buf230, mean_8, primals_182, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_8
        del primals_182
        buf234 = buf233[0]
        buf236 = empty_strided((768, 4), (1, 768), device='cuda', dtype=torch.float32)
        buf238 = empty_strided((768, 4), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_8, x_236], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_58.run(add_221, convolution_52, buf220, buf234, convolution_50, unsqueeze_478, buf236, buf238, 3072, 98, grid=grid(3072), stream=stream0)
        buf237 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_8, x_236], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_59.run(buf236, buf237, 768, 4, grid=grid(768), stream=stream0)
        del buf236
        buf239 = empty((768, ), device='cuda', dtype=torch.float32)
        buf241 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_8, x_236], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_60.run(buf238, squeeze_127, buf239, buf241, 768, 4, grid=grid(768), stream=stream0)
        del buf238
        buf240 = empty_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda', dtype=torch.float32)
        buf242 = buf240; del buf240  # reuse
        # Source Nodes: [sigmoid_8, x_236], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_61.run(buf242, add_221, convolution_52, buf220, buf234, convolution_50, unsqueeze_478, buf239, squeeze_127, buf237, primals_69, 392, 768, grid=grid(392, 768), stream=stream0)
        del add_221
        del buf220
        del buf234
        del convolution_50
        del convolution_52
        del primals_69
        del squeeze_127
        del unsqueeze_478
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf243 = aten.convolution_backward(buf242, mul_313, primals_181, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False])
        del buf242
        del mul_313
        del primals_181
        buf244 = buf243[0]
        buf246 = empty((768, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_62.run(buf244, mul_680, buf246, 9984, 121, grid=grid(9984), stream=stream0)
        buf247 = buf239; del buf239  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_63.run(buf246, buf247, 768, 13, grid=grid(768), stream=stream0)
        buf248 = reinterpret_tensor(buf246, (768, 13), (1, 768), 0); del buf246  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_64.run(buf244, mul_680, convolution_49, unsqueeze_490, buf248, 9984, 121, grid=grid(9984), stream=stream0)
        buf249 = empty((768, ), device='cuda', dtype=torch.float32)
        buf250 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_65.run(buf248, squeeze_124, buf249, buf250, 768, 13, grid=grid(768), stream=stream0)
        del buf248
        buf251 = buf244; del buf244  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_66.run(buf251, mul_680, convolution_49, unsqueeze_490, buf249, squeeze_124, buf247, primals_67, 6144, 196, grid=grid(6144, 196), stream=stream0)
        del convolution_49
        del mul_680
        del primals_67
        del squeeze_124
        del unsqueeze_490
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf252 = aten.convolution_backward(buf251, cat_6, primals_180, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf251
        del cat_6
        del primals_180
        buf253 = buf252[0]
        buf255 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]
        triton_red_fused_add_native_batch_norm_backward_slice_backward_67.run(buf253, buf255, 128, 1568, grid=grid(128), stream=stream0)
        buf256 = empty((128, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]
        triton_red_fused_add_native_batch_norm_backward_slice_backward_68.run(buf253, convolution_48, unsqueeze_502, buf256, 1664, 121, grid=grid(1664), stream=stream0)
        buf257 = empty((128, ), device='cuda', dtype=torch.float32)
        buf258 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]
        triton_per_fused_add_native_batch_norm_backward_slice_backward_69.run(buf256, squeeze_121, buf257, buf258, 128, 13, grid=grid(128), stream=stream0)
        del buf256
        buf259 = empty((8, 128, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_70.run(buf253, convolution_48, unsqueeze_502, buf257, squeeze_121, buf255, primals_65, buf259, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del buf257
        del convolution_48
        del primals_65
        del squeeze_121
        del unsqueeze_502
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]
        buf260 = aten.convolution_backward(buf259, clamp_max_10, primals_179, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf259
        del clamp_max_10
        del primals_179
        buf261 = buf260[0]
        buf263 = empty_strided((8, 702, 1, 1, 2), (1404, 2, 11232, 11232, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_7, x_215], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sum]
        triton_red_fused_hardtanh_backward_mul_sigmoid_sum_71.run(add_200, convolution_47, buf261, buf263, 11232, 98, grid=grid(11232), stream=stream0)
        buf264 = empty_strided((8, 702, 1, 1), (702, 1, 5616, 5616), device='cuda', dtype=torch.float32)
        buf265 = reinterpret_tensor(buf264, (8, 702, 1, 1), (702, 1, 1, 1), 0); del buf264  # reuse
        # Source Nodes: [sigmoid_7, x_215], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_72.run(buf265, buf263, convolution_47, 5616, 2, grid=grid(5616), stream=stream0)
        del buf263
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf267 = aten.convolution_backward(buf265, relu_7, primals_177, [702], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_177
        buf268 = buf267[0]
        buf22 = empty_strided((1, 58, 1, 1), (58, 1, 58, 58), device='cuda', dtype=torch.float32)
        buf21 = empty((1, 58, 1, 1), device='cuda', dtype=torch.float32)
        buf270 = empty((58, ), device='cuda', dtype=torch.float32)
        buf271 = empty((58, ), device='cuda', dtype=torch.float32)
        buf273 = empty((58, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___10___se_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_73.run(convolution_46, relu_7, buf268, buf22, buf21, buf270, buf271, buf273, 58, 8, grid=grid(58), stream=stream0)
        buf272 = buf268; del buf268  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_74.run(buf272, relu_7, convolution_46, buf21, buf271, buf22, buf270, primals_175, 464, grid=grid(464), stream=stream0)
        del buf21
        del buf22
        del convolution_46
        del primals_175
        del relu_7
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf275 = aten.convolution_backward(buf272, mean_7, primals_173, [58], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_7
        del primals_173
        buf276 = buf275[0]
        buf278 = empty((702, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_7, x_215], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_75.run(add_200, convolution_47, buf261, buf276, buf278, 9126, 121, grid=grid(9126), stream=stream0)
        buf279 = empty((702, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_7, x_215], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_76.run(buf278, buf279, 702, 13, grid=grid(702), stream=stream0)
        buf280 = reinterpret_tensor(buf278, (702, 13), (1, 702), 0); del buf278  # reuse
        # Source Nodes: [sigmoid_7, x_215], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_77.run(add_200, convolution_47, buf261, buf276, convolution_45, unsqueeze_526, buf280, 9126, 121, grid=grid(9126), stream=stream0)
        buf281 = empty((702, ), device='cuda', dtype=torch.float32)
        buf283 = empty((702, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_7, x_215], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_78.run(buf280, squeeze_115, buf281, buf283, 702, 13, grid=grid(702), stream=stream0)
        buf282 = empty_strided((8, 702, 14, 14), (137592, 1, 9828, 702), device='cuda', dtype=torch.float32)
        buf284 = buf282; del buf282  # reuse
        # Source Nodes: [sigmoid_7, x_215], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_79.run(buf284, add_200, convolution_47, buf261, buf276, convolution_45, unsqueeze_526, buf281, squeeze_115, buf279, primals_63, 1568, 702, grid=grid(1568, 702), stream=stream0)
        del add_200
        del buf261
        del convolution_45
        del convolution_47
        del primals_63
        del squeeze_115
        del unsqueeze_526
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf285 = aten.convolution_backward(buf284, mul_283, primals_172, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 702, [True, True, False])
        del buf284
        del mul_283
        del primals_172
        buf286 = buf285[0]
        buf288 = reinterpret_tensor(buf280, (702, 13), (13, 1), 0); del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_80.run(buf286, mul_723, buf288, 9126, 121, grid=grid(9126), stream=stream0)
        buf289 = buf281; del buf281  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_76.run(buf288, buf289, 702, 13, grid=grid(702), stream=stream0)
        buf290 = reinterpret_tensor(buf288, (702, 13), (1, 702), 0); del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_81.run(buf286, mul_723, convolution_44, unsqueeze_538, buf290, 9126, 121, grid=grid(9126), stream=stream0)
        buf291 = empty((702, ), device='cuda', dtype=torch.float32)
        buf292 = empty((702, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_78.run(buf290, squeeze_112, buf291, buf292, 702, 13, grid=grid(702), stream=stream0)
        del buf290
        buf293 = buf286; del buf286  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_82.run(buf293, mul_723, convolution_44, unsqueeze_538, buf291, squeeze_112, buf289, primals_61, 5616, 196, grid=grid(5616, 196), stream=stream0)
        del convolution_44
        del mul_723
        del primals_61
        del squeeze_112
        del unsqueeze_538
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf294 = aten.convolution_backward(buf293, cat_5, primals_171, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf293
        del cat_5
        del primals_171
        buf295 = buf294[0]
        buf297 = empty((117, ), device='cuda', dtype=torch.float32)
        buf298 = empty((117, ), device='cuda', dtype=torch.float32)
        buf299 = empty((117, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]
        triton_red_fused_add_native_batch_norm_backward_slice_backward_83.run(buf253, buf295, convolution_43, unsqueeze_550, squeeze_109, buf297, buf298, buf299, 117, 1568, grid=grid(117), stream=stream0)
        buf300 = empty((8, 117, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_84.run(buf253, buf295, convolution_43, unsqueeze_550, buf298, squeeze_109, buf297, primals_59, buf300, 936, 196, grid=grid(936, 196), stream=stream0)
        del buf298
        del convolution_43
        del primals_59
        del squeeze_109
        del unsqueeze_550
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]
        buf301 = aten.convolution_backward(buf300, clamp_max_9, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf300
        del clamp_max_9
        del primals_170
        buf302 = buf301[0]
        buf304 = empty_strided((8, 636, 1, 1, 2), (1272, 2, 10176, 10176, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_6, x_194], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sum]
        triton_red_fused_hardtanh_backward_mul_sigmoid_sum_85.run(add_179, convolution_42, buf302, buf304, 10176, 98, grid=grid(10176), stream=stream0)
        buf305 = empty_strided((8, 636, 1, 1), (636, 1, 5088, 5088), device='cuda', dtype=torch.float32)
        buf306 = reinterpret_tensor(buf305, (8, 636, 1, 1), (636, 1, 1, 1), 0); del buf305  # reuse
        # Source Nodes: [sigmoid_6, x_194], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_86.run(buf306, buf304, convolution_42, 5088, 2, grid=grid(5088), stream=stream0)
        del buf304
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf308 = aten.convolution_backward(buf306, relu_6, primals_168, [636], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_168
        buf309 = buf308[0]
        buf19 = empty_strided((1, 53, 1, 1), (53, 1, 53, 53), device='cuda', dtype=torch.float32)
        buf18 = empty((1, 53, 1, 1), device='cuda', dtype=torch.float32)
        buf311 = empty((53, ), device='cuda', dtype=torch.float32)
        buf312 = empty((53, ), device='cuda', dtype=torch.float32)
        buf314 = empty((53, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___9___se_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_87.run(convolution_41, relu_6, buf309, buf19, buf18, buf311, buf312, buf314, 53, 8, grid=grid(53), stream=stream0)
        buf313 = buf309; del buf309  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_88.run(buf313, relu_6, convolution_41, buf18, buf312, buf19, buf311, primals_166, 424, grid=grid(424), stream=stream0)
        del buf18
        del buf19
        del convolution_41
        del primals_166
        del relu_6
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf316 = aten.convolution_backward(buf313, mean_6, primals_164, [53], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_6
        del primals_164
        buf317 = buf316[0]
        buf319 = empty((636, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_6, x_194], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_89.run(add_179, convolution_42, buf302, buf317, buf319, 8268, 121, grid=grid(8268), stream=stream0)
        buf320 = empty((636, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_6, x_194], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_90.run(buf319, buf320, 636, 13, grid=grid(636), stream=stream0)
        buf321 = reinterpret_tensor(buf319, (636, 13), (1, 636), 0); del buf319  # reuse
        # Source Nodes: [sigmoid_6, x_194], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_91.run(add_179, convolution_42, buf302, buf317, convolution_40, unsqueeze_574, buf321, 8268, 121, grid=grid(8268), stream=stream0)
        buf322 = empty((636, ), device='cuda', dtype=torch.float32)
        buf324 = empty((636, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_6, x_194], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_92.run(buf321, squeeze_103, buf322, buf324, 636, 13, grid=grid(636), stream=stream0)
        buf323 = empty_strided((8, 636, 14, 14), (124656, 1, 8904, 636), device='cuda', dtype=torch.float32)
        buf325 = buf323; del buf323  # reuse
        # Source Nodes: [sigmoid_6, x_194], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_93.run(buf325, add_179, convolution_42, buf302, buf317, convolution_40, unsqueeze_574, buf322, squeeze_103, buf320, primals_57, 1568, 636, grid=grid(1568, 636), stream=stream0)
        del add_179
        del buf302
        del buf317
        del convolution_40
        del convolution_42
        del primals_57
        del squeeze_103
        del unsqueeze_574
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf326 = aten.convolution_backward(buf325, mul_253, primals_163, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 636, [True, True, False])
        del buf325
        del mul_253
        del primals_163
        buf327 = buf326[0]
        buf329 = reinterpret_tensor(buf321, (636, 13), (13, 1), 0); del buf321  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_94.run(buf327, mul_766, buf329, 8268, 121, grid=grid(8268), stream=stream0)
        buf330 = buf322; del buf322  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_90.run(buf329, buf330, 636, 13, grid=grid(636), stream=stream0)
        buf331 = reinterpret_tensor(buf329, (636, 13), (1, 636), 0); del buf329  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_95.run(buf327, mul_766, convolution_39, unsqueeze_586, buf331, 8268, 121, grid=grid(8268), stream=stream0)
        buf332 = empty((636, ), device='cuda', dtype=torch.float32)
        buf333 = empty((636, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_92.run(buf331, squeeze_100, buf332, buf333, 636, 13, grid=grid(636), stream=stream0)
        del buf331
        buf334 = buf327; del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_96.run(buf334, mul_766, convolution_39, unsqueeze_586, buf332, squeeze_100, buf330, primals_55, 5088, 196, grid=grid(5088, 196), stream=stream0)
        del convolution_39
        del mul_766
        del primals_55
        del squeeze_100
        del unsqueeze_586
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf335 = aten.convolution_backward(buf334, cat_4, primals_162, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf334
        del cat_4
        del primals_162
        buf336 = buf335[0]
        buf338 = empty((106, ), device='cuda', dtype=torch.float32)
        buf339 = empty((106, ), device='cuda', dtype=torch.float32)
        buf341 = empty((106, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]
        triton_red_fused_add_native_batch_norm_backward_slice_backward_97.run(buf253, buf295, buf336, convolution_38, unsqueeze_598, squeeze_97, buf338, buf339, buf341, 106, 1568, grid=grid(106), stream=stream0)
        buf340 = empty((8, 106, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]
        triton_poi_fused_add_native_batch_norm_backward_slice_backward_98.run(buf253, buf295, buf336, convolution_38, unsqueeze_598, buf339, squeeze_97, buf338, primals_53, buf340, 848, 196, grid=grid(848, 196), stream=stream0)
        del buf339
        del convolution_38
        del primals_53
        del squeeze_97
        del unsqueeze_598
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf342 = aten.convolution_backward(buf340, clamp_max_8, primals_161, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf340
        del clamp_max_8
        del primals_161
        buf343 = buf342[0]
        buf345 = empty_strided((8, 570, 1, 1, 2), (1140, 2, 9120, 9120, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_5, x_173], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sum]
        triton_red_fused_hardtanh_backward_mul_sigmoid_sum_99.run(add_158, convolution_37, buf343, buf345, 9120, 98, grid=grid(9120), stream=stream0)
        buf346 = empty_strided((8, 570, 1, 1), (570, 1, 4560, 4560), device='cuda', dtype=torch.float32)
        buf347 = reinterpret_tensor(buf346, (8, 570, 1, 1), (570, 1, 1, 1), 0); del buf346  # reuse
        # Source Nodes: [sigmoid_5, x_173], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_100.run(buf347, buf345, convolution_37, 4560, 2, grid=grid(4560), stream=stream0)
        del buf345
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf349 = aten.convolution_backward(buf347, relu_5, primals_159, [570], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_159
        buf350 = buf349[0]
        buf16 = empty_strided((1, 47, 1, 1), (47, 1, 47, 47), device='cuda', dtype=torch.float32)
        buf15 = empty((1, 47, 1, 1), device='cuda', dtype=torch.float32)
        buf352 = empty((47, ), device='cuda', dtype=torch.float32)
        buf353 = empty((47, ), device='cuda', dtype=torch.float32)
        buf355 = empty((47, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___8___se_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_101.run(convolution_36, relu_5, buf350, buf16, buf15, buf352, buf353, buf355, 47, 8, grid=grid(47), stream=stream0)
        buf354 = buf350; del buf350  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_102.run(buf354, relu_5, convolution_36, buf15, buf353, buf16, buf352, primals_157, 376, grid=grid(376), stream=stream0)
        del buf15
        del buf16
        del convolution_36
        del primals_157
        del relu_5
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf357 = aten.convolution_backward(buf354, mean_5, primals_155, [47], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_5
        del primals_155
        buf358 = buf357[0]
        buf360 = empty((570, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_5, x_173], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_103.run(add_158, convolution_37, buf343, buf358, buf360, 7410, 121, grid=grid(7410), stream=stream0)
        buf361 = empty((570, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_5, x_173], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_104.run(buf360, buf361, 570, 13, grid=grid(570), stream=stream0)
        buf362 = reinterpret_tensor(buf360, (570, 13), (1, 570), 0); del buf360  # reuse
        # Source Nodes: [sigmoid_5, x_173], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_105.run(add_158, convolution_37, buf343, buf358, convolution_35, unsqueeze_622, buf362, 7410, 121, grid=grid(7410), stream=stream0)
        buf363 = empty((570, ), device='cuda', dtype=torch.float32)
        buf365 = empty((570, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_5, x_173], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_106.run(buf362, squeeze_91, buf363, buf365, 570, 13, grid=grid(570), stream=stream0)
        buf364 = empty_strided((8, 570, 14, 14), (111720, 1, 7980, 570), device='cuda', dtype=torch.float32)
        buf366 = buf364; del buf364  # reuse
        # Source Nodes: [sigmoid_5, x_173], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_107.run(buf366, add_158, convolution_37, buf343, buf358, convolution_35, unsqueeze_622, buf363, squeeze_91, buf361, primals_51, 1568, 570, grid=grid(1568, 570), stream=stream0)
        del add_158
        del buf343
        del buf358
        del convolution_35
        del convolution_37
        del primals_51
        del squeeze_91
        del unsqueeze_622
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf367 = aten.convolution_backward(buf366, mul_223, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 570, [True, True, False])
        del buf366
        del mul_223
        del primals_154
        buf368 = buf367[0]
        buf370 = reinterpret_tensor(buf362, (570, 13), (13, 1), 0); del buf362  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_108.run(buf368, mul_809, buf370, 7410, 121, grid=grid(7410), stream=stream0)
        buf371 = buf363; del buf363  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_104.run(buf370, buf371, 570, 13, grid=grid(570), stream=stream0)
        buf372 = reinterpret_tensor(buf370, (570, 13), (1, 570), 0); del buf370  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_109.run(buf368, mul_809, convolution_34, unsqueeze_634, buf372, 7410, 121, grid=grid(7410), stream=stream0)
        buf373 = empty((570, ), device='cuda', dtype=torch.float32)
        buf374 = empty((570, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_106.run(buf372, squeeze_88, buf373, buf374, 570, 13, grid=grid(570), stream=stream0)
        del buf372
        buf375 = buf368; del buf368  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_110.run(buf375, mul_809, convolution_34, unsqueeze_634, buf373, squeeze_88, buf371, primals_49, 4560, 196, grid=grid(4560, 196), stream=stream0)
        del convolution_34
        del mul_809
        del primals_49
        del squeeze_88
        del unsqueeze_634
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf376 = aten.convolution_backward(buf375, cat_3, primals_153, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf375
        del cat_3
        del primals_153
        buf377 = buf376[0]
        buf379 = empty((95, ), device='cuda', dtype=torch.float32)
        buf380 = empty((95, ), device='cuda', dtype=torch.float32)
        buf382 = empty((95, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]
        triton_red_fused_add_native_batch_norm_backward_slice_backward_111.run(buf253, buf295, buf336, buf377, convolution_33, unsqueeze_646, squeeze_85, buf379, buf380, buf382, 95, 1568, grid=grid(95), stream=stream0)
        buf381 = empty((8, 95, 14, 14), device='cuda', dtype=torch.float32)
        buf383 = buf381; del buf381  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_112.run(buf383, buf253, buf295, buf336, buf377, convolution_33, unsqueeze_646, buf380, squeeze_85, buf379, primals_47, 760, 196, grid=grid(760, 196), stream=stream0)
        del buf380
        del convolution_33
        del primals_47
        del squeeze_85
        del unsqueeze_646
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf384 = aten.convolution_backward(buf383, clamp_max_7, primals_152, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf383
        del clamp_max_7
        del primals_152
        buf385 = buf384[0]
        buf387 = empty_strided((8, 504, 1, 1, 2), (1008, 2, 8064, 8064, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_4, x_152], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sum]
        triton_red_fused_hardtanh_backward_mul_sigmoid_sum_113.run(add_137, convolution_32, buf385, buf387, 8064, 98, grid=grid(8064), stream=stream0)
        buf388 = empty_strided((8, 504, 1, 1), (504, 1, 4032, 4032), device='cuda', dtype=torch.float32)
        buf389 = reinterpret_tensor(buf388, (8, 504, 1, 1), (504, 1, 1, 1), 0); del buf388  # reuse
        # Source Nodes: [sigmoid_4, x_152], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_114.run(buf389, buf387, convolution_32, 4032, 2, grid=grid(4032), stream=stream0)
        del buf387
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf391 = aten.convolution_backward(buf389, relu_4, primals_150, [504], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_150
        buf392 = buf391[0]
        buf13 = empty_strided((1, 42, 1, 1), (42, 1, 42, 42), device='cuda', dtype=torch.float32)
        buf12 = empty((1, 42, 1, 1), device='cuda', dtype=torch.float32)
        buf394 = empty((42, ), device='cuda', dtype=torch.float32)
        buf395 = empty((42, ), device='cuda', dtype=torch.float32)
        buf397 = empty((42, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___7___se_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_115.run(convolution_31, relu_4, buf392, buf13, buf12, buf394, buf395, buf397, 42, 8, grid=grid(42), stream=stream0)
        buf396 = buf392; del buf392  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_116.run(buf396, relu_4, convolution_31, buf12, buf395, buf13, buf394, primals_148, 336, grid=grid(336), stream=stream0)
        del buf12
        del buf13
        del convolution_31
        del primals_148
        del relu_4
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf399 = aten.convolution_backward(buf396, mean_4, primals_146, [42], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_4
        del primals_146
        buf400 = buf399[0]
        buf402 = empty((504, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_4, x_152], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_117.run(add_137, convolution_32, buf385, buf400, buf402, 6552, 121, grid=grid(6552), stream=stream0)
        buf403 = empty((504, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_4, x_152], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_118.run(buf402, buf403, 504, 13, grid=grid(504), stream=stream0)
        buf404 = reinterpret_tensor(buf402, (504, 13), (1, 504), 0); del buf402  # reuse
        # Source Nodes: [sigmoid_4, x_152], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_119.run(add_137, convolution_32, buf385, buf400, convolution_30, unsqueeze_670, buf404, 6552, 121, grid=grid(6552), stream=stream0)
        buf405 = empty((504, ), device='cuda', dtype=torch.float32)
        buf407 = empty((504, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_4, x_152], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_120.run(buf404, squeeze_79, buf405, buf407, 504, 13, grid=grid(504), stream=stream0)
        buf406 = empty_strided((8, 504, 14, 14), (98784, 1, 7056, 504), device='cuda', dtype=torch.float32)
        buf408 = buf406; del buf406  # reuse
        # Source Nodes: [sigmoid_4, x_152], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_121.run(buf408, add_137, convolution_32, buf385, buf400, convolution_30, unsqueeze_670, buf405, squeeze_79, buf403, primals_45, 1568, 504, grid=grid(1568, 504), stream=stream0)
        del add_137
        del buf385
        del buf400
        del convolution_30
        del convolution_32
        del primals_45
        del squeeze_79
        del unsqueeze_670
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf409 = aten.convolution_backward(buf408, mul_193, primals_145, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 504, [True, True, False])
        del buf408
        del mul_193
        del primals_145
        buf410 = buf409[0]
        buf412 = reinterpret_tensor(buf404, (504, 13), (13, 1), 0); del buf404  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_122.run(buf410, mul_852, buf412, 6552, 121, grid=grid(6552), stream=stream0)
        buf413 = buf405; del buf405  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_118.run(buf412, buf413, 504, 13, grid=grid(504), stream=stream0)
        buf414 = reinterpret_tensor(buf412, (504, 13), (1, 504), 0); del buf412  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_123.run(buf410, mul_852, convolution_29, unsqueeze_682, buf414, 6552, 121, grid=grid(6552), stream=stream0)
        buf415 = empty((504, ), device='cuda', dtype=torch.float32)
        buf416 = empty((504, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_120.run(buf414, squeeze_76, buf415, buf416, 504, 13, grid=grid(504), stream=stream0)
        del buf414
        buf417 = buf410; del buf410  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_124.run(buf417, mul_852, convolution_29, unsqueeze_682, buf415, squeeze_76, buf413, primals_43, 4032, 196, grid=grid(4032, 196), stream=stream0)
        del convolution_29
        del mul_852
        del primals_43
        del squeeze_76
        del unsqueeze_682
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf418 = aten.convolution_backward(buf417, cat_2, primals_144, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf417
        del cat_2
        del primals_144
        buf419 = buf418[0]
        buf421 = buf419; del buf419  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_125.run(buf421, buf253, buf295, buf336, buf377, 131712, grid=grid(131712), stream=stream0)
        del buf253
        del buf295
        del buf336
        del buf377
        buf422 = empty((84, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]
        triton_red_fused_add_native_batch_norm_backward_slice_backward_126.run(buf421, buf422, 84, 1568, grid=grid(84), stream=stream0)
        buf423 = empty((84, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]
        triton_red_fused_add_native_batch_norm_backward_slice_backward_127.run(buf421, convolution_28, unsqueeze_694, buf423, 1092, 121, grid=grid(1092), stream=stream0)
        buf424 = empty((84, ), device='cuda', dtype=torch.float32)
        buf425 = empty((84, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]
        triton_per_fused_add_native_batch_norm_backward_slice_backward_128.run(buf423, squeeze_73, buf424, buf425, 84, 13, grid=grid(84), stream=stream0)
        del buf423
        buf426 = empty((8, 84, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_129.run(buf421, convolution_28, unsqueeze_694, buf424, squeeze_73, buf422, primals_41, buf426, 672, 196, grid=grid(672, 196), stream=stream0)
        del buf424
        del convolution_28
        del primals_41
        del squeeze_73
        del unsqueeze_694
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]
        buf427 = aten.convolution_backward(buf426, clamp_max_6, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf426
        del clamp_max_6
        del primals_143
        buf428 = buf427[0]
        buf430 = empty_strided((8, 432, 1, 1, 2), (864, 2, 6912, 6912, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_3, x_131], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sum]
        triton_red_fused_hardtanh_backward_mul_sigmoid_sum_130.run(add_116, convolution_27, buf428, buf430, 6912, 98, grid=grid(6912), stream=stream0)
        buf431 = empty_strided((8, 432, 1, 1), (432, 1, 3456, 3456), device='cuda', dtype=torch.float32)
        buf432 = reinterpret_tensor(buf431, (8, 432, 1, 1), (432, 1, 1, 1), 0); del buf431  # reuse
        # Source Nodes: [sigmoid_3, x_131], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_131.run(buf432, buf430, convolution_27, 3456, 2, grid=grid(3456), stream=stream0)
        del buf430
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf434 = aten.convolution_backward(buf432, relu_3, primals_141, [432], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_141
        buf435 = buf434[0]
        buf10 = empty_strided((1, 36, 1, 1), (36, 1, 36, 36), device='cuda', dtype=torch.float32)
        buf9 = empty((1, 36, 1, 1), device='cuda', dtype=torch.float32)
        buf437 = empty((36, ), device='cuda', dtype=torch.float32)
        buf438 = empty((36, ), device='cuda', dtype=torch.float32)
        buf440 = empty((36, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___6___se_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_132.run(convolution_26, relu_3, buf435, buf10, buf9, buf437, buf438, buf440, 36, 8, grid=grid(36), stream=stream0)
        buf439 = buf435; del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_133.run(buf439, relu_3, convolution_26, buf9, buf438, buf10, buf437, primals_139, 288, grid=grid(288), stream=stream0)
        del buf10
        del buf438
        del convolution_26
        del primals_139
        del relu_3
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf442 = aten.convolution_backward(buf439, mean_3, primals_137, [36], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_3
        del primals_137
        buf443 = buf442[0]
        buf445 = reinterpret_tensor(buf276, (432, 13), (13, 1), 0); del buf276  # reuse
        # Source Nodes: [sigmoid_3, x_131], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_134.run(add_116, convolution_27, buf428, buf443, buf445, 5616, 121, grid=grid(5616), stream=stream0)
        buf446 = empty((432, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_3, x_131], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_135.run(buf445, buf446, 432, 13, grid=grid(432), stream=stream0)
        buf447 = reinterpret_tensor(buf445, (432, 13), (1, 432), 0); del buf445  # reuse
        # Source Nodes: [sigmoid_3, x_131], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_136.run(add_116, convolution_27, buf428, buf443, convolution_25, unsqueeze_718, buf447, 5616, 121, grid=grid(5616), stream=stream0)
        buf448 = empty((432, ), device='cuda', dtype=torch.float32)
        buf450 = empty((432, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_3, x_131], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_137.run(buf447, squeeze_67, buf448, buf450, 432, 13, grid=grid(432), stream=stream0)
        buf449 = empty_strided((8, 432, 14, 14), (84672, 1, 6048, 432), device='cuda', dtype=torch.float32)
        buf451 = buf449; del buf449  # reuse
        # Source Nodes: [sigmoid_3, x_131], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_138.run(buf451, add_116, convolution_27, buf428, buf443, convolution_25, unsqueeze_718, buf448, squeeze_67, buf446, primals_39, 1568, 432, grid=grid(1568, 432), stream=stream0)
        del add_116
        del buf428
        del buf443
        del convolution_25
        del convolution_27
        del primals_39
        del squeeze_67
        del unsqueeze_718
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf452 = aten.convolution_backward(buf451, mul_163, primals_136, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 432, [True, True, False])
        del buf451
        del mul_163
        del primals_136
        buf453 = buf452[0]
        buf455 = reinterpret_tensor(buf447, (432, 13), (13, 1), 0); del buf447  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_139.run(buf453, mul_895, buf455, 5616, 121, grid=grid(5616), stream=stream0)
        buf456 = buf448; del buf448  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_135.run(buf455, buf456, 432, 13, grid=grid(432), stream=stream0)
        buf457 = reinterpret_tensor(buf455, (432, 13), (1, 432), 0); del buf455  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_140.run(buf453, mul_895, convolution_24, unsqueeze_730, buf457, 5616, 121, grid=grid(5616), stream=stream0)
        buf458 = empty((432, ), device='cuda', dtype=torch.float32)
        buf459 = empty((432, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_137.run(buf457, squeeze_64, buf458, buf459, 432, 13, grid=grid(432), stream=stream0)
        del buf457
        buf460 = buf453; del buf453  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_141.run(buf460, mul_895, convolution_24, unsqueeze_730, buf458, squeeze_64, buf456, primals_37, 3456, 196, grid=grid(3456, 196), stream=stream0)
        del convolution_24
        del mul_895
        del primals_37
        del squeeze_64
        del unsqueeze_730
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf461 = aten.convolution_backward(buf460, add_106, primals_135, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_106
        del buf460
        del primals_135
        buf462 = buf461[0]
        buf464 = empty((72, ), device='cuda', dtype=torch.float32)
        buf465 = empty((72, ), device='cuda', dtype=torch.float32)
        buf466 = empty((72, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_142.run(buf421, buf462, convolution_23, unsqueeze_742, squeeze_61, buf464, buf465, buf466, 72, 1568, grid=grid(72), stream=stream0)
        buf467 = buf462; del buf462  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_143.run(buf467, buf421, convolution_23, unsqueeze_742, buf465, squeeze_61, buf464, primals_35, 576, 196, grid=grid(576, 196), stream=stream0)
        del buf421
        del buf465
        del convolution_23
        del primals_35
        del squeeze_61
        del unsqueeze_742
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf468 = aten.convolution_backward(buf467, clamp_max_5, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf467
        del clamp_max_5
        del primals_134
        buf469 = buf468[0]
        buf471 = empty_strided((8, 366, 1, 1, 2), (732, 2, 5856, 5856, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_2, x_111], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sum]
        triton_red_fused_hardtanh_backward_mul_sigmoid_sum_144.run(add_96, convolution_22, buf469, buf471, 5856, 98, grid=grid(5856), stream=stream0)
        buf472 = empty_strided((8, 366, 1, 1), (366, 1, 2928, 2928), device='cuda', dtype=torch.float32)
        buf473 = reinterpret_tensor(buf472, (8, 366, 1, 1), (366, 1, 1, 1), 0); del buf472  # reuse
        # Source Nodes: [sigmoid_2, x_111], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_145.run(buf473, buf471, convolution_22, 2928, 2, grid=grid(2928), stream=stream0)
        del buf471
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf475 = aten.convolution_backward(buf473, relu_2, primals_132, [366], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_132
        buf476 = buf475[0]
        buf7 = empty_strided((1, 30, 1, 1), (30, 1, 30, 30), device='cuda', dtype=torch.float32)
        buf6 = empty((1, 30, 1, 1), device='cuda', dtype=torch.float32)
        buf478 = empty((30, ), device='cuda', dtype=torch.float32)
        buf479 = empty((30, ), device='cuda', dtype=torch.float32)
        buf481 = empty((30, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___5___se_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_146.run(convolution_21, relu_2, buf476, buf7, buf6, buf478, buf479, buf481, 30, 8, grid=grid(30), stream=stream0)
        buf480 = buf476; del buf476  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_147.run(buf480, relu_2, convolution_21, buf6, buf479, buf7, buf478, primals_130, 240, grid=grid(240), stream=stream0)
        del buf479
        del buf6
        del convolution_21
        del primals_130
        del relu_2
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf483 = aten.convolution_backward(buf480, mean_2, primals_128, [30], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_2
        del primals_128
        buf484 = buf483[0]
        buf486 = empty((366, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_2, x_111], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_148.run(add_96, convolution_22, buf469, buf484, buf486, 4758, 121, grid=grid(4758), stream=stream0)
        buf487 = empty((366, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_2, x_111], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_149.run(buf486, buf487, 366, 13, grid=grid(366), stream=stream0)
        buf488 = reinterpret_tensor(buf486, (366, 13), (1, 366), 0); del buf486  # reuse
        # Source Nodes: [sigmoid_2, x_111], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_150.run(add_96, convolution_22, buf469, buf484, convolution_20, unsqueeze_766, buf488, 4758, 121, grid=grid(4758), stream=stream0)
        buf489 = empty((366, ), device='cuda', dtype=torch.float32)
        buf491 = empty((366, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_2, x_111], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_151.run(buf488, squeeze_55, buf489, buf491, 366, 13, grid=grid(366), stream=stream0)
        del buf488
        buf490 = empty_strided((8, 366, 14, 14), (71736, 1, 5124, 366), device='cuda', dtype=torch.float32)
        buf492 = buf490; del buf490  # reuse
        # Source Nodes: [sigmoid_2, x_111], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_152.run(buf492, add_96, convolution_22, buf469, buf484, convolution_20, unsqueeze_766, buf489, squeeze_55, buf487, primals_33, 1568, 366, grid=grid(1568, 366), stream=stream0)
        del add_96
        del buf469
        del buf484
        del convolution_20
        del convolution_22
        del primals_33
        del squeeze_55
        del unsqueeze_766
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf493 = aten.convolution_backward(buf492, mul_133, primals_127, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 366, [True, True, False])
        del buf492
        del mul_133
        del primals_127
        buf494 = buf493[0]
        buf496 = empty((366, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_153.run(buf494, mul_938, buf496, 17934, 128, grid=grid(17934), stream=stream0)
        buf497 = buf489; del buf489  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_154.run(buf496, buf497, 366, 49, grid=grid(366), stream=stream0)
        buf498 = reinterpret_tensor(buf496, (366, 49), (1, 366), 0); del buf496  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_155.run(buf494, mul_938, convolution_19, unsqueeze_778, buf498, 17934, 128, grid=grid(17934), stream=stream0)
        buf499 = empty((366, ), device='cuda', dtype=torch.float32)
        buf500 = empty((366, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_156.run(buf498, squeeze_52, buf499, buf500, 366, 49, grid=grid(366), stream=stream0)
        del buf498
        buf501 = buf494; del buf494  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_157.run(buf501, mul_938, convolution_19, unsqueeze_778, buf499, squeeze_52, buf497, primals_31, 2928, 784, grid=grid(2928, 784), stream=stream0)
        del convolution_19
        del mul_938
        del primals_31
        del squeeze_52
        del unsqueeze_778
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf502 = aten.convolution_backward(buf501, cat_1, primals_126, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf501
        del cat_1
        del primals_126
        buf503 = buf502[0]
        buf505 = empty((61, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]
        triton_red_fused_add_native_batch_norm_backward_slice_backward_158.run(buf503, buf505, 61, 6272, grid=grid(61), stream=stream0)
        buf506 = empty((61, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]
        triton_red_fused_add_native_batch_norm_backward_slice_backward_159.run(buf503, convolution_18, unsqueeze_790, buf506, 2989, 128, grid=grid(2989), stream=stream0)
        buf507 = empty((61, ), device='cuda', dtype=torch.float32)
        buf508 = empty((61, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]
        triton_per_fused_add_native_batch_norm_backward_slice_backward_160.run(buf506, squeeze_49, buf507, buf508, 61, 49, grid=grid(61), stream=stream0)
        del buf506
        buf509 = empty((8, 61, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_161.run(buf503, convolution_18, unsqueeze_790, buf507, squeeze_49, buf505, primals_29, buf509, 488, 784, grid=grid(488, 784), stream=stream0)
        del buf507
        del convolution_18
        del primals_29
        del squeeze_49
        del unsqueeze_790
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]
        buf510 = aten.convolution_backward(buf509, clamp_max_4, primals_125, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf509
        del clamp_max_4
        del primals_125
        buf511 = buf510[0]
        buf513 = empty_strided((8, 300, 1, 1, 7), (2100, 7, 16800, 16800, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_1, x_90], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sum]
        triton_red_fused_hardtanh_backward_mul_sigmoid_sum_162.run(add_75, convolution_17, buf511, buf513, 16800, 112, grid=grid(16800), stream=stream0)
        buf514 = empty_strided((8, 300, 1, 1), (300, 1, 2400, 2400), device='cuda', dtype=torch.float32)
        buf515 = reinterpret_tensor(buf514, (8, 300, 1, 1), (300, 1, 1, 1), 0); del buf514  # reuse
        # Source Nodes: [sigmoid_1, x_90], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_163.run(buf515, buf513, convolution_17, 2400, 7, grid=grid(2400), stream=stream0)
        del buf513
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf517 = aten.convolution_backward(buf515, relu_1, primals_123, [300], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_123
        buf518 = buf517[0]
        buf4 = empty_strided((1, 25, 1, 1), (25, 1, 25, 25), device='cuda', dtype=torch.float32)
        buf3 = empty((1, 25, 1, 1), device='cuda', dtype=torch.float32)
        buf520 = empty((25, ), device='cuda', dtype=torch.float32)
        buf521 = empty((25, ), device='cuda', dtype=torch.float32)
        buf523 = empty((25, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___4___se_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_164.run(convolution_16, relu_1, buf518, buf4, buf3, buf520, buf521, buf523, 25, 8, grid=grid(25), stream=stream0)
        buf522 = buf518; del buf518  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_165.run(buf522, relu_1, convolution_16, buf3, buf521, buf4, buf520, primals_121, 200, grid=grid(200), stream=stream0)
        del buf3
        del buf4
        del convolution_16
        del primals_121
        del relu_1
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf525 = aten.convolution_backward(buf522, mean_1, primals_119, [25], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean_1
        del primals_119
        buf526 = buf525[0]
        buf528 = empty((300, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_1, x_90], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_166.run(add_75, convolution_17, buf511, buf526, buf528, 14700, 128, grid=grid(14700), stream=stream0)
        buf529 = empty((300, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_1, x_90], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_167.run(buf528, buf529, 300, 49, grid=grid(300), stream=stream0)
        buf530 = reinterpret_tensor(buf528, (300, 49), (1, 300), 0); del buf528  # reuse
        # Source Nodes: [sigmoid_1, x_90], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_168.run(add_75, convolution_17, buf511, buf526, convolution_15, unsqueeze_814, buf530, 14700, 128, grid=grid(14700), stream=stream0)
        buf531 = empty((300, ), device='cuda', dtype=torch.float32)
        buf533 = empty((300, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid_1, x_90], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_169.run(buf530, squeeze_43, buf531, buf533, 300, 49, grid=grid(300), stream=stream0)
        buf532 = empty_strided((8, 300, 28, 28), (235200, 1, 8400, 300), device='cuda', dtype=torch.float32)
        buf534 = buf532; del buf532  # reuse
        # Source Nodes: [sigmoid_1, x_90], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_170.run(buf534, add_75, convolution_17, buf511, buf526, convolution_15, unsqueeze_814, buf531, squeeze_43, buf529, primals_27, 6272, 300, grid=grid(6272, 300), stream=stream0)
        del add_75
        del buf511
        del buf526
        del convolution_15
        del convolution_17
        del primals_27
        del squeeze_43
        del unsqueeze_814
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf535 = aten.convolution_backward(buf534, mul_103, primals_118, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 300, [True, True, False])
        del buf534
        del mul_103
        del primals_118
        buf536 = buf535[0]
        buf538 = reinterpret_tensor(buf530, (300, 49), (49, 1), 0); del buf530  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_171.run(buf536, mul_981, buf538, 14700, 128, grid=grid(14700), stream=stream0)
        buf539 = buf531; del buf531  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_167.run(buf538, buf539, 300, 49, grid=grid(300), stream=stream0)
        buf540 = reinterpret_tensor(buf538, (300, 49), (1, 300), 0); del buf538  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_172.run(buf536, mul_981, convolution_14, unsqueeze_826, buf540, 14700, 128, grid=grid(14700), stream=stream0)
        buf541 = empty((300, ), device='cuda', dtype=torch.float32)
        buf542 = empty((300, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_169.run(buf540, squeeze_40, buf541, buf542, 300, 49, grid=grid(300), stream=stream0)
        del buf540
        buf543 = buf536; del buf536  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_173.run(buf543, mul_981, convolution_14, unsqueeze_826, buf541, squeeze_40, buf539, primals_25, 2400, 784, grid=grid(2400, 784), stream=stream0)
        del convolution_14
        del mul_981
        del primals_25
        del squeeze_40
        del unsqueeze_826
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf544 = aten.convolution_backward(buf543, add_65, primals_117, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_65
        del buf543
        del primals_117
        buf545 = buf544[0]
        buf547 = empty((50, ), device='cuda', dtype=torch.float32)
        buf548 = empty((50, ), device='cuda', dtype=torch.float32)
        buf549 = empty((50, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_174.run(buf503, buf545, convolution_13, unsqueeze_838, squeeze_37, buf547, buf548, buf549, 50, 6272, grid=grid(50), stream=stream0)
        buf550 = buf545; del buf545  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_175.run(buf550, buf503, convolution_13, unsqueeze_838, buf548, squeeze_37, buf547, primals_23, 400, 784, grid=grid(400, 784), stream=stream0)
        del buf503
        del buf548
        del convolution_13
        del primals_23
        del squeeze_37
        del unsqueeze_838
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf551 = aten.convolution_backward(buf550, clamp_max_3, primals_116, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf550
        del clamp_max_3
        del primals_116
        buf552 = buf551[0]
        buf554 = empty_strided((8, 228, 1, 1, 7), (1596, 7, 12768, 12768, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid, x_70], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sum]
        triton_red_fused_hardtanh_backward_mul_sigmoid_sum_176.run(add_55, convolution_12, buf552, buf554, 12768, 112, grid=grid(12768), stream=stream0)
        buf555 = empty_strided((8, 228, 1, 1), (228, 1, 1824, 1824), device='cuda', dtype=torch.float32)
        buf556 = reinterpret_tensor(buf555, (8, 228, 1, 1), (228, 1, 1, 1), 0); del buf555  # reuse
        # Source Nodes: [sigmoid, x_70], Original ATen: [aten.hardtanh_backward, aten.mul, aten.sigmoid, aten.sigmoid_backward, aten.sum]
        triton_per_fused_hardtanh_backward_mul_sigmoid_sigmoid_backward_sum_177.run(buf556, buf554, convolution_12, 1824, 7, grid=grid(1824), stream=stream0)
        del buf554
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf558 = aten.convolution_backward(buf556, relu, primals_114, [228], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del primals_114
        buf559 = buf558[0]
        buf1 = empty_strided((1, 19, 1, 1), (19, 1, 19, 19), device='cuda', dtype=torch.float32)
        buf0 = empty((1, 19, 1, 1), device='cuda', dtype=torch.float32)
        buf561 = empty((19, ), device='cuda', dtype=torch.float32)
        buf562 = empty((19, ), device='cuda', dtype=torch.float32)
        buf564 = empty((19, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___3___se_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_hardtanh_backward_native_batch_norm_backward_threshold_backward_178.run(convolution_11, relu, buf559, buf1, buf0, buf561, buf562, buf564, 19, 8, grid=grid(19), stream=stream0)
        buf40 = empty((1000, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_17, out=buf40)
        del clone_17
        buf41 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_179.run(tangents_1, buf41, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf50 = buf48[1]
        del buf48
        buf58 = buf56[1]
        del buf56
        buf61 = buf86; del buf86  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_180.run(buf60, buf61, 1044, 8, grid=grid(1044), stream=stream0)
        del buf60
        buf64 = buf62[1]
        del buf62
        buf69 = buf66; del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_181.run(buf67, buf69, 87, 8, grid=grid(87), stream=stream0)
        del buf67
        buf72 = buf70[1]
        del buf70
        buf82 = buf80[1]
        del buf80
        buf91 = buf89[1]
        del buf89
        buf98 = buf96[1]
        del buf96
        buf101 = buf126; del buf126  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_182.run(buf100, buf101, 972, 8, grid=grid(972), stream=stream0)
        del buf100
        buf104 = buf102[1]
        del buf102
        buf109 = reinterpret_tensor(buf34, (81, ), (1, ), 0); del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_183.run(buf107, buf109, 81, 8, grid=grid(81), stream=stream0)
        del buf107
        buf112 = buf110[1]
        del buf110
        buf122 = buf120[1]
        del buf120
        buf131 = buf129[1]
        del buf129
        buf138 = buf136[1]
        del buf136
        buf141 = buf166; del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_184.run(buf140, buf141, 906, 8, grid=grid(906), stream=stream0)
        del buf140
        buf144 = buf142[1]
        del buf142
        buf149 = reinterpret_tensor(buf31, (75, ), (1, ), 0); del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_185.run(buf147, buf149, 75, 8, grid=grid(75), stream=stream0)
        del buf147
        buf152 = buf150[1]
        del buf150
        buf162 = buf160[1]
        del buf160
        buf171 = buf169[1]
        del buf169
        buf179 = buf177[1]
        del buf177
        buf182 = buf207; del buf207  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_186.run(buf181, buf182, 840, 8, grid=grid(840), stream=stream0)
        del buf181
        buf185 = buf183[1]
        del buf183
        buf190 = reinterpret_tensor(buf28, (70, ), (1, ), 0); del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_187.run(buf188, buf190, 70, 8, grid=grid(70), stream=stream0)
        del buf188
        buf193 = buf191[1]
        del buf191
        buf203 = buf201[1]
        del buf201
        buf212 = buf210[1]
        del buf210
        buf221 = buf219[1]
        del buf219
        buf224 = buf249; del buf249  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_188.run(buf223, buf224, 768, 8, grid=grid(768), stream=stream0)
        del buf223
        buf227 = buf225[1]
        del buf225
        buf232 = reinterpret_tensor(buf25, (64, ), (1, ), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_189.run(buf230, buf232, 64, 8, grid=grid(64), stream=stream0)
        del buf230
        buf235 = buf233[1]
        del buf233
        buf245 = buf243[1]
        del buf243
        buf254 = buf252[1]
        del buf252
        buf262 = buf260[1]
        del buf260
        buf266 = buf291; del buf291  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_190.run(buf265, buf266, 702, 8, grid=grid(702), stream=stream0)
        del buf265
        buf269 = buf267[1]
        del buf267
        buf274 = buf271; del buf271  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_191.run(buf272, buf274, 58, 8, grid=grid(58), stream=stream0)
        del buf272
        buf277 = buf275[1]
        del buf275
        buf287 = buf285[1]
        del buf285
        buf296 = buf294[1]
        del buf294
        buf303 = buf301[1]
        del buf301
        buf307 = buf332; del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_192.run(buf306, buf307, 636, 8, grid=grid(636), stream=stream0)
        del buf306
        buf310 = buf308[1]
        del buf308
        buf315 = buf312; del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_193.run(buf313, buf315, 53, 8, grid=grid(53), stream=stream0)
        del buf313
        buf318 = buf316[1]
        del buf316
        buf328 = buf326[1]
        del buf326
        buf337 = buf335[1]
        del buf335
        buf344 = buf342[1]
        del buf342
        buf348 = buf373; del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_194.run(buf347, buf348, 570, 8, grid=grid(570), stream=stream0)
        del buf347
        buf351 = buf349[1]
        del buf349
        buf356 = buf353; del buf353  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_195.run(buf354, buf356, 47, 8, grid=grid(47), stream=stream0)
        del buf354
        buf359 = buf357[1]
        del buf357
        buf369 = buf367[1]
        del buf367
        buf378 = buf376[1]
        del buf376
        buf386 = buf384[1]
        del buf384
        buf390 = buf415; del buf415  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_196.run(buf389, buf390, 504, 8, grid=grid(504), stream=stream0)
        del buf389
        buf393 = buf391[1]
        del buf391
        buf398 = buf395; del buf395  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_197.run(buf396, buf398, 42, 8, grid=grid(42), stream=stream0)
        del buf396
        buf401 = buf399[1]
        del buf399
        buf411 = buf409[1]
        del buf409
        buf420 = buf418[1]
        del buf418
        buf429 = buf427[1]
        del buf427
        buf433 = buf458; del buf458  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_198.run(buf432, buf433, 432, 8, grid=grid(432), stream=stream0)
        del buf432
        buf436 = buf434[1]
        del buf434
        buf441 = reinterpret_tensor(buf9, (36, ), (1, ), 0); del buf9  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_199.run(buf439, buf441, 36, 8, grid=grid(36), stream=stream0)
        del buf439
        buf444 = buf442[1]
        del buf442
        buf454 = buf452[1]
        del buf452
        buf463 = buf461[1]
        del buf461
        buf470 = buf468[1]
        del buf468
        buf474 = buf499; del buf499  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_200.run(buf473, buf474, 366, 8, grid=grid(366), stream=stream0)
        del buf473
        buf477 = buf475[1]
        del buf475
        buf482 = reinterpret_tensor(buf7, (30, ), (1, ), 0); del buf7  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_201.run(buf480, buf482, 30, 8, grid=grid(30), stream=stream0)
        del buf480
        buf485 = buf483[1]
        del buf483
        buf495 = buf493[1]
        del buf493
        buf504 = buf502[1]
        del buf502
        buf512 = buf510[1]
        del buf510
        buf516 = buf541; del buf541  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_202.run(buf515, buf516, 300, 8, grid=grid(300), stream=stream0)
        del buf515
        buf519 = buf517[1]
        del buf517
        buf524 = buf521; del buf521  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_203.run(buf522, buf524, 25, 8, grid=grid(25), stream=stream0)
        del buf522
        buf527 = buf525[1]
        del buf525
        buf537 = buf535[1]
        del buf535
        buf546 = buf544[1]
        del buf544
        buf553 = buf551[1]
        del buf551
        buf557 = empty((228, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_204.run(buf556, buf557, 228, 8, grid=grid(228), stream=stream0)
        del buf556
        buf560 = buf558[1]
        del buf558
        buf563 = buf559; del buf559  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_hardtanh_backward_native_batch_norm_backward_threshold_backward_205.run(buf563, relu, convolution_11, buf0, buf562, buf1, buf561, primals_112, 152, grid=grid(152), stream=stream0)
        del buf0
        del buf1
        del convolution_11
        del primals_112
        del relu
        buf565 = buf562; del buf562  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_206.run(buf563, buf565, 19, 8, grid=grid(19), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf566 = aten.convolution_backward(buf563, mean, primals_110, [19], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del mean
        del primals_110
        buf567 = buf566[0]
        buf568 = buf566[1]
        del buf566
        buf569 = empty((228, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid, x_70], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_207.run(add_55, convolution_12, buf552, buf567, buf569, 11172, 128, grid=grid(11172), stream=stream0)
        buf570 = empty((228, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid, x_70], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_208.run(buf569, buf570, 228, 49, grid=grid(228), stream=stream0)
        buf571 = reinterpret_tensor(buf569, (228, 49), (1, 228), 0); del buf569  # reuse
        # Source Nodes: [sigmoid, x_70], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_red_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_209.run(add_55, convolution_12, buf552, buf567, convolution_10, unsqueeze_862, buf571, 11172, 128, grid=grid(11172), stream=stream0)
        buf572 = empty((228, ), device='cuda', dtype=torch.float32)
        buf574 = empty((228, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sigmoid, x_70], Original ATen: [aten.add, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_per_fused_add_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_210.run(buf571, squeeze_31, buf572, buf574, 228, 49, grid=grid(228), stream=stream0)
        del buf571
        buf573 = empty_strided((8, 228, 28, 28), (178752, 1, 6384, 228), device='cuda', dtype=torch.float32)
        buf575 = buf573; del buf573  # reuse
        # Source Nodes: [sigmoid, x_70], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardtanh_backward, aten.mul, aten.native_batch_norm_backward, aten.sigmoid]
        triton_poi_fused_add_convolution_backward_div_hardtanh_backward_mul_native_batch_norm_backward_sigmoid_211.run(buf575, add_55, convolution_12, buf552, buf567, convolution_10, unsqueeze_862, buf572, squeeze_31, buf570, primals_21, 6272, 228, grid=grid(6272, 228), stream=stream0)
        del add_55
        del buf552
        del buf567
        del convolution_10
        del convolution_12
        del primals_21
        del squeeze_31
        del unsqueeze_862
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf576 = aten.convolution_backward(buf575, mul_73, primals_109, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 228, [True, True, False])
        del buf575
        del mul_73
        del primals_109
        buf577 = buf576[0]
        buf578 = buf576[1]
        del buf576
        buf579 = empty((228, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_212.run(buf577, mul_1024, buf579, 44688, 128, grid=grid(44688), stream=stream0)
        buf580 = buf572; del buf572  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_213.run(buf579, buf580, 228, 196, grid=grid(228), stream=stream0)
        buf581 = reinterpret_tensor(buf579, (228, 196), (1, 228), 0); del buf579  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_214.run(buf577, mul_1024, convolution_9, unsqueeze_874, buf581, 44688, 128, grid=grid(44688), stream=stream0)
        buf582 = empty((228, ), device='cuda', dtype=torch.float32)
        buf583 = empty((228, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_215.run(buf581, squeeze_28, buf582, buf583, 228, 196, grid=grid(228), stream=stream0)
        del buf581
        buf584 = buf577; del buf577  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_216.run(buf584, mul_1024, convolution_9, unsqueeze_874, buf582, squeeze_28, buf580, primals_19, 1824, 3136, grid=grid(1824, 3136), stream=stream0)
        del buf582
        del convolution_9
        del mul_1024
        del primals_19
        del squeeze_28
        del unsqueeze_874
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf585 = aten.convolution_backward(buf584, cat, primals_108, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf584
        del cat
        del primals_108
        buf586 = buf585[0]
        buf587 = buf585[1]
        del buf585
        buf588 = reinterpret_tensor(buf563, (38, 4), (1, 38), 0); del buf563  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]
        triton_red_fused_add_native_batch_norm_backward_slice_backward_217.run(buf586, buf588, 152, 6272, grid=grid(152), stream=stream0)
        buf589 = empty((38, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]
        triton_per_fused_add_native_batch_norm_backward_slice_backward_218.run(buf588, buf589, 38, 4, grid=grid(38), stream=stream0)
        del buf588
        buf590 = empty((38, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]
        triton_red_fused_add_native_batch_norm_backward_slice_backward_219.run(buf586, convolution_8, unsqueeze_886, buf590, 7448, 128, grid=grid(7448), stream=stream0)
        buf591 = empty((38, ), device='cuda', dtype=torch.float32)
        buf592 = empty((38, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward, aten.slice_backward]
        triton_per_fused_add_native_batch_norm_backward_slice_backward_220.run(buf590, squeeze_25, buf591, buf592, 38, 196, grid=grid(38), stream=stream0)
        del buf590
        buf593 = empty((8, 38, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_slice_backward_221.run(buf586, convolution_8, unsqueeze_886, buf591, squeeze_25, buf589, primals_17, buf593, 304, 3136, grid=grid(304, 3136), stream=stream0)
        del buf591
        del convolution_8
        del primals_17
        del squeeze_25
        del unsqueeze_886
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.slice_backward]
        buf594 = aten.convolution_backward(buf593, clamp_max_2, primals_107, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf593
        del clamp_max_2
        del primals_107
        buf595 = buf594[0]
        buf596 = buf594[1]
        del buf594
        buf597 = empty((162, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_222.run(bitwise_or_13, buf595, buf597, 31752, 128, grid=grid(31752), stream=stream0)
        buf598 = buf133; del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_223.run(buf597, buf598, 162, 196, grid=grid(162), stream=stream0)
        buf599 = reinterpret_tensor(buf597, (162, 196), (1, 162), 0); del buf597  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_224.run(bitwise_or_13, buf595, convolution_7, unsqueeze_898, buf599, 31752, 128, grid=grid(31752), stream=stream0)
        buf600 = empty((162, ), device='cuda', dtype=torch.float32)
        buf601 = empty((162, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_225.run(buf599, squeeze_22, buf600, buf601, 162, 196, grid=grid(162), stream=stream0)
        buf602 = empty_strided((8, 162, 56, 56), (508032, 1, 9072, 162), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_226.run(bitwise_or_13, buf595, convolution_7, unsqueeze_898, buf600, squeeze_22, buf598, primals_15, buf602, 25088, 162, grid=grid(25088, 162), stream=stream0)
        del bitwise_or_13
        del buf595
        del convolution_7
        del primals_15
        del squeeze_22
        del unsqueeze_898
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf603 = aten.convolution_backward(buf602, mul_51, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 162, [True, True, False])
        del buf602
        del mul_51
        del primals_106
        buf604 = buf603[0]
        buf605 = buf603[1]
        del buf603
        buf606 = reinterpret_tensor(buf599, (162, 196), (196, 1), 0); del buf599  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_227.run(buf604, mul_1054, buf606, 31752, 128, grid=grid(31752), stream=stream0)
        buf607 = buf600; del buf600  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_223.run(buf606, buf607, 162, 196, grid=grid(162), stream=stream0)
        buf608 = reinterpret_tensor(buf606, (162, 196), (1, 162), 0); del buf606  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_228.run(buf604, mul_1054, convolution_6, unsqueeze_910, buf608, 31752, 128, grid=grid(31752), stream=stream0)
        buf609 = empty((162, ), device='cuda', dtype=torch.float32)
        buf610 = empty((162, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_225.run(buf608, squeeze_19, buf609, buf610, 162, 196, grid=grid(162), stream=stream0)
        del buf608
        buf611 = buf604; del buf604  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_229.run(buf611, mul_1054, convolution_6, unsqueeze_910, buf609, squeeze_19, buf607, primals_13, 1296, 3136, grid=grid(1296, 3136), stream=stream0)
        del buf609
        del convolution_6
        del mul_1054
        del primals_13
        del squeeze_19
        del unsqueeze_910
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf612 = aten.convolution_backward(buf611, add_29, primals_105, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_29
        del buf611
        del primals_105
        buf613 = buf612[0]
        buf614 = buf612[1]
        del buf612
        buf615 = empty_strided((27, 4), (1, 27), device='cuda', dtype=torch.float32)
        buf617 = empty_strided((27, 4), (1, 27), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_red_fused_add_native_batch_norm_backward_230.run(buf586, buf613, convolution_5, unsqueeze_922, buf615, buf617, 108, 6272, grid=grid(108), stream=stream0)
        buf616 = empty((27, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_231.run(buf615, buf616, 27, 4, grid=grid(27), stream=stream0)
        del buf615
        buf618 = empty((27, ), device='cuda', dtype=torch.float32)
        buf619 = empty((27, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_per_fused_add_native_batch_norm_backward_232.run(buf617, squeeze_16, buf618, buf619, 27, 4, grid=grid(27), stream=stream0)
        del buf617
        buf620 = buf613; del buf613  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_add_convolution_backward_native_batch_norm_backward_233.run(buf620, buf586, convolution_5, unsqueeze_922, buf618, squeeze_16, buf616, primals_11, 216, 3136, grid=grid(216, 3136), stream=stream0)
        del buf586
        del buf618
        del convolution_5
        del primals_11
        del squeeze_16
        del unsqueeze_922
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
        buf621 = aten.convolution_backward(buf620, clamp_max_1, primals_104, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf620
        del clamp_max_1
        del primals_104
        buf622 = buf621[0]
        buf623 = buf621[1]
        del buf621
        buf624 = empty((96, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_234.run(bitwise_or_14, buf622, buf624, 18816, 128, grid=grid(18816), stream=stream0)
        buf625 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_235.run(buf624, buf625, 96, 196, grid=grid(96), stream=stream0)
        buf626 = reinterpret_tensor(buf624, (96, 196), (1, 96), 0); del buf624  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_236.run(bitwise_or_14, buf622, convolution_4, unsqueeze_934, buf626, 18816, 128, grid=grid(18816), stream=stream0)
        buf627 = empty((96, ), device='cuda', dtype=torch.float32)
        buf628 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_237.run(buf626, squeeze_13, buf627, buf628, 96, 196, grid=grid(96), stream=stream0)
        del buf626
        buf629 = empty_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_238.run(bitwise_or_14, buf622, convolution_4, unsqueeze_934, buf627, squeeze_13, buf625, primals_9, buf629, 25088, 96, grid=grid(25088, 96), stream=stream0)
        del bitwise_or_14
        del buf622
        del convolution_4
        del primals_9
        del squeeze_13
        del unsqueeze_934
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf630 = aten.convolution_backward(buf629, mul_29, primals_103, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 96, [True, True, False])
        del buf629
        del mul_29
        del primals_103
        buf631 = buf630[0]
        buf632 = buf630[1]
        del buf630
        buf633 = empty((96, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_239.run(buf631, mul_1084, buf633, 75264, 128, grid=grid(75264), stream=stream0)
        buf634 = buf627; del buf627  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_mul_native_batch_norm_backward_240.run(buf633, buf634, 96, 784, grid=grid(96), stream=stream0)
        buf635 = reinterpret_tensor(buf633, (96, 784), (1, 96), 0); del buf633  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_241.run(buf631, mul_1084, convolution_3, unsqueeze_946, buf635, 75264, 128, grid=grid(75264), stream=stream0)
        buf636 = empty((96, ), device='cuda', dtype=torch.float32)
        buf637 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_242.run(buf635, squeeze_10, buf636, buf637, 96, 784, grid=grid(96), stream=stream0)
        del buf635
        buf638 = buf631; del buf631  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_243.run(buf638, mul_1084, convolution_3, unsqueeze_946, buf636, squeeze_10, buf634, primals_7, 768, 12544, grid=grid(768, 12544), stream=stream0)
        del buf636
        del convolution_3
        del mul_1084
        del primals_7
        del squeeze_10
        del unsqueeze_946
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf639 = aten.convolution_backward(buf638, add_14, primals_102, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del add_14
        del buf638
        del primals_102
        buf640 = buf639[0]
        buf641 = buf639[1]
        del buf639
        buf642 = empty((16, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_244.run(buf640, buf642, 208, 7720, grid=grid(208), stream=stream0)
        buf643 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_245.run(buf642, buf643, 16, 13, grid=grid(16), stream=stream0)
        del buf642
        buf644 = empty((16, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_246.run(buf640, convolution_2, unsqueeze_958, buf644, 12544, 128, grid=grid(12544), stream=stream0)
        buf645 = empty((16, ), device='cuda', dtype=torch.float32)
        buf646 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_247.run(buf644, squeeze_7, buf645, buf646, 16, 784, grid=grid(16), stream=stream0)
        del buf644
        buf647 = buf640; del buf640  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_248.run(buf647, convolution_2, unsqueeze_958, buf645, squeeze_7, buf643, primals_5, 128, 12544, grid=grid(128, 12544), stream=stream0)
        del buf645
        del convolution_2
        del primals_5
        del squeeze_7
        del unsqueeze_958
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf648 = aten.convolution_backward(buf647, clamp_max, primals_101, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf647
        del clamp_max
        del primals_101
        buf649 = buf648[0]
        buf650 = buf648[1]
        del buf648
        buf651 = empty((32, 784), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_249.run(bitwise_or_15, buf649, buf651, 25088, 128, grid=grid(25088), stream=stream0)
        buf652 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_250.run(buf651, buf652, 32, 784, grid=grid(32), stream=stream0)
        buf653 = reinterpret_tensor(buf651, (32, 784), (1, 32), 0); del buf651  # reuse
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_251.run(bitwise_or_15, buf649, convolution_1, unsqueeze_970, buf653, 25088, 128, grid=grid(25088), stream=stream0)
        buf654 = empty((32, ), device='cuda', dtype=torch.float32)
        buf655 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_252.run(buf653, squeeze_4, buf654, buf655, 32, 784, grid=grid(32), stream=stream0)
        buf656 = empty_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardtanh_backward_native_batch_norm_backward_253.run(bitwise_or_15, buf649, convolution_1, unsqueeze_970, buf654, squeeze_4, buf652, primals_3, buf656, 100352, 32, grid=grid(100352, 32), stream=stream0)
        del bitwise_or_15
        del buf649
        del convolution_1
        del primals_3
        del squeeze_4
        del unsqueeze_970
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardtanh_backward, aten.native_batch_norm_backward]
        buf657 = aten.convolution_backward(buf656, mul_7, primals_100, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf656
        del mul_7
        del primals_100
        buf658 = buf657[0]
        buf659 = buf657[1]
        del buf657
        buf660 = reinterpret_tensor(buf653, (32, 784), (784, 1), 0); del buf653  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_254.run(buf658, mul_1114, buf660, 25088, 128, grid=grid(25088), stream=stream0)
        buf661 = buf654; del buf654  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_per_fused_hardtanh_backward_native_batch_norm_backward_250.run(buf660, buf661, 32, 784, grid=grid(32), stream=stream0)
        buf662 = reinterpret_tensor(buf660, (32, 784), (1, 32), 0); del buf660  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_mul_native_batch_norm_backward_255.run(buf658, mul_1114, convolution, unsqueeze_982, buf662, 25088, 128, grid=grid(25088), stream=stream0)
        buf663 = empty((32, ), device='cuda', dtype=torch.float32)
        buf664 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.native_batch_norm_backward]
        triton_red_fused_hardtanh_backward_native_batch_norm_backward_252.run(buf662, squeeze_1, buf663, buf664, 32, 784, grid=grid(32), stream=stream0)
        del buf662
        buf665 = buf658; del buf658  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_mul_native_batch_norm_backward_256.run(buf665, mul_1114, convolution, unsqueeze_982, buf663, squeeze_1, buf661, primals_1, 256, 12544, grid=grid(256, 12544), stream=stream0)
        del buf663
        del convolution
        del mul_1114
        del primals_1
        del squeeze_1
        del unsqueeze_982
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.mul, aten.native_batch_norm_backward]
        buf666 = aten.convolution_backward(buf665, primals_414, primals_99, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf665
        del primals_414
        del primals_99
        buf667 = buf666[1]
        return (buf664, buf661, buf655, buf652, buf646, buf643, buf637, buf634, buf628, buf625, buf619, buf616, buf610, buf607, buf601, buf598, buf592, buf589, buf583, buf580, buf574, buf570, buf549, buf547, buf542, buf539, buf533, buf529, buf508, buf505, buf500, buf497, buf491, buf487, buf466, buf464, buf459, buf456, buf450, buf446, buf425, buf422, buf416, buf413, buf407, buf403, buf382, buf379, buf374, buf371, buf365, buf361, buf341, buf338, buf333, buf330, buf324, buf320, buf299, buf297, buf292, buf289, buf283, buf279, buf258, buf255, buf250, buf247, buf241, buf237, buf217, buf214, buf208, buf205, buf199, buf195, buf175, buf172, buf167, buf164, buf158, buf154, buf135, buf132, buf127, buf124, buf118, buf114, buf94, buf92, buf87, buf84, buf78, buf74, buf54, buf51, buf46, buf43, buf667, buf659, buf650, buf641, buf632, buf623, buf614, buf605, buf596, buf587, buf578, buf568, buf565, buf564, buf561, buf560, buf557, buf553, buf546, buf537, buf527, buf524, buf523, buf520, buf519, buf516, buf512, buf504, buf495, buf485, buf482, buf481, buf478, buf477, buf474, buf470, buf463, buf454, buf444, buf441, buf440, buf437, buf436, buf433, buf429, buf420, buf411, buf401, buf398, buf397, buf394, buf393, buf390, buf386, buf378, buf369, buf359, buf356, buf355, buf352, buf351, buf348, buf344, buf337, buf328, buf318, buf315, buf314, buf311, buf310, buf307, buf303, buf296, buf287, buf277, buf274, buf273, buf270, buf269, buf266, buf262, buf254, buf245, buf235, buf232, buf231, buf228, buf227, buf224, buf221, buf212, buf203, buf193, buf190, buf189, buf186, buf185, buf182, buf179, buf171, buf162, buf152, buf149, buf148, buf145, buf144, buf141, buf138, buf131, buf122, buf112, buf109, buf108, buf105, buf104, buf101, buf98, buf91, buf82, buf72, buf69, buf68, buf65, buf64, buf61, buf58, buf50, reinterpret_tensor(buf40, (1000, 1280), (1280, 1), 0), reinterpret_tensor(buf41, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((27, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((38, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((50, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((61, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((84, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((95, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((106, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((117, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((140, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((151, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((174, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((185, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((27, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((162, 27, 1, 1), (27, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((162, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((38, 162, 1, 1), (162, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((228, 38, 1, 1), (38, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((228, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((19, 228, 1, 1), (228, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((19, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((228, 19, 1, 1), (19, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((50, 228, 1, 1), (228, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((300, 50, 1, 1), (50, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((300, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((25, 300, 1, 1), (300, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((25, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((300, 25, 1, 1), (25, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((61, 300, 1, 1), (300, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((366, 61, 1, 1), (61, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((366, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((30, 366, 1, 1), (366, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((30, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((366, 30, 1, 1), (30, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((72, 366, 1, 1), (366, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((432, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((36, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((36, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((432, 36, 1, 1), (36, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((84, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((504, 84, 1, 1), (84, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((504, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((42, 504, 1, 1), (504, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((504, 42, 1, 1), (42, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((95, 504, 1, 1), (504, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((570, 95, 1, 1), (95, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((570, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((47, 570, 1, 1), (570, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((47, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((570, 47, 1, 1), (47, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((106, 570, 1, 1), (570, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((636, 106, 1, 1), (106, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((636, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((53, 636, 1, 1), (636, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((53, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((636, 53, 1, 1), (53, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((117, 636, 1, 1), (636, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((702, 117, 1, 1), (117, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((702, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((58, 702, 1, 1), (702, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((702, 58, 1, 1), (58, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((128, 702, 1, 1), (702, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((768, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((64, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((768, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((140, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((840, 140, 1, 1), (140, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((840, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((70, 840, 1, 1), (840, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((840, 70, 1, 1), (70, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((151, 840, 1, 1), (840, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((906, 151, 1, 1), (151, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((906, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((75, 906, 1, 1), (906, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((906, 75, 1, 1), (75, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((162, 906, 1, 1), (906, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((972, 162, 1, 1), (162, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((972, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((81, 972, 1, 1), (972, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((81, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((972, 81, 1, 1), (81, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((174, 972, 1, 1), (972, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((1044, 174, 1, 1), (174, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((1044, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((87, 1044, 1, 1), (1044, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((87, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((1044, 87, 1, 1), (87, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((185, 1044, 1, 1), (1044, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((1280, 185, 1, 1), (185, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_7 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_14 = rand_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((8, 96, 112, 112), (1204224, 1, 10752, 96), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_29 = rand_strided((8, 96, 112, 112), (1204224, 1, 10752, 96), device='cuda:0', dtype=torch.float32)
    convolution_4 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_1 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.float32)
    convolution_5 = rand_strided((8, 27, 56, 56), (84672, 1, 1512, 27), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((27, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_29 = rand_strided((8, 27, 56, 56), (84672, 1, 1512, 27), device='cuda:0', dtype=torch.float32)
    convolution_6 = rand_strided((8, 162, 56, 56), (508032, 1, 9072, 162), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_51 = rand_strided((8, 162, 56, 56), (508032, 1, 9072, 162), device='cuda:0', dtype=torch.float32)
    convolution_7 = rand_strided((8, 162, 56, 56), (508032, 1, 9072, 162), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    clamp_max_2 = rand_strided((8, 162, 56, 56), (508032, 1, 9072, 162), device='cuda:0', dtype=torch.float32)
    convolution_8 = rand_strided((8, 38, 56, 56), (119168, 1, 2128, 38), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((38, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat = rand_strided((8, 38, 56, 56), (119168, 1, 2128, 38), device='cuda:0', dtype=torch.float32)
    convolution_9 = rand_strided((8, 228, 56, 56), (715008, 1, 12768, 228), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_73 = rand_strided((8, 228, 56, 56), (715008, 1, 12768, 228), device='cuda:0', dtype=torch.float32)
    convolution_10 = rand_strided((8, 228, 28, 28), (178752, 1, 6384, 228), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((228, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_55 = rand_strided((8, 228, 28, 28), (178752, 1, 6384, 228), device='cuda:0', dtype=torch.float32)
    mean = rand_strided((8, 228, 1, 1), (228, 1, 228, 228), device='cuda:0', dtype=torch.float32)
    convolution_11 = rand_strided((8, 19, 1, 1), (19, 1, 19, 19), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((8, 19, 1, 1), (19, 1, 19, 19), device='cuda:0', dtype=torch.float32)
    convolution_12 = rand_strided((8, 228, 1, 1), (228, 1, 228, 228), device='cuda:0', dtype=torch.float32)
    clamp_max_3 = rand_strided((8, 228, 28, 28), (178752, 1, 6384, 228), device='cuda:0', dtype=torch.float32)
    convolution_13 = rand_strided((8, 50, 28, 28), (39200, 1, 1400, 50), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((50, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_65 = rand_strided((8, 50, 28, 28), (39200, 1, 1400, 50), device='cuda:0', dtype=torch.float32)
    convolution_14 = rand_strided((8, 300, 28, 28), (235200, 1, 8400, 300), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_103 = rand_strided((8, 300, 28, 28), (235200, 1, 8400, 300), device='cuda:0', dtype=torch.float32)
    convolution_15 = rand_strided((8, 300, 28, 28), (235200, 1, 8400, 300), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((300, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_75 = rand_strided((8, 300, 28, 28), (235200, 1, 8400, 300), device='cuda:0', dtype=torch.float32)
    mean_1 = rand_strided((8, 300, 1, 1), (300, 1, 300, 300), device='cuda:0', dtype=torch.float32)
    convolution_16 = rand_strided((8, 25, 1, 1), (25, 1, 25, 25), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((8, 25, 1, 1), (25, 1, 25, 25), device='cuda:0', dtype=torch.float32)
    convolution_17 = rand_strided((8, 300, 1, 1), (300, 1, 300, 300), device='cuda:0', dtype=torch.float32)
    clamp_max_4 = rand_strided((8, 300, 28, 28), (235200, 1, 8400, 300), device='cuda:0', dtype=torch.float32)
    convolution_18 = rand_strided((8, 61, 28, 28), (47824, 1, 1708, 61), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((61, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_1 = rand_strided((8, 61, 28, 28), (47824, 1, 1708, 61), device='cuda:0', dtype=torch.float32)
    convolution_19 = rand_strided((8, 366, 28, 28), (286944, 1, 10248, 366), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_133 = rand_strided((8, 366, 28, 28), (286944, 1, 10248, 366), device='cuda:0', dtype=torch.float32)
    convolution_20 = rand_strided((8, 366, 14, 14), (71736, 1, 5124, 366), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((366, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_96 = rand_strided((8, 366, 14, 14), (71736, 1, 5124, 366), device='cuda:0', dtype=torch.float32)
    mean_2 = rand_strided((8, 366, 1, 1), (366, 1, 366, 366), device='cuda:0', dtype=torch.float32)
    convolution_21 = rand_strided((8, 30, 1, 1), (30, 1, 30, 30), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((8, 30, 1, 1), (30, 1, 30, 30), device='cuda:0', dtype=torch.float32)
    convolution_22 = rand_strided((8, 366, 1, 1), (366, 1, 366, 366), device='cuda:0', dtype=torch.float32)
    clamp_max_5 = rand_strided((8, 366, 14, 14), (71736, 1, 5124, 366), device='cuda:0', dtype=torch.float32)
    convolution_23 = rand_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_106 = rand_strided((8, 72, 14, 14), (14112, 1, 1008, 72), device='cuda:0', dtype=torch.float32)
    convolution_24 = rand_strided((8, 432, 14, 14), (84672, 1, 6048, 432), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_163 = rand_strided((8, 432, 14, 14), (84672, 1, 6048, 432), device='cuda:0', dtype=torch.float32)
    convolution_25 = rand_strided((8, 432, 14, 14), (84672, 1, 6048, 432), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_116 = rand_strided((8, 432, 14, 14), (84672, 1, 6048, 432), device='cuda:0', dtype=torch.float32)
    mean_3 = rand_strided((8, 432, 1, 1), (432, 1, 432, 432), device='cuda:0', dtype=torch.float32)
    convolution_26 = rand_strided((8, 36, 1, 1), (36, 1, 36, 36), device='cuda:0', dtype=torch.float32)
    relu_3 = rand_strided((8, 36, 1, 1), (36, 1, 36, 36), device='cuda:0', dtype=torch.float32)
    convolution_27 = rand_strided((8, 432, 1, 1), (432, 1, 432, 432), device='cuda:0', dtype=torch.float32)
    clamp_max_6 = rand_strided((8, 432, 14, 14), (84672, 1, 6048, 432), device='cuda:0', dtype=torch.float32)
    convolution_28 = rand_strided((8, 84, 14, 14), (16464, 1, 1176, 84), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((84, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_2 = rand_strided((8, 84, 14, 14), (16464, 1, 1176, 84), device='cuda:0', dtype=torch.float32)
    convolution_29 = rand_strided((8, 504, 14, 14), (98784, 1, 7056, 504), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_193 = rand_strided((8, 504, 14, 14), (98784, 1, 7056, 504), device='cuda:0', dtype=torch.float32)
    convolution_30 = rand_strided((8, 504, 14, 14), (98784, 1, 7056, 504), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_137 = rand_strided((8, 504, 14, 14), (98784, 1, 7056, 504), device='cuda:0', dtype=torch.float32)
    mean_4 = rand_strided((8, 504, 1, 1), (504, 1, 504, 504), device='cuda:0', dtype=torch.float32)
    convolution_31 = rand_strided((8, 42, 1, 1), (42, 1, 42, 42), device='cuda:0', dtype=torch.float32)
    relu_4 = rand_strided((8, 42, 1, 1), (42, 1, 42, 42), device='cuda:0', dtype=torch.float32)
    convolution_32 = rand_strided((8, 504, 1, 1), (504, 1, 504, 504), device='cuda:0', dtype=torch.float32)
    clamp_max_7 = rand_strided((8, 504, 14, 14), (98784, 1, 7056, 504), device='cuda:0', dtype=torch.float32)
    convolution_33 = rand_strided((8, 95, 14, 14), (18620, 1, 1330, 95), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((95, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_3 = rand_strided((8, 95, 14, 14), (18620, 1, 1330, 95), device='cuda:0', dtype=torch.float32)
    convolution_34 = rand_strided((8, 570, 14, 14), (111720, 1, 7980, 570), device='cuda:0', dtype=torch.float32)
    squeeze_88 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_223 = rand_strided((8, 570, 14, 14), (111720, 1, 7980, 570), device='cuda:0', dtype=torch.float32)
    convolution_35 = rand_strided((8, 570, 14, 14), (111720, 1, 7980, 570), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((570, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_158 = rand_strided((8, 570, 14, 14), (111720, 1, 7980, 570), device='cuda:0', dtype=torch.float32)
    mean_5 = rand_strided((8, 570, 1, 1), (570, 1, 570, 570), device='cuda:0', dtype=torch.float32)
    convolution_36 = rand_strided((8, 47, 1, 1), (47, 1, 47, 47), device='cuda:0', dtype=torch.float32)
    relu_5 = rand_strided((8, 47, 1, 1), (47, 1, 47, 47), device='cuda:0', dtype=torch.float32)
    convolution_37 = rand_strided((8, 570, 1, 1), (570, 1, 570, 570), device='cuda:0', dtype=torch.float32)
    clamp_max_8 = rand_strided((8, 570, 14, 14), (111720, 1, 7980, 570), device='cuda:0', dtype=torch.float32)
    convolution_38 = rand_strided((8, 106, 14, 14), (20776, 1, 1484, 106), device='cuda:0', dtype=torch.float32)
    squeeze_97 = rand_strided((106, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_4 = rand_strided((8, 106, 14, 14), (20776, 1, 1484, 106), device='cuda:0', dtype=torch.float32)
    convolution_39 = rand_strided((8, 636, 14, 14), (124656, 1, 8904, 636), device='cuda:0', dtype=torch.float32)
    squeeze_100 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_253 = rand_strided((8, 636, 14, 14), (124656, 1, 8904, 636), device='cuda:0', dtype=torch.float32)
    convolution_40 = rand_strided((8, 636, 14, 14), (124656, 1, 8904, 636), device='cuda:0', dtype=torch.float32)
    squeeze_103 = rand_strided((636, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_179 = rand_strided((8, 636, 14, 14), (124656, 1, 8904, 636), device='cuda:0', dtype=torch.float32)
    mean_6 = rand_strided((8, 636, 1, 1), (636, 1, 636, 636), device='cuda:0', dtype=torch.float32)
    convolution_41 = rand_strided((8, 53, 1, 1), (53, 1, 53, 53), device='cuda:0', dtype=torch.float32)
    relu_6 = rand_strided((8, 53, 1, 1), (53, 1, 53, 53), device='cuda:0', dtype=torch.float32)
    convolution_42 = rand_strided((8, 636, 1, 1), (636, 1, 636, 636), device='cuda:0', dtype=torch.float32)
    clamp_max_9 = rand_strided((8, 636, 14, 14), (124656, 1, 8904, 636), device='cuda:0', dtype=torch.float32)
    convolution_43 = rand_strided((8, 117, 14, 14), (22932, 1, 1638, 117), device='cuda:0', dtype=torch.float32)
    squeeze_109 = rand_strided((117, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_5 = rand_strided((8, 117, 14, 14), (22932, 1, 1638, 117), device='cuda:0', dtype=torch.float32)
    convolution_44 = rand_strided((8, 702, 14, 14), (137592, 1, 9828, 702), device='cuda:0', dtype=torch.float32)
    squeeze_112 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_283 = rand_strided((8, 702, 14, 14), (137592, 1, 9828, 702), device='cuda:0', dtype=torch.float32)
    convolution_45 = rand_strided((8, 702, 14, 14), (137592, 1, 9828, 702), device='cuda:0', dtype=torch.float32)
    squeeze_115 = rand_strided((702, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_200 = rand_strided((8, 702, 14, 14), (137592, 1, 9828, 702), device='cuda:0', dtype=torch.float32)
    mean_7 = rand_strided((8, 702, 1, 1), (702, 1, 702, 702), device='cuda:0', dtype=torch.float32)
    convolution_46 = rand_strided((8, 58, 1, 1), (58, 1, 58, 58), device='cuda:0', dtype=torch.float32)
    relu_7 = rand_strided((8, 58, 1, 1), (58, 1, 58, 58), device='cuda:0', dtype=torch.float32)
    convolution_47 = rand_strided((8, 702, 1, 1), (702, 1, 702, 702), device='cuda:0', dtype=torch.float32)
    clamp_max_10 = rand_strided((8, 702, 14, 14), (137592, 1, 9828, 702), device='cuda:0', dtype=torch.float32)
    convolution_48 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    squeeze_121 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_6 = rand_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cuda:0', dtype=torch.float32)
    convolution_49 = rand_strided((8, 768, 14, 14), (150528, 1, 10752, 768), device='cuda:0', dtype=torch.float32)
    squeeze_124 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_313 = rand_strided((8, 768, 14, 14), (150528, 1, 10752, 768), device='cuda:0', dtype=torch.float32)
    convolution_50 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
    squeeze_127 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_221 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
    mean_8 = rand_strided((8, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    convolution_51 = rand_strided((8, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    relu_8 = rand_strided((8, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    convolution_52 = rand_strided((8, 768, 1, 1), (768, 1, 768, 768), device='cuda:0', dtype=torch.float32)
    clamp_max_11 = rand_strided((8, 768, 7, 7), (37632, 1, 5376, 768), device='cuda:0', dtype=torch.float32)
    convolution_53 = rand_strided((8, 140, 7, 7), (6860, 1, 980, 140), device='cuda:0', dtype=torch.float32)
    squeeze_133 = rand_strided((140, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_231 = rand_strided((8, 140, 7, 7), (6860, 1, 980, 140), device='cuda:0', dtype=torch.float32)
    convolution_54 = rand_strided((8, 840, 7, 7), (41160, 1, 5880, 840), device='cuda:0', dtype=torch.float32)
    squeeze_136 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_343 = rand_strided((8, 840, 7, 7), (41160, 1, 5880, 840), device='cuda:0', dtype=torch.float32)
    convolution_55 = rand_strided((8, 840, 7, 7), (41160, 1, 5880, 840), device='cuda:0', dtype=torch.float32)
    squeeze_139 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_241 = rand_strided((8, 840, 7, 7), (41160, 1, 5880, 840), device='cuda:0', dtype=torch.float32)
    mean_9 = rand_strided((8, 840, 1, 1), (840, 1, 840, 840), device='cuda:0', dtype=torch.float32)
    convolution_56 = rand_strided((8, 70, 1, 1), (70, 1, 70, 70), device='cuda:0', dtype=torch.float32)
    relu_9 = rand_strided((8, 70, 1, 1), (70, 1, 70, 70), device='cuda:0', dtype=torch.float32)
    convolution_57 = rand_strided((8, 840, 1, 1), (840, 1, 840, 840), device='cuda:0', dtype=torch.float32)
    clamp_max_12 = rand_strided((8, 840, 7, 7), (41160, 1, 5880, 840), device='cuda:0', dtype=torch.float32)
    convolution_58 = rand_strided((8, 151, 7, 7), (7399, 1, 1057, 151), device='cuda:0', dtype=torch.float32)
    squeeze_145 = rand_strided((151, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_7 = rand_strided((8, 151, 7, 7), (7399, 1, 1057, 151), device='cuda:0', dtype=torch.float32)
    convolution_59 = rand_strided((8, 906, 7, 7), (44394, 1, 6342, 906), device='cuda:0', dtype=torch.float32)
    squeeze_148 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_373 = rand_strided((8, 906, 7, 7), (44394, 1, 6342, 906), device='cuda:0', dtype=torch.float32)
    convolution_60 = rand_strided((8, 906, 7, 7), (44394, 1, 6342, 906), device='cuda:0', dtype=torch.float32)
    squeeze_151 = rand_strided((906, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_262 = rand_strided((8, 906, 7, 7), (44394, 1, 6342, 906), device='cuda:0', dtype=torch.float32)
    mean_10 = rand_strided((8, 906, 1, 1), (906, 1, 906, 906), device='cuda:0', dtype=torch.float32)
    convolution_61 = rand_strided((8, 75, 1, 1), (75, 1, 75, 75), device='cuda:0', dtype=torch.float32)
    relu_10 = rand_strided((8, 75, 1, 1), (75, 1, 75, 75), device='cuda:0', dtype=torch.float32)
    convolution_62 = rand_strided((8, 906, 1, 1), (906, 1, 906, 906), device='cuda:0', dtype=torch.float32)
    clamp_max_13 = rand_strided((8, 906, 7, 7), (44394, 1, 6342, 906), device='cuda:0', dtype=torch.float32)
    convolution_63 = rand_strided((8, 162, 7, 7), (7938, 1, 1134, 162), device='cuda:0', dtype=torch.float32)
    squeeze_157 = rand_strided((162, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_8 = rand_strided((8, 162, 7, 7), (7938, 1, 1134, 162), device='cuda:0', dtype=torch.float32)
    convolution_64 = rand_strided((8, 972, 7, 7), (47628, 1, 6804, 972), device='cuda:0', dtype=torch.float32)
    squeeze_160 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_403 = rand_strided((8, 972, 7, 7), (47628, 1, 6804, 972), device='cuda:0', dtype=torch.float32)
    convolution_65 = rand_strided((8, 972, 7, 7), (47628, 1, 6804, 972), device='cuda:0', dtype=torch.float32)
    squeeze_163 = rand_strided((972, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_283 = rand_strided((8, 972, 7, 7), (47628, 1, 6804, 972), device='cuda:0', dtype=torch.float32)
    mean_11 = rand_strided((8, 972, 1, 1), (972, 1, 972, 972), device='cuda:0', dtype=torch.float32)
    convolution_66 = rand_strided((8, 81, 1, 1), (81, 1, 81, 81), device='cuda:0', dtype=torch.float32)
    relu_11 = rand_strided((8, 81, 1, 1), (81, 1, 81, 81), device='cuda:0', dtype=torch.float32)
    convolution_67 = rand_strided((8, 972, 1, 1), (972, 1, 972, 972), device='cuda:0', dtype=torch.float32)
    clamp_max_14 = rand_strided((8, 972, 7, 7), (47628, 1, 6804, 972), device='cuda:0', dtype=torch.float32)
    convolution_68 = rand_strided((8, 174, 7, 7), (8526, 1, 1218, 174), device='cuda:0', dtype=torch.float32)
    squeeze_169 = rand_strided((174, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_9 = rand_strided((8, 174, 7, 7), (8526, 1, 1218, 174), device='cuda:0', dtype=torch.float32)
    convolution_69 = rand_strided((8, 1044, 7, 7), (51156, 1, 7308, 1044), device='cuda:0', dtype=torch.float32)
    squeeze_172 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    mul_433 = rand_strided((8, 1044, 7, 7), (51156, 1, 7308, 1044), device='cuda:0', dtype=torch.float32)
    convolution_70 = rand_strided((8, 1044, 7, 7), (51156, 1, 7308, 1044), device='cuda:0', dtype=torch.float32)
    squeeze_175 = rand_strided((1044, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_304 = rand_strided((8, 1044, 7, 7), (51156, 1, 7308, 1044), device='cuda:0', dtype=torch.float32)
    mean_12 = rand_strided((8, 1044, 1, 1), (1044, 1, 1044, 1044), device='cuda:0', dtype=torch.float32)
    convolution_71 = rand_strided((8, 87, 1, 1), (87, 1, 87, 87), device='cuda:0', dtype=torch.float32)
    relu_12 = rand_strided((8, 87, 1, 1), (87, 1, 87, 87), device='cuda:0', dtype=torch.float32)
    convolution_72 = rand_strided((8, 1044, 1, 1), (1044, 1, 1044, 1044), device='cuda:0', dtype=torch.float32)
    clamp_max_15 = rand_strided((8, 1044, 7, 7), (51156, 1, 7308, 1044), device='cuda:0', dtype=torch.float32)
    convolution_73 = rand_strided((8, 185, 7, 7), (9065, 1, 1295, 185), device='cuda:0', dtype=torch.float32)
    squeeze_181 = rand_strided((185, ), (1, ), device='cuda:0', dtype=torch.float32)
    cat_10 = rand_strided((8, 185, 7, 7), (9065, 1, 1295, 185), device='cuda:0', dtype=torch.float32)
    convolution_74 = rand_strided((8, 1280, 7, 7), (62720, 1, 8960, 1280), device='cuda:0', dtype=torch.float32)
    squeeze_184 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    clone_17 = rand_strided((8, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    mul_465 = rand_strided((8, 1280, 7, 7), (62720, 1, 8960, 1280), device='cuda:0', dtype=torch.float32)
    unsqueeze_250 = rand_strided((1, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_262 = rand_strided((1, 185, 1, 1), (185, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_286 = rand_strided((1, 1044, 1, 1), (1044, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_508 = rand_strided((8, 1044, 7, 7), (51156, 1, 7308, 1044), device='cuda:0', dtype=torch.float32)
    unsqueeze_298 = rand_strided((1, 1044, 1, 1), (1044, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_310 = rand_strided((1, 174, 1, 1), (174, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_334 = rand_strided((1, 972, 1, 1), (972, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_551 = rand_strided((8, 972, 7, 7), (47628, 1, 6804, 972), device='cuda:0', dtype=torch.float32)
    unsqueeze_346 = rand_strided((1, 972, 1, 1), (972, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_358 = rand_strided((1, 162, 1, 1), (162, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_382 = rand_strided((1, 906, 1, 1), (906, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_594 = rand_strided((8, 906, 7, 7), (44394, 1, 6342, 906), device='cuda:0', dtype=torch.float32)
    unsqueeze_394 = rand_strided((1, 906, 1, 1), (906, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_406 = rand_strided((1, 151, 1, 1), (151, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_430 = rand_strided((1, 840, 1, 1), (840, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_637 = rand_strided((8, 840, 7, 7), (41160, 1, 5880, 840), device='cuda:0', dtype=torch.float32)
    unsqueeze_442 = rand_strided((1, 840, 1, 1), (840, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_454 = rand_strided((1, 140, 1, 1), (140, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_478 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_680 = rand_strided((8, 768, 14, 14), (150528, 1, 10752, 768), device='cuda:0', dtype=torch.float32)
    unsqueeze_490 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_502 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_526 = rand_strided((1, 702, 1, 1), (702, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_723 = rand_strided((8, 702, 14, 14), (137592, 1, 9828, 702), device='cuda:0', dtype=torch.float32)
    unsqueeze_538 = rand_strided((1, 702, 1, 1), (702, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_550 = rand_strided((1, 117, 1, 1), (117, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_574 = rand_strided((1, 636, 1, 1), (636, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_766 = rand_strided((8, 636, 14, 14), (124656, 1, 8904, 636), device='cuda:0', dtype=torch.float32)
    unsqueeze_586 = rand_strided((1, 636, 1, 1), (636, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_598 = rand_strided((1, 106, 1, 1), (106, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_622 = rand_strided((1, 570, 1, 1), (570, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_809 = rand_strided((8, 570, 14, 14), (111720, 1, 7980, 570), device='cuda:0', dtype=torch.float32)
    unsqueeze_634 = rand_strided((1, 570, 1, 1), (570, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_646 = rand_strided((1, 95, 1, 1), (95, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_670 = rand_strided((1, 504, 1, 1), (504, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_852 = rand_strided((8, 504, 14, 14), (98784, 1, 7056, 504), device='cuda:0', dtype=torch.float32)
    unsqueeze_682 = rand_strided((1, 504, 1, 1), (504, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_694 = rand_strided((1, 84, 1, 1), (84, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_718 = rand_strided((1, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_895 = rand_strided((8, 432, 14, 14), (84672, 1, 6048, 432), device='cuda:0', dtype=torch.float32)
    unsqueeze_730 = rand_strided((1, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_742 = rand_strided((1, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_766 = rand_strided((1, 366, 1, 1), (366, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_938 = rand_strided((8, 366, 28, 28), (286944, 1, 10248, 366), device='cuda:0', dtype=torch.float32)
    unsqueeze_778 = rand_strided((1, 366, 1, 1), (366, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_790 = rand_strided((1, 61, 1, 1), (61, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_814 = rand_strided((1, 300, 1, 1), (300, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_981 = rand_strided((8, 300, 28, 28), (235200, 1, 8400, 300), device='cuda:0', dtype=torch.float32)
    unsqueeze_826 = rand_strided((1, 300, 1, 1), (300, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_838 = rand_strided((1, 50, 1, 1), (50, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_862 = rand_strided((1, 228, 1, 1), (228, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_1024 = rand_strided((8, 228, 56, 56), (715008, 1, 12768, 228), device='cuda:0', dtype=torch.float32)
    unsqueeze_874 = rand_strided((1, 228, 1, 1), (228, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_886 = rand_strided((1, 38, 1, 1), (38, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_13 = rand_strided((8, 162, 56, 56), (508032, 1, 9072, 162), device='cuda:0', dtype=torch.bool)
    unsqueeze_898 = rand_strided((1, 162, 1, 1), (162, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_1054 = rand_strided((8, 162, 56, 56), (508032, 1, 9072, 162), device='cuda:0', dtype=torch.float32)
    unsqueeze_910 = rand_strided((1, 162, 1, 1), (162, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_922 = rand_strided((1, 27, 1, 1), (27, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_14 = rand_strided((8, 96, 56, 56), (301056, 1, 5376, 96), device='cuda:0', dtype=torch.bool)
    unsqueeze_934 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_1084 = rand_strided((8, 96, 112, 112), (1204224, 1, 10752, 96), device='cuda:0', dtype=torch.float32)
    unsqueeze_946 = rand_strided((1, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_958 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_or_15 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.bool)
    unsqueeze_970 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_1114 = rand_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda:0', dtype=torch.float32)
    unsqueeze_982 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_112, primals_114, primals_116, primals_117, primals_118, primals_119, primals_121, primals_123, primals_125, primals_126, primals_127, primals_128, primals_130, primals_132, primals_134, primals_135, primals_136, primals_137, primals_139, primals_141, primals_143, primals_144, primals_145, primals_146, primals_148, primals_150, primals_152, primals_153, primals_154, primals_155, primals_157, primals_159, primals_161, primals_162, primals_163, primals_164, primals_166, primals_168, primals_170, primals_171, primals_172, primals_173, primals_175, primals_177, primals_179, primals_180, primals_181, primals_182, primals_184, primals_186, primals_188, primals_189, primals_190, primals_191, primals_193, primals_195, primals_197, primals_198, primals_199, primals_200, primals_202, primals_204, primals_206, primals_207, primals_208, primals_209, primals_211, primals_213, primals_215, primals_216, primals_217, primals_218, primals_220, primals_222, primals_224, primals_225, primals_414, convolution, squeeze_1, mul_7, convolution_1, squeeze_4, clamp_max, convolution_2, squeeze_7, add_14, convolution_3, squeeze_10, mul_29, convolution_4, squeeze_13, clamp_max_1, convolution_5, squeeze_16, add_29, convolution_6, squeeze_19, mul_51, convolution_7, squeeze_22, clamp_max_2, convolution_8, squeeze_25, cat, convolution_9, squeeze_28, mul_73, convolution_10, squeeze_31, add_55, mean, convolution_11, relu, convolution_12, clamp_max_3, convolution_13, squeeze_37, add_65, convolution_14, squeeze_40, mul_103, convolution_15, squeeze_43, add_75, mean_1, convolution_16, relu_1, convolution_17, clamp_max_4, convolution_18, squeeze_49, cat_1, convolution_19, squeeze_52, mul_133, convolution_20, squeeze_55, add_96, mean_2, convolution_21, relu_2, convolution_22, clamp_max_5, convolution_23, squeeze_61, add_106, convolution_24, squeeze_64, mul_163, convolution_25, squeeze_67, add_116, mean_3, convolution_26, relu_3, convolution_27, clamp_max_6, convolution_28, squeeze_73, cat_2, convolution_29, squeeze_76, mul_193, convolution_30, squeeze_79, add_137, mean_4, convolution_31, relu_4, convolution_32, clamp_max_7, convolution_33, squeeze_85, cat_3, convolution_34, squeeze_88, mul_223, convolution_35, squeeze_91, add_158, mean_5, convolution_36, relu_5, convolution_37, clamp_max_8, convolution_38, squeeze_97, cat_4, convolution_39, squeeze_100, mul_253, convolution_40, squeeze_103, add_179, mean_6, convolution_41, relu_6, convolution_42, clamp_max_9, convolution_43, squeeze_109, cat_5, convolution_44, squeeze_112, mul_283, convolution_45, squeeze_115, add_200, mean_7, convolution_46, relu_7, convolution_47, clamp_max_10, convolution_48, squeeze_121, cat_6, convolution_49, squeeze_124, mul_313, convolution_50, squeeze_127, add_221, mean_8, convolution_51, relu_8, convolution_52, clamp_max_11, convolution_53, squeeze_133, add_231, convolution_54, squeeze_136, mul_343, convolution_55, squeeze_139, add_241, mean_9, convolution_56, relu_9, convolution_57, clamp_max_12, convolution_58, squeeze_145, cat_7, convolution_59, squeeze_148, mul_373, convolution_60, squeeze_151, add_262, mean_10, convolution_61, relu_10, convolution_62, clamp_max_13, convolution_63, squeeze_157, cat_8, convolution_64, squeeze_160, mul_403, convolution_65, squeeze_163, add_283, mean_11, convolution_66, relu_11, convolution_67, clamp_max_14, convolution_68, squeeze_169, cat_9, convolution_69, squeeze_172, mul_433, convolution_70, squeeze_175, add_304, mean_12, convolution_71, relu_12, convolution_72, clamp_max_15, convolution_73, squeeze_181, cat_10, convolution_74, squeeze_184, clone_17, permute_1, mul_465, unsqueeze_250, unsqueeze_262, unsqueeze_286, mul_508, unsqueeze_298, unsqueeze_310, unsqueeze_334, mul_551, unsqueeze_346, unsqueeze_358, unsqueeze_382, mul_594, unsqueeze_394, unsqueeze_406, unsqueeze_430, mul_637, unsqueeze_442, unsqueeze_454, unsqueeze_478, mul_680, unsqueeze_490, unsqueeze_502, unsqueeze_526, mul_723, unsqueeze_538, unsqueeze_550, unsqueeze_574, mul_766, unsqueeze_586, unsqueeze_598, unsqueeze_622, mul_809, unsqueeze_634, unsqueeze_646, unsqueeze_670, mul_852, unsqueeze_682, unsqueeze_694, unsqueeze_718, mul_895, unsqueeze_730, unsqueeze_742, unsqueeze_766, mul_938, unsqueeze_778, unsqueeze_790, unsqueeze_814, mul_981, unsqueeze_826, unsqueeze_838, unsqueeze_862, mul_1024, unsqueeze_874, unsqueeze_886, bitwise_or_13, unsqueeze_898, mul_1054, unsqueeze_910, unsqueeze_922, bitwise_or_14, unsqueeze_934, mul_1084, unsqueeze_946, unsqueeze_958, bitwise_or_15, unsqueeze_970, mul_1114, unsqueeze_982, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('rexnet_100', benchmark_compiled_module)
