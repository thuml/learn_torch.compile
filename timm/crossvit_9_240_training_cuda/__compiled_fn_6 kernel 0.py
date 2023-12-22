
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


# kernel path: /tmp/torchinductor_youkaichao/px/cpxms57nlnk34hkfu4pf3p46vbqir3aph4mqu6jat6nsbuxo6w6u.py
# Source Nodes: [], Original ATen: [aten.div, aten.select]

triton_poi_fused_div_select_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_select_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 2.0
    tmp2 = tmp0 / tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tl.store(out_ptr1 + (x0), tmp2, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/od/coddoiqm2f76oxxdljt4qt65v3naxmjftdz6atkmmxip7twqqwfk.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_1', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/6j/c6j5acjbkq6qiwpyry2e26hjdghlawptrbbenvvhqdwcu3xyiqop.py
# Source Nodes: [x_172], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward]
# x_172 => mul_246, sub_95
triton_red_fused_native_layer_norm_native_layer_norm_backward_select_backward_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_select_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1576
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 197
    x1 = (xindex // 197)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp12 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp3 = tl.load(in_ptr0 + (r2 + (256*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr2 + (r2 + (256*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp4 = 0.0
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp7 = tmp5 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp13 = tmp11 - tmp12
        tmp15 = tmp13 * tmp14
        tmp16 = tmp7 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp25 = tl.load(in_ptr0 + (r2 + (256*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp32 = tl.load(in_ptr2 + (r2 + (256*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = 256.0
        tmp21 = tmp14 / tmp20
        tmp22 = x0
        tmp23 = tl.full([1, 1], 0, tl.int32)
        tmp24 = tmp22 == tmp23
        tmp26 = 0.0
        tmp27 = tl.where(tmp24, tmp25, tmp26)
        tmp29 = tmp27 * tmp28
        tmp30 = tmp29 * tmp20
        tmp31 = tmp30 - tmp9
        tmp33 = tmp32 - tmp12
        tmp34 = tmp33 * tmp14
        tmp35 = tmp34 * tmp18
        tmp36 = tmp31 - tmp35
        tmp37 = tmp21 * tmp36
        tl.store(out_ptr2 + (r2 + (256*x3)), tmp37, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o7/co7qpnc7okm5xqvvtpcazvtlpvijvjhmbzt5k4mz4urykf36kf5t.py
# Source Nodes: [x_172], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward]
# x_172 => mul_246, sub_95
triton_red_fused_native_layer_norm_native_layer_norm_backward_select_backward_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_select_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3328
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 256)
    x0 = xindex % 256
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 1576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = (r2 + (122*x1)) % 197
        tmp4 = tl.full([1, 1], 0, tl.int32)
        tmp5 = tmp3 == tmp4
        tmp6 = tl.load(in_ptr0 + (x0 + (256*(((r2 + (122*x1)) // 197) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = 0.0
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = tl.load(in_ptr1 + (x0 + (256*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr2 + ((r2 + (122*x1)) % 1576), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 - tmp10
        tmp12 = tl.load(in_ptr3 + ((r2 + (122*x1)) % 1576), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 * tmp12
        tmp14 = tmp8 * tmp13
        tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
        tmp16 = tl.where(tmp2, tmp14, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gc/cgcxnyatgnot5mvs3ywad3333m54hirolwetf74g64jvynm63njv.py
# Source Nodes: [x_172], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward]
# x_172 => mul_246, sub_95
triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/35/c35k2vauuek7aed6fu5iadfwwzntvuqcusgfglhg3gji7o4rr4p2.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]

triton_red_fused_native_layer_norm_backward_select_backward_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_select_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 1576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 197
        r2 = (rindex // 197)
        tmp3 = tl.load(in_ptr0 + (x0 + (256*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = r1
        tmp1 = tl.full([1, 1], 0, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp4 = 0.0
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ge/cgevxd63ddjiyxdv6kitckizurilaamp4cgsrd7rmt5jrity2nx7.py
# Source Nodes: [x_171], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward]
# x_171 => mul_244, sub_94
triton_red_fused_native_layer_norm_native_layer_norm_backward_select_backward_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_select_backward_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3208
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 401
    x1 = (xindex // 401)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp12 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp3 = tl.load(in_ptr0 + (r2 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr2 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp4 = 0.0
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp7 = tmp5 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp13 = tmp11 - tmp12
        tmp15 = tmp13 * tmp14
        tmp16 = tmp7 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp25 = tl.load(in_ptr0 + (r2 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp32 = tl.load(in_ptr2 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = 128.0
        tmp21 = tmp14 / tmp20
        tmp22 = x0
        tmp23 = tl.full([1, 1], 0, tl.int32)
        tmp24 = tmp22 == tmp23
        tmp26 = 0.0
        tmp27 = tl.where(tmp24, tmp25, tmp26)
        tmp29 = tmp27 * tmp28
        tmp30 = tmp29 * tmp20
        tmp31 = tmp30 - tmp9
        tmp33 = tmp32 - tmp12
        tmp34 = tmp33 * tmp14
        tmp35 = tmp34 * tmp18
        tmp36 = tmp31 - tmp35
        tmp37 = tmp21 * tmp36
        tl.store(out_ptr2 + (r2 + (128*x3)), tmp37, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6c/c6ccvpdpnbcarxkjvt66wzyhy5jrjvuhxc3p64hvighjg37ojrsd.py
# Source Nodes: [x_171], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward]
# x_171 => mul_244, sub_94
triton_red_fused_native_layer_norm_native_layer_norm_backward_select_backward_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_select_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3328
    rnumel = 124
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 128)
    x0 = xindex % 128
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (124*x1)
        tmp1 = tl.full([1, 1], 3208, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = (r2 + (124*x1)) % 401
        tmp4 = tl.full([1, 1], 0, tl.int32)
        tmp5 = tmp3 == tmp4
        tmp6 = tl.load(in_ptr0 + (x0 + (128*(((r2 + (124*x1)) // 401) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = 0.0
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = tl.load(in_ptr1 + (x0 + (128*((r2 + (124*x1)) % 3208))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr2 + ((r2 + (124*x1)) % 3208), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 - tmp10
        tmp12 = tl.load(in_ptr3 + ((r2 + (124*x1)) % 3208), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 * tmp12
        tmp14 = tmp8 * tmp13
        tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
        tmp16 = tl.where(tmp2, tmp14, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6c/c6c43who7pgph3mizsbyvciumwpk32k2tgwr4oprbeafyw4qzofa.py
# Source Nodes: [x_171], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward]
# x_171 => mul_244, sub_94
triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 26
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


# kernel path: /tmp/torchinductor_youkaichao/f2/cf2vxxlzaxnthvovlytycddzmy63iez5sd3rl3lwnqwecy53gflc.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]

triton_red_fused_native_layer_norm_backward_select_backward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 4096],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_select_backward_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 3208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 401
        r2 = (rindex // 401)
        tmp3 = tl.load(in_ptr0 + (x0 + (128*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = r1
        tmp1 = tl.full([1, 1], 0, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp4 = 0.0
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b6/cb6ap2im7lwrpqt7ltnectqbnpi7s44yoninetxskxkeatblhzrw.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_10', 'mutated_arg_names': []}
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
    tmp0 = tl.load(in_ptr0 + (x0 + (50432*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4v/c4v7zkx7rdliprclpg4lejmtwfbjnl2sunfvesr5jvbyqjcngeoe.py
# Source Nodes: [l__mod___blocks_2_revert_projs_1_0, l__mod___blocks_2_revert_projs_1_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
# l__mod___blocks_2_revert_projs_1_0 => add_190, mul_240
# l__mod___blocks_2_revert_projs_1_1 => add_191, erf_23, mul_242
triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = 0.7071067811865476
    tmp7 = tmp5 * tmp6
    tmp8 = tl.math.erf(tmp7)
    tmp9 = 1.0
    tmp10 = tmp8 + tmp9
    tmp11 = 0.5
    tmp12 = tmp10 * tmp11
    tmp13 = tmp5 * tmp5
    tmp14 = -0.5
    tmp15 = tmp13 * tmp14
    tmp16 = tl.exp(tmp15)
    tmp17 = 0.3989422804014327
    tmp18 = tmp16 * tmp17
    tmp19 = tmp5 * tmp18
    tmp20 = tmp12 + tmp19
    tmp21 = tmp0 * tmp20
    tmp22 = tmp21 * tmp2
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None]
    tmp27 = tmp22 * tmp1
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None]
    tmp33 = 128.0
    tmp34 = tmp22 * tmp33
    tmp35 = tmp34 - tmp26
    tmp36 = tmp1 * tmp31
    tmp37 = tmp35 - tmp36
    tmp38 = tmp32 * tmp37
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp38, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7b/c7bnqzaeac4qbidvopgjduvizcujrfzabdgom4qwoyqbhhn7nf5j.py
# Source Nodes: [l__mod___blocks_2_revert_projs_1_0, l__mod___blocks_2_revert_projs_1_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
# l__mod___blocks_2_revert_projs_1_0 => add_190, mul_240
# l__mod___blocks_2_revert_projs_1_1 => add_191, erf_23, mul_242
triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = 0.7071067811865476
    tmp7 = tmp5 * tmp6
    tmp8 = tl.math.erf(tmp7)
    tmp9 = 1.0
    tmp10 = tmp8 + tmp9
    tmp11 = 0.5
    tmp12 = tmp10 * tmp11
    tmp13 = tmp5 * tmp5
    tmp14 = -0.5
    tmp15 = tmp13 * tmp14
    tmp16 = tl.exp(tmp15)
    tmp17 = 0.3989422804014327
    tmp18 = tmp16 * tmp17
    tmp19 = tmp5 * tmp18
    tmp20 = tmp12 + tmp19
    tmp21 = tmp0 * tmp20
    tmp22 = tmp21 * tmp1
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None]
    tmp27 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp29 = tl.where(rmask & xmask, tmp27, 0)
    tmp30 = tl.sum(tmp29, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp26, xmask)
    tl.store(out_ptr1 + (x0), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tj/ctjwhezu3ow5o6hacujajtoconpdkk2zgqo52pgxwbulzam7xvhg.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_13', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/be/cbegmggbyl4rnfum3jxv5ru2d6gael52ogzad6chzkwvpoxzrvtt.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]

triton_per_fused__softmax_backward_data_mul_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_mul_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel):
    xnumel = 32
    XBLOCK: tl.constexpr = 1
    rnumel = 401
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (401*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (401*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp7 = tmp1 * tmp6
    tmp8 = tmp2 - tmp7
    tmp9 = 0.1767766952966369
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr1 + (r1 + (401*x0)), tmp10, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ag/cagkngn3sp2sm2gyjai62axqpi54iytizcweaby5fgrxw54dbfwm.py
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
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 410624
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((32*(x1 % 401)) + (12832*(x0 // 32)) + (51328*(x1 // 401)) + (x0 % 32)), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/67/c67vbdjr5ojsp4btsvchnelazkbwiyysvqdzudsbr4qkxw5ilqor.py
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
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3328
    rnumel = 124
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
        tmp0 = r2 + (124*x1)
        tmp1 = tl.full([1, 1], 3208, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (128*r2) + (15872*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ox/coxopree43nb6he4iajp7erony4orkrhp3bvluqfakn7jc65cyxo.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]

triton_poi_fused__unsafe_view_clone_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3208
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + ((401*x1) + (51328*(y0 // 401)) + (y0 % 401)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + (128*y0)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p6/cp6ruyovrepwznd6wyt7yj5lueuibglbobcnqec4juqdkilljjrx.py
# Source Nodes: [l__mod___blocks_2_revert_projs_0_0, l__mod___blocks_2_revert_projs_0_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
# l__mod___blocks_2_revert_projs_0_0 => add_183, mul_232
# l__mod___blocks_2_revert_projs_0_1 => add_184, erf_22, mul_234
triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 8
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
    tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = 0.7071067811865476
    tmp7 = tmp5 * tmp6
    tmp8 = tl.math.erf(tmp7)
    tmp9 = 1.0
    tmp10 = tmp8 + tmp9
    tmp11 = 0.5
    tmp12 = tmp10 * tmp11
    tmp13 = tmp5 * tmp5
    tmp14 = -0.5
    tmp15 = tmp13 * tmp14
    tmp16 = tl.exp(tmp15)
    tmp17 = 0.3989422804014327
    tmp18 = tmp16 * tmp17
    tmp19 = tmp5 * tmp18
    tmp20 = tmp12 + tmp19
    tmp21 = tmp0 * tmp20
    tmp22 = tmp21 * tmp2
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp27 = tmp22 * tmp1
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp33 = 256.0
    tmp34 = tmp22 * tmp33
    tmp35 = tmp34 - tmp26
    tmp36 = tmp1 * tmp31
    tmp37 = tmp35 - tmp36
    tmp38 = tmp32 * tmp37
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp38, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sv/csv4pgjypshjafygifop3pi7jcyugkbriera4oqhom7znnb33ayg.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 403456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*(x1 % 197)) + (12608*(x0 // 64)) + (50432*(x1 // 197)) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/d3/cd3re5y6rgr2oborwwaymncwbmfi4cszbnbgexa2calfm7xyauyt.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]

triton_per_fused__softmax_backward_data_mul_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_mul_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 197
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (197*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (197*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tmp1 * tmp6
    tmp8 = tmp2 - tmp7
    tmp9 = 0.125
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr1 + (r1 + (197*x0)), tmp10, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zx/czxjc6mw2lndldhvkoryw54hzhspabedpp525kuq3rmzblasxcnd.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]

triton_poi_fused__unsafe_view_clone_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1576
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + ((197*x1) + (50432*(y0 // 197)) + (y0 % 197)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + (256*y0)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5j/c5jhx6b47xyk55aomxmvclbykon3mz3hq7nhpfvogu2djsnryl47.py
# Source Nodes: [l__mod___blocks_2_fusion_0_norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
# l__mod___blocks_2_fusion_0_norm1 => mul_228, sub_88
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 197
    x1 = (xindex // 197)
    tmp0 = tl.load(in_ptr0 + (r2 + (256*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (256*x3)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr4 + (r2 + (256*x3)), rmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tl.load(in_ptr2 + (r2 + (256*x1)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = 0.0
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tmp11 = tmp2 + tmp10
    tmp13 = tmp11 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp20 = tmp18 - tmp19
    tmp22 = tmp20 * tmp21
    tmp23 = tmp13 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp26 = tl.where(rmask & xmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp28 = 256.0
    tmp29 = tmp13 * tmp28
    tmp30 = tmp29 - tmp17
    tmp31 = tmp22 * tmp27
    tmp32 = tmp30 - tmp31
    tmp33 = tmp21 / tmp28
    tmp34 = tmp33 * tmp32
    tmp35 = tl.load(in_ptr7 + (r2 + (256*x1)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp5, tmp35, tmp36)
    tmp38 = tl.where(tmp5, tmp37, tmp9)
    tmp39 = tmp34 + tmp38
    tl.store(out_ptr2 + (r2 + (256*x3)), tmp32, rmask & xmask)
    tl.store(out_ptr3 + (r2 + (256*x3)), tmp39, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/le/clee6kkzyilccdbdwdy47swx4nqx2mca45h5tp473n27swwlq5wj.py
# Source Nodes: [l__mod___blocks_2_projs_0_0, l__mod___blocks_2_projs_0_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
# l__mod___blocks_2_projs_0_0 => add_173, mul_219
# l__mod___blocks_2_projs_0_1 => add_174, erf_20, mul_221
triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = 0.7071067811865476
    tmp7 = tmp5 * tmp6
    tmp8 = tl.math.erf(tmp7)
    tmp9 = 1.0
    tmp10 = tmp8 + tmp9
    tmp11 = 0.5
    tmp12 = tmp10 * tmp11
    tmp13 = tmp5 * tmp5
    tmp14 = -0.5
    tmp15 = tmp13 * tmp14
    tmp16 = tl.exp(tmp15)
    tmp17 = 0.3989422804014327
    tmp18 = tmp16 * tmp17
    tmp19 = tmp5 * tmp18
    tmp20 = tmp12 + tmp19
    tmp21 = tmp0 * tmp20
    tmp22 = tmp21 * tmp2
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None]
    tmp27 = tmp22 * tmp1
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp26, xmask)
    tl.store(out_ptr1 + (x0), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mb/cmbt5yczwdkploh33tt62subbchs5ov3yy264bohv6igdtatt3h3.py
# Source Nodes: [l__mod___blocks_2_fusion_1_norm1, l__mod___blocks_2_projs_0_0, l__mod___blocks_2_projs_0_1], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
# l__mod___blocks_2_fusion_1_norm1 => mul_236, sub_91
# l__mod___blocks_2_projs_0_0 => add_173, mul_219
# l__mod___blocks_2_projs_0_1 => add_174, erf_20, mul_221
triton_per_fused_add_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: 'i32', 19: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(18, 19))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3208
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 401
    x1 = (xindex // 401)
    tmp0 = tl.load(in_ptr0 + (r2 + (128*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (128*x3)), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr4 + (r2 + (128*x3)), rmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = x0
    tmp4 = tl.full([1, 1], 1, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tl.load(in_ptr2 + (r2 + (128*x1)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = 0.0
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tmp11 = tmp2 + tmp10
    tmp13 = tmp11 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp20 = tmp18 - tmp19
    tmp22 = tmp20 * tmp21
    tmp23 = tmp13 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
    tmp26 = tl.where(rmask & xmask, tmp24, 0)
    tmp27 = tl.sum(tmp26, 1)[:, None]
    tmp28 = 128.0
    tmp29 = tmp13 * tmp28
    tmp30 = tmp29 - tmp17
    tmp31 = tmp22 * tmp27
    tmp32 = tmp30 - tmp31
    tmp33 = tmp21 / tmp28
    tmp34 = tmp33 * tmp32
    tmp35 = tl.load(in_ptr7 + (r2 + (128*x1)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp5, tmp35, tmp36)
    tmp38 = tl.where(tmp5, tmp37, tmp9)
    tmp39 = tmp34 + tmp38
    tmp40 = tmp3 >= tmp4
    tmp41 = tl.load(in_ptr6 + (tl.broadcast_to(x3, [XBLOCK, RBLOCK])), rmask & tmp40 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tmp41 / tmp28
    tmp43 = tmp42 * tmp32
    tmp44 = tl.broadcast_to(x0, [XBLOCK, RBLOCK])
    tmp45 = tmp44 < tmp4
    tmp46 = tmp45 & tmp40
    tmp47 = tl.load(in_ptr7 + (r2 + (128*x1)), rmask & tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tl.where(tmp45, tmp49, tmp9)
    tmp51 = tmp43 + tmp50
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp40, tmp51, tmp52)
    tmp54 = tl.where(tmp40, tmp53, tmp9)
    tmp55 = tl.load(in_ptr8 + (r2 + (128*x3)), rmask & tmp40 & xmask, other=0.0)
    tmp56 = tl.full(tmp55.shape, 0.0, tmp55.dtype)
    tmp57 = tl.where(tmp40, tmp55, tmp56)
    tmp58 = tl.where(tmp40, tmp57, tmp9)
    tmp59 = tmp54 + tmp58
    tmp60 = tl.load(in_ptr9 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp61 = tl.load(in_ptr10 + (r2 + (128*x1)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.load(in_ptr11 + (r2 + (128*x1)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp63 = tl.load(in_ptr12 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp64 = tmp62 * tmp63
    tmp65 = tl.load(in_ptr13 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp66 = tmp64 + tmp65
    tmp67 = 0.7071067811865476
    tmp68 = tmp66 * tmp67
    tmp69 = tl.math.erf(tmp68)
    tmp70 = 1.0
    tmp71 = tmp69 + tmp70
    tmp72 = 0.5
    tmp73 = tmp71 * tmp72
    tmp74 = tmp66 * tmp66
    tmp75 = -0.5
    tmp76 = tmp74 * tmp75
    tmp77 = tl.exp(tmp76)
    tmp78 = 0.3989422804014327
    tmp79 = tmp77 * tmp78
    tmp80 = tmp66 * tmp79
    tmp81 = tmp73 + tmp80
    tmp82 = tmp61 * tmp81
    tmp83 = tmp82 * tmp63
    tmp84 = tmp83 * tmp28
    tmp85 = tl.load(in_ptr14 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp86 = tmp84 - tmp85
    tmp87 = tl.load(in_ptr15 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp88 = tmp62 * tmp87
    tmp89 = tmp86 - tmp88
    tmp90 = tmp60 * tmp89
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp5, tmp90, tmp91)
    tmp93 = tl.where(tmp5, tmp92, tmp9)
    tmp94 = tmp59 + tmp93
    tl.store(out_ptr3 + (r2 + (128*x3)), tmp39, rmask & xmask)
    tl.store(out_ptr4 + (r2 + (128*x3)), tmp94, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v4/cv47wprennpepx6oiincipppifdul6lcozj5o4k2tzenwipgrdld.py
# Source Nodes: [l__mod___blocks_2_fusion_1_norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
# l__mod___blocks_2_fusion_1_norm1 => mul_236, sub_91
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3328
    rnumel = 124
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 128)
    x0 = xindex % 128
    _tmp25 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp30 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (124*x1)
        tmp1 = tl.full([1, 1], 3208, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (128*((r2 + (124*x1)) % 3208))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (128*((r2 + (124*x1)) % 3208))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = (r2 + (124*x1)) % 401
        tmp7 = tl.full([1, 1], 1, tl.int64)
        tmp8 = tmp6 < tmp7
        tmp9 = tmp8 & tmp2
        tmp10 = tl.load(in_ptr2 + (x0 + (128*(((r2 + (124*x1)) // 401) % 8))), rmask & tmp9 & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
        tmp12 = tl.where(tmp9, tmp10, tmp11)
        tmp13 = 0.0
        tmp14 = tl.where(tmp8, tmp12, tmp13)
        tmp15 = tmp5 + tmp14
        tmp16 = tl.load(in_ptr3 + (x0 + (128*((r2 + (124*x1)) % 3208))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr4 + ((r2 + (124*x1)) % 3208), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tmp16 - tmp17
        tmp19 = tl.load(in_ptr5 + ((r2 + (124*x1)) % 3208), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tmp18 * tmp19
        tmp21 = tmp15 * tmp20
        tmp22 = tl.full(tmp21.shape, 0, tmp21.dtype)
        tmp23 = tl.where(tmp2, tmp21, tmp22)
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(rmask & xmask, tmp26, _tmp25)
        tmp27 = tl.full(tmp15.shape, 0, tmp15.dtype)
        tmp28 = tl.where(tmp2, tmp15, tmp27)
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
        tmp31 = _tmp30 + tmp29
        _tmp30 = tl.where(rmask & xmask, tmp31, _tmp30)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp25, xmask)
    tmp30 = tl.sum(_tmp30, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/go/cgovzlleai35tybd3ioncjas42bh2gaeujmwnmm6aesjclyssfph.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_26', 'mutated_arg_names': []}
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
    tmp0 = tl.load(in_ptr0 + (x0 + (51328*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yc/cycgtziicnjeepmbuiie34y46mhb2j32tzs2etmgxealsktq3trz.py
# Source Nodes: [l__mod___blocks_2_revert_projs_0_0, l__mod___blocks_2_revert_projs_0_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
# l__mod___blocks_2_revert_projs_0_0 => add_183, mul_232
# l__mod___blocks_2_revert_projs_0_1 => add_184, erf_22, mul_234
triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = 0.7071067811865476
    tmp7 = tmp5 * tmp6
    tmp8 = tl.math.erf(tmp7)
    tmp9 = 1.0
    tmp10 = tmp8 + tmp9
    tmp11 = 0.5
    tmp12 = tmp10 * tmp11
    tmp13 = tmp5 * tmp5
    tmp14 = -0.5
    tmp15 = tmp13 * tmp14
    tmp16 = tl.exp(tmp15)
    tmp17 = 0.3989422804014327
    tmp18 = tmp16 * tmp17
    tmp19 = tmp5 * tmp18
    tmp20 = tmp12 + tmp19
    tmp21 = tmp0 * tmp20
    tmp22 = tmp21 * tmp1
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None]
    tmp27 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp29 = tl.where(rmask & xmask, tmp27, 0)
    tmp30 = tl.sum(tmp29, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp26, xmask)
    tl.store(out_ptr1 + (x0), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t5/ct5b7kffdhqlt2m52hindh3qcknqyhsyf6ay3wg3ydrdw6r5bh3d.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_28', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/b4/cb4fvh6m47j37wezndpjxitv3hw3dhajzcxhubwibfmfnuv6cxev.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3328
    rnumel = 122
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
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 1576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (256*r2) + (31232*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qh/cqhsavdtdjk5uzxo36wjdep774qhzi2kvrjo4shafqmrokbi76rx.py
# Source Nodes: [l__mod___blocks_2_fusion_0_norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
# l__mod___blocks_2_fusion_0_norm1 => mul_228, sub_88
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3328
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 256)
    x0 = xindex % 256
    _tmp25 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp30 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 1576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (256*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (256*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = (r2 + (122*x1)) % 197
        tmp7 = tl.full([1, 1], 1, tl.int64)
        tmp8 = tmp6 < tmp7
        tmp9 = tmp8 & tmp2
        tmp10 = tl.load(in_ptr2 + (x0 + (256*(((r2 + (122*x1)) // 197) % 8))), rmask & tmp9 & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
        tmp12 = tl.where(tmp9, tmp10, tmp11)
        tmp13 = 0.0
        tmp14 = tl.where(tmp8, tmp12, tmp13)
        tmp15 = tmp5 + tmp14
        tmp16 = tl.load(in_ptr3 + (x0 + (256*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr4 + ((r2 + (122*x1)) % 1576), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tmp16 - tmp17
        tmp19 = tl.load(in_ptr5 + ((r2 + (122*x1)) % 1576), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tmp18 * tmp19
        tmp21 = tmp15 * tmp20
        tmp22 = tl.full(tmp21.shape, 0, tmp21.dtype)
        tmp23 = tl.where(tmp2, tmp21, tmp22)
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(rmask & xmask, tmp26, _tmp25)
        tmp27 = tl.full(tmp15.shape, 0, tmp15.dtype)
        tmp28 = tl.where(tmp2, tmp15, tmp27)
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
        tmp31 = _tmp30 + tmp29
        _tmp30 = tl.where(rmask & xmask, tmp31, _tmp30)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp25, xmask)
    tmp30 = tl.sum(_tmp30, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp30, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wj/cwjqkkht4tcs7b5gyoo47ifdkw77dzwom4izvbnwcpepdcoro2gq.py
# Source Nodes: [l__mod___blocks_2_projs_1_0, l__mod___blocks_2_projs_1_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
# l__mod___blocks_2_projs_1_0 => add_176, mul_224
# l__mod___blocks_2_projs_1_1 => add_177, erf_21, mul_226
triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 8
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
    tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = 0.7071067811865476
    tmp7 = tmp5 * tmp6
    tmp8 = tl.math.erf(tmp7)
    tmp9 = 1.0
    tmp10 = tmp8 + tmp9
    tmp11 = 0.5
    tmp12 = tmp10 * tmp11
    tmp13 = tmp5 * tmp5
    tmp14 = -0.5
    tmp15 = tmp13 * tmp14
    tmp16 = tl.exp(tmp15)
    tmp17 = 0.3989422804014327
    tmp18 = tmp16 * tmp17
    tmp19 = tmp5 * tmp18
    tmp20 = tmp12 + tmp19
    tmp21 = tmp0 * tmp20
    tmp22 = tmp21 * tmp2
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp27 = tmp22 * tmp1
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask & xmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tl.store(out_ptr0 + (x0), tmp26, xmask)
    tl.store(out_ptr1 + (x0), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5s/c5s6gt6aivdtjmufan3kq7jmqexywwznkgcom5vsqcm7bm5ruypc.py
# Source Nodes: [l__mod___blocks_2_projs_1_0, l__mod___blocks_2_projs_1_1], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
# l__mod___blocks_2_projs_1_0 => add_176, mul_224
# l__mod___blocks_2_projs_1_1 => add_177, erf_21, mul_226
triton_poi_fused_add_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_32 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 403456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 256) % 197
    x3 = xindex
    x4 = (xindex // 256)
    x0 = xindex % 256
    x2 = (xindex // 50432)
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_out_ptr0 + (x3), tmp2, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = tl.load(in_ptr0 + (x4), tmp2, eviction_policy='evict_last', other=0.0)
    tmp9 = 256.0
    tmp10 = tmp8 / tmp9
    tmp11 = tl.load(in_ptr1 + (x3), tmp2, other=0.0)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp0 < tmp1
    tmp14 = tmp13 & tmp2
    tmp15 = tl.load(in_ptr2 + (x0 + (256*x2)), tmp14, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp14, tmp15, tmp16)
    tmp18 = tl.where(tmp13, tmp17, tmp6)
    tmp19 = tmp12 + tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp2, tmp19, tmp20)
    tmp22 = tl.where(tmp2, tmp21, tmp6)
    tmp23 = tmp7 + tmp22
    tmp24 = tl.load(in_ptr3 + (x2), tmp13, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.load(in_ptr4 + (x0 + (256*x2)), tmp13, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr5 + (x0 + (256*x2)), tmp13, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.load(in_ptr6 + (x0), tmp13, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp26 * tmp27
    tmp29 = tl.load(in_ptr7 + (x0), tmp13, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 + tmp29
    tmp31 = 0.7071067811865476
    tmp32 = tmp30 * tmp31
    tmp33 = tl.math.erf(tmp32)
    tmp34 = 1.0
    tmp35 = tmp33 + tmp34
    tmp36 = 0.5
    tmp37 = tmp35 * tmp36
    tmp38 = tmp30 * tmp30
    tmp39 = -0.5
    tmp40 = tmp38 * tmp39
    tmp41 = tl.exp(tmp40)
    tmp42 = 0.3989422804014327
    tmp43 = tmp41 * tmp42
    tmp44 = tmp30 * tmp43
    tmp45 = tmp37 + tmp44
    tmp46 = tmp25 * tmp45
    tmp47 = tmp46 * tmp27
    tmp48 = tmp47 * tmp9
    tmp49 = tl.load(in_ptr8 + (x2), tmp13, eviction_policy='evict_last', other=0.0)
    tmp50 = tmp48 - tmp49
    tmp51 = tl.load(in_ptr9 + (x2), tmp13, eviction_policy='evict_last', other=0.0)
    tmp52 = tmp26 * tmp51
    tmp53 = tmp50 - tmp52
    tmp54 = tmp24 * tmp53
    tmp55 = tl.full(tmp54.shape, 0.0, tmp54.dtype)
    tmp56 = tl.where(tmp13, tmp54, tmp55)
    tmp57 = tl.where(tmp13, tmp56, tmp6)
    tmp58 = tmp23 + tmp57
    tl.store(in_out_ptr0 + (x3), tmp58, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lh/clhyul4dfxodztqtnssarkgu3p66gaaeod4n25fvih6wdyimvw2m.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3328
    rnumel = 122
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
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 1576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (256*r2) + (31232*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n4/cn4x5mms4kjmdqugk6o6nrhy42v4ksohaxxn7v76cse62y756hqb.py
# Source Nodes: [x_157], Original ATen: [aten.gelu, aten.gelu_backward]
# x_157 => add_170, erf_19, mul_216
triton_poi_fused_gelu_gelu_backward_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_34', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1210368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = 0.7071067811865476
    tmp3 = tmp1 * tmp2
    tmp4 = tl.math.erf(tmp3)
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp1 * tmp1
    tmp10 = -0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = 0.3989422804014327
    tmp14 = tmp12 * tmp13
    tmp15 = tmp1 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tmp0 * tmp16
    tl.store(in_out_ptr0 + (x0), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2b/c2bw2gvxorvvth5ro57bwhestjqxx3frfju6nc2vbtjq7jd4cspg.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9984
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 768)
    x0 = xindex % 768
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 1576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (768*r2) + (93696*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sy/csy26fnlfia3f7bhxfm7o673sgr77iy2h33wy6wt3bjgwen54tig.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_36', 'mutated_arg_names': []}
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
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wl/cwlhlalex576gm7q57whmb3rtahgra5gteacf366vrlmq5d7noht.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_37', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 1576
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


# kernel path: /tmp/torchinductor_youkaichao/k3/ck3t6ca53qeahhhvv2jrh7bdqtmqicapu6xvhmfi47xionswftak.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3328
    rnumel = 122
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
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 1576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (256*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (256*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/yd/cydf22kmnozpcxqed5i7mbqz4zfahjmmyqd3ltynqmeugl6j3gdx.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1210368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x5 = (xindex // 50432)
    x6 = xindex
    x0 = xindex % 256
    x3 = (xindex // 403456)
    x7 = (xindex // 256) % 1576
    tmp0 = x5
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x6), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 16, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-403456) + x6), tmp11, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 24, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-806912) + x6), tmp15, other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tl.where(tmp11, tmp14, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tl.store(out_ptr0 + (x0 + (256*x3) + (768*x7)), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/s5/cs534rh3fg5jg4po7bg5x57gaf6s7f7xhiah6gxfd7b6ba2tkjyc.py
# Source Nodes: [getattr_l__mod___blocks_2_blocks_1___0___norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# getattr_l__mod___blocks_2_blocks_1___0___norm1 => mul_197, sub_80
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_40', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel):
    xnumel = 1576
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
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_out_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp9 = tmp7 - tmp8
    tmp11 = tmp9 * tmp10
    tmp12 = tmp2 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = 256.0
    tmp19 = tmp10 / tmp18
    tmp20 = tmp2 * tmp18
    tmp21 = tmp20 - tmp6
    tmp22 = tmp11 * tmp16
    tmp23 = tmp21 - tmp22
    tmp24 = tmp19 * tmp23
    tmp25 = tmp17 + tmp24
    tl.store(in_out_ptr0 + (r1 + (256*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7a/c7afnbhu7fpm4me4pejysp4iwhkjfn57fix6c2aklgpevj7pdl5h.py
# Source Nodes: [getattr_l__mod___blocks_2_blocks_1___0___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# getattr_l__mod___blocks_2_blocks_1___0___norm1 => mul_197, sub_80
triton_red_fused_native_layer_norm_native_layer_norm_backward_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3328
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 256)
    x0 = xindex % 256
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 1576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (256*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (256*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + ((r2 + (122*x1)) % 1576), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 - tmp5
        tmp7 = tl.load(in_ptr3 + ((r2 + (122*x1)) % 1576), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 * tmp7
        tmp9 = tmp3 * tmp8
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
        tmp15 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp16 = tl.where(tmp2, tmp3, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zm/czmc4xtup73b53rss5dstal4c753id467yn7t35h6pnjqq3d47js.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3328
    rnumel = 124
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
        tmp0 = r2 + (124*x1)
        tmp1 = tl.full([1, 1], 3208, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (128*r2) + (15872*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6z/c6zcapivpbezcrct7ezf3s6wcthwhl52htiidynz5lt5amr6lyou.py
# Source Nodes: [x_121], Original ATen: [aten.gelu, aten.gelu_backward]
# x_121 => add_149, erf_16, mul_195
triton_poi_fused_gelu_gelu_backward_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_43', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1231872
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = 0.7071067811865476
    tmp3 = tmp1 * tmp2
    tmp4 = tl.math.erf(tmp3)
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp1 * tmp1
    tmp10 = -0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tl.exp(tmp11)
    tmp13 = 0.3989422804014327
    tmp14 = tmp12 * tmp13
    tmp15 = tmp1 * tmp14
    tmp16 = tmp8 + tmp15
    tmp17 = tmp0 * tmp16
    tl.store(in_out_ptr0 + (x0), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m7/cm7cfduakybilqsmcb2po5c7qvzh3b4gyntrzsnl2macrip2pw7m.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9984
    rnumel = 124
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (124*x1)
        tmp1 = tl.full([1, 1], 3208, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*r2) + (47616*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6r/c6rkj3bctgukt2gc6ty6wvmhn5qy7uavesb5ohjodywclhjimxht.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 26
    RBLOCK: tl.constexpr = 32
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


# kernel path: /tmp/torchinductor_youkaichao/24/c24bnpbkofjf4bhurghwbif5vwjw2daptb672un2sldsjvrykg3l.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_46', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3208
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp15 = 128.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(in_out_ptr0 + (r1 + (128*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x7/cx7ne7gt23bz7yruspbllaljwclvbtcxlynztiadspshx2vgls4h.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3328
    rnumel = 124
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 128)
    x0 = xindex % 128
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (124*x1)
        tmp1 = tl.full([1, 1], 3208, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (128*((r2 + (124*x1)) % 3208))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (128*((r2 + (124*x1)) % 3208))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/x5/cx5ogvwwnahwgl46prtiy6uehocd7cmrwnpatmiowfy4lo47bhw3.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1231872
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x5 = (xindex // 51328)
    x6 = xindex
    x0 = xindex % 128
    x3 = (xindex // 410624)
    x7 = (xindex // 128) % 3208
    tmp0 = x5
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x6), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 16, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-410624) + x6), tmp11 & xmask, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 24, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-821248) + x6), tmp15 & xmask, other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tl.where(tmp11, tmp14, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tl.store(out_ptr0 + (x0 + (128*x3) + (384*x7)), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r5/cr5lhbxszlfas5obysxvmseasa6heifkj6ylkjpfpcf5ddwnp67f.py
# Source Nodes: [getattr_l__mod___blocks_2_blocks_0___0___norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# getattr_l__mod___blocks_2_blocks_0___0___norm1 => mul_190, sub_78
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_49', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3208
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_out_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp9 = tmp7 - tmp8
    tmp11 = tmp9 * tmp10
    tmp12 = tmp2 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 128.0
    tmp19 = tmp10 / tmp18
    tmp20 = tmp2 * tmp18
    tmp21 = tmp20 - tmp6
    tmp22 = tmp11 * tmp16
    tmp23 = tmp21 - tmp22
    tmp24 = tmp19 * tmp23
    tmp25 = tmp17 + tmp24
    tl.store(in_out_ptr0 + (r1 + (128*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nh/cnho644ajprth4wb3bbzmr3zwm4stdcdmvlc23ec3mnflbwouijx.py
# Source Nodes: [getattr_l__mod___blocks_2_blocks_0___0___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# getattr_l__mod___blocks_2_blocks_0___0___norm1 => mul_190, sub_78
triton_red_fused_native_layer_norm_native_layer_norm_backward_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3328
    rnumel = 124
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 128)
    x0 = xindex % 128
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (124*x1)
        tmp1 = tl.full([1, 1], 3208, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (128*((r2 + (124*x1)) % 3208))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (128*((r2 + (124*x1)) % 3208))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + ((r2 + (124*x1)) % 3208), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 - tmp5
        tmp7 = tl.load(in_ptr3 + ((r2 + (124*x1)) % 3208), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 * tmp7
        tmp9 = tmp3 * tmp8
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
        tmp15 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp16 = tl.where(tmp2, tmp3, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o4/co462ki5l7wn6xa4dp5v3xjbxyxxkxhc57dtzfyh5htna5m22rbx.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[65536, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 50432
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (50432*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ah/cahdzslxrpthlu5hb3tbtlmpwve4hz3j5yuq7voclxfxrykohgfg.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_52', 'mutated_arg_names': []}
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
        tmp3 = tl.load(in_ptr0 + (256 + x0 + (256*((r2 + (121*x1)) % 196)) + (50432*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lr/clr2t7mmxqga46j5aitl3dvssd4yb2o6g6ukphtowx7lwn64tvgr.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_53 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[65536, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 51328
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (51328*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nb/cnbomqejg423gjbr6two3xfl3morbsdkz4vblnlokuuq4zednxx5.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_54 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_54', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3200
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (128 + x0 + (128*((r2 + (128*x1)) % 400)) + (51328*((r2 + (128*x1)) // 400))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kq/ckqkurjhu3uh3mkwfailsoyvkv7huldep7wofbbteek2qgzhspfp.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_55', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 25
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_5, primals_7, primals_9, primals_15, primals_21, primals_27, primals_33, primals_39, primals_45, primals_51, primals_57, primals_58, primals_61, primals_62, primals_65, primals_75, primals_76, primals_79, primals_89, primals_90, primals_93, primals_99, primals_105, primals_111, primals_117, primals_123, primals_129, primals_135, primals_141, primals_142, primals_145, primals_146, primals_149, primals_159, primals_160, primals_163, primals_173, primals_174, primals_177, primals_183, primals_189, primals_195, primals_201, primals_207, primals_213, primals_219, primals_225, primals_226, primals_229, primals_230, primals_233, primals_243, primals_244, primals_247, primals_257, primals_258, primals_261, primals_263, primals_269, add_46, mul_82, view_6, getitem_2, getitem_3, getitem_4, getitem_6, getitem_7, getitem_8, view_10, mul_84, view_12, addmm_2, view_14, mul_89, view_16, getitem_13, getitem_14, getitem_15, getitem_17, getitem_18, getitem_19, view_20, mul_91, view_22, addmm_6, view_24, mul_96, view_26, getitem_24, getitem_25, getitem_26, getitem_28, getitem_29, getitem_30, view_30, mul_98, view_32, addmm_10, view_34, mul_103, view_36, getitem_35, getitem_36, getitem_37, getitem_39, getitem_40, getitem_41, view_40, mul_105, view_42, addmm_14, view_44, mul_110, view_46, mul_115, view_48, cat_2, getitem_49, rsqrt_10, view_50, view_53, view_66, mul_123, view_68, cat_3, cat_4, getitem_53, rsqrt_12, view_70, view_73, view_86, mul_131, view_88, cat_5, getitem_57, rsqrt_14, view_90, getitem_58, getitem_59, getitem_60, getitem_62, getitem_63, getitem_64, view_94, mul_138, view_96, addmm_28, view_98, getitem_68, rsqrt_16, view_100, getitem_69, getitem_70, getitem_71, getitem_73, getitem_74, getitem_75, view_104, mul_145, view_106, addmm_32, view_108, mul_150, view_110, getitem_80, getitem_81, getitem_82, getitem_84, getitem_85, getitem_86, view_114, mul_152, view_116, addmm_36, view_118, mul_157, view_120, getitem_91, getitem_92, getitem_93, getitem_95, getitem_96, getitem_97, view_124, mul_159, view_126, addmm_40, view_128, mul_164, view_130, mul_169, view_132, cat_6, getitem_105, rsqrt_24, view_134, view_137, view_150, mul_177, view_152, cat_7, cat_8, getitem_109, rsqrt_26, view_154, view_157, view_170, mul_185, view_172, cat_9, getitem_113, rsqrt_28, view_174, getitem_114, getitem_115, getitem_116, getitem_118, getitem_119, getitem_120, view_178, mul_192, view_180, addmm_54, view_182, getitem_124, rsqrt_30, view_184, getitem_125, getitem_126, getitem_127, getitem_129, getitem_130, getitem_131, view_188, mul_199, view_190, addmm_58, view_192, mul_204, view_194, getitem_136, getitem_137, getitem_138, getitem_140, getitem_141, getitem_142, view_198, mul_206, view_200, addmm_62, view_202, mul_211, view_204, getitem_147, getitem_148, getitem_149, getitem_151, getitem_152, getitem_153, view_208, mul_213, view_210, addmm_66, view_212, mul_218, view_214, mul_223, view_216, cat_10, getitem_161, rsqrt_38, view_218, view_221, view_234, mul_231, view_236, cat_11, cat_12, getitem_165, rsqrt_40, view_238, view_241, view_254, mul_239, view_256, cat_13, getitem_169, rsqrt_42, getitem_171, rsqrt_43, clone_68, clone_69, permute_142, permute_146, permute_150, div_9, permute_154, permute_159, permute_160, alias_18, permute_161, permute_162, permute_165, permute_170, permute_177, permute_179, div_11, permute_183, permute_188, permute_189, alias_19, permute_190, permute_191, permute_194, permute_199, permute_206, permute_208, div_13, permute_212, div_14, permute_216, permute_220, div_15, permute_224, alias_20, permute_230, div_16, permute_234, permute_238, div_17, permute_242, alias_21, permute_248, div_18, permute_252, permute_256, div_19, permute_260, alias_22, permute_266, permute_270, permute_274, div_21, permute_278, alias_23, permute_284, permute_288, div_23, permute_292, permute_297, permute_298, alias_24, permute_299, permute_300, permute_303, permute_308, permute_315, permute_317, div_25, permute_321, permute_326, permute_327, alias_25, permute_328, permute_329, permute_332, permute_337, permute_344, permute_346, div_27, permute_350, div_28, permute_354, permute_358, div_29, permute_362, alias_26, permute_368, div_30, permute_372, permute_376, div_31, permute_380, alias_27, permute_386, div_32, permute_390, permute_394, div_33, permute_398, alias_28, permute_404, permute_408, permute_412, div_35, permute_416, alias_29, permute_422, permute_426, div_37, permute_430, permute_435, permute_436, alias_30, permute_437, permute_438, permute_441, permute_446, permute_453, permute_455, div_39, permute_459, permute_464, permute_465, alias_31, permute_466, permute_467, permute_470, permute_475, permute_482, permute_484, div_41, permute_488, div_42, permute_492, permute_496, div_43, permute_500, alias_32, permute_506, div_44, permute_510, permute_514, div_45, permute_518, alias_33, permute_524, div_46, permute_528, permute_532, div_47, permute_536, alias_34, permute_542, div_48, permute_546, permute_550, div_49, permute_554, alias_35, permute_560, div_50, tangents_1 = args
    args.clear()
    assert_size_stride(primals_5, (128, 3, 12, 12), (432, 144, 12, 1))
    assert_size_stride(primals_7, (256, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_21, (256, ), (1, ))
    assert_size_stride(primals_27, (256, ), (1, ))
    assert_size_stride(primals_33, (256, ), (1, ))
    assert_size_stride(primals_39, (256, ), (1, ))
    assert_size_stride(primals_45, (256, ), (1, ))
    assert_size_stride(primals_51, (256, ), (1, ))
    assert_size_stride(primals_57, (128, ), (1, ))
    assert_size_stride(primals_58, (128, ), (1, ))
    assert_size_stride(primals_61, (256, ), (1, ))
    assert_size_stride(primals_62, (256, ), (1, ))
    assert_size_stride(primals_65, (256, ), (1, ))
    assert_size_stride(primals_75, (256, ), (1, ))
    assert_size_stride(primals_76, (256, ), (1, ))
    assert_size_stride(primals_79, (128, ), (1, ))
    assert_size_stride(primals_89, (128, ), (1, ))
    assert_size_stride(primals_90, (128, ), (1, ))
    assert_size_stride(primals_93, (128, ), (1, ))
    assert_size_stride(primals_99, (128, ), (1, ))
    assert_size_stride(primals_105, (256, ), (1, ))
    assert_size_stride(primals_111, (256, ), (1, ))
    assert_size_stride(primals_117, (256, ), (1, ))
    assert_size_stride(primals_123, (256, ), (1, ))
    assert_size_stride(primals_129, (256, ), (1, ))
    assert_size_stride(primals_135, (256, ), (1, ))
    assert_size_stride(primals_141, (128, ), (1, ))
    assert_size_stride(primals_142, (128, ), (1, ))
    assert_size_stride(primals_145, (256, ), (1, ))
    assert_size_stride(primals_146, (256, ), (1, ))
    assert_size_stride(primals_149, (256, ), (1, ))
    assert_size_stride(primals_159, (256, ), (1, ))
    assert_size_stride(primals_160, (256, ), (1, ))
    assert_size_stride(primals_163, (128, ), (1, ))
    assert_size_stride(primals_173, (128, ), (1, ))
    assert_size_stride(primals_174, (128, ), (1, ))
    assert_size_stride(primals_177, (128, ), (1, ))
    assert_size_stride(primals_183, (128, ), (1, ))
    assert_size_stride(primals_189, (256, ), (1, ))
    assert_size_stride(primals_195, (256, ), (1, ))
    assert_size_stride(primals_201, (256, ), (1, ))
    assert_size_stride(primals_207, (256, ), (1, ))
    assert_size_stride(primals_213, (256, ), (1, ))
    assert_size_stride(primals_219, (256, ), (1, ))
    assert_size_stride(primals_225, (128, ), (1, ))
    assert_size_stride(primals_226, (128, ), (1, ))
    assert_size_stride(primals_229, (256, ), (1, ))
    assert_size_stride(primals_230, (256, ), (1, ))
    assert_size_stride(primals_233, (256, ), (1, ))
    assert_size_stride(primals_243, (256, ), (1, ))
    assert_size_stride(primals_244, (256, ), (1, ))
    assert_size_stride(primals_247, (128, ), (1, ))
    assert_size_stride(primals_257, (128, ), (1, ))
    assert_size_stride(primals_258, (128, ), (1, ))
    assert_size_stride(primals_261, (128, ), (1, ))
    assert_size_stride(primals_263, (256, ), (1, ))
    assert_size_stride(primals_269, (8, 3, 240, 240), (172800, 57600, 240, 1))
    assert_size_stride(add_46, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(mul_82, (8, 401, 128), (51328, 128, 1))
    assert_size_stride(view_6, (3208, 128), (128, 1))
    assert_size_stride(getitem_2, (8, 4, 401, 32), (153984, 32, 384, 1))
    assert_size_stride(getitem_3, (8, 4, 401, 32), (153984, 32, 384, 1))
    assert_size_stride(getitem_4, (8, 4, 401, 32), (153984, 32, 384, 1))
    assert_size_stride(getitem_6, (8, 4, 416), (1664, 416, 1))
    assert_size_stride(getitem_7, (), ())
    assert_size_stride(getitem_8, (), ())
    assert_size_stride(view_10, (3208, 128), (128, 1))
    assert_size_stride(mul_84, (8, 401, 128), (51328, 128, 1))
    assert_size_stride(view_12, (3208, 128), (128, 1))
    assert_size_stride(addmm_2, (3208, 384), (384, 1))
    assert_size_stride(view_14, (3208, 384), (384, 1))
    assert_size_stride(mul_89, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_16, (1576, 256), (256, 1))
    assert_size_stride(getitem_13, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_14, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_15, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_17, (8, 4, 224), (896, 224, 1))
    assert_size_stride(getitem_18, (), ())
    assert_size_stride(getitem_19, (), ())
    assert_size_stride(view_20, (1576, 256), (256, 1))
    assert_size_stride(mul_91, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_22, (1576, 256), (256, 1))
    assert_size_stride(addmm_6, (1576, 768), (768, 1))
    assert_size_stride(view_24, (1576, 768), (768, 1))
    assert_size_stride(mul_96, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_26, (1576, 256), (256, 1))
    assert_size_stride(getitem_24, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_25, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_26, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_28, (8, 4, 224), (896, 224, 1))
    assert_size_stride(getitem_29, (), ())
    assert_size_stride(getitem_30, (), ())
    assert_size_stride(view_30, (1576, 256), (256, 1))
    assert_size_stride(mul_98, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_32, (1576, 256), (256, 1))
    assert_size_stride(addmm_10, (1576, 768), (768, 1))
    assert_size_stride(view_34, (1576, 768), (768, 1))
    assert_size_stride(mul_103, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_36, (1576, 256), (256, 1))
    assert_size_stride(getitem_35, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_36, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_37, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_39, (8, 4, 224), (896, 224, 1))
    assert_size_stride(getitem_40, (), ())
    assert_size_stride(getitem_41, (), ())
    assert_size_stride(view_40, (1576, 256), (256, 1))
    assert_size_stride(mul_105, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_42, (1576, 256), (256, 1))
    assert_size_stride(addmm_14, (1576, 768), (768, 1))
    assert_size_stride(view_44, (1576, 768), (768, 1))
    assert_size_stride(mul_110, (8, 1, 128), (128, 128, 1))
    assert_size_stride(view_46, (8, 128), (128, 1))
    assert_size_stride(mul_115, (8, 1, 256), (256, 256, 1))
    assert_size_stride(view_48, (8, 256), (256, 1))
    assert_size_stride(cat_2, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(getitem_49, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_10, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_50, (8, 256), (50432, 1))
    assert_size_stride(view_53, (1576, 256), (256, 1))
    assert_size_stride(view_66, (8, 256), (256, 1))
    assert_size_stride(mul_123, (8, 1, 256), (256, 256, 1))
    assert_size_stride(view_68, (8, 256), (256, 1))
    assert_size_stride(cat_3, (8, 401, 128), (51328, 128, 1))
    assert_size_stride(cat_4, (8, 401, 128), (51328, 128, 1))
    assert_size_stride(getitem_53, (8, 401, 1), (401, 1, 1))
    assert_size_stride(rsqrt_12, (8, 401, 1), (401, 1, 1))
    assert_size_stride(view_70, (8, 128), (51328, 1))
    assert_size_stride(view_73, (3208, 128), (128, 1))
    assert_size_stride(view_86, (8, 128), (128, 1))
    assert_size_stride(mul_131, (8, 1, 128), (128, 128, 1))
    assert_size_stride(view_88, (8, 128), (128, 1))
    assert_size_stride(cat_5, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(getitem_57, (8, 401, 1), (401, 1, 1))
    assert_size_stride(rsqrt_14, (8, 401, 1), (401, 1, 1))
    assert_size_stride(view_90, (3208, 128), (128, 1))
    assert_size_stride(getitem_58, (8, 4, 401, 32), (153984, 32, 384, 1))
    assert_size_stride(getitem_59, (8, 4, 401, 32), (153984, 32, 384, 1))
    assert_size_stride(getitem_60, (8, 4, 401, 32), (153984, 32, 384, 1))
    assert_size_stride(getitem_62, (8, 4, 416), (1664, 416, 1))
    assert_size_stride(getitem_63, (), ())
    assert_size_stride(getitem_64, (), ())
    assert_size_stride(view_94, (3208, 128), (128, 1))
    assert_size_stride(mul_138, (8, 401, 128), (51328, 128, 1))
    assert_size_stride(view_96, (3208, 128), (128, 1))
    assert_size_stride(addmm_28, (3208, 384), (384, 1))
    assert_size_stride(view_98, (3208, 384), (384, 1))
    assert_size_stride(getitem_68, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_16, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_100, (1576, 256), (256, 1))
    assert_size_stride(getitem_69, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_70, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_71, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_73, (8, 4, 224), (896, 224, 1))
    assert_size_stride(getitem_74, (), ())
    assert_size_stride(getitem_75, (), ())
    assert_size_stride(view_104, (1576, 256), (256, 1))
    assert_size_stride(mul_145, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_106, (1576, 256), (256, 1))
    assert_size_stride(addmm_32, (1576, 768), (768, 1))
    assert_size_stride(view_108, (1576, 768), (768, 1))
    assert_size_stride(mul_150, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_110, (1576, 256), (256, 1))
    assert_size_stride(getitem_80, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_81, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_82, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_84, (8, 4, 224), (896, 224, 1))
    assert_size_stride(getitem_85, (), ())
    assert_size_stride(getitem_86, (), ())
    assert_size_stride(view_114, (1576, 256), (256, 1))
    assert_size_stride(mul_152, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_116, (1576, 256), (256, 1))
    assert_size_stride(addmm_36, (1576, 768), (768, 1))
    assert_size_stride(view_118, (1576, 768), (768, 1))
    assert_size_stride(mul_157, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_120, (1576, 256), (256, 1))
    assert_size_stride(getitem_91, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_92, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_93, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_95, (8, 4, 224), (896, 224, 1))
    assert_size_stride(getitem_96, (), ())
    assert_size_stride(getitem_97, (), ())
    assert_size_stride(view_124, (1576, 256), (256, 1))
    assert_size_stride(mul_159, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_126, (1576, 256), (256, 1))
    assert_size_stride(addmm_40, (1576, 768), (768, 1))
    assert_size_stride(view_128, (1576, 768), (768, 1))
    assert_size_stride(mul_164, (8, 1, 128), (128, 128, 1))
    assert_size_stride(view_130, (8, 128), (128, 1))
    assert_size_stride(mul_169, (8, 1, 256), (256, 256, 1))
    assert_size_stride(view_132, (8, 256), (256, 1))
    assert_size_stride(cat_6, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(getitem_105, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_24, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_134, (8, 256), (50432, 1))
    assert_size_stride(view_137, (1576, 256), (256, 1))
    assert_size_stride(view_150, (8, 256), (256, 1))
    assert_size_stride(mul_177, (8, 1, 256), (256, 256, 1))
    assert_size_stride(view_152, (8, 256), (256, 1))
    assert_size_stride(cat_7, (8, 401, 128), (51328, 128, 1))
    assert_size_stride(cat_8, (8, 401, 128), (51328, 128, 1))
    assert_size_stride(getitem_109, (8, 401, 1), (401, 1, 1))
    assert_size_stride(rsqrt_26, (8, 401, 1), (401, 1, 1))
    assert_size_stride(view_154, (8, 128), (51328, 1))
    assert_size_stride(view_157, (3208, 128), (128, 1))
    assert_size_stride(view_170, (8, 128), (128, 1))
    assert_size_stride(mul_185, (8, 1, 128), (128, 128, 1))
    assert_size_stride(view_172, (8, 128), (128, 1))
    assert_size_stride(cat_9, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(getitem_113, (8, 401, 1), (401, 1, 1))
    assert_size_stride(rsqrt_28, (8, 401, 1), (401, 1, 1))
    assert_size_stride(view_174, (3208, 128), (128, 1))
    assert_size_stride(getitem_114, (8, 4, 401, 32), (153984, 32, 384, 1))
    assert_size_stride(getitem_115, (8, 4, 401, 32), (153984, 32, 384, 1))
    assert_size_stride(getitem_116, (8, 4, 401, 32), (153984, 32, 384, 1))
    assert_size_stride(getitem_118, (8, 4, 416), (1664, 416, 1))
    assert_size_stride(getitem_119, (), ())
    assert_size_stride(getitem_120, (), ())
    assert_size_stride(view_178, (3208, 128), (128, 1))
    assert_size_stride(mul_192, (8, 401, 128), (51328, 128, 1))
    assert_size_stride(view_180, (3208, 128), (128, 1))
    assert_size_stride(addmm_54, (3208, 384), (384, 1))
    assert_size_stride(view_182, (3208, 384), (384, 1))
    assert_size_stride(getitem_124, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_30, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_184, (1576, 256), (256, 1))
    assert_size_stride(getitem_125, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_126, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_127, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_129, (8, 4, 224), (896, 224, 1))
    assert_size_stride(getitem_130, (), ())
    assert_size_stride(getitem_131, (), ())
    assert_size_stride(view_188, (1576, 256), (256, 1))
    assert_size_stride(mul_199, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_190, (1576, 256), (256, 1))
    assert_size_stride(addmm_58, (1576, 768), (768, 1))
    assert_size_stride(view_192, (1576, 768), (768, 1))
    assert_size_stride(mul_204, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_194, (1576, 256), (256, 1))
    assert_size_stride(getitem_136, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_137, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_138, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_140, (8, 4, 224), (896, 224, 1))
    assert_size_stride(getitem_141, (), ())
    assert_size_stride(getitem_142, (), ())
    assert_size_stride(view_198, (1576, 256), (256, 1))
    assert_size_stride(mul_206, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_200, (1576, 256), (256, 1))
    assert_size_stride(addmm_62, (1576, 768), (768, 1))
    assert_size_stride(view_202, (1576, 768), (768, 1))
    assert_size_stride(mul_211, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_204, (1576, 256), (256, 1))
    assert_size_stride(getitem_147, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_148, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_149, (8, 4, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(getitem_151, (8, 4, 224), (896, 224, 1))
    assert_size_stride(getitem_152, (), ())
    assert_size_stride(getitem_153, (), ())
    assert_size_stride(view_208, (1576, 256), (256, 1))
    assert_size_stride(mul_213, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(view_210, (1576, 256), (256, 1))
    assert_size_stride(addmm_66, (1576, 768), (768, 1))
    assert_size_stride(view_212, (1576, 768), (768, 1))
    assert_size_stride(mul_218, (8, 1, 128), (128, 128, 1))
    assert_size_stride(view_214, (8, 128), (128, 1))
    assert_size_stride(mul_223, (8, 1, 256), (256, 256, 1))
    assert_size_stride(view_216, (8, 256), (256, 1))
    assert_size_stride(cat_10, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(getitem_161, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_38, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_218, (8, 256), (50432, 1))
    assert_size_stride(view_221, (1576, 256), (256, 1))
    assert_size_stride(view_234, (8, 256), (256, 1))
    assert_size_stride(mul_231, (8, 1, 256), (256, 256, 1))
    assert_size_stride(view_236, (8, 256), (256, 1))
    assert_size_stride(cat_11, (8, 401, 128), (51328, 128, 1))
    assert_size_stride(cat_12, (8, 401, 128), (51328, 128, 1))
    assert_size_stride(getitem_165, (8, 401, 1), (401, 1, 1))
    assert_size_stride(rsqrt_40, (8, 401, 1), (401, 1, 1))
    assert_size_stride(view_238, (8, 128), (51328, 1))
    assert_size_stride(view_241, (3208, 128), (128, 1))
    assert_size_stride(view_254, (8, 128), (128, 1))
    assert_size_stride(mul_239, (8, 1, 128), (128, 128, 1))
    assert_size_stride(view_256, (8, 128), (128, 1))
    assert_size_stride(cat_13, (8, 197, 256), (50432, 256, 1))
    assert_size_stride(getitem_169, (8, 401, 1), (401, 1, 1))
    assert_size_stride(rsqrt_42, (8, 401, 1), (401, 1, 1))
    assert_size_stride(getitem_171, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_43, (8, 197, 1), (197, 1, 1))
    assert_size_stride(clone_68, (8, 128), (128, 1))
    assert_size_stride(clone_69, (8, 256), (256, 1))
    assert_size_stride(permute_142, (1000, 256), (256, 1))
    assert_size_stride(permute_146, (1000, 128), (128, 1))
    assert_size_stride(permute_150, (256, 128), (128, 1))
    assert_size_stride(div_9, (8, 1, 1), (1, 1, 1))
    assert_size_stride(permute_154, (128, 128), (128, 1))
    assert_size_stride(permute_159, (32, 401, 1), (401, 1, 0))
    assert_size_stride(permute_160, (32, 32, 401), (12832, 1, 32))
    assert_size_stride(alias_18, (8, 4, 1, 401), (1604, 401, 401, 1))
    assert_size_stride(permute_161, (32, 32, 1), (32, 1, 0))
    assert_size_stride(permute_162, (32, 401, 32), (12832, 1, 401))
    assert_size_stride(permute_165, (128, 128), (128, 1))
    assert_size_stride(permute_170, (128, 128), (128, 1))
    assert_size_stride(permute_177, (128, 128), (128, 1))
    assert_size_stride(permute_179, (128, 256), (256, 1))
    assert_size_stride(div_11, (8, 1, 1), (1, 1, 1))
    assert_size_stride(permute_183, (256, 256), (256, 1))
    assert_size_stride(permute_188, (32, 197, 1), (197, 1, 0))
    assert_size_stride(permute_189, (32, 64, 197), (12608, 1, 64))
    assert_size_stride(alias_19, (8, 4, 1, 197), (788, 197, 197, 1))
    assert_size_stride(permute_190, (32, 64, 1), (64, 1, 0))
    assert_size_stride(permute_191, (32, 197, 64), (12608, 1, 197))
    assert_size_stride(permute_194, (256, 256), (256, 1))
    assert_size_stride(permute_199, (256, 256), (256, 1))
    assert_size_stride(permute_206, (256, 256), (256, 1))
    assert_size_stride(permute_208, (128, 256), (256, 1))
    assert_size_stride(div_13, (8, 1, 1), (1, 1, 1))
    assert_size_stride(permute_212, (256, 128), (128, 1))
    assert_size_stride(div_14, (8, 1, 1), (1, 1, 1))
    assert_size_stride(permute_216, (256, 768), (768, 1))
    assert_size_stride(permute_220, (768, 256), (256, 1))
    assert_size_stride(div_15, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_224, (256, 256), (256, 1))
    assert_size_stride(alias_20, (8, 4, 197, 64), (50432, 64, 256, 1))
    assert_size_stride(permute_230, (768, 256), (256, 1))
    assert_size_stride(div_16, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_234, (256, 768), (768, 1))
    assert_size_stride(permute_238, (768, 256), (256, 1))
    assert_size_stride(div_17, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_242, (256, 256), (256, 1))
    assert_size_stride(alias_21, (8, 4, 197, 64), (50432, 64, 256, 1))
    assert_size_stride(permute_248, (768, 256), (256, 1))
    assert_size_stride(div_18, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_252, (256, 768), (768, 1))
    assert_size_stride(permute_256, (768, 256), (256, 1))
    assert_size_stride(div_19, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_260, (256, 256), (256, 1))
    assert_size_stride(alias_22, (8, 4, 197, 64), (50432, 64, 256, 1))
    assert_size_stride(permute_266, (768, 256), (256, 1))
    assert_size_stride(permute_270, (128, 384), (384, 1))
    assert_size_stride(permute_274, (384, 128), (128, 1))
    assert_size_stride(div_21, (8, 401, 1), (401, 1, 1))
    assert_size_stride(permute_278, (128, 128), (128, 1))
    assert_size_stride(alias_23, (8, 4, 401, 32), (51328, 32, 128, 1))
    assert_size_stride(permute_284, (384, 128), (128, 1))
    assert_size_stride(permute_288, (256, 128), (128, 1))
    assert_size_stride(div_23, (8, 1, 1), (1, 1, 1))
    assert_size_stride(permute_292, (128, 128), (128, 1))
    assert_size_stride(permute_297, (32, 401, 1), (401, 1, 0))
    assert_size_stride(permute_298, (32, 32, 401), (12832, 1, 32))
    assert_size_stride(alias_24, (8, 4, 1, 401), (1604, 401, 401, 1))
    assert_size_stride(permute_299, (32, 32, 1), (32, 1, 0))
    assert_size_stride(permute_300, (32, 401, 32), (12832, 1, 401))
    assert_size_stride(permute_303, (128, 128), (128, 1))
    assert_size_stride(permute_308, (128, 128), (128, 1))
    assert_size_stride(permute_315, (128, 128), (128, 1))
    assert_size_stride(permute_317, (128, 256), (256, 1))
    assert_size_stride(div_25, (8, 1, 1), (1, 1, 1))
    assert_size_stride(permute_321, (256, 256), (256, 1))
    assert_size_stride(permute_326, (32, 197, 1), (197, 1, 0))
    assert_size_stride(permute_327, (32, 64, 197), (12608, 1, 64))
    assert_size_stride(alias_25, (8, 4, 1, 197), (788, 197, 197, 1))
    assert_size_stride(permute_328, (32, 64, 1), (64, 1, 0))
    assert_size_stride(permute_329, (32, 197, 64), (12608, 1, 197))
    assert_size_stride(permute_332, (256, 256), (256, 1))
    assert_size_stride(permute_337, (256, 256), (256, 1))
    assert_size_stride(permute_344, (256, 256), (256, 1))
    assert_size_stride(permute_346, (128, 256), (256, 1))
    assert_size_stride(div_27, (8, 1, 1), (1, 1, 1))
    assert_size_stride(permute_350, (256, 128), (128, 1))
    assert_size_stride(div_28, (8, 1, 1), (1, 1, 1))
    assert_size_stride(permute_354, (256, 768), (768, 1))
    assert_size_stride(permute_358, (768, 256), (256, 1))
    assert_size_stride(div_29, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_362, (256, 256), (256, 1))
    assert_size_stride(alias_26, (8, 4, 197, 64), (50432, 64, 256, 1))
    assert_size_stride(permute_368, (768, 256), (256, 1))
    assert_size_stride(div_30, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_372, (256, 768), (768, 1))
    assert_size_stride(permute_376, (768, 256), (256, 1))
    assert_size_stride(div_31, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_380, (256, 256), (256, 1))
    assert_size_stride(alias_27, (8, 4, 197, 64), (50432, 64, 256, 1))
    assert_size_stride(permute_386, (768, 256), (256, 1))
    assert_size_stride(div_32, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_390, (256, 768), (768, 1))
    assert_size_stride(permute_394, (768, 256), (256, 1))
    assert_size_stride(div_33, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_398, (256, 256), (256, 1))
    assert_size_stride(alias_28, (8, 4, 197, 64), (50432, 64, 256, 1))
    assert_size_stride(permute_404, (768, 256), (256, 1))
    assert_size_stride(permute_408, (128, 384), (384, 1))
    assert_size_stride(permute_412, (384, 128), (128, 1))
    assert_size_stride(div_35, (8, 401, 1), (401, 1, 1))
    assert_size_stride(permute_416, (128, 128), (128, 1))
    assert_size_stride(alias_29, (8, 4, 401, 32), (51328, 32, 128, 1))
    assert_size_stride(permute_422, (384, 128), (128, 1))
    assert_size_stride(permute_426, (256, 128), (128, 1))
    assert_size_stride(div_37, (8, 1, 1), (1, 1, 1))
    assert_size_stride(permute_430, (128, 128), (128, 1))
    assert_size_stride(permute_435, (32, 401, 1), (401, 1, 0))
    assert_size_stride(permute_436, (32, 32, 401), (12832, 1, 32))
    assert_size_stride(alias_30, (8, 4, 1, 401), (1604, 401, 401, 1))
    assert_size_stride(permute_437, (32, 32, 1), (32, 1, 0))
    assert_size_stride(permute_438, (32, 401, 32), (12832, 1, 401))
    assert_size_stride(permute_441, (128, 128), (128, 1))
    assert_size_stride(permute_446, (128, 128), (128, 1))
    assert_size_stride(permute_453, (128, 128), (128, 1))
    assert_size_stride(permute_455, (128, 256), (256, 1))
    assert_size_stride(div_39, (8, 1, 1), (1, 1, 1))
    assert_size_stride(permute_459, (256, 256), (256, 1))
    assert_size_stride(permute_464, (32, 197, 1), (197, 1, 0))
    assert_size_stride(permute_465, (32, 64, 197), (12608, 1, 64))
    assert_size_stride(alias_31, (8, 4, 1, 197), (788, 197, 197, 1))
    assert_size_stride(permute_466, (32, 64, 1), (64, 1, 0))
    assert_size_stride(permute_467, (32, 197, 64), (12608, 1, 197))
    assert_size_stride(permute_470, (256, 256), (256, 1))
    assert_size_stride(permute_475, (256, 256), (256, 1))
    assert_size_stride(permute_482, (256, 256), (256, 1))
    assert_size_stride(permute_484, (128, 256), (256, 1))
    assert_size_stride(div_41, (8, 1, 1), (1, 1, 1))
    assert_size_stride(permute_488, (256, 128), (128, 1))
    assert_size_stride(div_42, (8, 1, 1), (1, 1, 1))
    assert_size_stride(permute_492, (256, 768), (768, 1))
    assert_size_stride(permute_496, (768, 256), (256, 1))
    assert_size_stride(div_43, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_500, (256, 256), (256, 1))
    assert_size_stride(alias_32, (8, 4, 197, 64), (50432, 64, 256, 1))
    assert_size_stride(permute_506, (768, 256), (256, 1))
    assert_size_stride(div_44, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_510, (256, 768), (768, 1))
    assert_size_stride(permute_514, (768, 256), (256, 1))
    assert_size_stride(div_45, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_518, (256, 256), (256, 1))
    assert_size_stride(alias_33, (8, 4, 197, 64), (50432, 64, 256, 1))
    assert_size_stride(permute_524, (768, 256), (256, 1))
    assert_size_stride(div_46, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_528, (256, 768), (768, 1))
    assert_size_stride(permute_532, (768, 256), (256, 1))
    assert_size_stride(div_47, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_536, (256, 256), (256, 1))
    assert_size_stride(alias_34, (8, 4, 197, 64), (50432, 64, 256, 1))
    assert_size_stride(permute_542, (768, 256), (256, 1))
    assert_size_stride(div_48, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_546, (128, 384), (384, 1))
    assert_size_stride(permute_550, (384, 128), (128, 1))
    assert_size_stride(div_49, (8, 401, 1), (401, 1, 1))
    assert_size_stride(permute_554, (128, 128), (128, 1))
    assert_size_stride(alias_35, (8, 4, 401, 32), (51328, 32, 128, 1))
    assert_size_stride(permute_560, (384, 128), (128, 1))
    assert_size_stride(div_50, (8, 401, 1), (401, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 1000), device='cuda', dtype=torch.float32)
        buf1 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.select]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_div_select_0.run(tangents_1, buf0, buf1, 8000, grid=grid(8000), stream=stream0)
        del tangents_1
        buf2 = empty((8, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1, permute_142, out=buf2)
        del permute_142
        buf3 = empty((1000, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (1000, 8), (1, 1000), 0), clone_69, out=buf3)
        del clone_69
        buf4 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf1, buf4, 1000, 8, grid=grid(1000), stream=stream0)
        del buf1
        buf5 = empty((8, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf0, permute_146, out=buf5)
        del permute_146
        buf6 = empty((1000, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (1000, 8), (1, 1000), 0), clone_68, out=buf6)
        del clone_68
        buf7 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf0, buf7, 1000, 8, grid=grid(1000), stream=stream0)
        del buf0
        buf10 = empty((8, 197, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_172], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_select_backward_2.run(buf2, primals_263, cat_13, getitem_171, rsqrt_43, buf10, 1576, 256, grid=grid(1576), stream=stream0)
        del primals_263
        buf11 = empty_strided((256, 13), (1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_172], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_select_backward_3.run(buf2, cat_13, getitem_171, rsqrt_43, buf11, 3328, 122, grid=grid(3328), stream=stream0)
        del cat_13
        del getitem_171
        del rsqrt_43
        buf12 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_172], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf11, buf12, 256, 13, grid=grid(256), stream=stream0)
        buf13 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]
        triton_red_fused_native_layer_norm_backward_select_backward_5.run(buf2, buf13, 256, 1576, grid=grid(256), stream=stream0)
        buf16 = empty((8, 401, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_171], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_select_backward_6.run(buf5, primals_261, cat_11, getitem_169, rsqrt_42, buf16, 3208, 128, grid=grid(3208), stream=stream0)
        del primals_261
        buf17 = reinterpret_tensor(buf11, (128, 26), (1, 128), 0); del buf11  # reuse
        # Source Nodes: [x_171], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_select_backward_7.run(buf5, cat_11, getitem_169, rsqrt_42, buf17, 3328, 124, grid=grid(3328), stream=stream0)
        del cat_11
        del getitem_169
        del rsqrt_42
        buf18 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_171], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf17, buf18, 128, 26, grid=grid(128), stream=stream0)
        buf19 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]
        triton_red_fused_native_layer_norm_backward_select_backward_9.run(buf5, buf19, 128, 3208, grid=grid(128), stream=stream0)
        buf20 = buf5; del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (8, 256), (50432, 1), 0), permute_150, out=buf20)
        del permute_150
        buf21 = empty((256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (256, 8), (1, 50432), 0), view_256, out=buf21)
        del view_256
        buf22 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf10, buf22, 256, 8, grid=grid(256), stream=stream0)
        buf25 = empty((8, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_revert_projs_1_0, l__mod___blocks_2_revert_projs_1_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_11.run(buf20, mul_239, primals_257, primals_258, div_9, buf25, 8, 128, grid=grid(8), stream=stream0)
        del div_9
        buf26 = empty((128, ), device='cuda', dtype=torch.float32)
        buf27 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_revert_projs_1_0, l__mod___blocks_2_revert_projs_1_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_12.run(buf20, mul_239, primals_257, primals_258, buf26, buf27, 128, 8, grid=grid(128), stream=stream0)
        del mul_239
        del primals_257
        del primals_258
        buf28 = buf20; del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf25, (8, 128), (128, 1), 0), permute_154, out=buf28)
        del permute_154
        buf29 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf25, (128, 8), (1, 128), 0), view_254, out=buf29)
        del view_254
        buf30 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf25, buf30, 128, 8, grid=grid(128), stream=stream0)
        buf31 = empty((32, 401, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_159, reinterpret_tensor(buf28, (32, 1, 32), (32, 32, 1), 0), out=buf31)
        del permute_159
        buf32 = empty((32, 1, 401), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf28, (32, 1, 32), (32, 32, 1), 0), permute_160, out=buf32)
        del permute_160
        buf34 = empty((8, 4, 1, 401), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_14.run(buf32, alias_18, buf34, 32, 401, grid=grid(32), stream=stream0)
        del alias_18
        buf35 = empty((32, 32, 401), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_161, reinterpret_tensor(buf34, (32, 1, 401), (401, 0, 1), 0), out=buf35)
        del permute_161
        buf36 = reinterpret_tensor(buf28, (32, 1, 32), (32, 32, 1), 0); del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf34, (32, 1, 401), (401, 0, 1), 0), permute_162, out=buf36)
        del permute_162
        buf37 = empty((3208, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf31, buf37, 410624, grid=grid(410624), stream=stream0)
        buf38 = reinterpret_tensor(buf31, (3208, 128), (128, 1), 0); del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf37, permute_165, out=buf38)
        del permute_165
        buf39 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf37, (128, 3208), (1, 128), 0), view_241, out=buf39)
        buf40 = reinterpret_tensor(buf17, (1, 128, 26), (3328, 1, 128), 0); del buf17  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf37, buf40, 3328, 124, grid=grid(3328), stream=stream0)
        buf41 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf40, buf41, 128, 26, grid=grid(128), stream=stream0)
        buf42 = buf37; del buf37  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_17.run(buf35, buf42, 3208, 128, grid=grid(3208, 128), stream=stream0)
        buf43 = reinterpret_tensor(buf35, (3208, 128), (128, 1), 0); del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf42, permute_170, out=buf43)
        del permute_170
        buf44 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf42, (128, 3208), (1, 128), 0), view_241, out=buf44)
        del view_241
        buf45 = buf40; del buf40  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf42, buf45, 3328, 124, grid=grid(3328), stream=stream0)
        buf46 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf45, buf46, 128, 26, grid=grid(128), stream=stream0)
        buf47 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf36, buf47, 128, 8, grid=grid(128), stream=stream0)
        buf48 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (128, 8), (1, 128), 0), view_238, out=buf48)
        del view_238
        buf49 = empty((8, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (8, 128), (128, 1), 0), permute_177, out=buf49)
        del permute_177
        buf57 = buf2; del buf2  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf16, (8, 128), (51328, 1), 0), permute_179, out=buf57)
        del permute_179
        buf62 = empty((8, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_revert_projs_0_0, l__mod___blocks_2_revert_projs_0_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_18.run(buf57, mul_231, primals_243, primals_244, div_11, buf62, 8, 256, grid=grid(8), stream=stream0)
        del div_11
        buf65 = empty((8, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf62, (8, 256), (256, 1), 0), permute_183, out=buf65)
        del permute_183
        buf68 = empty((32, 197, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_188, reinterpret_tensor(buf65, (32, 1, 64), (64, 64, 1), 0), out=buf68)
        del permute_188
        buf74 = empty((1576, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf68, buf74, 403456, grid=grid(403456), stream=stream0)
        buf75 = reinterpret_tensor(buf68, (1576, 256), (256, 1), 0); del buf68  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf74, permute_194, out=buf75)
        del permute_194
        buf69 = empty((32, 1, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf65, (32, 1, 64), (64, 64, 1), 0), permute_189, out=buf69)
        del permute_189
        buf71 = empty((8, 4, 1, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_20.run(buf69, alias_19, buf71, 32, 197, grid=grid(32), stream=stream0)
        del alias_19
        buf72 = empty((32, 64, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_190, reinterpret_tensor(buf71, (32, 1, 197), (197, 0, 1), 0), out=buf72)
        del permute_190
        buf79 = empty((1576, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_21.run(buf72, buf79, 1576, 256, grid=grid(1576, 256), stream=stream0)
        buf80 = reinterpret_tensor(buf72, (1576, 256), (256, 1), 0); del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf79, permute_199, out=buf80)
        del permute_199
        buf73 = reinterpret_tensor(buf65, (32, 1, 64), (64, 64, 1), 0); del buf65  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf71, (32, 1, 197), (197, 0, 1), 0), permute_191, out=buf73)
        del permute_191
        buf86 = empty((8, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf73, (8, 256), (256, 1), 0), permute_206, out=buf86)
        del permute_206
        buf89 = empty((8, 197, 256), device='cuda', dtype=torch.float32)
        buf103 = empty((8, 197, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_fusion_0_norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_22.run(buf75, buf80, buf86, primals_233, cat_10, getitem_161, rsqrt_38, buf62, buf89, buf103, 1576, 256, grid=grid(1576), stream=stream0)
        del primals_233
        buf104 = reinterpret_tensor(buf36, (8, 128), (128, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf103, (8, 256), (50432, 1), 0), permute_212, out=buf104)
        del permute_212
        buf107 = empty_strided((8, 1, 1), (1, 8, 8), device='cuda', dtype=torch.float32)
        buf108 = empty_strided((8, 1, 1), (1, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_projs_0_0, l__mod___blocks_2_projs_0_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_23.run(buf104, mul_218, primals_225, primals_226, buf107, buf108, 8, 128, grid=grid(8), stream=stream0)
        buf94 = reinterpret_tensor(buf42, (8, 401, 128), (51328, 128, 1), 0); del buf42  # reuse
        buf111 = empty((8, 401, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_fusion_1_norm1, l__mod___blocks_2_projs_0_0, l__mod___blocks_2_projs_0_1], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_per_fused_add_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_24.run(buf38, buf43, buf49, primals_247, cat_12, getitem_165, rsqrt_40, buf25, buf16, div_14, buf104, mul_218, primals_225, primals_226, buf107, buf108, buf94, buf111, 3208, 128, grid=grid(3208), stream=stream0)
        del div_14
        del primals_247
        buf53 = reinterpret_tensor(buf45, (128, 26), (1, 128), 0); del buf45  # reuse
        buf55 = empty_strided((128, 26), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_fusion_1_norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_25.run(buf38, buf43, buf49, cat_12, getitem_165, rsqrt_40, buf53, buf55, 3328, 124, grid=grid(3328), stream=stream0)
        del buf38
        del buf43
        del cat_12
        del getitem_165
        del rsqrt_40
        buf54 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_fusion_1_norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf53, buf54, 128, 26, grid=grid(128), stream=stream0)
        buf56 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf55, buf56, 128, 26, grid=grid(128), stream=stream0)
        buf58 = empty((128, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf16, (128, 8), (1, 51328), 0), view_236, out=buf58)
        del view_236
        buf59 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf16, buf59, 128, 8, grid=grid(128), stream=stream0)
        del buf16
        buf63 = empty((256, ), device='cuda', dtype=torch.float32)
        buf64 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_revert_projs_0_0, l__mod___blocks_2_revert_projs_0_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_27.run(buf57, mul_231, primals_243, primals_244, buf63, buf64, 256, 8, grid=grid(256), stream=stream0)
        del mul_231
        del primals_243
        del primals_244
        buf66 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf62, (256, 8), (1, 256), 0), view_234, out=buf66)
        del view_234
        buf67 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_28.run(buf62, buf67, 256, 8, grid=grid(256), stream=stream0)
        buf76 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf74, (256, 1576), (1, 256), 0), view_221, out=buf76)
        buf77 = reinterpret_tensor(buf55, (1, 256, 13), (3328, 1, 256), 0); del buf55  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_29.run(buf74, buf77, 3328, 122, grid=grid(3328), stream=stream0)
        del buf74
        buf78 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf77, buf78, 256, 13, grid=grid(256), stream=stream0)
        buf81 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf79, (256, 1576), (1, 256), 0), view_221, out=buf81)
        del view_221
        buf82 = buf77; del buf77  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_29.run(buf79, buf82, 3328, 122, grid=grid(3328), stream=stream0)
        del buf79
        buf83 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf82, buf83, 256, 13, grid=grid(256), stream=stream0)
        buf84 = empty((1, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_28.run(buf73, buf84, 256, 8, grid=grid(256), stream=stream0)
        buf85 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf73, (256, 8), (1, 256), 0), view_218, out=buf85)
        del view_218
        buf90 = reinterpret_tensor(buf82, (256, 13), (1, 256), 0); del buf82  # reuse
        buf92 = reinterpret_tensor(buf53, (256, 13), (1, 256), 0); del buf53  # reuse
        # Source Nodes: [l__mod___blocks_2_fusion_0_norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_30.run(buf75, buf80, buf86, cat_10, getitem_161, rsqrt_38, buf90, buf92, 3328, 122, grid=grid(3328), stream=stream0)
        del buf75
        del buf80
        del cat_10
        del getitem_161
        buf91 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_fusion_0_norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf90, buf91, 256, 13, grid=grid(256), stream=stream0)
        buf93 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf92, buf93, 256, 13, grid=grid(256), stream=stream0)
        buf95 = buf86; del buf86  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf94, (8, 128), (51328, 1), 0), permute_208, out=buf95)
        del permute_208
        buf96 = empty((128, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf94, (128, 8), (1, 51328), 0), view_216, out=buf96)
        del view_216
        buf97 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf94, buf97, 128, 8, grid=grid(128), stream=stream0)
        buf98 = buf108; del buf108  # reuse
        buf99 = buf107; del buf107  # reuse
        # Source Nodes: [l__mod___blocks_2_projs_1_0, l__mod___blocks_2_projs_1_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_31.run(buf95, mul_223, primals_229, primals_230, buf98, buf99, 8, 256, grid=grid(8), stream=stream0)
        buf100 = empty((256, ), device='cuda', dtype=torch.float32)
        buf101 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_projs_1_0, l__mod___blocks_2_projs_1_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_27.run(buf95, mul_223, primals_229, primals_230, buf100, buf101, 256, 8, grid=grid(256), stream=stream0)
        buf102 = buf10; del buf10  # reuse
        # Source Nodes: [l__mod___blocks_2_projs_1_0, l__mod___blocks_2_projs_1_1], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_poi_fused_add_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_32.run(buf102, rsqrt_38, buf89, buf62, div_13, buf95, mul_223, primals_229, primals_230, buf98, buf99, 403456, grid=grid(403456), stream=stream0)
        del buf89
        del div_13
        del mul_223
        del primals_229
        del primals_230
        del rsqrt_38
        buf105 = empty((256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf103, (256, 8), (1, 50432), 0), view_214, out=buf105)
        del view_214
        buf106 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf103, buf106, 256, 8, grid=grid(256), stream=stream0)
        buf109 = empty((128, ), device='cuda', dtype=torch.float32)
        buf110 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_2_projs_0_0, l__mod___blocks_2_projs_0_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_12.run(buf104, mul_218, primals_225, primals_226, buf109, buf110, 128, 8, grid=grid(128), stream=stream0)
        del mul_218
        del primals_225
        del primals_226
        buf112 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf102, (1576, 256), (256, 1), 0), permute_216, out=buf112)
        del permute_216
        buf113 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf102, (256, 1576), (1, 256), 0), view_212, out=buf113)
        del view_212
        buf114 = reinterpret_tensor(buf92, (1, 256, 13), (3328, 1, 256), 0); del buf92  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_33.run(buf102, buf114, 3328, 122, grid=grid(3328), stream=stream0)
        buf115 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf114, buf115, 256, 13, grid=grid(256), stream=stream0)
        buf116 = reinterpret_tensor(buf112, (8, 197, 768), (151296, 768, 1), 0); del buf112  # reuse
        # Source Nodes: [x_157], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_34.run(buf116, addmm_66, 1210368, grid=grid(1210368), stream=stream0)
        del addmm_66
        buf117 = reinterpret_tensor(buf103, (1576, 256), (256, 1), 0); del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (1576, 768), (768, 1), 0), permute_220, out=buf117)
        del permute_220
        buf118 = empty((768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (768, 1576), (1, 768), 0), view_210, out=buf118)
        del view_210
        buf119 = empty_strided((1, 768, 13), (9984, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_35.run(buf116, buf119, 9984, 122, grid=grid(9984), stream=stream0)
        buf120 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_36.run(buf119, buf120, 768, 13, grid=grid(768), stream=stream0)
        buf127 = buf102; del buf102  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_37.run(buf127, buf117, primals_219, mul_213, div_15, 1576, 256, grid=grid(1576), stream=stream0)
        del div_15
        del primals_219
        buf123 = reinterpret_tensor(buf114, (256, 13), (1, 256), 0); del buf114  # reuse
        buf125 = buf90; del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf117, mul_213, buf123, buf125, 3328, 122, grid=grid(3328), stream=stream0)
        del mul_213
        buf124 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf123, buf124, 256, 13, grid=grid(256), stream=stream0)
        buf126 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf125, buf126, 256, 13, grid=grid(256), stream=stream0)
        buf128 = buf117; del buf117  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf127, (1576, 256), (256, 1), 0), permute_224, out=buf128)
        del permute_224
        buf129 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf127, (256, 1576), (1, 256), 0), view_208, out=buf129)
        del view_208
        buf130 = reinterpret_tensor(buf125, (1, 256, 13), (3328, 1, 256), 0); del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_33.run(buf127, buf130, 3328, 122, grid=grid(3328), stream=stream0)
        buf131 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf130, buf131, 256, 13, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf132 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf128, (8, 4, 197, 64), (50432, 64, 256, 1), 0), getitem_147, getitem_148, getitem_149, None, alias_20, getitem_151, getitem_152, getitem_153, 0.0, [True, True, True, False])
        del alias_20
        del buf128
        del getitem_147
        del getitem_148
        del getitem_149
        del getitem_151
        del getitem_152
        del getitem_153
        buf133 = buf132[0]
        buf134 = buf132[1]
        buf135 = buf132[2]
        del buf132
        buf136 = reinterpret_tensor(buf116, (8, 197, 3, 4, 64), (151296, 768, 256, 64, 1), 0); del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf133, buf134, buf135, buf136, 1210368, grid=grid(1210368), stream=stream0)
        del buf133
        del buf134
        buf137 = reinterpret_tensor(buf135, (1576, 256), (256, 1), 0); del buf135  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf136, (1576, 768), (768, 1), 0), permute_230, out=buf137)
        del permute_230
        buf138 = empty((768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf136, (768, 1576), (1, 768), 0), view_204, out=buf138)
        del view_204
        buf139 = buf119; del buf119  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_35.run(buf136, buf139, 9984, 122, grid=grid(9984), stream=stream0)
        buf140 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_36.run(buf139, buf140, 768, 13, grid=grid(768), stream=stream0)
        buf147 = buf127; del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_37.run(buf147, buf137, primals_213, mul_211, div_16, 1576, 256, grid=grid(1576), stream=stream0)
        del div_16
        del primals_213
        buf143 = reinterpret_tensor(buf130, (256, 13), (1, 256), 0); del buf130  # reuse
        buf145 = buf123; del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf137, mul_211, buf143, buf145, 3328, 122, grid=grid(3328), stream=stream0)
        del mul_211
        buf144 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf143, buf144, 256, 13, grid=grid(256), stream=stream0)
        buf146 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf145, buf146, 256, 13, grid=grid(256), stream=stream0)
        buf148 = reinterpret_tensor(buf136, (1576, 768), (768, 1), 0); del buf136  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf147, (1576, 256), (256, 1), 0), permute_234, out=buf148)
        del permute_234
        buf149 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf147, (256, 1576), (1, 256), 0), view_202, out=buf149)
        del view_202
        buf150 = reinterpret_tensor(buf145, (1, 256, 13), (3328, 1, 256), 0); del buf145  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_33.run(buf147, buf150, 3328, 122, grid=grid(3328), stream=stream0)
        buf151 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf150, buf151, 256, 13, grid=grid(256), stream=stream0)
        buf152 = reinterpret_tensor(buf148, (8, 197, 768), (151296, 768, 1), 0); del buf148  # reuse
        # Source Nodes: [x_145], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_34.run(buf152, addmm_62, 1210368, grid=grid(1210368), stream=stream0)
        del addmm_62
        buf153 = buf137; del buf137  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf152, (1576, 768), (768, 1), 0), permute_238, out=buf153)
        del permute_238
        buf154 = empty((768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf152, (768, 1576), (1, 768), 0), view_200, out=buf154)
        del view_200
        buf155 = buf139; del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_35.run(buf152, buf155, 9984, 122, grid=grid(9984), stream=stream0)
        buf156 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_36.run(buf155, buf156, 768, 13, grid=grid(768), stream=stream0)
        buf163 = buf147; del buf147  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_37.run(buf163, buf153, primals_207, mul_206, div_17, 1576, 256, grid=grid(1576), stream=stream0)
        del div_17
        del primals_207
        buf159 = reinterpret_tensor(buf150, (256, 13), (1, 256), 0); del buf150  # reuse
        buf161 = buf143; del buf143  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf153, mul_206, buf159, buf161, 3328, 122, grid=grid(3328), stream=stream0)
        del mul_206
        buf160 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf159, buf160, 256, 13, grid=grid(256), stream=stream0)
        buf162 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf161, buf162, 256, 13, grid=grid(256), stream=stream0)
        buf164 = buf153; del buf153  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (1576, 256), (256, 1), 0), permute_242, out=buf164)
        del permute_242
        buf165 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (256, 1576), (1, 256), 0), view_198, out=buf165)
        del view_198
        buf166 = reinterpret_tensor(buf161, (1, 256, 13), (3328, 1, 256), 0); del buf161  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_33.run(buf163, buf166, 3328, 122, grid=grid(3328), stream=stream0)
        buf167 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf166, buf167, 256, 13, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf168 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf164, (8, 4, 197, 64), (50432, 64, 256, 1), 0), getitem_136, getitem_137, getitem_138, None, alias_21, getitem_140, getitem_141, getitem_142, 0.0, [True, True, True, False])
        del alias_21
        del buf164
        del getitem_136
        del getitem_137
        del getitem_138
        del getitem_140
        del getitem_141
        del getitem_142
        buf169 = buf168[0]
        buf170 = buf168[1]
        buf171 = buf168[2]
        del buf168
        buf172 = reinterpret_tensor(buf152, (8, 197, 3, 4, 64), (151296, 768, 256, 64, 1), 0); del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf169, buf170, buf171, buf172, 1210368, grid=grid(1210368), stream=stream0)
        buf173 = reinterpret_tensor(buf171, (1576, 256), (256, 1), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf172, (1576, 768), (768, 1), 0), permute_248, out=buf173)
        del permute_248
        buf174 = empty((768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf172, (768, 1576), (1, 768), 0), view_194, out=buf174)
        del view_194
        buf175 = buf155; del buf155  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_35.run(buf172, buf175, 9984, 122, grid=grid(9984), stream=stream0)
        buf176 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_36.run(buf175, buf176, 768, 13, grid=grid(768), stream=stream0)
        buf183 = buf163; del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_37.run(buf183, buf173, primals_201, mul_204, div_18, 1576, 256, grid=grid(1576), stream=stream0)
        del div_18
        del primals_201
        buf179 = reinterpret_tensor(buf166, (256, 13), (1, 256), 0); del buf166  # reuse
        buf181 = buf159; del buf159  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf173, mul_204, buf179, buf181, 3328, 122, grid=grid(3328), stream=stream0)
        del mul_204
        buf180 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf179, buf180, 256, 13, grid=grid(256), stream=stream0)
        buf182 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf181, buf182, 256, 13, grid=grid(256), stream=stream0)
        buf184 = reinterpret_tensor(buf172, (1576, 768), (768, 1), 0); del buf172  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf183, (1576, 256), (256, 1), 0), permute_252, out=buf184)
        del permute_252
        buf185 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf183, (256, 1576), (1, 256), 0), view_192, out=buf185)
        del view_192
        buf186 = reinterpret_tensor(buf181, (1, 256, 13), (3328, 1, 256), 0); del buf181  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_33.run(buf183, buf186, 3328, 122, grid=grid(3328), stream=stream0)
        buf187 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf186, buf187, 256, 13, grid=grid(256), stream=stream0)
        buf188 = reinterpret_tensor(buf184, (8, 197, 768), (151296, 768, 1), 0); del buf184  # reuse
        # Source Nodes: [x_133], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_34.run(buf188, addmm_58, 1210368, grid=grid(1210368), stream=stream0)
        del addmm_58
        buf189 = buf173; del buf173  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf188, (1576, 768), (768, 1), 0), permute_256, out=buf189)
        del permute_256
        buf190 = empty((768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf188, (768, 1576), (1, 768), 0), view_190, out=buf190)
        del view_190
        buf191 = buf175; del buf175  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_35.run(buf188, buf191, 9984, 122, grid=grid(9984), stream=stream0)
        buf192 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_36.run(buf191, buf192, 768, 13, grid=grid(768), stream=stream0)
        buf199 = buf183; del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_37.run(buf199, buf189, primals_195, mul_199, div_19, 1576, 256, grid=grid(1576), stream=stream0)
        del div_19
        del primals_195
        buf195 = reinterpret_tensor(buf186, (256, 13), (1, 256), 0); del buf186  # reuse
        buf197 = buf179; del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf189, mul_199, buf195, buf197, 3328, 122, grid=grid(3328), stream=stream0)
        del mul_199
        buf196 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf195, buf196, 256, 13, grid=grid(256), stream=stream0)
        buf198 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf197, buf198, 256, 13, grid=grid(256), stream=stream0)
        buf200 = buf189; del buf189  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf199, (1576, 256), (256, 1), 0), permute_260, out=buf200)
        del permute_260
        buf201 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf199, (256, 1576), (1, 256), 0), view_188, out=buf201)
        del view_188
        buf202 = reinterpret_tensor(buf197, (1, 256, 13), (3328, 1, 256), 0); del buf197  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_33.run(buf199, buf202, 3328, 122, grid=grid(3328), stream=stream0)
        buf203 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf202, buf203, 256, 13, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf204 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf200, (8, 4, 197, 64), (50432, 64, 256, 1), 0), getitem_125, getitem_126, getitem_127, None, alias_22, getitem_129, getitem_130, getitem_131, 0.0, [True, True, True, False])
        del alias_22
        del getitem_125
        del getitem_126
        del getitem_127
        del getitem_129
        del getitem_130
        del getitem_131
        buf205 = buf204[0]
        buf206 = buf204[1]
        buf207 = buf204[2]
        del buf204
        buf208 = reinterpret_tensor(buf188, (8, 197, 3, 4, 64), (151296, 768, 256, 64, 1), 0); del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf205, buf206, buf207, buf208, 1210368, grid=grid(1210368), stream=stream0)
        buf209 = reinterpret_tensor(buf207, (1576, 256), (256, 1), 0); del buf207  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (1576, 768), (768, 1), 0), permute_266, out=buf209)
        del permute_266
        buf210 = empty((768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (768, 1576), (1, 768), 0), view_184, out=buf210)
        del view_184
        buf211 = buf191; del buf191  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_35.run(buf208, buf211, 9984, 122, grid=grid(9984), stream=stream0)
        buf212 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_36.run(buf211, buf212, 768, 13, grid=grid(768), stream=stream0)
        buf219 = buf199; del buf199  # reuse
        # Source Nodes: [getattr_l__mod___blocks_2_blocks_1___0___norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_40.run(buf219, buf209, primals_189, cat_9, getitem_124, rsqrt_30, 1576, 256, grid=grid(1576), stream=stream0)
        del primals_189
        buf215 = reinterpret_tensor(buf202, (256, 13), (1, 256), 0); del buf202  # reuse
        buf217 = buf195; del buf195  # reuse
        # Source Nodes: [getattr_l__mod___blocks_2_blocks_1___0___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_41.run(buf209, cat_9, getitem_124, rsqrt_30, buf215, buf217, 3328, 122, grid=grid(3328), stream=stream0)
        del cat_9
        del getitem_124
        del rsqrt_30
        buf216 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_2_blocks_1___0___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf215, buf216, 256, 13, grid=grid(256), stream=stream0)
        buf218 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf217, buf218, 256, 13, grid=grid(256), stream=stream0)
        buf220 = empty((3208, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf111, (3208, 128), (128, 1), 0), permute_270, out=buf220)
        del permute_270
        buf221 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf111, (128, 3208), (1, 128), 0), view_182, out=buf221)
        del view_182
        buf222 = reinterpret_tensor(buf217, (1, 128, 26), (3328, 1, 128), 0); del buf217  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_42.run(buf111, buf222, 3328, 124, grid=grid(3328), stream=stream0)
        buf223 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf222, buf223, 128, 26, grid=grid(128), stream=stream0)
        buf224 = reinterpret_tensor(buf220, (8, 401, 384), (153984, 384, 1), 0); del buf220  # reuse
        # Source Nodes: [x_121], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_43.run(buf224, addmm_54, 1231872, grid=grid(1231872), stream=stream0)
        del addmm_54
        buf225 = reinterpret_tensor(buf94, (3208, 128), (128, 1), 0); del buf94  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf224, (3208, 384), (384, 1), 0), permute_274, out=buf225)
        del permute_274
        buf226 = empty((384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf224, (384, 3208), (1, 384), 0), view_180, out=buf226)
        del view_180
        buf227 = reinterpret_tensor(buf211, (1, 384, 26), (9984, 1, 384), 0); del buf211  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_44.run(buf224, buf227, 9984, 124, grid=grid(9984), stream=stream0)
        buf228 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_45.run(buf227, buf228, 384, 26, grid=grid(384), stream=stream0)
        buf235 = buf111; del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_46.run(buf235, buf225, primals_183, mul_192, div_21, 3208, 128, grid=grid(3208), stream=stream0)
        del div_21
        del primals_183
        buf231 = reinterpret_tensor(buf222, (128, 26), (1, 128), 0); del buf222  # reuse
        buf233 = reinterpret_tensor(buf215, (128, 26), (1, 128), 0); del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_47.run(buf225, mul_192, buf231, buf233, 3328, 124, grid=grid(3328), stream=stream0)
        del mul_192
        buf232 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf231, buf232, 128, 26, grid=grid(128), stream=stream0)
        buf234 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf233, buf234, 128, 26, grid=grid(128), stream=stream0)
        buf236 = buf225; del buf225  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf235, (3208, 128), (128, 1), 0), permute_278, out=buf236)
        del permute_278
        buf237 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf235, (128, 3208), (1, 128), 0), view_178, out=buf237)
        del view_178
        buf238 = reinterpret_tensor(buf233, (1, 128, 26), (3328, 1, 128), 0); del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_42.run(buf235, buf238, 3328, 124, grid=grid(3328), stream=stream0)
        buf239 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf238, buf239, 128, 26, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf240 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf236, (8, 4, 401, 32), (51328, 32, 128, 1), 0), getitem_114, getitem_115, getitem_116, None, alias_23, getitem_118, getitem_119, getitem_120, 0.0, [True, True, True, False])
        del alias_23
        del getitem_114
        del getitem_115
        del getitem_116
        del getitem_118
        del getitem_119
        del getitem_120
        buf241 = buf240[0]
        buf242 = buf240[1]
        buf243 = buf240[2]
        del buf240
        buf244 = reinterpret_tensor(buf224, (8, 401, 3, 4, 32), (153984, 384, 128, 32, 1), 0); del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_48.run(buf241, buf242, buf243, buf244, 1231872, grid=grid(1231872), stream=stream0)
        buf245 = reinterpret_tensor(buf243, (3208, 128), (128, 1), 0); del buf243  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf244, (3208, 384), (384, 1), 0), permute_284, out=buf245)
        del permute_284
        buf246 = empty((384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf244, (384, 3208), (1, 384), 0), view_174, out=buf246)
        del view_174
        buf247 = buf227; del buf227  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_44.run(buf244, buf247, 9984, 124, grid=grid(9984), stream=stream0)
        buf248 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_45.run(buf247, buf248, 384, 26, grid=grid(384), stream=stream0)
        buf255 = buf235; del buf235  # reuse
        # Source Nodes: [getattr_l__mod___blocks_2_blocks_0___0___norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_49.run(buf255, buf245, primals_177, cat_7, getitem_113, rsqrt_28, 3208, 128, grid=grid(3208), stream=stream0)
        del primals_177
        buf251 = reinterpret_tensor(buf238, (128, 26), (1, 128), 0); del buf238  # reuse
        buf253 = buf231; del buf231  # reuse
        # Source Nodes: [getattr_l__mod___blocks_2_blocks_0___0___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_50.run(buf245, cat_7, getitem_113, rsqrt_28, buf251, buf253, 3328, 124, grid=grid(3328), stream=stream0)
        del cat_7
        del getitem_113
        del rsqrt_28
        buf252 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_2_blocks_0___0___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf251, buf252, 128, 26, grid=grid(128), stream=stream0)
        buf254 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf253, buf254, 128, 26, grid=grid(128), stream=stream0)
        buf256 = buf104; del buf104  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf219, (8, 256), (50432, 1), 0), permute_288, out=buf256)
        del permute_288
        buf257 = empty((256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf219, (256, 8), (1, 50432), 0), view_172, out=buf257)
        del view_172
        buf258 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf219, buf258, 256, 8, grid=grid(256), stream=stream0)
        buf261 = reinterpret_tensor(buf49, (8, 1, 128), (128, 128, 1), 0); del buf49  # reuse
        # Source Nodes: [l__mod___blocks_1_revert_projs_1_0, l__mod___blocks_1_revert_projs_1_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_11.run(buf256, mul_185, primals_173, primals_174, div_23, buf261, 8, 128, grid=grid(8), stream=stream0)
        del div_23
        buf262 = empty((128, ), device='cuda', dtype=torch.float32)
        buf263 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_revert_projs_1_0, l__mod___blocks_1_revert_projs_1_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_12.run(buf256, mul_185, primals_173, primals_174, buf262, buf263, 128, 8, grid=grid(128), stream=stream0)
        del mul_185
        del primals_173
        del primals_174
        buf264 = buf256; del buf256  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf261, (8, 128), (128, 1), 0), permute_292, out=buf264)
        del permute_292
        buf265 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf261, (128, 8), (1, 128), 0), view_170, out=buf265)
        del view_170
        buf266 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf261, buf266, 128, 8, grid=grid(128), stream=stream0)
        buf267 = reinterpret_tensor(buf245, (32, 401, 32), (12832, 32, 1), 0); del buf245  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_297, reinterpret_tensor(buf264, (32, 1, 32), (32, 32, 1), 0), out=buf267)
        del permute_297
        buf268 = reinterpret_tensor(buf34, (32, 1, 401), (401, 401, 1), 0); del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf264, (32, 1, 32), (32, 32, 1), 0), permute_298, out=buf268)
        del permute_298
        buf270 = reinterpret_tensor(buf32, (8, 4, 1, 401), (1604, 401, 401, 1), 0); del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_14.run(buf268, alias_24, buf270, 32, 401, grid=grid(32), stream=stream0)
        del alias_24
        buf271 = reinterpret_tensor(buf242, (32, 32, 401), (12832, 401, 1), 0); del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_299, reinterpret_tensor(buf270, (32, 1, 401), (401, 0, 1), 0), out=buf271)
        del permute_299
        buf272 = reinterpret_tensor(buf264, (32, 1, 32), (32, 32, 1), 0); del buf264  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf270, (32, 1, 401), (401, 0, 1), 0), permute_300, out=buf272)
        del permute_300
        buf273 = reinterpret_tensor(buf241, (3208, 128), (128, 1), 0); del buf241  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf267, buf273, 410624, grid=grid(410624), stream=stream0)
        buf274 = reinterpret_tensor(buf267, (3208, 128), (128, 1), 0); del buf267  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf273, permute_303, out=buf274)
        del permute_303
        buf275 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf273, (128, 3208), (1, 128), 0), view_157, out=buf275)
        buf276 = reinterpret_tensor(buf253, (1, 128, 26), (3328, 1, 128), 0); del buf253  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf273, buf276, 3328, 124, grid=grid(3328), stream=stream0)
        buf277 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf276, buf277, 128, 26, grid=grid(128), stream=stream0)
        buf278 = buf273; del buf273  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_17.run(buf271, buf278, 3208, 128, grid=grid(3208, 128), stream=stream0)
        buf279 = reinterpret_tensor(buf271, (3208, 128), (128, 1), 0); del buf271  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf278, permute_308, out=buf279)
        del permute_308
        buf280 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf278, (128, 3208), (1, 128), 0), view_157, out=buf280)
        del view_157
        buf281 = buf276; del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf278, buf281, 3328, 124, grid=grid(3328), stream=stream0)
        buf282 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf281, buf282, 128, 26, grid=grid(128), stream=stream0)
        buf283 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf272, buf283, 128, 8, grid=grid(128), stream=stream0)
        buf284 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf272, (128, 8), (1, 128), 0), view_154, out=buf284)
        del view_154
        buf285 = reinterpret_tensor(buf25, (8, 128), (128, 1), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf272, (8, 128), (128, 1), 0), permute_315, out=buf285)
        del permute_315
        buf293 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf255, (8, 128), (51328, 1), 0), permute_317, out=buf293)
        del permute_317
        buf298 = buf62; del buf62  # reuse
        # Source Nodes: [l__mod___blocks_1_revert_projs_0_0, l__mod___blocks_1_revert_projs_0_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_18.run(buf293, mul_177, primals_159, primals_160, div_25, buf298, 8, 256, grid=grid(8), stream=stream0)
        del div_25
        buf301 = reinterpret_tensor(buf73, (8, 256), (256, 1), 0); del buf73  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf298, (8, 256), (256, 1), 0), permute_321, out=buf301)
        del permute_321
        buf304 = reinterpret_tensor(buf209, (32, 197, 64), (12608, 64, 1), 0); del buf209  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_326, reinterpret_tensor(buf301, (32, 1, 64), (64, 64, 1), 0), out=buf304)
        del permute_326
        buf310 = reinterpret_tensor(buf206, (1576, 256), (256, 1), 0); del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf304, buf310, 403456, grid=grid(403456), stream=stream0)
        buf311 = reinterpret_tensor(buf304, (1576, 256), (256, 1), 0); del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf310, permute_332, out=buf311)
        del permute_332
        buf305 = reinterpret_tensor(buf71, (32, 1, 197), (197, 197, 1), 0); del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf301, (32, 1, 64), (64, 64, 1), 0), permute_327, out=buf305)
        del permute_327
        buf307 = reinterpret_tensor(buf69, (8, 4, 1, 197), (788, 197, 197, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_20.run(buf305, alias_25, buf307, 32, 197, grid=grid(32), stream=stream0)
        del alias_25
        buf308 = reinterpret_tensor(buf205, (32, 64, 197), (12608, 197, 1), 0); del buf205  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_328, reinterpret_tensor(buf307, (32, 1, 197), (197, 0, 1), 0), out=buf308)
        del permute_328
        buf315 = buf200; del buf200  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_21.run(buf308, buf315, 1576, 256, grid=grid(1576, 256), stream=stream0)
        buf316 = reinterpret_tensor(buf308, (1576, 256), (256, 1), 0); del buf308  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf315, permute_337, out=buf316)
        del permute_337
        buf309 = reinterpret_tensor(buf301, (32, 1, 64), (64, 64, 1), 0); del buf301  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf307, (32, 1, 197), (197, 0, 1), 0), permute_329, out=buf309)
        del permute_329
        buf322 = buf57; del buf57  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf309, (8, 256), (256, 1), 0), permute_344, out=buf322)
        del permute_344
        buf325 = reinterpret_tensor(buf170, (8, 197, 256), (50432, 256, 1), 0); del buf170  # reuse
        buf339 = reinterpret_tensor(buf169, (8, 197, 256), (50432, 256, 1), 0); del buf169  # reuse
        # Source Nodes: [l__mod___blocks_1_fusion_0_norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_22.run(buf311, buf316, buf322, primals_149, cat_6, getitem_105, rsqrt_24, buf298, buf325, buf339, 1576, 256, grid=grid(1576), stream=stream0)
        del primals_149
        buf340 = reinterpret_tensor(buf272, (8, 128), (128, 1), 0); del buf272  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf339, (8, 256), (50432, 1), 0), permute_350, out=buf340)
        del permute_350
        buf343 = buf99; del buf99  # reuse
        buf344 = buf98; del buf98  # reuse
        # Source Nodes: [l__mod___blocks_1_projs_0_0, l__mod___blocks_1_projs_0_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_23.run(buf340, mul_164, primals_141, primals_142, buf343, buf344, 8, 128, grid=grid(8), stream=stream0)
        buf330 = reinterpret_tensor(buf278, (8, 401, 128), (51328, 128, 1), 0); del buf278  # reuse
        buf347 = reinterpret_tensor(buf236, (8, 401, 128), (51328, 128, 1), 0); del buf236  # reuse
        # Source Nodes: [l__mod___blocks_1_fusion_1_norm1, l__mod___blocks_1_projs_0_0, l__mod___blocks_1_projs_0_1], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_per_fused_add_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_24.run(buf274, buf279, buf285, primals_163, cat_8, getitem_109, rsqrt_26, buf261, buf255, div_28, buf340, mul_164, primals_141, primals_142, buf343, buf344, buf330, buf347, 3208, 128, grid=grid(3208), stream=stream0)
        del div_28
        del primals_163
        buf289 = reinterpret_tensor(buf281, (128, 26), (1, 128), 0); del buf281  # reuse
        buf291 = buf251; del buf251  # reuse
        # Source Nodes: [l__mod___blocks_1_fusion_1_norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_25.run(buf274, buf279, buf285, cat_8, getitem_109, rsqrt_26, buf289, buf291, 3328, 124, grid=grid(3328), stream=stream0)
        del buf274
        del buf279
        del cat_8
        del getitem_109
        del rsqrt_26
        buf290 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_fusion_1_norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf289, buf290, 128, 26, grid=grid(128), stream=stream0)
        buf292 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf291, buf292, 128, 26, grid=grid(128), stream=stream0)
        buf294 = empty((128, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf255, (128, 8), (1, 51328), 0), view_152, out=buf294)
        del view_152
        buf295 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf255, buf295, 128, 8, grid=grid(128), stream=stream0)
        del buf255
        buf299 = empty((256, ), device='cuda', dtype=torch.float32)
        buf300 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_revert_projs_0_0, l__mod___blocks_1_revert_projs_0_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_27.run(buf293, mul_177, primals_159, primals_160, buf299, buf300, 256, 8, grid=grid(256), stream=stream0)
        del mul_177
        del primals_159
        del primals_160
        buf302 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf298, (256, 8), (1, 256), 0), view_150, out=buf302)
        del view_150
        buf303 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_28.run(buf298, buf303, 256, 8, grid=grid(256), stream=stream0)
        buf312 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf310, (256, 1576), (1, 256), 0), view_137, out=buf312)
        buf313 = reinterpret_tensor(buf291, (1, 256, 13), (3328, 1, 256), 0); del buf291  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_29.run(buf310, buf313, 3328, 122, grid=grid(3328), stream=stream0)
        del buf310
        buf314 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf313, buf314, 256, 13, grid=grid(256), stream=stream0)
        buf317 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf315, (256, 1576), (1, 256), 0), view_137, out=buf317)
        del view_137
        buf318 = buf313; del buf313  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_29.run(buf315, buf318, 3328, 122, grid=grid(3328), stream=stream0)
        del buf315
        buf319 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf318, buf319, 256, 13, grid=grid(256), stream=stream0)
        buf320 = empty((1, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_28.run(buf309, buf320, 256, 8, grid=grid(256), stream=stream0)
        buf321 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf309, (256, 8), (1, 256), 0), view_134, out=buf321)
        del view_134
        buf326 = reinterpret_tensor(buf318, (256, 13), (1, 256), 0); del buf318  # reuse
        buf328 = reinterpret_tensor(buf289, (256, 13), (1, 256), 0); del buf289  # reuse
        # Source Nodes: [l__mod___blocks_1_fusion_0_norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_30.run(buf311, buf316, buf322, cat_6, getitem_105, rsqrt_24, buf326, buf328, 3328, 122, grid=grid(3328), stream=stream0)
        del buf311
        del buf316
        del cat_6
        del getitem_105
        buf327 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_fusion_0_norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf326, buf327, 256, 13, grid=grid(256), stream=stream0)
        buf329 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf328, buf329, 256, 13, grid=grid(256), stream=stream0)
        buf331 = buf322; del buf322  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf330, (8, 128), (51328, 1), 0), permute_346, out=buf331)
        del permute_346
        buf332 = empty((128, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf330, (128, 8), (1, 51328), 0), view_132, out=buf332)
        del view_132
        buf333 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf330, buf333, 128, 8, grid=grid(128), stream=stream0)
        buf334 = buf344; del buf344  # reuse
        buf335 = buf343; del buf343  # reuse
        # Source Nodes: [l__mod___blocks_1_projs_1_0, l__mod___blocks_1_projs_1_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_31.run(buf331, mul_169, primals_145, primals_146, buf334, buf335, 8, 256, grid=grid(8), stream=stream0)
        buf336 = empty((256, ), device='cuda', dtype=torch.float32)
        buf337 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_projs_1_0, l__mod___blocks_1_projs_1_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_27.run(buf331, mul_169, primals_145, primals_146, buf336, buf337, 256, 8, grid=grid(256), stream=stream0)
        buf338 = buf219; del buf219  # reuse
        # Source Nodes: [l__mod___blocks_1_projs_1_0, l__mod___blocks_1_projs_1_1], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_poi_fused_add_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_32.run(buf338, rsqrt_24, buf325, buf298, div_27, buf331, mul_169, primals_145, primals_146, buf334, buf335, 403456, grid=grid(403456), stream=stream0)
        del buf325
        del div_27
        del mul_169
        del primals_145
        del primals_146
        del rsqrt_24
        buf341 = empty((256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf339, (256, 8), (1, 50432), 0), view_130, out=buf341)
        del view_130
        buf342 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf339, buf342, 256, 8, grid=grid(256), stream=stream0)
        buf345 = empty((128, ), device='cuda', dtype=torch.float32)
        buf346 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_1_projs_0_0, l__mod___blocks_1_projs_0_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_12.run(buf340, mul_164, primals_141, primals_142, buf345, buf346, 128, 8, grid=grid(128), stream=stream0)
        del mul_164
        del primals_141
        del primals_142
        buf348 = reinterpret_tensor(buf208, (1576, 768), (768, 1), 0); del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf338, (1576, 256), (256, 1), 0), permute_354, out=buf348)
        del permute_354
        buf349 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf338, (256, 1576), (1, 256), 0), view_128, out=buf349)
        del view_128
        buf350 = reinterpret_tensor(buf328, (1, 256, 13), (3328, 1, 256), 0); del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_33.run(buf338, buf350, 3328, 122, grid=grid(3328), stream=stream0)
        buf351 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf350, buf351, 256, 13, grid=grid(256), stream=stream0)
        buf352 = reinterpret_tensor(buf348, (8, 197, 768), (151296, 768, 1), 0); del buf348  # reuse
        # Source Nodes: [x_101], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_34.run(buf352, addmm_40, 1210368, grid=grid(1210368), stream=stream0)
        del addmm_40
        buf353 = reinterpret_tensor(buf339, (1576, 256), (256, 1), 0); del buf339  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf352, (1576, 768), (768, 1), 0), permute_358, out=buf353)
        del permute_358
        buf354 = empty((768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf352, (768, 1576), (1, 768), 0), view_126, out=buf354)
        del view_126
        buf355 = reinterpret_tensor(buf247, (1, 768, 13), (9984, 1, 768), 0); del buf247  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_35.run(buf352, buf355, 9984, 122, grid=grid(9984), stream=stream0)
        buf356 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_36.run(buf355, buf356, 768, 13, grid=grid(768), stream=stream0)
        buf363 = buf338; del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_37.run(buf363, buf353, primals_135, mul_159, div_29, 1576, 256, grid=grid(1576), stream=stream0)
        del div_29
        del primals_135
        buf359 = reinterpret_tensor(buf350, (256, 13), (1, 256), 0); del buf350  # reuse
        buf361 = buf326; del buf326  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf353, mul_159, buf359, buf361, 3328, 122, grid=grid(3328), stream=stream0)
        del mul_159
        buf360 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf359, buf360, 256, 13, grid=grid(256), stream=stream0)
        buf362 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf361, buf362, 256, 13, grid=grid(256), stream=stream0)
        buf364 = buf353; del buf353  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf363, (1576, 256), (256, 1), 0), permute_362, out=buf364)
        del permute_362
        buf365 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf363, (256, 1576), (1, 256), 0), view_124, out=buf365)
        del view_124
        buf366 = reinterpret_tensor(buf361, (1, 256, 13), (3328, 1, 256), 0); del buf361  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_33.run(buf363, buf366, 3328, 122, grid=grid(3328), stream=stream0)
        buf367 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf366, buf367, 256, 13, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf368 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf364, (8, 4, 197, 64), (50432, 64, 256, 1), 0), getitem_91, getitem_92, getitem_93, None, alias_26, getitem_95, getitem_96, getitem_97, 0.0, [True, True, True, False])
        del alias_26
        del buf364
        del getitem_91
        del getitem_92
        del getitem_93
        del getitem_95
        del getitem_96
        del getitem_97
        buf369 = buf368[0]
        buf370 = buf368[1]
        buf371 = buf368[2]
        del buf368
        buf372 = reinterpret_tensor(buf352, (8, 197, 3, 4, 64), (151296, 768, 256, 64, 1), 0); del buf352  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf369, buf370, buf371, buf372, 1210368, grid=grid(1210368), stream=stream0)
        del buf369
        del buf370
        buf373 = reinterpret_tensor(buf371, (1576, 256), (256, 1), 0); del buf371  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf372, (1576, 768), (768, 1), 0), permute_368, out=buf373)
        del permute_368
        buf374 = empty((768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf372, (768, 1576), (1, 768), 0), view_120, out=buf374)
        del view_120
        buf375 = buf355; del buf355  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_35.run(buf372, buf375, 9984, 122, grid=grid(9984), stream=stream0)
        buf376 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_36.run(buf375, buf376, 768, 13, grid=grid(768), stream=stream0)
        buf383 = buf363; del buf363  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_37.run(buf383, buf373, primals_129, mul_157, div_30, 1576, 256, grid=grid(1576), stream=stream0)
        del div_30
        del primals_129
        buf379 = reinterpret_tensor(buf366, (256, 13), (1, 256), 0); del buf366  # reuse
        buf381 = buf359; del buf359  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf373, mul_157, buf379, buf381, 3328, 122, grid=grid(3328), stream=stream0)
        del mul_157
        buf380 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf379, buf380, 256, 13, grid=grid(256), stream=stream0)
        buf382 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf381, buf382, 256, 13, grid=grid(256), stream=stream0)
        buf384 = reinterpret_tensor(buf372, (1576, 768), (768, 1), 0); del buf372  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf383, (1576, 256), (256, 1), 0), permute_372, out=buf384)
        del permute_372
        buf385 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf383, (256, 1576), (1, 256), 0), view_118, out=buf385)
        del view_118
        buf386 = reinterpret_tensor(buf381, (1, 256, 13), (3328, 1, 256), 0); del buf381  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_33.run(buf383, buf386, 3328, 122, grid=grid(3328), stream=stream0)
        buf387 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf386, buf387, 256, 13, grid=grid(256), stream=stream0)
        buf388 = reinterpret_tensor(buf384, (8, 197, 768), (151296, 768, 1), 0); del buf384  # reuse
        # Source Nodes: [x_89], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_34.run(buf388, addmm_36, 1210368, grid=grid(1210368), stream=stream0)
        del addmm_36
        buf389 = buf373; del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf388, (1576, 768), (768, 1), 0), permute_376, out=buf389)
        del permute_376
        buf390 = empty((768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf388, (768, 1576), (1, 768), 0), view_116, out=buf390)
        del view_116
        buf391 = buf375; del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_35.run(buf388, buf391, 9984, 122, grid=grid(9984), stream=stream0)
        buf392 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_36.run(buf391, buf392, 768, 13, grid=grid(768), stream=stream0)
        buf399 = buf383; del buf383  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_37.run(buf399, buf389, primals_123, mul_152, div_31, 1576, 256, grid=grid(1576), stream=stream0)
        del div_31
        del primals_123
        buf395 = reinterpret_tensor(buf386, (256, 13), (1, 256), 0); del buf386  # reuse
        buf397 = buf379; del buf379  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf389, mul_152, buf395, buf397, 3328, 122, grid=grid(3328), stream=stream0)
        del mul_152
        buf396 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf395, buf396, 256, 13, grid=grid(256), stream=stream0)
        buf398 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf397, buf398, 256, 13, grid=grid(256), stream=stream0)
        buf400 = buf389; del buf389  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf399, (1576, 256), (256, 1), 0), permute_380, out=buf400)
        del permute_380
        buf401 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf399, (256, 1576), (1, 256), 0), view_114, out=buf401)
        del view_114
        buf402 = reinterpret_tensor(buf397, (1, 256, 13), (3328, 1, 256), 0); del buf397  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_33.run(buf399, buf402, 3328, 122, grid=grid(3328), stream=stream0)
        buf403 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf402, buf403, 256, 13, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf404 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf400, (8, 4, 197, 64), (50432, 64, 256, 1), 0), getitem_80, getitem_81, getitem_82, None, alias_27, getitem_84, getitem_85, getitem_86, 0.0, [True, True, True, False])
        del alias_27
        del buf400
        del getitem_80
        del getitem_81
        del getitem_82
        del getitem_84
        del getitem_85
        del getitem_86
        buf405 = buf404[0]
        buf406 = buf404[1]
        buf407 = buf404[2]
        del buf404
        buf408 = reinterpret_tensor(buf388, (8, 197, 3, 4, 64), (151296, 768, 256, 64, 1), 0); del buf388  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf405, buf406, buf407, buf408, 1210368, grid=grid(1210368), stream=stream0)
        buf409 = reinterpret_tensor(buf407, (1576, 256), (256, 1), 0); del buf407  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf408, (1576, 768), (768, 1), 0), permute_386, out=buf409)
        del permute_386
        buf410 = empty((768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf408, (768, 1576), (1, 768), 0), view_110, out=buf410)
        del view_110
        buf411 = buf391; del buf391  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_35.run(buf408, buf411, 9984, 122, grid=grid(9984), stream=stream0)
        buf412 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_36.run(buf411, buf412, 768, 13, grid=grid(768), stream=stream0)
        buf419 = buf399; del buf399  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_37.run(buf419, buf409, primals_117, mul_150, div_32, 1576, 256, grid=grid(1576), stream=stream0)
        del div_32
        del primals_117
        buf415 = reinterpret_tensor(buf402, (256, 13), (1, 256), 0); del buf402  # reuse
        buf417 = buf395; del buf395  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf409, mul_150, buf415, buf417, 3328, 122, grid=grid(3328), stream=stream0)
        del mul_150
        buf416 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf415, buf416, 256, 13, grid=grid(256), stream=stream0)
        buf418 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf417, buf418, 256, 13, grid=grid(256), stream=stream0)
        buf420 = reinterpret_tensor(buf408, (1576, 768), (768, 1), 0); del buf408  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf419, (1576, 256), (256, 1), 0), permute_390, out=buf420)
        del permute_390
        buf421 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf419, (256, 1576), (1, 256), 0), view_108, out=buf421)
        del view_108
        buf422 = reinterpret_tensor(buf417, (1, 256, 13), (3328, 1, 256), 0); del buf417  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_33.run(buf419, buf422, 3328, 122, grid=grid(3328), stream=stream0)
        buf423 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf422, buf423, 256, 13, grid=grid(256), stream=stream0)
        buf424 = reinterpret_tensor(buf420, (8, 197, 768), (151296, 768, 1), 0); del buf420  # reuse
        # Source Nodes: [x_77], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_34.run(buf424, addmm_32, 1210368, grid=grid(1210368), stream=stream0)
        del addmm_32
        buf425 = buf409; del buf409  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf424, (1576, 768), (768, 1), 0), permute_394, out=buf425)
        del permute_394
        buf426 = empty((768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf424, (768, 1576), (1, 768), 0), view_106, out=buf426)
        del view_106
        buf427 = buf411; del buf411  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_35.run(buf424, buf427, 9984, 122, grid=grid(9984), stream=stream0)
        buf428 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_36.run(buf427, buf428, 768, 13, grid=grid(768), stream=stream0)
        buf435 = buf419; del buf419  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_37.run(buf435, buf425, primals_111, mul_145, div_33, 1576, 256, grid=grid(1576), stream=stream0)
        del div_33
        del primals_111
        buf431 = reinterpret_tensor(buf422, (256, 13), (1, 256), 0); del buf422  # reuse
        buf433 = buf415; del buf415  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf425, mul_145, buf431, buf433, 3328, 122, grid=grid(3328), stream=stream0)
        del mul_145
        buf432 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf431, buf432, 256, 13, grid=grid(256), stream=stream0)
        buf434 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf433, buf434, 256, 13, grid=grid(256), stream=stream0)
        buf436 = buf425; del buf425  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf435, (1576, 256), (256, 1), 0), permute_398, out=buf436)
        del permute_398
        buf437 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf435, (256, 1576), (1, 256), 0), view_104, out=buf437)
        del view_104
        buf438 = reinterpret_tensor(buf433, (1, 256, 13), (3328, 1, 256), 0); del buf433  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_33.run(buf435, buf438, 3328, 122, grid=grid(3328), stream=stream0)
        buf439 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf438, buf439, 256, 13, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf440 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf436, (8, 4, 197, 64), (50432, 64, 256, 1), 0), getitem_69, getitem_70, getitem_71, None, alias_28, getitem_73, getitem_74, getitem_75, 0.0, [True, True, True, False])
        del alias_28
        del getitem_69
        del getitem_70
        del getitem_71
        del getitem_73
        del getitem_74
        del getitem_75
        buf441 = buf440[0]
        buf442 = buf440[1]
        buf443 = buf440[2]
        del buf440
        buf444 = reinterpret_tensor(buf424, (8, 197, 3, 4, 64), (151296, 768, 256, 64, 1), 0); del buf424  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf441, buf442, buf443, buf444, 1210368, grid=grid(1210368), stream=stream0)
        buf445 = reinterpret_tensor(buf443, (1576, 256), (256, 1), 0); del buf443  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf444, (1576, 768), (768, 1), 0), permute_404, out=buf445)
        del permute_404
        buf446 = empty((768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf444, (768, 1576), (1, 768), 0), view_100, out=buf446)
        del view_100
        buf447 = buf427; del buf427  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_35.run(buf444, buf447, 9984, 122, grid=grid(9984), stream=stream0)
        buf448 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_36.run(buf447, buf448, 768, 13, grid=grid(768), stream=stream0)
        buf455 = buf435; del buf435  # reuse
        # Source Nodes: [getattr_l__mod___blocks_1_blocks_1___0___norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_40.run(buf455, buf445, primals_105, cat_5, getitem_68, rsqrt_16, 1576, 256, grid=grid(1576), stream=stream0)
        del primals_105
        buf451 = reinterpret_tensor(buf438, (256, 13), (1, 256), 0); del buf438  # reuse
        buf453 = buf431; del buf431  # reuse
        # Source Nodes: [getattr_l__mod___blocks_1_blocks_1___0___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_41.run(buf445, cat_5, getitem_68, rsqrt_16, buf451, buf453, 3328, 122, grid=grid(3328), stream=stream0)
        del cat_5
        del getitem_68
        del rsqrt_16
        buf452 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_1_blocks_1___0___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf451, buf452, 256, 13, grid=grid(256), stream=stream0)
        buf454 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf453, buf454, 256, 13, grid=grid(256), stream=stream0)
        buf456 = reinterpret_tensor(buf244, (3208, 384), (384, 1), 0); del buf244  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf347, (3208, 128), (128, 1), 0), permute_408, out=buf456)
        del permute_408
        buf457 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf347, (128, 3208), (1, 128), 0), view_98, out=buf457)
        del view_98
        buf458 = reinterpret_tensor(buf453, (1, 128, 26), (3328, 1, 128), 0); del buf453  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_42.run(buf347, buf458, 3328, 124, grid=grid(3328), stream=stream0)
        buf459 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf458, buf459, 128, 26, grid=grid(128), stream=stream0)
        buf460 = reinterpret_tensor(buf456, (8, 401, 384), (153984, 384, 1), 0); del buf456  # reuse
        # Source Nodes: [x_65], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_43.run(buf460, addmm_28, 1231872, grid=grid(1231872), stream=stream0)
        del addmm_28
        buf461 = reinterpret_tensor(buf330, (3208, 128), (128, 1), 0); del buf330  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf460, (3208, 384), (384, 1), 0), permute_412, out=buf461)
        del permute_412
        buf462 = empty((384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf460, (384, 3208), (1, 384), 0), view_96, out=buf462)
        del view_96
        buf463 = reinterpret_tensor(buf447, (1, 384, 26), (9984, 1, 384), 0); del buf447  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_44.run(buf460, buf463, 9984, 124, grid=grid(9984), stream=stream0)
        buf464 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_45.run(buf463, buf464, 384, 26, grid=grid(384), stream=stream0)
        buf471 = buf347; del buf347  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_46.run(buf471, buf461, primals_99, mul_138, div_35, 3208, 128, grid=grid(3208), stream=stream0)
        del div_35
        del primals_99
        buf467 = reinterpret_tensor(buf458, (128, 26), (1, 128), 0); del buf458  # reuse
        buf469 = reinterpret_tensor(buf451, (128, 26), (1, 128), 0); del buf451  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_47.run(buf461, mul_138, buf467, buf469, 3328, 124, grid=grid(3328), stream=stream0)
        del mul_138
        buf468 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf467, buf468, 128, 26, grid=grid(128), stream=stream0)
        buf470 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf469, buf470, 128, 26, grid=grid(128), stream=stream0)
        buf472 = buf461; del buf461  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf471, (3208, 128), (128, 1), 0), permute_416, out=buf472)
        del permute_416
        buf473 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf471, (128, 3208), (1, 128), 0), view_94, out=buf473)
        del view_94
        buf474 = reinterpret_tensor(buf469, (1, 128, 26), (3328, 1, 128), 0); del buf469  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_42.run(buf471, buf474, 3328, 124, grid=grid(3328), stream=stream0)
        buf475 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf474, buf475, 128, 26, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf476 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf472, (8, 4, 401, 32), (51328, 32, 128, 1), 0), getitem_58, getitem_59, getitem_60, None, alias_29, getitem_62, getitem_63, getitem_64, 0.0, [True, True, True, False])
        del alias_29
        del getitem_58
        del getitem_59
        del getitem_60
        del getitem_62
        del getitem_63
        del getitem_64
        buf477 = buf476[0]
        buf478 = buf476[1]
        buf479 = buf476[2]
        del buf476
        buf480 = reinterpret_tensor(buf460, (8, 401, 3, 4, 32), (153984, 384, 128, 32, 1), 0); del buf460  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_48.run(buf477, buf478, buf479, buf480, 1231872, grid=grid(1231872), stream=stream0)
        buf481 = reinterpret_tensor(buf479, (3208, 128), (128, 1), 0); del buf479  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf480, (3208, 384), (384, 1), 0), permute_422, out=buf481)
        del permute_422
        buf482 = empty((384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf480, (384, 3208), (1, 384), 0), view_90, out=buf482)
        del view_90
        buf483 = buf463; del buf463  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_44.run(buf480, buf483, 9984, 124, grid=grid(9984), stream=stream0)
        buf484 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_45.run(buf483, buf484, 384, 26, grid=grid(384), stream=stream0)
        buf491 = buf471; del buf471  # reuse
        # Source Nodes: [getattr_l__mod___blocks_1_blocks_0___0___norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_49.run(buf491, buf481, primals_93, cat_3, getitem_57, rsqrt_14, 3208, 128, grid=grid(3208), stream=stream0)
        del primals_93
        buf487 = reinterpret_tensor(buf474, (128, 26), (1, 128), 0); del buf474  # reuse
        buf489 = buf467; del buf467  # reuse
        # Source Nodes: [getattr_l__mod___blocks_1_blocks_0___0___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_50.run(buf481, cat_3, getitem_57, rsqrt_14, buf487, buf489, 3328, 124, grid=grid(3328), stream=stream0)
        del cat_3
        del getitem_57
        del rsqrt_14
        buf488 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___blocks_1_blocks_0___0___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf487, buf488, 128, 26, grid=grid(128), stream=stream0)
        buf490 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf489, buf490, 128, 26, grid=grid(128), stream=stream0)
        buf492 = buf340; del buf340  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf455, (8, 256), (50432, 1), 0), permute_426, out=buf492)
        del permute_426
        buf493 = empty((256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf455, (256, 8), (1, 50432), 0), view_88, out=buf493)
        del view_88
        buf494 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf455, buf494, 256, 8, grid=grid(256), stream=stream0)
        buf497 = reinterpret_tensor(buf285, (8, 1, 128), (128, 128, 1), 0); del buf285  # reuse
        # Source Nodes: [l__mod___blocks_0_revert_projs_1_0, l__mod___blocks_0_revert_projs_1_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_11.run(buf492, mul_131, primals_89, primals_90, div_37, buf497, 8, 128, grid=grid(8), stream=stream0)
        del div_37
        buf498 = empty((128, ), device='cuda', dtype=torch.float32)
        buf499 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_revert_projs_1_0, l__mod___blocks_0_revert_projs_1_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_12.run(buf492, mul_131, primals_89, primals_90, buf498, buf499, 128, 8, grid=grid(128), stream=stream0)
        del mul_131
        del primals_89
        del primals_90
        buf500 = buf492; del buf492  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf497, (8, 128), (128, 1), 0), permute_430, out=buf500)
        del permute_430
        buf501 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf497, (128, 8), (1, 128), 0), view_86, out=buf501)
        del view_86
        buf502 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf497, buf502, 128, 8, grid=grid(128), stream=stream0)
        buf503 = reinterpret_tensor(buf481, (32, 401, 32), (12832, 32, 1), 0); del buf481  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_435, reinterpret_tensor(buf500, (32, 1, 32), (32, 32, 1), 0), out=buf503)
        del permute_435
        buf504 = reinterpret_tensor(buf270, (32, 1, 401), (401, 401, 1), 0); del buf270  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf500, (32, 1, 32), (32, 32, 1), 0), permute_436, out=buf504)
        del permute_436
        buf506 = reinterpret_tensor(buf268, (8, 4, 1, 401), (1604, 401, 401, 1), 0); del buf268  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_14.run(buf504, alias_30, buf506, 32, 401, grid=grid(32), stream=stream0)
        del alias_30
        del buf504
        buf507 = reinterpret_tensor(buf478, (32, 32, 401), (12832, 401, 1), 0); del buf478  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_437, reinterpret_tensor(buf506, (32, 1, 401), (401, 0, 1), 0), out=buf507)
        del permute_437
        buf508 = reinterpret_tensor(buf500, (32, 1, 32), (32, 32, 1), 0); del buf500  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf506, (32, 1, 401), (401, 0, 1), 0), permute_438, out=buf508)
        del buf506
        del permute_438
        buf509 = reinterpret_tensor(buf477, (3208, 128), (128, 1), 0); del buf477  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf503, buf509, 410624, grid=grid(410624), stream=stream0)
        buf510 = reinterpret_tensor(buf503, (3208, 128), (128, 1), 0); del buf503  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf509, permute_441, out=buf510)
        del permute_441
        buf511 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf509, (128, 3208), (1, 128), 0), view_73, out=buf511)
        buf512 = reinterpret_tensor(buf489, (1, 128, 26), (3328, 1, 128), 0); del buf489  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf509, buf512, 3328, 124, grid=grid(3328), stream=stream0)
        buf513 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf512, buf513, 128, 26, grid=grid(128), stream=stream0)
        buf514 = buf509; del buf509  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_17.run(buf507, buf514, 3208, 128, grid=grid(3208, 128), stream=stream0)
        buf515 = reinterpret_tensor(buf507, (3208, 128), (128, 1), 0); del buf507  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf514, permute_446, out=buf515)
        del permute_446
        buf516 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf514, (128, 3208), (1, 128), 0), view_73, out=buf516)
        del view_73
        buf517 = buf512; del buf512  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf514, buf517, 3328, 124, grid=grid(3328), stream=stream0)
        buf518 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf517, buf518, 128, 26, grid=grid(128), stream=stream0)
        buf519 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf508, buf519, 128, 8, grid=grid(128), stream=stream0)
        buf520 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf508, (128, 8), (1, 128), 0), view_70, out=buf520)
        del view_70
        buf521 = reinterpret_tensor(buf261, (8, 128), (128, 1), 0); del buf261  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf508, (8, 128), (128, 1), 0), permute_453, out=buf521)
        del permute_453
        buf529 = buf331; del buf331  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf491, (8, 128), (51328, 1), 0), permute_455, out=buf529)
        del permute_455
        buf534 = buf298; del buf298  # reuse
        # Source Nodes: [l__mod___blocks_0_revert_projs_0_0, l__mod___blocks_0_revert_projs_0_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_18.run(buf529, mul_123, primals_75, primals_76, div_39, buf534, 8, 256, grid=grid(8), stream=stream0)
        del div_39
        buf537 = reinterpret_tensor(buf309, (8, 256), (256, 1), 0); del buf309  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf534, (8, 256), (256, 1), 0), permute_459, out=buf537)
        del permute_459
        buf540 = reinterpret_tensor(buf445, (32, 197, 64), (12608, 64, 1), 0); del buf445  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_464, reinterpret_tensor(buf537, (32, 1, 64), (64, 64, 1), 0), out=buf540)
        del permute_464
        buf546 = reinterpret_tensor(buf442, (1576, 256), (256, 1), 0); del buf442  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf540, buf546, 403456, grid=grid(403456), stream=stream0)
        buf547 = reinterpret_tensor(buf540, (1576, 256), (256, 1), 0); del buf540  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf546, permute_470, out=buf547)
        del permute_470
        buf541 = reinterpret_tensor(buf307, (32, 1, 197), (197, 197, 1), 0); del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf537, (32, 1, 64), (64, 64, 1), 0), permute_465, out=buf541)
        del permute_465
        buf543 = reinterpret_tensor(buf305, (8, 4, 1, 197), (788, 197, 197, 1), 0); del buf305  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_20.run(buf541, alias_31, buf543, 32, 197, grid=grid(32), stream=stream0)
        del alias_31
        del buf541
        buf544 = reinterpret_tensor(buf441, (32, 64, 197), (12608, 197, 1), 0); del buf441  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_466, reinterpret_tensor(buf543, (32, 1, 197), (197, 0, 1), 0), out=buf544)
        del permute_466
        buf551 = buf436; del buf436  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_21.run(buf544, buf551, 1576, 256, grid=grid(1576, 256), stream=stream0)
        buf552 = reinterpret_tensor(buf544, (1576, 256), (256, 1), 0); del buf544  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf551, permute_475, out=buf552)
        del permute_475
        buf545 = reinterpret_tensor(buf537, (32, 1, 64), (64, 64, 1), 0); del buf537  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf543, (32, 1, 197), (197, 0, 1), 0), permute_467, out=buf545)
        del buf543
        del permute_467
        buf558 = buf293; del buf293  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf545, (8, 256), (256, 1), 0), permute_482, out=buf558)
        del permute_482
        buf561 = reinterpret_tensor(buf406, (8, 197, 256), (50432, 256, 1), 0); del buf406  # reuse
        buf575 = reinterpret_tensor(buf405, (8, 197, 256), (50432, 256, 1), 0); del buf405  # reuse
        # Source Nodes: [l__mod___blocks_0_fusion_0_norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_22.run(buf547, buf552, buf558, primals_65, cat_2, getitem_49, rsqrt_10, buf534, buf561, buf575, 1576, 256, grid=grid(1576), stream=stream0)
        del primals_65
        buf576 = reinterpret_tensor(buf508, (8, 128), (128, 1), 0); del buf508  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf575, (8, 256), (50432, 1), 0), permute_488, out=buf576)
        del permute_488
        buf579 = buf335; del buf335  # reuse
        buf580 = buf334; del buf334  # reuse
        # Source Nodes: [l__mod___blocks_0_projs_0_0, l__mod___blocks_0_projs_0_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_23.run(buf576, mul_110, primals_57, primals_58, buf579, buf580, 8, 128, grid=grid(8), stream=stream0)
        buf566 = reinterpret_tensor(buf514, (8, 401, 128), (51328, 128, 1), 0); del buf514  # reuse
        buf583 = reinterpret_tensor(buf472, (8, 401, 128), (51328, 128, 1), 0); del buf472  # reuse
        # Source Nodes: [l__mod___blocks_0_fusion_1_norm1, l__mod___blocks_0_projs_0_0, l__mod___blocks_0_projs_0_1], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_per_fused_add_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_24.run(buf510, buf515, buf521, primals_79, cat_4, getitem_53, rsqrt_12, buf497, buf491, div_42, buf576, mul_110, primals_57, primals_58, buf579, buf580, buf566, buf583, 3208, 128, grid=grid(3208), stream=stream0)
        del buf497
        del div_42
        del primals_79
        buf525 = reinterpret_tensor(buf517, (128, 26), (1, 128), 0); del buf517  # reuse
        buf527 = buf487; del buf487  # reuse
        # Source Nodes: [l__mod___blocks_0_fusion_1_norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_25.run(buf510, buf515, buf521, cat_4, getitem_53, rsqrt_12, buf525, buf527, 3328, 124, grid=grid(3328), stream=stream0)
        del buf510
        del buf515
        del buf521
        del cat_4
        del getitem_53
        del rsqrt_12
        buf526 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_fusion_1_norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf525, buf526, 128, 26, grid=grid(128), stream=stream0)
        buf528 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf527, buf528, 128, 26, grid=grid(128), stream=stream0)
        buf530 = empty((128, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf491, (128, 8), (1, 51328), 0), view_68, out=buf530)
        del view_68
        buf531 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf491, buf531, 128, 8, grid=grid(128), stream=stream0)
        del buf491
        buf535 = empty((256, ), device='cuda', dtype=torch.float32)
        buf536 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_revert_projs_0_0, l__mod___blocks_0_revert_projs_0_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_27.run(buf529, mul_123, primals_75, primals_76, buf535, buf536, 256, 8, grid=grid(256), stream=stream0)
        del buf529
        del mul_123
        del primals_75
        del primals_76
        buf538 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf534, (256, 8), (1, 256), 0), view_66, out=buf538)
        del view_66
        buf539 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_28.run(buf534, buf539, 256, 8, grid=grid(256), stream=stream0)
        buf548 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf546, (256, 1576), (1, 256), 0), view_53, out=buf548)
        buf549 = reinterpret_tensor(buf527, (1, 256, 13), (3328, 1, 256), 0); del buf527  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_29.run(buf546, buf549, 3328, 122, grid=grid(3328), stream=stream0)
        del buf546
        buf550 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf549, buf550, 256, 13, grid=grid(256), stream=stream0)
        buf553 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf551, (256, 1576), (1, 256), 0), view_53, out=buf553)
        del view_53
        buf554 = buf549; del buf549  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_29.run(buf551, buf554, 3328, 122, grid=grid(3328), stream=stream0)
        del buf551
        buf555 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf554, buf555, 256, 13, grid=grid(256), stream=stream0)
        buf556 = empty((1, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_28.run(buf545, buf556, 256, 8, grid=grid(256), stream=stream0)
        buf557 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf545, (256, 8), (1, 256), 0), view_50, out=buf557)
        del buf545
        del view_50
        buf562 = reinterpret_tensor(buf554, (256, 13), (1, 256), 0); del buf554  # reuse
        buf564 = reinterpret_tensor(buf525, (256, 13), (1, 256), 0); del buf525  # reuse
        # Source Nodes: [l__mod___blocks_0_fusion_0_norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_30.run(buf547, buf552, buf558, cat_2, getitem_49, rsqrt_10, buf562, buf564, 3328, 122, grid=grid(3328), stream=stream0)
        del buf547
        del buf552
        del cat_2
        del getitem_49
        buf563 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_fusion_0_norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf562, buf563, 256, 13, grid=grid(256), stream=stream0)
        buf565 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf564, buf565, 256, 13, grid=grid(256), stream=stream0)
        buf567 = buf558; del buf558  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf566, (8, 128), (51328, 1), 0), permute_484, out=buf567)
        del permute_484
        buf568 = empty((128, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf566, (128, 8), (1, 51328), 0), view_48, out=buf568)
        del view_48
        buf569 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf566, buf569, 128, 8, grid=grid(128), stream=stream0)
        buf570 = buf580; del buf580  # reuse
        buf571 = buf579; del buf579  # reuse
        # Source Nodes: [l__mod___blocks_0_projs_1_0, l__mod___blocks_0_projs_1_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_31.run(buf567, mul_115, primals_61, primals_62, buf570, buf571, 8, 256, grid=grid(8), stream=stream0)
        buf572 = empty((256, ), device='cuda', dtype=torch.float32)
        buf573 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_projs_1_0, l__mod___blocks_0_projs_1_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_27.run(buf567, mul_115, primals_61, primals_62, buf572, buf573, 256, 8, grid=grid(256), stream=stream0)
        buf574 = buf455; del buf455  # reuse
        # Source Nodes: [l__mod___blocks_0_projs_1_0, l__mod___blocks_0_projs_1_1], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_poi_fused_add_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_32.run(buf574, rsqrt_10, buf561, buf534, div_41, buf567, mul_115, primals_61, primals_62, buf570, buf571, 403456, grid=grid(403456), stream=stream0)
        del buf534
        del buf561
        del buf567
        del buf570
        del buf571
        del div_41
        del mul_115
        del primals_61
        del primals_62
        del rsqrt_10
        buf577 = empty((256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf575, (256, 8), (1, 50432), 0), view_46, out=buf577)
        del view_46
        buf578 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf575, buf578, 256, 8, grid=grid(256), stream=stream0)
        buf581 = empty((128, ), device='cuda', dtype=torch.float32)
        buf582 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___blocks_0_projs_0_0, l__mod___blocks_0_projs_0_1], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_12.run(buf576, mul_110, primals_57, primals_58, buf581, buf582, 128, 8, grid=grid(128), stream=stream0)
        del buf576
        del mul_110
        del primals_57
        del primals_58
        buf584 = reinterpret_tensor(buf444, (1576, 768), (768, 1), 0); del buf444  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf574, (1576, 256), (256, 1), 0), permute_492, out=buf584)
        del permute_492
        buf585 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf574, (256, 1576), (1, 256), 0), view_44, out=buf585)
        del view_44
        buf586 = reinterpret_tensor(buf564, (1, 256, 13), (3328, 1, 256), 0); del buf564  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_33.run(buf574, buf586, 3328, 122, grid=grid(3328), stream=stream0)
        buf587 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf586, buf587, 256, 13, grid=grid(256), stream=stream0)
        buf588 = reinterpret_tensor(buf584, (8, 197, 768), (151296, 768, 1), 0); del buf584  # reuse
        # Source Nodes: [x_45], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_34.run(buf588, addmm_14, 1210368, grid=grid(1210368), stream=stream0)
        del addmm_14
        buf589 = reinterpret_tensor(buf575, (1576, 256), (256, 1), 0); del buf575  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf588, (1576, 768), (768, 1), 0), permute_496, out=buf589)
        del permute_496
        buf590 = empty((768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf588, (768, 1576), (1, 768), 0), view_42, out=buf590)
        del view_42
        buf591 = reinterpret_tensor(buf483, (1, 768, 13), (9984, 1, 768), 0); del buf483  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_35.run(buf588, buf591, 9984, 122, grid=grid(9984), stream=stream0)
        buf592 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_36.run(buf591, buf592, 768, 13, grid=grid(768), stream=stream0)
        buf599 = buf574; del buf574  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_37.run(buf599, buf589, primals_51, mul_105, div_43, 1576, 256, grid=grid(1576), stream=stream0)
        del div_43
        del primals_51
        buf595 = reinterpret_tensor(buf586, (256, 13), (1, 256), 0); del buf586  # reuse
        buf597 = buf562; del buf562  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf589, mul_105, buf595, buf597, 3328, 122, grid=grid(3328), stream=stream0)
        del mul_105
        buf596 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf595, buf596, 256, 13, grid=grid(256), stream=stream0)
        buf598 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf597, buf598, 256, 13, grid=grid(256), stream=stream0)
        buf600 = buf589; del buf589  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf599, (1576, 256), (256, 1), 0), permute_500, out=buf600)
        del permute_500
        buf601 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf599, (256, 1576), (1, 256), 0), view_40, out=buf601)
        del view_40
        buf602 = reinterpret_tensor(buf597, (1, 256, 13), (3328, 1, 256), 0); del buf597  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_33.run(buf599, buf602, 3328, 122, grid=grid(3328), stream=stream0)
        buf603 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf602, buf603, 256, 13, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf604 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf600, (8, 4, 197, 64), (50432, 64, 256, 1), 0), getitem_35, getitem_36, getitem_37, None, alias_32, getitem_39, getitem_40, getitem_41, 0.0, [True, True, True, False])
        del alias_32
        del buf600
        del getitem_35
        del getitem_36
        del getitem_37
        del getitem_39
        del getitem_40
        del getitem_41
        buf605 = buf604[0]
        buf606 = buf604[1]
        buf607 = buf604[2]
        del buf604
        buf608 = reinterpret_tensor(buf588, (8, 197, 3, 4, 64), (151296, 768, 256, 64, 1), 0); del buf588  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf605, buf606, buf607, buf608, 1210368, grid=grid(1210368), stream=stream0)
        del buf605
        del buf606
        buf609 = reinterpret_tensor(buf607, (1576, 256), (256, 1), 0); del buf607  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf608, (1576, 768), (768, 1), 0), permute_506, out=buf609)
        del permute_506
        buf610 = empty((768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf608, (768, 1576), (1, 768), 0), view_36, out=buf610)
        del view_36
        buf611 = buf591; del buf591  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_35.run(buf608, buf611, 9984, 122, grid=grid(9984), stream=stream0)
        buf612 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_36.run(buf611, buf612, 768, 13, grid=grid(768), stream=stream0)
        buf619 = buf599; del buf599  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_37.run(buf619, buf609, primals_45, mul_103, div_44, 1576, 256, grid=grid(1576), stream=stream0)
        del div_44
        del primals_45
        buf615 = reinterpret_tensor(buf602, (256, 13), (1, 256), 0); del buf602  # reuse
        buf617 = buf595; del buf595  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf609, mul_103, buf615, buf617, 3328, 122, grid=grid(3328), stream=stream0)
        del mul_103
        buf616 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf615, buf616, 256, 13, grid=grid(256), stream=stream0)
        buf618 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf617, buf618, 256, 13, grid=grid(256), stream=stream0)
        buf620 = reinterpret_tensor(buf608, (1576, 768), (768, 1), 0); del buf608  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf619, (1576, 256), (256, 1), 0), permute_510, out=buf620)
        del permute_510
        buf621 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf619, (256, 1576), (1, 256), 0), view_34, out=buf621)
        del view_34
        buf622 = reinterpret_tensor(buf617, (1, 256, 13), (3328, 1, 256), 0); del buf617  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_33.run(buf619, buf622, 3328, 122, grid=grid(3328), stream=stream0)
        buf623 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf622, buf623, 256, 13, grid=grid(256), stream=stream0)
        buf624 = reinterpret_tensor(buf620, (8, 197, 768), (151296, 768, 1), 0); del buf620  # reuse
        # Source Nodes: [x_33], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_34.run(buf624, addmm_10, 1210368, grid=grid(1210368), stream=stream0)
        del addmm_10
        buf625 = buf609; del buf609  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf624, (1576, 768), (768, 1), 0), permute_514, out=buf625)
        del permute_514
        buf626 = empty((768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf624, (768, 1576), (1, 768), 0), view_32, out=buf626)
        del view_32
        buf627 = buf611; del buf611  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_35.run(buf624, buf627, 9984, 122, grid=grid(9984), stream=stream0)
        buf628 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_36.run(buf627, buf628, 768, 13, grid=grid(768), stream=stream0)
        buf635 = buf619; del buf619  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_37.run(buf635, buf625, primals_39, mul_98, div_45, 1576, 256, grid=grid(1576), stream=stream0)
        del div_45
        del primals_39
        buf631 = reinterpret_tensor(buf622, (256, 13), (1, 256), 0); del buf622  # reuse
        buf633 = buf615; del buf615  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf625, mul_98, buf631, buf633, 3328, 122, grid=grid(3328), stream=stream0)
        del mul_98
        buf632 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf631, buf632, 256, 13, grid=grid(256), stream=stream0)
        buf634 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf633, buf634, 256, 13, grid=grid(256), stream=stream0)
        buf636 = buf625; del buf625  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf635, (1576, 256), (256, 1), 0), permute_518, out=buf636)
        del permute_518
        buf637 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf635, (256, 1576), (1, 256), 0), view_30, out=buf637)
        del view_30
        buf638 = reinterpret_tensor(buf633, (1, 256, 13), (3328, 1, 256), 0); del buf633  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_33.run(buf635, buf638, 3328, 122, grid=grid(3328), stream=stream0)
        buf639 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf638, buf639, 256, 13, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf640 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf636, (8, 4, 197, 64), (50432, 64, 256, 1), 0), getitem_24, getitem_25, getitem_26, None, alias_33, getitem_28, getitem_29, getitem_30, 0.0, [True, True, True, False])
        del alias_33
        del buf636
        del getitem_24
        del getitem_25
        del getitem_26
        del getitem_28
        del getitem_29
        del getitem_30
        buf641 = buf640[0]
        buf642 = buf640[1]
        buf643 = buf640[2]
        del buf640
        buf644 = reinterpret_tensor(buf624, (8, 197, 3, 4, 64), (151296, 768, 256, 64, 1), 0); del buf624  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf641, buf642, buf643, buf644, 1210368, grid=grid(1210368), stream=stream0)
        del buf641
        del buf642
        buf645 = reinterpret_tensor(buf643, (1576, 256), (256, 1), 0); del buf643  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf644, (1576, 768), (768, 1), 0), permute_524, out=buf645)
        del permute_524
        buf646 = empty((768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf644, (768, 1576), (1, 768), 0), view_26, out=buf646)
        del view_26
        buf647 = buf627; del buf627  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_35.run(buf644, buf647, 9984, 122, grid=grid(9984), stream=stream0)
        buf648 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_36.run(buf647, buf648, 768, 13, grid=grid(768), stream=stream0)
        buf655 = buf635; del buf635  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_37.run(buf655, buf645, primals_33, mul_96, div_46, 1576, 256, grid=grid(1576), stream=stream0)
        del div_46
        del primals_33
        buf651 = reinterpret_tensor(buf638, (256, 13), (1, 256), 0); del buf638  # reuse
        buf653 = buf631; del buf631  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf645, mul_96, buf651, buf653, 3328, 122, grid=grid(3328), stream=stream0)
        del mul_96
        buf652 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf651, buf652, 256, 13, grid=grid(256), stream=stream0)
        buf654 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf653, buf654, 256, 13, grid=grid(256), stream=stream0)
        buf656 = reinterpret_tensor(buf644, (1576, 768), (768, 1), 0); del buf644  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf655, (1576, 256), (256, 1), 0), permute_528, out=buf656)
        del permute_528
        buf657 = empty((256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf655, (256, 1576), (1, 256), 0), view_24, out=buf657)
        del view_24
        buf658 = reinterpret_tensor(buf653, (1, 256, 13), (3328, 1, 256), 0); del buf653  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_33.run(buf655, buf658, 3328, 122, grid=grid(3328), stream=stream0)
        buf659 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf658, buf659, 256, 13, grid=grid(256), stream=stream0)
        buf660 = reinterpret_tensor(buf656, (8, 197, 768), (151296, 768, 1), 0); del buf656  # reuse
        # Source Nodes: [x_21], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_34.run(buf660, addmm_6, 1210368, grid=grid(1210368), stream=stream0)
        del addmm_6
        buf661 = buf645; del buf645  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf660, (1576, 768), (768, 1), 0), permute_532, out=buf661)
        del permute_532
        buf662 = empty((768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf660, (768, 1576), (1, 768), 0), view_22, out=buf662)
        del view_22
        buf663 = buf647; del buf647  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_35.run(buf660, buf663, 9984, 122, grid=grid(9984), stream=stream0)
        buf664 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_36.run(buf663, buf664, 768, 13, grid=grid(768), stream=stream0)
        buf671 = buf655; del buf655  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_37.run(buf671, buf661, primals_27, mul_91, div_47, 1576, 256, grid=grid(1576), stream=stream0)
        del div_47
        del primals_27
        buf667 = reinterpret_tensor(buf658, (256, 13), (1, 256), 0); del buf658  # reuse
        buf669 = buf651; del buf651  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf661, mul_91, buf667, buf669, 3328, 122, grid=grid(3328), stream=stream0)
        del mul_91
        buf668 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf667, buf668, 256, 13, grid=grid(256), stream=stream0)
        buf670 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf669, buf670, 256, 13, grid=grid(256), stream=stream0)
        buf672 = buf661; del buf661  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf671, (1576, 256), (256, 1), 0), permute_536, out=buf672)
        del permute_536
        buf673 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf671, (256, 1576), (1, 256), 0), view_20, out=buf673)
        del view_20
        buf674 = reinterpret_tensor(buf669, (1, 256, 13), (3328, 1, 256), 0); del buf669  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_33.run(buf671, buf674, 3328, 122, grid=grid(3328), stream=stream0)
        buf675 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf674, buf675, 256, 13, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf676 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf672, (8, 4, 197, 64), (50432, 64, 256, 1), 0), getitem_13, getitem_14, getitem_15, None, alias_34, getitem_17, getitem_18, getitem_19, 0.0, [True, True, True, False])
        del alias_34
        del buf672
        del getitem_13
        del getitem_14
        del getitem_15
        del getitem_17
        del getitem_18
        del getitem_19
        buf677 = buf676[0]
        buf678 = buf676[1]
        buf679 = buf676[2]
        del buf676
        buf680 = reinterpret_tensor(buf660, (8, 197, 3, 4, 64), (151296, 768, 256, 64, 1), 0); del buf660  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf677, buf678, buf679, buf680, 1210368, grid=grid(1210368), stream=stream0)
        del buf677
        del buf678
        buf681 = reinterpret_tensor(buf679, (1576, 256), (256, 1), 0); del buf679  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf680, (1576, 768), (768, 1), 0), permute_542, out=buf681)
        del permute_542
        buf682 = empty((768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf680, (768, 1576), (1, 768), 0), view_16, out=buf682)
        del view_16
        buf683 = buf663; del buf663  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_35.run(buf680, buf683, 9984, 122, grid=grid(9984), stream=stream0)
        del buf680
        buf684 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_36.run(buf683, buf684, 768, 13, grid=grid(768), stream=stream0)
        buf691 = buf671; del buf671  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_37.run(buf691, buf681, primals_21, mul_89, div_48, 1576, 256, grid=grid(1576), stream=stream0)
        del div_48
        del primals_21
        buf687 = reinterpret_tensor(buf674, (256, 13), (1, 256), 0); del buf674  # reuse
        buf689 = buf667; del buf667  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_38.run(buf681, mul_89, buf687, buf689, 3328, 122, grid=grid(3328), stream=stream0)
        del buf681
        del mul_89
        buf688 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf687, buf688, 256, 13, grid=grid(256), stream=stream0)
        buf690 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf689, buf690, 256, 13, grid=grid(256), stream=stream0)
        buf692 = reinterpret_tensor(buf480, (3208, 384), (384, 1), 0); del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf583, (3208, 128), (128, 1), 0), permute_546, out=buf692)
        del permute_546
        buf693 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf583, (128, 3208), (1, 128), 0), view_14, out=buf693)
        del view_14
        buf694 = reinterpret_tensor(buf689, (1, 128, 26), (3328, 1, 128), 0); del buf689  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_42.run(buf583, buf694, 3328, 124, grid=grid(3328), stream=stream0)
        buf695 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf694, buf695, 128, 26, grid=grid(128), stream=stream0)
        buf696 = reinterpret_tensor(buf692, (8, 401, 384), (153984, 384, 1), 0); del buf692  # reuse
        # Source Nodes: [x_9], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_43.run(buf696, addmm_2, 1231872, grid=grid(1231872), stream=stream0)
        del addmm_2
        buf697 = reinterpret_tensor(buf566, (3208, 128), (128, 1), 0); del buf566  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf696, (3208, 384), (384, 1), 0), permute_550, out=buf697)
        del permute_550
        buf698 = empty((384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf696, (384, 3208), (1, 384), 0), view_12, out=buf698)
        del view_12
        buf699 = reinterpret_tensor(buf683, (1, 384, 26), (9984, 1, 384), 0); del buf683  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_44.run(buf696, buf699, 9984, 124, grid=grid(9984), stream=stream0)
        buf700 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_45.run(buf699, buf700, 384, 26, grid=grid(384), stream=stream0)
        buf707 = buf583; del buf583  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_46.run(buf707, buf697, primals_15, mul_84, div_49, 3208, 128, grid=grid(3208), stream=stream0)
        del div_49
        del primals_15
        buf703 = reinterpret_tensor(buf694, (128, 26), (1, 128), 0); del buf694  # reuse
        buf705 = reinterpret_tensor(buf687, (128, 26), (1, 128), 0); del buf687  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_47.run(buf697, mul_84, buf703, buf705, 3328, 124, grid=grid(3328), stream=stream0)
        del mul_84
        buf704 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf703, buf704, 128, 26, grid=grid(128), stream=stream0)
        buf706 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf705, buf706, 128, 26, grid=grid(128), stream=stream0)
        buf708 = buf697; del buf697  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf707, (3208, 128), (128, 1), 0), permute_554, out=buf708)
        del permute_554
        buf709 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf707, (128, 3208), (1, 128), 0), view_10, out=buf709)
        del view_10
        buf710 = reinterpret_tensor(buf705, (1, 128, 26), (3328, 1, 128), 0); del buf705  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_42.run(buf707, buf710, 3328, 124, grid=grid(3328), stream=stream0)
        buf711 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf710, buf711, 128, 26, grid=grid(128), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf712 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf708, (8, 4, 401, 32), (51328, 32, 128, 1), 0), getitem_2, getitem_3, getitem_4, None, alias_35, getitem_6, getitem_7, getitem_8, 0.0, [True, True, True, False])
        del alias_35
        del buf708
        del getitem_2
        del getitem_3
        del getitem_4
        del getitem_6
        del getitem_7
        del getitem_8
        buf713 = buf712[0]
        buf714 = buf712[1]
        buf715 = buf712[2]
        del buf712
        buf716 = reinterpret_tensor(buf696, (8, 401, 3, 4, 32), (153984, 384, 128, 32, 1), 0); del buf696  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_48.run(buf713, buf714, buf715, buf716, 1231872, grid=grid(1231872), stream=stream0)
        del buf713
        del buf714
        buf717 = reinterpret_tensor(buf715, (3208, 128), (128, 1), 0); del buf715  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf716, (3208, 384), (384, 1), 0), permute_560, out=buf717)
        del permute_560
        buf718 = empty((384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf716, (384, 3208), (1, 384), 0), view_6, out=buf718)
        del view_6
        buf719 = buf699; del buf699  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_44.run(buf716, buf719, 9984, 124, grid=grid(9984), stream=stream0)
        del buf716
        buf720 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_45.run(buf719, buf720, 384, 26, grid=grid(384), stream=stream0)
        del buf719
        buf727 = buf707; del buf707  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_46.run(buf727, buf717, primals_9, mul_82, div_50, 3208, 128, grid=grid(3208), stream=stream0)
        del div_50
        del primals_9
        buf723 = reinterpret_tensor(buf710, (128, 26), (1, 128), 0); del buf710  # reuse
        buf725 = buf703; del buf703  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_47.run(buf717, mul_82, buf723, buf725, 3328, 124, grid=grid(3328), stream=stream0)
        del buf717
        del mul_82
        buf724 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf723, buf724, 128, 26, grid=grid(128), stream=stream0)
        del buf723
        buf726 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_8.run(buf725, buf726, 128, 26, grid=grid(128), stream=stream0)
        buf728 = empty((1, 197, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_51.run(buf691, buf728, 50432, 8, grid=grid(50432), stream=stream0)
        buf729 = empty((1, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf691, buf729, 256, 8, grid=grid(256), stream=stream0)
        buf730 = reinterpret_tensor(buf725, (256, 13), (1, 256), 0); del buf725  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_52.run(buf691, buf730, 3328, 121, grid=grid(3328), stream=stream0)
        buf731 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_select_backward_4.run(buf730, buf731, 256, 13, grid=grid(256), stream=stream0)
        del buf730
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf732 = aten.convolution_backward(reinterpret_tensor(buf691, (8, 256, 14, 14), (50432, 1, 3584, 256), 256), add_46, primals_7, [256], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])
        del add_46
        del buf691
        del primals_7
        buf733 = buf732[1]
        del buf732
        buf734 = empty((1, 401, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_53.run(buf727, buf734, 51328, 8, grid=grid(51328), stream=stream0)
        buf735 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_26.run(buf727, buf735, 128, 8, grid=grid(128), stream=stream0)
        buf736 = empty_strided((128, 25), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_54.run(buf727, buf736, 3200, 128, grid=grid(3200), stream=stream0)
        buf737 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_55.run(buf736, buf737, 128, 25, grid=grid(128), stream=stream0)
        del buf736
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf738 = aten.convolution_backward(reinterpret_tensor(buf727, (8, 128, 20, 20), (51328, 1, 2560, 128), 128), primals_269, primals_5, [128], [12, 12], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf727
        del primals_269
        del primals_5
        buf739 = buf738[1]
        return (buf735, buf734, buf729, buf728, buf739, buf737, buf733, buf731, buf724, buf726, reinterpret_tensor(buf718, (384, 128), (128, 1), 0), reinterpret_tensor(buf720, (384, ), (1, ), 0), reinterpret_tensor(buf709, (128, 128), (128, 1), 0), reinterpret_tensor(buf711, (128, ), (1, ), 0), buf704, buf706, reinterpret_tensor(buf698, (384, 128), (128, 1), 0), reinterpret_tensor(buf700, (384, ), (1, ), 0), reinterpret_tensor(buf693, (128, 384), (384, 1), 0), reinterpret_tensor(buf695, (128, ), (1, ), 0), buf688, buf690, reinterpret_tensor(buf682, (768, 256), (256, 1), 0), reinterpret_tensor(buf684, (768, ), (1, ), 0), reinterpret_tensor(buf673, (256, 256), (256, 1), 0), reinterpret_tensor(buf675, (256, ), (1, ), 0), buf668, buf670, reinterpret_tensor(buf662, (768, 256), (256, 1), 0), reinterpret_tensor(buf664, (768, ), (1, ), 0), reinterpret_tensor(buf657, (256, 768), (768, 1), 0), reinterpret_tensor(buf659, (256, ), (1, ), 0), buf652, buf654, reinterpret_tensor(buf646, (768, 256), (256, 1), 0), reinterpret_tensor(buf648, (768, ), (1, ), 0), reinterpret_tensor(buf637, (256, 256), (256, 1), 0), reinterpret_tensor(buf639, (256, ), (1, ), 0), buf632, buf634, reinterpret_tensor(buf626, (768, 256), (256, 1), 0), reinterpret_tensor(buf628, (768, ), (1, ), 0), reinterpret_tensor(buf621, (256, 768), (768, 1), 0), reinterpret_tensor(buf623, (256, ), (1, ), 0), buf616, buf618, reinterpret_tensor(buf610, (768, 256), (256, 1), 0), reinterpret_tensor(buf612, (768, ), (1, ), 0), reinterpret_tensor(buf601, (256, 256), (256, 1), 0), reinterpret_tensor(buf603, (256, ), (1, ), 0), buf596, buf598, reinterpret_tensor(buf590, (768, 256), (256, 1), 0), reinterpret_tensor(buf592, (768, ), (1, ), 0), reinterpret_tensor(buf585, (256, 768), (768, 1), 0), reinterpret_tensor(buf587, (256, ), (1, ), 0), buf581, buf582, reinterpret_tensor(buf577, (256, 128), (128, 1), 0), reinterpret_tensor(buf578, (256, ), (1, ), 0), buf572, buf573, reinterpret_tensor(buf568, (128, 256), (256, 1), 0), reinterpret_tensor(buf569, (128, ), (1, ), 0), buf563, buf565, reinterpret_tensor(buf557, (256, 256), (256, 1), 0), reinterpret_tensor(buf556, (256, ), (1, ), 0), reinterpret_tensor(buf553, (256, 256), (256, 1), 0), reinterpret_tensor(buf555, (256, ), (1, ), 0), reinterpret_tensor(buf548, (256, 256), (256, 1), 0), reinterpret_tensor(buf550, (256, ), (1, ), 0), reinterpret_tensor(buf538, (256, 256), (256, 1), 0), reinterpret_tensor(buf539, (256, ), (1, ), 0), buf535, buf536, reinterpret_tensor(buf530, (128, 256), (256, 1), 0), reinterpret_tensor(buf531, (128, ), (1, ), 0), buf526, buf528, reinterpret_tensor(buf520, (128, 128), (128, 1), 0), reinterpret_tensor(buf519, (128, ), (1, ), 0), reinterpret_tensor(buf516, (128, 128), (128, 1), 0), reinterpret_tensor(buf518, (128, ), (1, ), 0), reinterpret_tensor(buf511, (128, 128), (128, 1), 0), reinterpret_tensor(buf513, (128, ), (1, ), 0), reinterpret_tensor(buf501, (128, 128), (128, 1), 0), reinterpret_tensor(buf502, (128, ), (1, ), 0), buf498, buf499, reinterpret_tensor(buf493, (256, 128), (128, 1), 0), reinterpret_tensor(buf494, (256, ), (1, ), 0), buf488, buf490, reinterpret_tensor(buf482, (384, 128), (128, 1), 0), reinterpret_tensor(buf484, (384, ), (1, ), 0), reinterpret_tensor(buf473, (128, 128), (128, 1), 0), reinterpret_tensor(buf475, (128, ), (1, ), 0), buf468, buf470, reinterpret_tensor(buf462, (384, 128), (128, 1), 0), reinterpret_tensor(buf464, (384, ), (1, ), 0), reinterpret_tensor(buf457, (128, 384), (384, 1), 0), reinterpret_tensor(buf459, (128, ), (1, ), 0), buf452, buf454, reinterpret_tensor(buf446, (768, 256), (256, 1), 0), reinterpret_tensor(buf448, (768, ), (1, ), 0), reinterpret_tensor(buf437, (256, 256), (256, 1), 0), reinterpret_tensor(buf439, (256, ), (1, ), 0), buf432, buf434, reinterpret_tensor(buf426, (768, 256), (256, 1), 0), reinterpret_tensor(buf428, (768, ), (1, ), 0), reinterpret_tensor(buf421, (256, 768), (768, 1), 0), reinterpret_tensor(buf423, (256, ), (1, ), 0), buf416, buf418, reinterpret_tensor(buf410, (768, 256), (256, 1), 0), reinterpret_tensor(buf412, (768, ), (1, ), 0), reinterpret_tensor(buf401, (256, 256), (256, 1), 0), reinterpret_tensor(buf403, (256, ), (1, ), 0), buf396, buf398, reinterpret_tensor(buf390, (768, 256), (256, 1), 0), reinterpret_tensor(buf392, (768, ), (1, ), 0), reinterpret_tensor(buf385, (256, 768), (768, 1), 0), reinterpret_tensor(buf387, (256, ), (1, ), 0), buf380, buf382, reinterpret_tensor(buf374, (768, 256), (256, 1), 0), reinterpret_tensor(buf376, (768, ), (1, ), 0), reinterpret_tensor(buf365, (256, 256), (256, 1), 0), reinterpret_tensor(buf367, (256, ), (1, ), 0), buf360, buf362, reinterpret_tensor(buf354, (768, 256), (256, 1), 0), reinterpret_tensor(buf356, (768, ), (1, ), 0), reinterpret_tensor(buf349, (256, 768), (768, 1), 0), reinterpret_tensor(buf351, (256, ), (1, ), 0), buf345, buf346, reinterpret_tensor(buf341, (256, 128), (128, 1), 0), reinterpret_tensor(buf342, (256, ), (1, ), 0), buf336, buf337, reinterpret_tensor(buf332, (128, 256), (256, 1), 0), reinterpret_tensor(buf333, (128, ), (1, ), 0), buf327, buf329, reinterpret_tensor(buf321, (256, 256), (256, 1), 0), reinterpret_tensor(buf320, (256, ), (1, ), 0), reinterpret_tensor(buf317, (256, 256), (256, 1), 0), reinterpret_tensor(buf319, (256, ), (1, ), 0), reinterpret_tensor(buf312, (256, 256), (256, 1), 0), reinterpret_tensor(buf314, (256, ), (1, ), 0), reinterpret_tensor(buf302, (256, 256), (256, 1), 0), reinterpret_tensor(buf303, (256, ), (1, ), 0), buf299, buf300, reinterpret_tensor(buf294, (128, 256), (256, 1), 0), reinterpret_tensor(buf295, (128, ), (1, ), 0), buf290, buf292, reinterpret_tensor(buf284, (128, 128), (128, 1), 0), reinterpret_tensor(buf283, (128, ), (1, ), 0), reinterpret_tensor(buf280, (128, 128), (128, 1), 0), reinterpret_tensor(buf282, (128, ), (1, ), 0), reinterpret_tensor(buf275, (128, 128), (128, 1), 0), reinterpret_tensor(buf277, (128, ), (1, ), 0), reinterpret_tensor(buf265, (128, 128), (128, 1), 0), reinterpret_tensor(buf266, (128, ), (1, ), 0), buf262, buf263, reinterpret_tensor(buf257, (256, 128), (128, 1), 0), reinterpret_tensor(buf258, (256, ), (1, ), 0), buf252, buf254, reinterpret_tensor(buf246, (384, 128), (128, 1), 0), reinterpret_tensor(buf248, (384, ), (1, ), 0), reinterpret_tensor(buf237, (128, 128), (128, 1), 0), reinterpret_tensor(buf239, (128, ), (1, ), 0), buf232, buf234, reinterpret_tensor(buf226, (384, 128), (128, 1), 0), reinterpret_tensor(buf228, (384, ), (1, ), 0), reinterpret_tensor(buf221, (128, 384), (384, 1), 0), reinterpret_tensor(buf223, (128, ), (1, ), 0), buf216, buf218, reinterpret_tensor(buf210, (768, 256), (256, 1), 0), reinterpret_tensor(buf212, (768, ), (1, ), 0), reinterpret_tensor(buf201, (256, 256), (256, 1), 0), reinterpret_tensor(buf203, (256, ), (1, ), 0), buf196, buf198, reinterpret_tensor(buf190, (768, 256), (256, 1), 0), reinterpret_tensor(buf192, (768, ), (1, ), 0), reinterpret_tensor(buf185, (256, 768), (768, 1), 0), reinterpret_tensor(buf187, (256, ), (1, ), 0), buf180, buf182, reinterpret_tensor(buf174, (768, 256), (256, 1), 0), reinterpret_tensor(buf176, (768, ), (1, ), 0), reinterpret_tensor(buf165, (256, 256), (256, 1), 0), reinterpret_tensor(buf167, (256, ), (1, ), 0), buf160, buf162, reinterpret_tensor(buf154, (768, 256), (256, 1), 0), reinterpret_tensor(buf156, (768, ), (1, ), 0), reinterpret_tensor(buf149, (256, 768), (768, 1), 0), reinterpret_tensor(buf151, (256, ), (1, ), 0), buf144, buf146, reinterpret_tensor(buf138, (768, 256), (256, 1), 0), reinterpret_tensor(buf140, (768, ), (1, ), 0), reinterpret_tensor(buf129, (256, 256), (256, 1), 0), reinterpret_tensor(buf131, (256, ), (1, ), 0), buf124, buf126, reinterpret_tensor(buf118, (768, 256), (256, 1), 0), reinterpret_tensor(buf120, (768, ), (1, ), 0), reinterpret_tensor(buf113, (256, 768), (768, 1), 0), reinterpret_tensor(buf115, (256, ), (1, ), 0), buf109, buf110, reinterpret_tensor(buf105, (256, 128), (128, 1), 0), reinterpret_tensor(buf106, (256, ), (1, ), 0), buf100, buf101, reinterpret_tensor(buf96, (128, 256), (256, 1), 0), reinterpret_tensor(buf97, (128, ), (1, ), 0), buf91, buf93, reinterpret_tensor(buf85, (256, 256), (256, 1), 0), reinterpret_tensor(buf84, (256, ), (1, ), 0), reinterpret_tensor(buf81, (256, 256), (256, 1), 0), reinterpret_tensor(buf83, (256, ), (1, ), 0), reinterpret_tensor(buf76, (256, 256), (256, 1), 0), reinterpret_tensor(buf78, (256, ), (1, ), 0), reinterpret_tensor(buf66, (256, 256), (256, 1), 0), reinterpret_tensor(buf67, (256, ), (1, ), 0), buf63, buf64, reinterpret_tensor(buf58, (128, 256), (256, 1), 0), reinterpret_tensor(buf59, (128, ), (1, ), 0), buf54, buf56, reinterpret_tensor(buf48, (128, 128), (128, 1), 0), reinterpret_tensor(buf47, (128, ), (1, ), 0), reinterpret_tensor(buf44, (128, 128), (128, 1), 0), reinterpret_tensor(buf46, (128, ), (1, ), 0), reinterpret_tensor(buf39, (128, 128), (128, 1), 0), reinterpret_tensor(buf41, (128, ), (1, ), 0), reinterpret_tensor(buf29, (128, 128), (128, 1), 0), reinterpret_tensor(buf30, (128, ), (1, ), 0), buf26, buf27, reinterpret_tensor(buf21, (256, 128), (128, 1), 0), reinterpret_tensor(buf22, (256, ), (1, ), 0), buf18, buf19, buf12, buf13, reinterpret_tensor(buf6, (1000, 128), (128, 1), 0), reinterpret_tensor(buf7, (1000, ), (1, ), 0), reinterpret_tensor(buf3, (1000, 256), (256, 1), 0), reinterpret_tensor(buf4, (1000, ), (1, ), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_5 = rand_strided((128, 3, 12, 12), (432, 144, 12, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((256, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((8, 3, 240, 240), (172800, 57600, 240, 1), device='cuda:0', dtype=torch.float32)
    add_46 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    mul_82 = rand_strided((8, 401, 128), (51328, 128, 1), device='cuda:0', dtype=torch.float32)
    view_6 = rand_strided((3208, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    getitem_2 = rand_strided((8, 4, 401, 32), (153984, 32, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((8, 4, 401, 32), (153984, 32, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_4 = rand_strided((8, 4, 401, 32), (153984, 32, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_6 = rand_strided((8, 4, 416), (1664, 416, 1), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_8 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_10 = rand_strided((3208, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mul_84 = rand_strided((8, 401, 128), (51328, 128, 1), device='cuda:0', dtype=torch.float32)
    view_12 = rand_strided((3208, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_2 = rand_strided((3208, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_14 = rand_strided((3208, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_89 = rand_strided((8, 197, 256), (50432, 256, 1), device='cuda:0', dtype=torch.float32)
    view_16 = rand_strided((1576, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    getitem_13 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_14 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_15 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_17 = rand_strided((8, 4, 224), (896, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_18 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_19 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_20 = rand_strided((1576, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mul_91 = rand_strided((8, 197, 256), (50432, 256, 1), device='cuda:0', dtype=torch.float32)
    view_22 = rand_strided((1576, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_6 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_24 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_96 = rand_strided((8, 197, 256), (50432, 256, 1), device='cuda:0', dtype=torch.float32)
    view_26 = rand_strided((1576, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    getitem_24 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_25 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_26 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_28 = rand_strided((8, 4, 224), (896, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_29 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_30 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_30 = rand_strided((1576, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mul_98 = rand_strided((8, 197, 256), (50432, 256, 1), device='cuda:0', dtype=torch.float32)
    view_32 = rand_strided((1576, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_34 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_103 = rand_strided((8, 197, 256), (50432, 256, 1), device='cuda:0', dtype=torch.float32)
    view_36 = rand_strided((1576, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    getitem_35 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_36 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_37 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_39 = rand_strided((8, 4, 224), (896, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_40 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_41 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_40 = rand_strided((1576, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mul_105 = rand_strided((8, 197, 256), (50432, 256, 1), device='cuda:0', dtype=torch.float32)
    view_42 = rand_strided((1576, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_14 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_44 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_110 = rand_strided((8, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.float32)
    view_46 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mul_115 = rand_strided((8, 1, 256), (256, 256, 1), device='cuda:0', dtype=torch.float32)
    view_48 = rand_strided((8, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    cat_2 = rand_strided((8, 197, 256), (50432, 256, 1), device='cuda:0', dtype=torch.float32)
    getitem_49 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_10 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_50 = rand_strided((8, 256), (50432, 1), device='cuda:0', dtype=torch.float32)
    view_53 = rand_strided((1576, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    view_66 = rand_strided((8, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mul_123 = rand_strided((8, 1, 256), (256, 256, 1), device='cuda:0', dtype=torch.float32)
    view_68 = rand_strided((8, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    cat_3 = rand_strided((8, 401, 128), (51328, 128, 1), device='cuda:0', dtype=torch.float32)
    cat_4 = rand_strided((8, 401, 128), (51328, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_53 = rand_strided((8, 401, 1), (401, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_12 = rand_strided((8, 401, 1), (401, 1, 1), device='cuda:0', dtype=torch.float32)
    view_70 = rand_strided((8, 128), (51328, 1), device='cuda:0', dtype=torch.float32)
    view_73 = rand_strided((3208, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_86 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mul_131 = rand_strided((8, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.float32)
    view_88 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    cat_5 = rand_strided((8, 197, 256), (50432, 256, 1), device='cuda:0', dtype=torch.float32)
    getitem_57 = rand_strided((8, 401, 1), (401, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_14 = rand_strided((8, 401, 1), (401, 1, 1), device='cuda:0', dtype=torch.float32)
    view_90 = rand_strided((3208, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    getitem_58 = rand_strided((8, 4, 401, 32), (153984, 32, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_59 = rand_strided((8, 4, 401, 32), (153984, 32, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_60 = rand_strided((8, 4, 401, 32), (153984, 32, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_62 = rand_strided((8, 4, 416), (1664, 416, 1), device='cuda:0', dtype=torch.float32)
    getitem_63 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_64 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_94 = rand_strided((3208, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mul_138 = rand_strided((8, 401, 128), (51328, 128, 1), device='cuda:0', dtype=torch.float32)
    view_96 = rand_strided((3208, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_28 = rand_strided((3208, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_98 = rand_strided((3208, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_68 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_16 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_100 = rand_strided((1576, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    getitem_69 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_70 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_71 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_73 = rand_strided((8, 4, 224), (896, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_74 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_75 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_104 = rand_strided((1576, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mul_145 = rand_strided((8, 197, 256), (50432, 256, 1), device='cuda:0', dtype=torch.float32)
    view_106 = rand_strided((1576, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_32 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_108 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_150 = rand_strided((8, 197, 256), (50432, 256, 1), device='cuda:0', dtype=torch.float32)
    view_110 = rand_strided((1576, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    getitem_80 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_81 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_82 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_84 = rand_strided((8, 4, 224), (896, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_85 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_86 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_114 = rand_strided((1576, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mul_152 = rand_strided((8, 197, 256), (50432, 256, 1), device='cuda:0', dtype=torch.float32)
    view_116 = rand_strided((1576, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_36 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_118 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_157 = rand_strided((8, 197, 256), (50432, 256, 1), device='cuda:0', dtype=torch.float32)
    view_120 = rand_strided((1576, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    getitem_91 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_92 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_93 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_95 = rand_strided((8, 4, 224), (896, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_96 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_97 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_124 = rand_strided((1576, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mul_159 = rand_strided((8, 197, 256), (50432, 256, 1), device='cuda:0', dtype=torch.float32)
    view_126 = rand_strided((1576, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_40 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_128 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_164 = rand_strided((8, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.float32)
    view_130 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mul_169 = rand_strided((8, 1, 256), (256, 256, 1), device='cuda:0', dtype=torch.float32)
    view_132 = rand_strided((8, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    cat_6 = rand_strided((8, 197, 256), (50432, 256, 1), device='cuda:0', dtype=torch.float32)
    getitem_105 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_24 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_134 = rand_strided((8, 256), (50432, 1), device='cuda:0', dtype=torch.float32)
    view_137 = rand_strided((1576, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    view_150 = rand_strided((8, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mul_177 = rand_strided((8, 1, 256), (256, 256, 1), device='cuda:0', dtype=torch.float32)
    view_152 = rand_strided((8, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    cat_7 = rand_strided((8, 401, 128), (51328, 128, 1), device='cuda:0', dtype=torch.float32)
    cat_8 = rand_strided((8, 401, 128), (51328, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_109 = rand_strided((8, 401, 1), (401, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_26 = rand_strided((8, 401, 1), (401, 1, 1), device='cuda:0', dtype=torch.float32)
    view_154 = rand_strided((8, 128), (51328, 1), device='cuda:0', dtype=torch.float32)
    view_157 = rand_strided((3208, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_170 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mul_185 = rand_strided((8, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.float32)
    view_172 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    cat_9 = rand_strided((8, 197, 256), (50432, 256, 1), device='cuda:0', dtype=torch.float32)
    getitem_113 = rand_strided((8, 401, 1), (401, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_28 = rand_strided((8, 401, 1), (401, 1, 1), device='cuda:0', dtype=torch.float32)
    view_174 = rand_strided((3208, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    getitem_114 = rand_strided((8, 4, 401, 32), (153984, 32, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_115 = rand_strided((8, 4, 401, 32), (153984, 32, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_116 = rand_strided((8, 4, 401, 32), (153984, 32, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_118 = rand_strided((8, 4, 416), (1664, 416, 1), device='cuda:0', dtype=torch.float32)
    getitem_119 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_120 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_178 = rand_strided((3208, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mul_192 = rand_strided((8, 401, 128), (51328, 128, 1), device='cuda:0', dtype=torch.float32)
    view_180 = rand_strided((3208, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_54 = rand_strided((3208, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_182 = rand_strided((3208, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_124 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_30 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_184 = rand_strided((1576, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    getitem_125 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_126 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_127 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_129 = rand_strided((8, 4, 224), (896, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_130 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_131 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_188 = rand_strided((1576, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mul_199 = rand_strided((8, 197, 256), (50432, 256, 1), device='cuda:0', dtype=torch.float32)
    view_190 = rand_strided((1576, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_58 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_192 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_204 = rand_strided((8, 197, 256), (50432, 256, 1), device='cuda:0', dtype=torch.float32)
    view_194 = rand_strided((1576, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    getitem_136 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_137 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_138 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_140 = rand_strided((8, 4, 224), (896, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_141 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_142 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_198 = rand_strided((1576, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mul_206 = rand_strided((8, 197, 256), (50432, 256, 1), device='cuda:0', dtype=torch.float32)
    view_200 = rand_strided((1576, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_62 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_202 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_211 = rand_strided((8, 197, 256), (50432, 256, 1), device='cuda:0', dtype=torch.float32)
    view_204 = rand_strided((1576, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    getitem_147 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_148 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_149 = rand_strided((8, 4, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_151 = rand_strided((8, 4, 224), (896, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_152 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_153 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_208 = rand_strided((1576, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mul_213 = rand_strided((8, 197, 256), (50432, 256, 1), device='cuda:0', dtype=torch.float32)
    view_210 = rand_strided((1576, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_66 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_212 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_218 = rand_strided((8, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.float32)
    view_214 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mul_223 = rand_strided((8, 1, 256), (256, 256, 1), device='cuda:0', dtype=torch.float32)
    view_216 = rand_strided((8, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    cat_10 = rand_strided((8, 197, 256), (50432, 256, 1), device='cuda:0', dtype=torch.float32)
    getitem_161 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_38 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_218 = rand_strided((8, 256), (50432, 1), device='cuda:0', dtype=torch.float32)
    view_221 = rand_strided((1576, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    view_234 = rand_strided((8, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mul_231 = rand_strided((8, 1, 256), (256, 256, 1), device='cuda:0', dtype=torch.float32)
    view_236 = rand_strided((8, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    cat_11 = rand_strided((8, 401, 128), (51328, 128, 1), device='cuda:0', dtype=torch.float32)
    cat_12 = rand_strided((8, 401, 128), (51328, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_165 = rand_strided((8, 401, 1), (401, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_40 = rand_strided((8, 401, 1), (401, 1, 1), device='cuda:0', dtype=torch.float32)
    view_238 = rand_strided((8, 128), (51328, 1), device='cuda:0', dtype=torch.float32)
    view_241 = rand_strided((3208, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_254 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mul_239 = rand_strided((8, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.float32)
    view_256 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    cat_13 = rand_strided((8, 197, 256), (50432, 256, 1), device='cuda:0', dtype=torch.float32)
    getitem_169 = rand_strided((8, 401, 1), (401, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_42 = rand_strided((8, 401, 1), (401, 1, 1), device='cuda:0', dtype=torch.float32)
    getitem_171 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_43 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    clone_68 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    clone_69 = rand_strided((8, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_142 = rand_strided((1000, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_146 = rand_strided((1000, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_150 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_9 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_154 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_159 = rand_strided((32, 401, 1), (401, 1, 0), device='cuda:0', dtype=torch.float32)
    permute_160 = rand_strided((32, 32, 401), (12832, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_18 = rand_strided((8, 4, 1, 401), (1604, 401, 401, 1), device='cuda:0', dtype=torch.float32)
    permute_161 = rand_strided((32, 32, 1), (32, 1, 0), device='cuda:0', dtype=torch.float32)
    permute_162 = rand_strided((32, 401, 32), (12832, 1, 401), device='cuda:0', dtype=torch.float32)
    permute_165 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_170 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_177 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_179 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_11 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_183 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_188 = rand_strided((32, 197, 1), (197, 1, 0), device='cuda:0', dtype=torch.float32)
    permute_189 = rand_strided((32, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_19 = rand_strided((8, 4, 1, 197), (788, 197, 197, 1), device='cuda:0', dtype=torch.float32)
    permute_190 = rand_strided((32, 64, 1), (64, 1, 0), device='cuda:0', dtype=torch.float32)
    permute_191 = rand_strided((32, 197, 64), (12608, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_194 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_199 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_206 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_208 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_13 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_212 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_14 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_216 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_220 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_15 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_224 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    alias_20 = rand_strided((8, 4, 197, 64), (50432, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    permute_230 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_16 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_234 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_238 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_17 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_242 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    alias_21 = rand_strided((8, 4, 197, 64), (50432, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    permute_248 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_18 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_252 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_256 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_19 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_260 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    alias_22 = rand_strided((8, 4, 197, 64), (50432, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    permute_266 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_270 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_274 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_21 = rand_strided((8, 401, 1), (401, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_278 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    alias_23 = rand_strided((8, 4, 401, 32), (51328, 32, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_284 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_288 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_23 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_292 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_297 = rand_strided((32, 401, 1), (401, 1, 0), device='cuda:0', dtype=torch.float32)
    permute_298 = rand_strided((32, 32, 401), (12832, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_24 = rand_strided((8, 4, 1, 401), (1604, 401, 401, 1), device='cuda:0', dtype=torch.float32)
    permute_299 = rand_strided((32, 32, 1), (32, 1, 0), device='cuda:0', dtype=torch.float32)
    permute_300 = rand_strided((32, 401, 32), (12832, 1, 401), device='cuda:0', dtype=torch.float32)
    permute_303 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_308 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_315 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_317 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_25 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_321 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_326 = rand_strided((32, 197, 1), (197, 1, 0), device='cuda:0', dtype=torch.float32)
    permute_327 = rand_strided((32, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_25 = rand_strided((8, 4, 1, 197), (788, 197, 197, 1), device='cuda:0', dtype=torch.float32)
    permute_328 = rand_strided((32, 64, 1), (64, 1, 0), device='cuda:0', dtype=torch.float32)
    permute_329 = rand_strided((32, 197, 64), (12608, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_332 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_337 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_344 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_346 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_27 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_350 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_28 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_354 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_358 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_29 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_362 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    alias_26 = rand_strided((8, 4, 197, 64), (50432, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    permute_368 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_30 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_372 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_376 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_31 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_380 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    alias_27 = rand_strided((8, 4, 197, 64), (50432, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    permute_386 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_32 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_390 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_394 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_33 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_398 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    alias_28 = rand_strided((8, 4, 197, 64), (50432, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    permute_404 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_408 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_412 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_35 = rand_strided((8, 401, 1), (401, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_416 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    alias_29 = rand_strided((8, 4, 401, 32), (51328, 32, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_422 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_426 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_37 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_430 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_435 = rand_strided((32, 401, 1), (401, 1, 0), device='cuda:0', dtype=torch.float32)
    permute_436 = rand_strided((32, 32, 401), (12832, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_30 = rand_strided((8, 4, 1, 401), (1604, 401, 401, 1), device='cuda:0', dtype=torch.float32)
    permute_437 = rand_strided((32, 32, 1), (32, 1, 0), device='cuda:0', dtype=torch.float32)
    permute_438 = rand_strided((32, 401, 32), (12832, 1, 401), device='cuda:0', dtype=torch.float32)
    permute_441 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_446 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_453 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_455 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_39 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_459 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_464 = rand_strided((32, 197, 1), (197, 1, 0), device='cuda:0', dtype=torch.float32)
    permute_465 = rand_strided((32, 64, 197), (12608, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_31 = rand_strided((8, 4, 1, 197), (788, 197, 197, 1), device='cuda:0', dtype=torch.float32)
    permute_466 = rand_strided((32, 64, 1), (64, 1, 0), device='cuda:0', dtype=torch.float32)
    permute_467 = rand_strided((32, 197, 64), (12608, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_470 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_475 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_482 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_484 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_41 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_488 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_42 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_492 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_496 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_43 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_500 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    alias_32 = rand_strided((8, 4, 197, 64), (50432, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    permute_506 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_44 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_510 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_514 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_45 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_518 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    alias_33 = rand_strided((8, 4, 197, 64), (50432, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    permute_524 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_46 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_528 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_532 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_47 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_536 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    alias_34 = rand_strided((8, 4, 197, 64), (50432, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    permute_542 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_48 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_546 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_550 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_49 = rand_strided((8, 401, 1), (401, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_554 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    alias_35 = rand_strided((8, 4, 401, 32), (51328, 32, 128, 1), device='cuda:0', dtype=torch.float32)
    permute_560 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_50 = rand_strided((8, 401, 1), (401, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_5, primals_7, primals_9, primals_15, primals_21, primals_27, primals_33, primals_39, primals_45, primals_51, primals_57, primals_58, primals_61, primals_62, primals_65, primals_75, primals_76, primals_79, primals_89, primals_90, primals_93, primals_99, primals_105, primals_111, primals_117, primals_123, primals_129, primals_135, primals_141, primals_142, primals_145, primals_146, primals_149, primals_159, primals_160, primals_163, primals_173, primals_174, primals_177, primals_183, primals_189, primals_195, primals_201, primals_207, primals_213, primals_219, primals_225, primals_226, primals_229, primals_230, primals_233, primals_243, primals_244, primals_247, primals_257, primals_258, primals_261, primals_263, primals_269, add_46, mul_82, view_6, getitem_2, getitem_3, getitem_4, getitem_6, getitem_7, getitem_8, view_10, mul_84, view_12, addmm_2, view_14, mul_89, view_16, getitem_13, getitem_14, getitem_15, getitem_17, getitem_18, getitem_19, view_20, mul_91, view_22, addmm_6, view_24, mul_96, view_26, getitem_24, getitem_25, getitem_26, getitem_28, getitem_29, getitem_30, view_30, mul_98, view_32, addmm_10, view_34, mul_103, view_36, getitem_35, getitem_36, getitem_37, getitem_39, getitem_40, getitem_41, view_40, mul_105, view_42, addmm_14, view_44, mul_110, view_46, mul_115, view_48, cat_2, getitem_49, rsqrt_10, view_50, view_53, view_66, mul_123, view_68, cat_3, cat_4, getitem_53, rsqrt_12, view_70, view_73, view_86, mul_131, view_88, cat_5, getitem_57, rsqrt_14, view_90, getitem_58, getitem_59, getitem_60, getitem_62, getitem_63, getitem_64, view_94, mul_138, view_96, addmm_28, view_98, getitem_68, rsqrt_16, view_100, getitem_69, getitem_70, getitem_71, getitem_73, getitem_74, getitem_75, view_104, mul_145, view_106, addmm_32, view_108, mul_150, view_110, getitem_80, getitem_81, getitem_82, getitem_84, getitem_85, getitem_86, view_114, mul_152, view_116, addmm_36, view_118, mul_157, view_120, getitem_91, getitem_92, getitem_93, getitem_95, getitem_96, getitem_97, view_124, mul_159, view_126, addmm_40, view_128, mul_164, view_130, mul_169, view_132, cat_6, getitem_105, rsqrt_24, view_134, view_137, view_150, mul_177, view_152, cat_7, cat_8, getitem_109, rsqrt_26, view_154, view_157, view_170, mul_185, view_172, cat_9, getitem_113, rsqrt_28, view_174, getitem_114, getitem_115, getitem_116, getitem_118, getitem_119, getitem_120, view_178, mul_192, view_180, addmm_54, view_182, getitem_124, rsqrt_30, view_184, getitem_125, getitem_126, getitem_127, getitem_129, getitem_130, getitem_131, view_188, mul_199, view_190, addmm_58, view_192, mul_204, view_194, getitem_136, getitem_137, getitem_138, getitem_140, getitem_141, getitem_142, view_198, mul_206, view_200, addmm_62, view_202, mul_211, view_204, getitem_147, getitem_148, getitem_149, getitem_151, getitem_152, getitem_153, view_208, mul_213, view_210, addmm_66, view_212, mul_218, view_214, mul_223, view_216, cat_10, getitem_161, rsqrt_38, view_218, view_221, view_234, mul_231, view_236, cat_11, cat_12, getitem_165, rsqrt_40, view_238, view_241, view_254, mul_239, view_256, cat_13, getitem_169, rsqrt_42, getitem_171, rsqrt_43, clone_68, clone_69, permute_142, permute_146, permute_150, div_9, permute_154, permute_159, permute_160, alias_18, permute_161, permute_162, permute_165, permute_170, permute_177, permute_179, div_11, permute_183, permute_188, permute_189, alias_19, permute_190, permute_191, permute_194, permute_199, permute_206, permute_208, div_13, permute_212, div_14, permute_216, permute_220, div_15, permute_224, alias_20, permute_230, div_16, permute_234, permute_238, div_17, permute_242, alias_21, permute_248, div_18, permute_252, permute_256, div_19, permute_260, alias_22, permute_266, permute_270, permute_274, div_21, permute_278, alias_23, permute_284, permute_288, div_23, permute_292, permute_297, permute_298, alias_24, permute_299, permute_300, permute_303, permute_308, permute_315, permute_317, div_25, permute_321, permute_326, permute_327, alias_25, permute_328, permute_329, permute_332, permute_337, permute_344, permute_346, div_27, permute_350, div_28, permute_354, permute_358, div_29, permute_362, alias_26, permute_368, div_30, permute_372, permute_376, div_31, permute_380, alias_27, permute_386, div_32, permute_390, permute_394, div_33, permute_398, alias_28, permute_404, permute_408, permute_412, div_35, permute_416, alias_29, permute_422, permute_426, div_37, permute_430, permute_435, permute_436, alias_30, permute_437, permute_438, permute_441, permute_446, permute_453, permute_455, div_39, permute_459, permute_464, permute_465, alias_31, permute_466, permute_467, permute_470, permute_475, permute_482, permute_484, div_41, permute_488, div_42, permute_492, permute_496, div_43, permute_500, alias_32, permute_506, div_44, permute_510, permute_514, div_45, permute_518, alias_33, permute_524, div_46, permute_528, permute_532, div_47, permute_536, alias_34, permute_542, div_48, permute_546, permute_550, div_49, permute_554, alias_35, permute_560, div_50, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('crossvit_9_240', benchmark_compiled_module)
