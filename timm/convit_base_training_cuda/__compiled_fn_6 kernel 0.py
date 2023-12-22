
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


# kernel path: /tmp/torchinductor_youkaichao/mg/cmgp7vqgrwkcictvyqxbxyrz6dk5dqlv5t3iawxfqfunzovpwl6z.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]

triton_red_fused_native_layer_norm_backward_select_backward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_select_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1576
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 197
    x1 = (xindex // 197)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp3 = tl.load(in_ptr0 + (r2 + (768*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr2 + (r2 + (768*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp4 = 0.0
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp7 = tmp5 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp12 = tmp7 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp16 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp20 = tl.load(in_ptr0 + (r2 + (768*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.load(in_ptr2 + (r2 + (768*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = x0
        tmp18 = tl.full([1, 1], 0, tl.int32)
        tmp19 = tmp17 == tmp18
        tmp21 = 0.0
        tmp22 = tl.where(tmp19, tmp20, tmp21)
        tmp24 = tmp22 * tmp23
        tmp25 = 768.0
        tmp26 = tmp24 * tmp25
        tmp27 = tmp26 - tmp9
        tmp29 = tmp28 * tmp14
        tmp30 = tmp27 - tmp29
        tmp31 = tmp16 * tmp30
        tl.store(out_ptr2 + (r2 + (768*x3)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ur/cur4mi7oxatlweakndnk4623khy62xy3mwg543tgvtihlcmvacsx.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]

triton_red_fused_native_layer_norm_backward_select_backward_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_select_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9984
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 768)
    x0 = xindex % 768
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
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
        tmp6 = tl.load(in_ptr0 + (x0 + (768*(((r2 + (122*x1)) // 197) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = 0.0
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = tl.load(in_ptr1 + (x0 + (768*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tmp8 * tmp9
        tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bz/cbz7trmfeyzs5asnh5jp4532fhueiqeo7etaajsuw546yzfveodm.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]

triton_per_fused_native_layer_norm_backward_select_backward_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_select_backward_3', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/2c/c2cr6aookuoyl2lagwldx3ggwympdx57htbucslwwajylz7z3dln.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]

triton_red_fused_native_layer_norm_backward_select_backward_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_select_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
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
        tmp3 = tl.load(in_ptr0 + (x0 + (768*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/xg/cxgpcztj4blcbxn4ybt6526kjv2w5y7s2iiwnkbaxyaqini3xt7y.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_5', 'mutated_arg_names': []}
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
        tmp3 = tl.load(in_ptr0 + (x0 + (768*r2) + (93696*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yi/cyipukgo5gngueyfstqnlvgkibsbpdpbhhwohh4lwl6yrhjqxv37.py
# Source Nodes: [x_168], Original ATen: [aten.gelu, aten.gelu_backward]
# x_168 => add_113, erf_11, mul_114
triton_poi_fused_gelu_gelu_backward_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4841472
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


# kernel path: /tmp/torchinductor_youkaichao/kx/ckxdbuvjcv3gfyiyzamgc6ofkmkbrndrdfbp7yegrtzzjftkqnx5.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 39936
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 3072)
    x0 = xindex % 3072
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 1576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (3072*r2) + (374784*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/le/cle7vwfvitcycxhpunez2a3gpcw4izrllonrhyph5ewupq3jjubq.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3072*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jt/cjthhkqmrylj37hcb2r2mldky5y5aftfyjzcyn2di4z3hdtetppq.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_9', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 768.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4i/c4irnn45k6hcwtm3ir62ujlmvowu7cayrlh54t7okurjkte3o2e5.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9984
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 768)
    x0 = xindex % 768
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
        tmp3 = tl.load(in_ptr0 + (x0 + (768*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (768*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/5y/c5y4dhjjhijnjbbdexiuii5unvozblp5uaemelqkooaliawcflsr.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1210368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 48
    x1 = (xindex // 48) % 197
    x2 = (xindex // 9456) % 16
    x3 = (xindex // 151296)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (48*x2) + (768*x1) + (151296*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/w7/cw75fvdyjcoonkvwhpuq5nsjjfpiyedkfjm6oj5v6tf5e3sarwge.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]

triton_per_fused__softmax_backward_data_mul_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_mul_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25216
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
    tmp9 = 0.14433756729740643
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr1 + (r1 + (197*x0)), tmp10, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2s/c2s3qdknbjdkaeeubabyyp34apetmwjjpbl2fw75p4zuekknq4sl.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 75648
    xnumel = 48
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y6 = (yindex // 3152)
    x4 = xindex
    y7 = yindex
    y0 = yindex % 197
    y8 = (yindex // 197)
    y1 = (yindex // 197) % 16
    y2 = (yindex // 3152) % 8
    y3 = (yindex // 25216)
    tmp0 = y6
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + (48*y7)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 16, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-1210368) + y0 + (197*x4) + (9456*y8)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 24, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-2420736) + x4 + (48*y7)), tmp15 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tl.where(tmp11, tmp14, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tl.store(out_ptr0 + (x4 + (48*y1) + (768*y3) + (2304*y0) + (453888*y2)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/md/cmdwbzjefcimsamjt3iodr6g6xohdsjp3kntpjx3vl6r2kpaxzdj.py
# Source Nodes: [x_147], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# x_147 => mul_100, sub_60
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_14', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
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
    tmp18 = 768.0
    tmp19 = tmp10 / tmp18
    tmp20 = tmp2 * tmp18
    tmp21 = tmp20 - tmp6
    tmp22 = tmp11 * tmp16
    tmp23 = tmp21 - tmp22
    tmp24 = tmp19 * tmp23
    tmp25 = tmp17 + tmp24
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e3/ce3yyq6ensnay7mix5yewwfcqu7qlzdj4xuy772bykyazpmwajmr.py
# Source Nodes: [x_147], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# x_147 => mul_100, sub_60
triton_red_fused_native_layer_norm_native_layer_norm_backward_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9984
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 768)
    x0 = xindex % 768
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
        tmp3 = tl.load(in_ptr0 + (x0 + (768*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (768*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/4p/c4pepn64gqfj3k3nok7cg6kqmeacvihgi3k36sbu3jmwyhf2leia.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]

triton_poi_fused__unsafe_view_clone_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (768 + x0 + (768*(x1 % 196)) + (151296*(x1 // 196))), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hy/chyvvyuqurjerbwpqmsbywucjt7ujog5irc2qgr5upaibla35cbe.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9984
    rnumel = 121
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
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (768*r2) + (92928*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sl/csldsauagza2cex4pxuqjmiat5hg767ibugjuysqetwu7x4rsci7.py
# Source Nodes: [x_139], Original ATen: [aten.gelu, aten.gelu_backward]
# x_139 => add_99, erf_9, mul_98
triton_poi_fused_gelu_gelu_backward_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
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


# kernel path: /tmp/torchinductor_youkaichao/2i/c2icgovrcwvpifm4yvkbme5b6pyivie7rwt7ewrszjzjvbk77vms.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 39936
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 3072)
    x0 = xindex % 3072
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (3072*r2) + (371712*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/df/cdf3z5kpvg7krbhsegnty3vmmoqpz43kdc3o4uiqo6kmogeb3wz6.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 196
    x3 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (768 + r1 + (768*x2) + (151296*x3)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 768.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yg/cygy25zkv65tfcthj42hyzajzhygbtzbiyyiatbs2bvv6egleaqh.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9984
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 768)
    x0 = xindex % 768
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
        tmp3 = tl.load(in_ptr0 + (x0 + (768*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (768*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/5k/c5kbiaodnfe7vaqrf3k5mf4tb4cuevbw5ikm6g7vuuakug67wjin.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9984
    rnumel = 121
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
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (768*r2) + (92928*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f2/cf2rcqc3oa5m2kw5xaes33tt3x4zftpba26bruvcmkjcqqc7czzw.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 48
    x1 = (xindex // 48) % 196
    x2 = (xindex // 9408) % 16
    x3 = (xindex // 150528)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (48*x2) + (768*x1) + (150528*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hg/chgcttya3vt7bik6qkrlnv2empyw5bhxbkrjbwsznhgvcy6oot6d.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((48*(x1 % 196)) + (9408*(x0 // 48)) + (150528*(x1 // 196)) + (x0 % 48)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/j3/cj3owecpf7b4kzowlop2ykegrncdm25rcr2et6zw3izgto2npe3k.py
# Source Nodes: [attn_36, mul_28, mul_29, sigmoid_18, sub_19], Original ATen: [aten._softmax_backward_data, aten.add, aten.div, aten.mul, aten.neg, aten.rsub, aten.sigmoid, aten.sum]
# attn_36 => add_95
# mul_28 => mul_93
# mul_29 => mul_94
# sigmoid_18 => sigmoid_18
# sub_19 => sub_58
triton_per_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sum_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sum_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr3, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x1 = (xindex // 196) % 16
    tmp0 = tl.load(in_ptr0 + (r3 + (196*x4)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (r3 + (196*x4)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (r3 + (196*x4)), rmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr4 + (x4), xmask, eviction_policy='evict_last')
    tmp1 = -tmp0
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = 1.0
    tmp5 = tmp4 - tmp3
    tmp7 = tmp5 * tmp6
    tmp9 = tmp3 * tmp8
    tmp10 = tmp7 + tmp9
    tmp12 = tmp10 / tmp11
    tmp13 = tmp12 / tmp11
    tmp14 = tmp1 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp0 / tmp11
    tmp20 = tmp19 + tmp18
    tmp21 = tmp20 * tmp3
    tmp22 = tmp21 * tmp8
    tmp23 = tmp20 * tmp5
    tmp24 = tmp23 * tmp6
    tmp25 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp27 = tl.where(rmask & xmask, tmp25, 0)
    tmp28 = tl.sum(tmp27, 1)[:, None]
    tmp29 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = tl.sum(tmp31, 1)[:, None]
    tmp33 = tmp6 * tmp32
    tmp34 = tmp24 - tmp33
    tmp35 = 0.14433756729740643
    tmp36 = tmp34 * tmp35
    tl.store(out_ptr1 + (r3 + (196*x4)), tmp22, rmask & xmask)
    tl.store(out_ptr5 + (r3 + (196*x4)), tmp36, rmask & xmask)
    tl.store(out_ptr0 + (x4), tmp18, xmask)
    tl.store(out_ptr3 + (x4), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/li/cligruis6bvro2x5r5jpkvwa7hvucyabmf56xiy42zvb2x52lil2.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]

triton_red_fused_add_div_mul_sum_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_div_mul_sum_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 608
    rnumel = 8088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 38
    x1 = (xindex // 38)
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp29 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (8088*x0)
        tmp1 = tl.full([1, 1], 307328, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((38416*x1) + (614656*(((r2 + (8088*x0)) // 38416) % 8)) + ((r2 + (8088*x0)) % 38416)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + ((196*x1) + (3136*(((r2 + (8088*x0)) // 38416) % 8)) + (((r2 + (8088*x0)) // 196) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 / tmp4
        tmp6 = tl.load(in_ptr2 + ((196*x1) + (3136*(((r2 + (8088*x0)) // 38416) % 8)) + (((r2 + (8088*x0)) // 196) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.load(in_ptr3 + ((38416*x1) + (614656*(((r2 + (8088*x0)) // 38416) % 8)) + ((r2 + (8088*x0)) % 38416)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp7 * tmp8
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
        tmp15 = tl.load(in_ptr4 + ((38416*x1) + (614656*(((r2 + (8088*x0)) // 38416) % 8)) + ((r2 + (8088*x0)) % 38416)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tmp7 * tmp15
        tmp17 = tl.full(tmp16.shape, 0, tmp16.dtype)
        tmp18 = tl.where(tmp2, tmp16, tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
        tmp22 = tl.load(in_ptr5 + ((38416*x1) + (614656*(((r2 + (8088*x0)) // 38416) % 8)) + ((r2 + (8088*x0)) % 38416)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr6 + ((196*x1) + (3136*(((r2 + (8088*x0)) // 38416) % 8)) + (((r2 + (8088*x0)) // 196) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tmp8 * tmp23
        tmp25 = tmp22 - tmp24
        tmp26 = tl.full(tmp25.shape, 0, tmp25.dtype)
        tmp27 = tl.where(tmp2, tmp25, tmp26)
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
        tmp30 = _tmp29 + tmp28
        _tmp29 = tl.where(rmask & xmask, tmp30, _tmp29)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp20, xmask)
    tmp29 = tl.sum(_tmp29, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5f/c5fmi63b4nt7qtydyyhpnv57ngtfjhngkv6hmjlozf4sq3l3s3ko.py
# Source Nodes: [sigmoid_18], Original ATen: [aten.add, aten.div, aten.mul, aten.neg, aten.sigmoid, aten.sigmoid_backward, aten.sum, aten.view]
# sigmoid_18 => sigmoid_18
triton_per_fused_add_div_mul_neg_sigmoid_sigmoid_backward_sum_view_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_neg_sigmoid_sigmoid_backward_sum_view_27', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 38
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (38*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + (38*x0)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp11 = tl.sigmoid(tmp10)
    tmp12 = 1.0
    tmp13 = tmp12 - tmp11
    tmp14 = tmp11 * tmp13
    tmp15 = tmp4 * tmp14
    tmp16 = -tmp9
    tmp17 = tmp16 * tmp14
    tmp18 = tmp15 + tmp17
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fu/cfu5urr6avnkenbb2dutegjodjzlkbczvheovgd6bix2al6b4yff.py
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
    size_hints=[16, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 38
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (38*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t2/ct276q3wce7jbo7ajfutx7bnjgupp4uxlwh766d3xzzomrvqrstw.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 65536], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 38416
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x5 = xindex
    y4 = yindex
    x3 = (xindex // 196)
    y0 = yindex % 16
    y1 = (yindex // 16)
    tmp0 = tl.load(in_ptr0 + (x5 + (38416*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x5 + (38416*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x3 + (196*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 - tmp3
    tl.store(out_ptr0 + (y0 + (16*x5) + (614656*y1)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pn/cpnteebirzdnxsea7qfmysxmtcs2e2rkrvqpwf3tkgkacga3qmnw.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 1536
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x4 = (xindex // 768)
    x5 = xindex % 768
    y0 = yindex % 196
    y1 = (yindex // 196)
    x2 = xindex % 48
    x3 = (xindex // 48) % 16
    x7 = xindex
    y6 = yindex
    tmp3 = tl.load(in_ptr0 + (y0 + (196*x5) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x2 + (48*y0) + (9408*x3) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp0 = x4
    tmp1 = tl.full([1, 1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = tl.full([1, 1], 0, tl.int32)
    tmp7 = tmp0 == tmp6
    tmp9 = tl.where(tmp7, tmp8, tmp4)
    tmp10 = tmp5 + tmp9
    tl.store(out_ptr0 + (x7 + (1536*y6)), tmp10, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4a/c4a2224oeteobj3q73njhmew4xx7vb4nu6xgpf3i6po67y4lx5vn.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_31', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp15 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp17 = 768.0
    tmp18 = tmp4 * tmp17
    tmp19 = tmp18 - tmp8
    tmp20 = tmp9 * tmp14
    tmp21 = tmp19 - tmp20
    tmp22 = tmp16 * tmp21
    tmp23 = tmp15 + tmp22
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp23, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kc/ckcghk5a6ztxpfaiygvqve4joc276gby7b7e23w5qdhnbddhe4a6.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_red_fused_add_native_layer_norm_backward_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9984
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 768)
    x0 = xindex % 768
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (768*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (768*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.load(in_ptr2 + (x0 + (768*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tmp5 * tmp6
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
        tmp13 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp14 = tl.where(tmp2, tmp5, tmp13)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ve/cveusuennuyl6n36dar5a7nodoqzxtef3aiu7dmnhrzupu25mlko.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_33', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 768.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lo/clojoh4pb2srbfaun7z5ru45kuwvxq3ykogquld4fct6bkgig7mo.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_34', 'mutated_arg_names': []}
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
    tmp0 = tl.load(in_ptr0 + (x0 + (151296*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jg/cjghwbfnyqpgzqsizlhwrnu4y4pib5xwfhqugfwrarzdbvqzls4n.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[262144, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (150528*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q3/cq3rtrmu4rtn2nyh6kmer5twpf3ihjvoax6tyiy7hjigwyh2qm2g.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9984
    rnumel = 121
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
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (768*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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
    primals_3, primals_5, primals_6, primals_8, primals_10, primals_11, primals_13, primals_15, primals_16, primals_18, primals_20, primals_21, primals_23, primals_25, primals_26, primals_28, primals_30, primals_31, primals_33, primals_35, primals_36, primals_38, primals_40, primals_41, primals_43, primals_45, primals_46, primals_48, primals_50, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_181, mul, view_5, view_8, div, div_1, unsqueeze_5, view_21, mul_5, view_23, addmm_1, view_25, mul_10, view_31, div_3, div_4, unsqueeze_11, view_47, mul_15, view_49, addmm_4, view_51, mul_20, view_57, div_6, div_7, unsqueeze_17, view_73, mul_25, view_75, addmm_7, view_77, mul_30, view_83, div_9, div_10, unsqueeze_23, view_99, mul_35, view_101, addmm_10, view_103, mul_40, view_109, div_12, div_13, unsqueeze_29, view_125, mul_45, view_127, addmm_13, view_129, mul_50, view_135, div_15, div_16, unsqueeze_35, view_151, mul_55, view_153, addmm_16, view_155, mul_60, view_161, div_18, div_19, unsqueeze_41, view_177, mul_65, view_179, addmm_19, view_181, mul_70, view_187, div_21, div_22, unsqueeze_47, view_203, mul_75, view_205, addmm_22, view_207, mul_80, view_213, div_24, div_25, unsqueeze_53, view_229, mul_85, view_231, addmm_25, view_233, mul_90, view_239, div_27, div_28, unsqueeze_59, view_255, mul_95, view_257, addmm_28, view_259, cat, getitem_41, rsqrt_20, view_261, view_271, mul_103, view_273, addmm_31, view_275, mul_108, view_277, view_287, mul_111, view_289, addmm_34, view_291, mul_116, clone_167, permute_126, div_32, permute_130, permute_134, div_33, permute_138, permute_143, permute_144, alias_42, permute_145, permute_146, permute_151, div_34, permute_153, permute_157, div_35, permute_161, permute_166, permute_167, alias_43, permute_168, permute_169, permute_174, permute_176, permute_180, div_37, permute_184, permute_189, permute_190, permute_194, permute_196, permute_197, permute_206, div_41, permute_208, permute_212, div_42, permute_216, permute_221, permute_222, permute_226, permute_228, permute_229, permute_238, div_46, permute_240, permute_244, div_47, permute_248, permute_253, permute_254, permute_258, permute_260, permute_261, permute_270, div_51, permute_272, permute_276, div_52, permute_280, permute_285, permute_286, permute_290, permute_292, permute_293, permute_302, div_56, permute_304, permute_308, div_57, permute_312, permute_317, permute_318, permute_322, permute_324, permute_325, permute_334, div_61, permute_336, permute_340, div_62, permute_344, permute_349, permute_350, permute_354, permute_356, permute_357, permute_366, div_66, permute_368, permute_372, div_67, permute_376, permute_381, permute_382, permute_386, permute_388, permute_389, permute_398, div_71, permute_400, permute_404, div_72, permute_408, permute_413, permute_414, permute_418, permute_420, permute_421, permute_430, div_76, permute_432, permute_436, div_77, permute_440, permute_445, permute_446, permute_450, permute_452, permute_453, permute_462, div_81, permute_464, permute_468, div_82, permute_472, permute_477, permute_478, permute_482, permute_484, permute_485, permute_494, div_86, tangents_1 = args
    args.clear()
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_6, (768, ), (1, ))
    assert_size_stride(primals_8, (768, ), (1, ))
    assert_size_stride(primals_10, (16, ), (1, ))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_15, (16, ), (1, ))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_18, (768, ), (1, ))
    assert_size_stride(primals_20, (16, ), (1, ))
    assert_size_stride(primals_21, (768, ), (1, ))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_25, (16, ), (1, ))
    assert_size_stride(primals_26, (768, ), (1, ))
    assert_size_stride(primals_28, (768, ), (1, ))
    assert_size_stride(primals_30, (16, ), (1, ))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_33, (768, ), (1, ))
    assert_size_stride(primals_35, (16, ), (1, ))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_38, (768, ), (1, ))
    assert_size_stride(primals_40, (16, ), (1, ))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_45, (16, ), (1, ))
    assert_size_stride(primals_46, (768, ), (1, ))
    assert_size_stride(primals_48, (768, ), (1, ))
    assert_size_stride(primals_50, (16, ), (1, ))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_57, (768, ), (1, ))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_63, (768, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(primals_181, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(mul, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_5, (1568, 768), (768, 1))
    assert_size_stride(view_8, (307328, 3), (3, 1))
    assert_size_stride(div, (8, 16, 196, 196), (614656, 38416, 196, 1))
    assert_size_stride(div_1, (8, 16, 196, 196), (614656, 38416, 196, 1))
    assert_size_stride(unsqueeze_5, (8, 16, 196, 1), (3136, 196, 1, 1))
    assert_size_stride(view_21, (1568, 768), (768, 1))
    assert_size_stride(mul_5, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_23, (1568, 768), (768, 1))
    assert_size_stride(addmm_1, (1568, 3072), (3072, 1))
    assert_size_stride(view_25, (1568, 3072), (3072, 1))
    assert_size_stride(mul_10, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_31, (1568, 768), (768, 1))
    assert_size_stride(div_3, (8, 16, 196, 196), (614656, 38416, 196, 1))
    assert_size_stride(div_4, (8, 16, 196, 196), (614656, 38416, 196, 1))
    assert_size_stride(unsqueeze_11, (8, 16, 196, 1), (3136, 196, 1, 1))
    assert_size_stride(view_47, (1568, 768), (768, 1))
    assert_size_stride(mul_15, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_49, (1568, 768), (768, 1))
    assert_size_stride(addmm_4, (1568, 3072), (3072, 1))
    assert_size_stride(view_51, (1568, 3072), (3072, 1))
    assert_size_stride(mul_20, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_57, (1568, 768), (768, 1))
    assert_size_stride(div_6, (8, 16, 196, 196), (614656, 38416, 196, 1))
    assert_size_stride(div_7, (8, 16, 196, 196), (614656, 38416, 196, 1))
    assert_size_stride(unsqueeze_17, (8, 16, 196, 1), (3136, 196, 1, 1))
    assert_size_stride(view_73, (1568, 768), (768, 1))
    assert_size_stride(mul_25, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_75, (1568, 768), (768, 1))
    assert_size_stride(addmm_7, (1568, 3072), (3072, 1))
    assert_size_stride(view_77, (1568, 3072), (3072, 1))
    assert_size_stride(mul_30, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_83, (1568, 768), (768, 1))
    assert_size_stride(div_9, (8, 16, 196, 196), (614656, 38416, 196, 1))
    assert_size_stride(div_10, (8, 16, 196, 196), (614656, 38416, 196, 1))
    assert_size_stride(unsqueeze_23, (8, 16, 196, 1), (3136, 196, 1, 1))
    assert_size_stride(view_99, (1568, 768), (768, 1))
    assert_size_stride(mul_35, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_101, (1568, 768), (768, 1))
    assert_size_stride(addmm_10, (1568, 3072), (3072, 1))
    assert_size_stride(view_103, (1568, 3072), (3072, 1))
    assert_size_stride(mul_40, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_109, (1568, 768), (768, 1))
    assert_size_stride(div_12, (8, 16, 196, 196), (614656, 38416, 196, 1))
    assert_size_stride(div_13, (8, 16, 196, 196), (614656, 38416, 196, 1))
    assert_size_stride(unsqueeze_29, (8, 16, 196, 1), (3136, 196, 1, 1))
    assert_size_stride(view_125, (1568, 768), (768, 1))
    assert_size_stride(mul_45, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_127, (1568, 768), (768, 1))
    assert_size_stride(addmm_13, (1568, 3072), (3072, 1))
    assert_size_stride(view_129, (1568, 3072), (3072, 1))
    assert_size_stride(mul_50, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_135, (1568, 768), (768, 1))
    assert_size_stride(div_15, (8, 16, 196, 196), (614656, 38416, 196, 1))
    assert_size_stride(div_16, (8, 16, 196, 196), (614656, 38416, 196, 1))
    assert_size_stride(unsqueeze_35, (8, 16, 196, 1), (3136, 196, 1, 1))
    assert_size_stride(view_151, (1568, 768), (768, 1))
    assert_size_stride(mul_55, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_153, (1568, 768), (768, 1))
    assert_size_stride(addmm_16, (1568, 3072), (3072, 1))
    assert_size_stride(view_155, (1568, 3072), (3072, 1))
    assert_size_stride(mul_60, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_161, (1568, 768), (768, 1))
    assert_size_stride(div_18, (8, 16, 196, 196), (614656, 38416, 196, 1))
    assert_size_stride(div_19, (8, 16, 196, 196), (614656, 38416, 196, 1))
    assert_size_stride(unsqueeze_41, (8, 16, 196, 1), (3136, 196, 1, 1))
    assert_size_stride(view_177, (1568, 768), (768, 1))
    assert_size_stride(mul_65, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_179, (1568, 768), (768, 1))
    assert_size_stride(addmm_19, (1568, 3072), (3072, 1))
    assert_size_stride(view_181, (1568, 3072), (3072, 1))
    assert_size_stride(mul_70, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_187, (1568, 768), (768, 1))
    assert_size_stride(div_21, (8, 16, 196, 196), (614656, 38416, 196, 1))
    assert_size_stride(div_22, (8, 16, 196, 196), (614656, 38416, 196, 1))
    assert_size_stride(unsqueeze_47, (8, 16, 196, 1), (3136, 196, 1, 1))
    assert_size_stride(view_203, (1568, 768), (768, 1))
    assert_size_stride(mul_75, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_205, (1568, 768), (768, 1))
    assert_size_stride(addmm_22, (1568, 3072), (3072, 1))
    assert_size_stride(view_207, (1568, 3072), (3072, 1))
    assert_size_stride(mul_80, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_213, (1568, 768), (768, 1))
    assert_size_stride(div_24, (8, 16, 196, 196), (614656, 38416, 196, 1))
    assert_size_stride(div_25, (8, 16, 196, 196), (614656, 38416, 196, 1))
    assert_size_stride(unsqueeze_53, (8, 16, 196, 1), (3136, 196, 1, 1))
    assert_size_stride(view_229, (1568, 768), (768, 1))
    assert_size_stride(mul_85, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_231, (1568, 768), (768, 1))
    assert_size_stride(addmm_25, (1568, 3072), (3072, 1))
    assert_size_stride(view_233, (1568, 3072), (3072, 1))
    assert_size_stride(mul_90, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_239, (1568, 768), (768, 1))
    assert_size_stride(div_27, (8, 16, 196, 196), (614656, 38416, 196, 1))
    assert_size_stride(div_28, (8, 16, 196, 196), (614656, 38416, 196, 1))
    assert_size_stride(unsqueeze_59, (8, 16, 196, 1), (3136, 196, 1, 1))
    assert_size_stride(view_255, (1568, 768), (768, 1))
    assert_size_stride(mul_95, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_257, (1568, 768), (768, 1))
    assert_size_stride(addmm_28, (1568, 3072), (3072, 1))
    assert_size_stride(view_259, (1568, 3072), (3072, 1))
    assert_size_stride(cat, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(getitem_41, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_20, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_261, (1576, 768), (768, 1))
    assert_size_stride(view_271, (1576, 768), (768, 1))
    assert_size_stride(mul_103, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_273, (1576, 768), (768, 1))
    assert_size_stride(addmm_31, (1576, 3072), (3072, 1))
    assert_size_stride(view_275, (1576, 3072), (3072, 1))
    assert_size_stride(mul_108, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_277, (1576, 768), (768, 1))
    assert_size_stride(view_287, (1576, 768), (768, 1))
    assert_size_stride(mul_111, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_289, (1576, 768), (768, 1))
    assert_size_stride(addmm_34, (1576, 3072), (3072, 1))
    assert_size_stride(view_291, (1576, 3072), (3072, 1))
    assert_size_stride(mul_116, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(clone_167, (8, 768), (768, 1))
    assert_size_stride(permute_126, (1000, 768), (768, 1))
    assert_size_stride(div_32, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_130, (768, 3072), (3072, 1))
    assert_size_stride(permute_134, (3072, 768), (768, 1))
    assert_size_stride(div_33, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_138, (768, 768), (768, 1))
    assert_size_stride(permute_143, (128, 197, 197), (38809, 1, 197))
    assert_size_stride(permute_144, (128, 48, 197), (9456, 1, 48))
    assert_size_stride(alias_42, (8, 16, 197, 197), (620944, 38809, 197, 1))
    assert_size_stride(permute_145, (128, 48, 197), (9456, 1, 48))
    assert_size_stride(permute_146, (128, 197, 48), (9456, 1, 197))
    assert_size_stride(permute_151, (2304, 768), (768, 1))
    assert_size_stride(div_34, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_153, (768, 3072), (3072, 1))
    assert_size_stride(permute_157, (3072, 768), (768, 1))
    assert_size_stride(div_35, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_161, (768, 768), (768, 1))
    assert_size_stride(permute_166, (128, 197, 197), (38809, 1, 197))
    assert_size_stride(permute_167, (128, 48, 197), (9456, 1, 48))
    assert_size_stride(alias_43, (8, 16, 197, 197), (620944, 38809, 197, 1))
    assert_size_stride(permute_168, (128, 48, 197), (9456, 1, 48))
    assert_size_stride(permute_169, (128, 197, 48), (9456, 1, 197))
    assert_size_stride(permute_174, (2304, 768), (768, 1))
    assert_size_stride(permute_176, (768, 3072), (3072, 1))
    assert_size_stride(permute_180, (3072, 768), (768, 1))
    assert_size_stride(div_37, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_184, (768, 768), (768, 1))
    assert_size_stride(permute_189, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_190, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_194, (768, 768), (768, 1))
    assert_size_stride(permute_196, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_197, (128, 196, 48), (9408, 1, 196))
    assert_size_stride(permute_206, (1536, 768), (768, 1))
    assert_size_stride(div_41, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_208, (768, 3072), (3072, 1))
    assert_size_stride(permute_212, (3072, 768), (768, 1))
    assert_size_stride(div_42, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_216, (768, 768), (768, 1))
    assert_size_stride(permute_221, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_222, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_226, (768, 768), (768, 1))
    assert_size_stride(permute_228, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_229, (128, 196, 48), (9408, 1, 196))
    assert_size_stride(permute_238, (1536, 768), (768, 1))
    assert_size_stride(div_46, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_240, (768, 3072), (3072, 1))
    assert_size_stride(permute_244, (3072, 768), (768, 1))
    assert_size_stride(div_47, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_248, (768, 768), (768, 1))
    assert_size_stride(permute_253, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_254, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_258, (768, 768), (768, 1))
    assert_size_stride(permute_260, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_261, (128, 196, 48), (9408, 1, 196))
    assert_size_stride(permute_270, (1536, 768), (768, 1))
    assert_size_stride(div_51, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_272, (768, 3072), (3072, 1))
    assert_size_stride(permute_276, (3072, 768), (768, 1))
    assert_size_stride(div_52, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_280, (768, 768), (768, 1))
    assert_size_stride(permute_285, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_286, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_290, (768, 768), (768, 1))
    assert_size_stride(permute_292, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_293, (128, 196, 48), (9408, 1, 196))
    assert_size_stride(permute_302, (1536, 768), (768, 1))
    assert_size_stride(div_56, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_304, (768, 3072), (3072, 1))
    assert_size_stride(permute_308, (3072, 768), (768, 1))
    assert_size_stride(div_57, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_312, (768, 768), (768, 1))
    assert_size_stride(permute_317, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_318, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_322, (768, 768), (768, 1))
    assert_size_stride(permute_324, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_325, (128, 196, 48), (9408, 1, 196))
    assert_size_stride(permute_334, (1536, 768), (768, 1))
    assert_size_stride(div_61, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_336, (768, 3072), (3072, 1))
    assert_size_stride(permute_340, (3072, 768), (768, 1))
    assert_size_stride(div_62, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_344, (768, 768), (768, 1))
    assert_size_stride(permute_349, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_350, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_354, (768, 768), (768, 1))
    assert_size_stride(permute_356, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_357, (128, 196, 48), (9408, 1, 196))
    assert_size_stride(permute_366, (1536, 768), (768, 1))
    assert_size_stride(div_66, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_368, (768, 3072), (3072, 1))
    assert_size_stride(permute_372, (3072, 768), (768, 1))
    assert_size_stride(div_67, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_376, (768, 768), (768, 1))
    assert_size_stride(permute_381, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_382, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_386, (768, 768), (768, 1))
    assert_size_stride(permute_388, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_389, (128, 196, 48), (9408, 1, 196))
    assert_size_stride(permute_398, (1536, 768), (768, 1))
    assert_size_stride(div_71, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_400, (768, 3072), (3072, 1))
    assert_size_stride(permute_404, (3072, 768), (768, 1))
    assert_size_stride(div_72, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_408, (768, 768), (768, 1))
    assert_size_stride(permute_413, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_414, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_418, (768, 768), (768, 1))
    assert_size_stride(permute_420, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_421, (128, 196, 48), (9408, 1, 196))
    assert_size_stride(permute_430, (1536, 768), (768, 1))
    assert_size_stride(div_76, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_432, (768, 3072), (3072, 1))
    assert_size_stride(permute_436, (3072, 768), (768, 1))
    assert_size_stride(div_77, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_440, (768, 768), (768, 1))
    assert_size_stride(permute_445, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_446, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_450, (768, 768), (768, 1))
    assert_size_stride(permute_452, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_453, (128, 196, 48), (9408, 1, 196))
    assert_size_stride(permute_462, (1536, 768), (768, 1))
    assert_size_stride(div_81, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_464, (768, 3072), (3072, 1))
    assert_size_stride(permute_468, (3072, 768), (768, 1))
    assert_size_stride(div_82, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_472, (768, 768), (768, 1))
    assert_size_stride(permute_477, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_478, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_482, (768, 768), (768, 1))
    assert_size_stride(permute_484, (128, 48, 196), (9408, 1, 48))
    assert_size_stride(permute_485, (128, 196, 48), (9408, 1, 196))
    assert_size_stride(permute_494, (1536, 768), (768, 1))
    assert_size_stride(div_86, (8, 196, 1), (196, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_126, out=buf0)
        del permute_126
        buf1 = empty((1000, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_167, out=buf1)
        del clone_167
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf5 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]
        triton_red_fused_native_layer_norm_backward_select_backward_1.run(buf0, primals_61, mul_116, div_32, buf5, 1576, 768, grid=grid(1576), stream=stream0)
        del div_32
        del primals_61
        buf6 = empty_strided((768, 13), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]
        triton_red_fused_native_layer_norm_backward_select_backward_2.run(buf0, mul_116, buf6, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_116
        buf7 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf6, buf7, 768, 13, grid=grid(768), stream=stream0)
        buf8 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]
        triton_red_fused_native_layer_norm_backward_select_backward_4.run(buf0, buf8, 768, 1576, grid=grid(768), stream=stream0)
        del buf0
        buf9 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (1576, 768), (768, 1), 0), permute_130, out=buf9)
        del permute_130
        buf10 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (768, 1576), (1, 768), 0), view_291, out=buf10)
        del view_291
        buf11 = reinterpret_tensor(buf6, (1, 768, 13), (9984, 1, 768), 0); del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf5, buf11, 9984, 122, grid=grid(9984), stream=stream0)
        buf12 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf11, buf12, 768, 13, grid=grid(768), stream=stream0)
        buf13 = reinterpret_tensor(buf9, (8, 197, 3072), (605184, 3072, 1), 0); del buf9  # reuse
        # Source Nodes: [x_168], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf13, addmm_34, 4841472, grid=grid(4841472), stream=stream0)
        del addmm_34
        buf14 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (1576, 3072), (3072, 1), 0), permute_134, out=buf14)
        del permute_134
        buf15 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (3072, 1576), (1, 3072), 0), view_289, out=buf15)
        del view_289
        buf16 = empty_strided((1, 3072, 13), (39936, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf13, buf16, 39936, 122, grid=grid(39936), stream=stream0)
        buf17 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf16, buf17, 3072, 13, grid=grid(3072), stream=stream0)
        buf24 = buf5; del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf24, buf14, primals_59, mul_111, div_33, 1576, 768, grid=grid(1576), stream=stream0)
        del div_33
        del primals_59
        buf20 = reinterpret_tensor(buf11, (768, 13), (1, 768), 0); del buf11  # reuse
        buf22 = empty_strided((768, 13), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf14, mul_111, buf20, buf22, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_111
        buf21 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf20, buf21, 768, 13, grid=grid(768), stream=stream0)
        buf23 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf22, buf23, 768, 13, grid=grid(768), stream=stream0)
        buf25 = buf14; del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf24, (1576, 768), (768, 1), 0), permute_138, out=buf25)
        del permute_138
        buf26 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf24, (768, 1576), (1, 768), 0), view_287, out=buf26)
        del view_287
        buf27 = reinterpret_tensor(buf22, (1, 768, 13), (9984, 1, 768), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf24, buf27, 9984, 122, grid=grid(9984), stream=stream0)
        buf28 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf27, buf28, 768, 13, grid=grid(768), stream=stream0)
        buf29 = empty((8, 16, 197, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf25, buf29, 1210368, grid=grid(1210368), stream=stream0)
        buf30 = reinterpret_tensor(buf25, (128, 197, 48), (9456, 48, 1), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_143, reinterpret_tensor(buf29, (128, 197, 48), (9456, 48, 1), 0), out=buf30)
        del permute_143
        buf31 = empty((128, 197, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf29, (128, 197, 48), (9456, 48, 1), 0), permute_144, out=buf31)
        del permute_144
        buf33 = empty((8, 16, 197, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_12.run(buf31, alias_42, buf33, 25216, 197, grid=grid(25216), stream=stream0)
        del alias_42
        buf34 = reinterpret_tensor(buf29, (128, 48, 197), (9456, 197, 1), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_145, reinterpret_tensor(buf33, (128, 197, 197), (38809, 197, 1), 0), out=buf34)
        del permute_145
        buf35 = empty((128, 197, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf33, (128, 197, 197), (38809, 197, 1), 0), permute_146, out=buf35)
        del permute_146
        buf36 = empty((8, 197, 3, 16, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf35, buf34, buf30, buf36, 75648, 48, grid=grid(75648, 48), stream=stream0)
        buf37 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (2304, 1576), (1, 2304), 0), view_277, out=buf37)
        del view_277
        buf38 = reinterpret_tensor(buf35, (1576, 768), (768, 1), 0); del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (1576, 2304), (2304, 1), 0), permute_151, out=buf38)
        del permute_151
        buf45 = buf24; del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf45, buf38, primals_57, mul_108, div_34, 1576, 768, grid=grid(1576), stream=stream0)
        del div_34
        del primals_57
        buf41 = reinterpret_tensor(buf27, (768, 13), (1, 768), 0); del buf27  # reuse
        buf43 = buf20; del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf38, mul_108, buf41, buf43, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_108
        buf42 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf41, buf42, 768, 13, grid=grid(768), stream=stream0)
        buf44 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf43, buf44, 768, 13, grid=grid(768), stream=stream0)
        buf46 = reinterpret_tensor(buf13, (1576, 3072), (3072, 1), 0); del buf13  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (1576, 768), (768, 1), 0), permute_153, out=buf46)
        del permute_153
        buf47 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (768, 1576), (1, 768), 0), view_275, out=buf47)
        del view_275
        buf48 = reinterpret_tensor(buf43, (1, 768, 13), (9984, 1, 768), 0); del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf45, buf48, 9984, 122, grid=grid(9984), stream=stream0)
        buf49 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf48, buf49, 768, 13, grid=grid(768), stream=stream0)
        buf50 = reinterpret_tensor(buf46, (8, 197, 3072), (605184, 3072, 1), 0); del buf46  # reuse
        # Source Nodes: [x_154], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf50, addmm_31, 4841472, grid=grid(4841472), stream=stream0)
        del addmm_31
        buf51 = buf38; del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf50, (1576, 3072), (3072, 1), 0), permute_157, out=buf51)
        del permute_157
        buf52 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf50, (3072, 1576), (1, 3072), 0), view_273, out=buf52)
        del view_273
        buf53 = buf16; del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf50, buf53, 39936, 122, grid=grid(39936), stream=stream0)
        del buf50
        buf54 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf53, buf54, 3072, 13, grid=grid(3072), stream=stream0)
        buf61 = buf45; del buf45  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf61, buf51, primals_55, mul_103, div_35, 1576, 768, grid=grid(1576), stream=stream0)
        del div_35
        del primals_55
        buf57 = reinterpret_tensor(buf48, (768, 13), (1, 768), 0); del buf48  # reuse
        buf59 = buf41; del buf41  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf51, mul_103, buf57, buf59, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_103
        buf58 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf57, buf58, 768, 13, grid=grid(768), stream=stream0)
        buf60 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf59, buf60, 768, 13, grid=grid(768), stream=stream0)
        buf62 = buf51; del buf51  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf61, (1576, 768), (768, 1), 0), permute_161, out=buf62)
        del permute_161
        buf63 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf61, (768, 1576), (1, 768), 0), view_271, out=buf63)
        del view_271
        buf64 = reinterpret_tensor(buf59, (1, 768, 13), (9984, 1, 768), 0); del buf59  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf61, buf64, 9984, 122, grid=grid(9984), stream=stream0)
        buf65 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf64, buf65, 768, 13, grid=grid(768), stream=stream0)
        buf66 = reinterpret_tensor(buf34, (8, 16, 197, 48), (151296, 9456, 48, 1), 0); del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf62, buf66, 1210368, grid=grid(1210368), stream=stream0)
        buf67 = reinterpret_tensor(buf62, (128, 197, 48), (9456, 48, 1), 0); del buf62  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_166, reinterpret_tensor(buf66, (128, 197, 48), (9456, 48, 1), 0), out=buf67)
        del permute_166
        buf68 = reinterpret_tensor(buf33, (128, 197, 197), (38809, 197, 1), 0); del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf66, (128, 197, 48), (9456, 48, 1), 0), permute_167, out=buf68)
        del permute_167
        buf70 = reinterpret_tensor(buf31, (8, 16, 197, 197), (620944, 38809, 197, 1), 0); del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_12.run(buf68, alias_43, buf70, 25216, 197, grid=grid(25216), stream=stream0)
        del alias_43
        del buf68
        buf71 = reinterpret_tensor(buf66, (128, 48, 197), (9456, 197, 1), 0); del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_168, reinterpret_tensor(buf70, (128, 197, 197), (38809, 197, 1), 0), out=buf71)
        del permute_168
        buf72 = buf30; del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf70, (128, 197, 197), (38809, 197, 1), 0), permute_169, out=buf72)
        del buf70
        del permute_169
        buf73 = buf36; del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf72, buf71, buf67, buf73, 75648, 48, grid=grid(75648, 48), stream=stream0)
        del buf67
        del buf71
        buf74 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf73, (2304, 1576), (1, 2304), 0), view_261, out=buf74)
        del view_261
        buf75 = reinterpret_tensor(buf72, (1576, 768), (768, 1), 0); del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf73, (1576, 2304), (2304, 1), 0), permute_174, out=buf75)
        del buf73
        del permute_174
        buf82 = buf61; del buf61  # reuse
        # Source Nodes: [x_147], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_14.run(buf82, buf75, primals_53, cat, getitem_41, rsqrt_20, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_53
        buf78 = reinterpret_tensor(buf64, (768, 13), (1, 768), 0); del buf64  # reuse
        buf80 = buf57; del buf57  # reuse
        # Source Nodes: [x_147], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_15.run(buf75, cat, getitem_41, rsqrt_20, buf78, buf80, 9984, 122, grid=grid(9984), stream=stream0)
        del buf75
        del cat
        del getitem_41
        del rsqrt_20
        buf79 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_147], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf78, buf79, 768, 13, grid=grid(768), stream=stream0)
        buf81 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf80, buf81, 768, 13, grid=grid(768), stream=stream0)
        buf83 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_16.run(buf82, buf83, 1204224, grid=grid(1204224), stream=stream0)
        buf84 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf83, permute_176, out=buf84)
        del permute_176
        buf85 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf83, (768, 1568), (1, 768), 0), view_259, out=buf85)
        del view_259
        buf86 = reinterpret_tensor(buf80, (1, 768, 13), (9984, 1, 768), 0); del buf80  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_17.run(buf83, buf86, 9984, 121, grid=grid(9984), stream=stream0)
        buf87 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf86, buf87, 768, 13, grid=grid(768), stream=stream0)
        buf88 = reinterpret_tensor(buf84, (8, 196, 3072), (602112, 3072, 1), 0); del buf84  # reuse
        # Source Nodes: [x_139], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_18.run(buf88, addmm_28, 4816896, grid=grid(4816896), stream=stream0)
        del addmm_28
        buf89 = buf83; del buf83  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf88, (1568, 3072), (3072, 1), 0), permute_180, out=buf89)
        del permute_180
        buf90 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf88, (3072, 1568), (1, 3072), 0), view_257, out=buf90)
        del view_257
        buf91 = buf53; del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf88, buf91, 39936, 121, grid=grid(39936), stream=stream0)
        buf92 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf91, buf92, 3072, 13, grid=grid(3072), stream=stream0)
        buf99 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_20.run(buf89, primals_51, mul_95, buf82, div_37, buf99, 1568, 768, grid=grid(1568), stream=stream0)
        del div_37
        del primals_51
        buf95 = reinterpret_tensor(buf86, (768, 13), (1, 768), 0); del buf86  # reuse
        buf97 = buf78; del buf78  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf89, mul_95, buf95, buf97, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_95
        buf96 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf95, buf96, 768, 13, grid=grid(768), stream=stream0)
        buf98 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf97, buf98, 768, 13, grid=grid(768), stream=stream0)
        buf100 = buf89; del buf89  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (1568, 768), (768, 1), 0), permute_184, out=buf100)
        del permute_184
        buf101 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (768, 1568), (1, 768), 0), view_255, out=buf101)
        del view_255
        buf102 = reinterpret_tensor(buf97, (1, 768, 13), (9984, 1, 768), 0); del buf97  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf99, buf102, 9984, 121, grid=grid(9984), stream=stream0)
        buf103 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf102, buf103, 768, 13, grid=grid(768), stream=stream0)
        buf104 = empty((8, 16, 196, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_23.run(buf100, buf104, 1204224, grid=grid(1204224), stream=stream0)
        buf105 = reinterpret_tensor(buf100, (128, 196, 48), (9408, 48, 1), 0); del buf100  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_189, reinterpret_tensor(buf104, (128, 196, 48), (9408, 48, 1), 0), out=buf105)
        del permute_189
        buf106 = empty((128, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf104, (128, 196, 48), (9408, 48, 1), 0), permute_190, out=buf106)
        del permute_190
        buf107 = reinterpret_tensor(buf104, (1568, 768), (768, 1), 0); del buf104  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_24.run(buf105, buf107, 1204224, grid=grid(1204224), stream=stream0)
        buf108 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf107, (768, 1568), (1, 768), 0), view_239, out=buf108)
        buf109 = reinterpret_tensor(buf105, (1568, 768), (768, 1), 0); del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf107, permute_194, out=buf109)
        del permute_194
        buf110 = empty((8, 16, 196, 1), device='cuda', dtype=torch.float32)
        buf116 = empty((8, 16, 196, 196), device='cuda', dtype=torch.float32)
        buf117 = empty_strided((8, 16, 196, 1), (3136, 196, 1, 25088), device='cuda', dtype=torch.float32)
        buf120 = empty((8, 16, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_36, mul_28, mul_29, sigmoid_18, sub_19], Original ATen: [aten._softmax_backward_data, aten.add, aten.div, aten.mul, aten.neg, aten.rsub, aten.sigmoid, aten.sum]
        triton_per_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sum_25.run(buf106, primals_50, div_27, div_28, unsqueeze_59, buf110, buf116, buf117, buf120, 25088, 196, grid=grid(25088), stream=stream0)
        buf111 = empty_strided((1, 16, 1, 1, 38), (608, 38, 608, 608, 1), device='cuda', dtype=torch.float32)
        buf113 = empty_strided((1, 16, 1, 1, 38), (608, 38, 608, 608, 1), device='cuda', dtype=torch.float32)
        buf123 = empty((1, 1, 1, 16, 38), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_26.run(buf106, unsqueeze_59, buf110, div_28, div_27, buf116, buf117, buf111, buf113, buf123, 608, 8088, grid=grid(608), stream=stream0)
        del div_27
        del unsqueeze_59
        buf112 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf115 = reinterpret_tensor(buf112, (16, ), (1, ), 0); del buf112  # reuse
        # Source Nodes: [sigmoid_18], Original ATen: [aten.add, aten.div, aten.mul, aten.neg, aten.sigmoid, aten.sigmoid_backward, aten.sum, aten.view]
        triton_per_fused_add_div_mul_neg_sigmoid_sigmoid_backward_sum_view_27.run(buf115, buf111, buf113, primals_50, 16, 38, grid=grid(16), stream=stream0)
        del primals_50
        buf121 = reinterpret_tensor(buf107, (128, 48, 196), (9408, 196, 1), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_196, reinterpret_tensor(buf120, (128, 196, 196), (38416, 196, 1), 0), out=buf121)
        del permute_196
        buf122 = empty((128, 196, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf120, (128, 196, 196), (38416, 196, 1), 0), permute_197, out=buf122)
        del permute_197
        buf124 = empty((1, 1, 1, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_28.run(buf123, buf124, 16, 38, grid=grid(16), stream=stream0)
        buf125 = reinterpret_tensor(buf120, (8, 196, 196, 16), (614656, 3136, 16, 1), 0); del buf120  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf116, div_28, buf117, buf125, 128, 38416, grid=grid(128, 38416), stream=stream0)
        del div_28
        buf126 = empty((16, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (16, 307328), (1, 16), 0), view_8, out=buf126)
        buf127 = empty((8, 196, 2, 16, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_30.run(buf121, buf122, buf127, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        buf128 = empty((1536, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf127, (1536, 1568), (1, 1536), 0), view_239, out=buf128)
        del view_239
        buf129 = reinterpret_tensor(buf122, (1568, 768), (768, 1), 0); del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf127, (1568, 1536), (1536, 1), 0), permute_206, out=buf129)
        del permute_206
        buf136 = buf99; del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_31.run(buf136, buf109, buf129, primals_48, mul_90, div_41, 1568, 768, grid=grid(1568), stream=stream0)
        del div_41
        del primals_48
        buf132 = reinterpret_tensor(buf102, (768, 13), (1, 768), 0); del buf102  # reuse
        buf134 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_32.run(buf109, buf129, mul_90, buf132, buf134, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_90
        buf133 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf132, buf133, 768, 13, grid=grid(768), stream=stream0)
        buf135 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf134, buf135, 768, 13, grid=grid(768), stream=stream0)
        buf137 = reinterpret_tensor(buf88, (1568, 3072), (3072, 1), 0); del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf136, (1568, 768), (768, 1), 0), permute_208, out=buf137)
        del permute_208
        buf138 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf136, (768, 1568), (1, 768), 0), view_233, out=buf138)
        del view_233
        buf139 = reinterpret_tensor(buf134, (1, 768, 13), (9984, 1, 768), 0); del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf136, buf139, 9984, 121, grid=grid(9984), stream=stream0)
        buf140 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf139, buf140, 768, 13, grid=grid(768), stream=stream0)
        buf141 = reinterpret_tensor(buf137, (8, 196, 3072), (602112, 3072, 1), 0); del buf137  # reuse
        # Source Nodes: [x_125], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_18.run(buf141, addmm_25, 4816896, grid=grid(4816896), stream=stream0)
        del addmm_25
        buf142 = buf129; del buf129  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf141, (1568, 3072), (3072, 1), 0), permute_212, out=buf142)
        del permute_212
        buf143 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf141, (3072, 1568), (1, 3072), 0), view_231, out=buf143)
        del view_231
        buf144 = buf91; del buf91  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf141, buf144, 39936, 121, grid=grid(39936), stream=stream0)
        buf145 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf144, buf145, 3072, 13, grid=grid(3072), stream=stream0)
        buf152 = buf136; del buf136  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_33.run(buf152, buf142, primals_46, mul_85, div_42, 1568, 768, grid=grid(1568), stream=stream0)
        del div_42
        del primals_46
        buf148 = reinterpret_tensor(buf139, (768, 13), (1, 768), 0); del buf139  # reuse
        buf150 = buf132; del buf132  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf142, mul_85, buf148, buf150, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_85
        buf149 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf148, buf149, 768, 13, grid=grid(768), stream=stream0)
        buf151 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf150, buf151, 768, 13, grid=grid(768), stream=stream0)
        buf153 = buf142; del buf142  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf152, (1568, 768), (768, 1), 0), permute_216, out=buf153)
        del permute_216
        buf154 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf152, (768, 1568), (1, 768), 0), view_229, out=buf154)
        del view_229
        buf155 = reinterpret_tensor(buf150, (1, 768, 13), (9984, 1, 768), 0); del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf152, buf155, 9984, 121, grid=grid(9984), stream=stream0)
        buf156 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf155, buf156, 768, 13, grid=grid(768), stream=stream0)
        buf157 = reinterpret_tensor(buf109, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf109  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_23.run(buf153, buf157, 1204224, grid=grid(1204224), stream=stream0)
        buf158 = reinterpret_tensor(buf153, (128, 196, 48), (9408, 48, 1), 0); del buf153  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_221, reinterpret_tensor(buf157, (128, 196, 48), (9408, 48, 1), 0), out=buf158)
        del permute_221
        buf159 = reinterpret_tensor(buf125, (128, 196, 196), (38416, 196, 1), 0); del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf157, (128, 196, 48), (9408, 48, 1), 0), permute_222, out=buf159)
        del permute_222
        buf160 = reinterpret_tensor(buf157, (1568, 768), (768, 1), 0); del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_24.run(buf158, buf160, 1204224, grid=grid(1204224), stream=stream0)
        buf161 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf160, (768, 1568), (1, 768), 0), view_213, out=buf161)
        buf162 = reinterpret_tensor(buf158, (1568, 768), (768, 1), 0); del buf158  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf160, permute_226, out=buf162)
        del permute_226
        buf163 = reinterpret_tensor(buf117, (8, 16, 196, 1), (3136, 196, 1, 1), 0); del buf117  # reuse
        buf169 = buf116; del buf116  # reuse
        buf170 = reinterpret_tensor(buf110, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf110  # reuse
        buf173 = reinterpret_tensor(buf106, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf106  # reuse
        # Source Nodes: [attn_32, mul_25, mul_26, sigmoid_16, sub_17], Original ATen: [aten._softmax_backward_data, aten.add, aten.div, aten.mul, aten.neg, aten.rsub, aten.sigmoid, aten.sum]
        triton_per_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sum_25.run(buf159, primals_45, div_24, div_25, unsqueeze_53, buf163, buf169, buf170, buf173, 25088, 196, grid=grid(25088), stream=stream0)
        buf164 = reinterpret_tensor(buf123, (1, 16, 1, 1, 38), (608, 38, 608, 608, 1), 0); del buf123  # reuse
        buf166 = buf113; del buf113  # reuse
        buf176 = reinterpret_tensor(buf111, (1, 1, 1, 16, 38), (608, 608, 608, 38, 1), 0); del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_26.run(buf159, unsqueeze_53, buf163, div_25, div_24, buf169, buf170, buf164, buf166, buf176, 608, 8088, grid=grid(608), stream=stream0)
        del div_24
        del unsqueeze_53
        buf165 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf168 = reinterpret_tensor(buf165, (16, ), (1, ), 0); del buf165  # reuse
        # Source Nodes: [sigmoid_16], Original ATen: [aten.add, aten.div, aten.mul, aten.neg, aten.sigmoid, aten.sigmoid_backward, aten.sum, aten.view]
        triton_per_fused_add_div_mul_neg_sigmoid_sigmoid_backward_sum_view_27.run(buf168, buf164, buf166, primals_45, 16, 38, grid=grid(16), stream=stream0)
        del primals_45
        buf174 = reinterpret_tensor(buf160, (128, 48, 196), (9408, 196, 1), 0); del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_228, reinterpret_tensor(buf173, (128, 196, 196), (38416, 196, 1), 0), out=buf174)
        del permute_228
        buf175 = reinterpret_tensor(buf121, (128, 196, 48), (9408, 48, 1), 0); del buf121  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf173, (128, 196, 196), (38416, 196, 1), 0), permute_229, out=buf175)
        del permute_229
        buf177 = empty((1, 1, 1, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_28.run(buf176, buf177, 16, 38, grid=grid(16), stream=stream0)
        buf178 = reinterpret_tensor(buf173, (8, 196, 196, 16), (614656, 3136, 16, 1), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf169, div_25, buf170, buf178, 128, 38416, grid=grid(128, 38416), stream=stream0)
        del div_25
        buf179 = empty((16, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf178, (16, 307328), (1, 16), 0), view_8, out=buf179)
        buf180 = buf127; del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_30.run(buf174, buf175, buf180, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        buf181 = empty((1536, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf180, (1536, 1568), (1, 1536), 0), view_213, out=buf181)
        del view_213
        buf182 = reinterpret_tensor(buf175, (1568, 768), (768, 1), 0); del buf175  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf180, (1568, 1536), (1536, 1), 0), permute_238, out=buf182)
        del permute_238
        buf189 = buf152; del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_31.run(buf189, buf162, buf182, primals_43, mul_80, div_46, 1568, 768, grid=grid(1568), stream=stream0)
        del div_46
        del primals_43
        buf185 = reinterpret_tensor(buf155, (768, 13), (1, 768), 0); del buf155  # reuse
        buf187 = buf148; del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_32.run(buf162, buf182, mul_80, buf185, buf187, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_80
        buf186 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf185, buf186, 768, 13, grid=grid(768), stream=stream0)
        buf188 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf187, buf188, 768, 13, grid=grid(768), stream=stream0)
        buf190 = reinterpret_tensor(buf141, (1568, 3072), (3072, 1), 0); del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf189, (1568, 768), (768, 1), 0), permute_240, out=buf190)
        del permute_240
        buf191 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf189, (768, 1568), (1, 768), 0), view_207, out=buf191)
        del view_207
        buf192 = reinterpret_tensor(buf187, (1, 768, 13), (9984, 1, 768), 0); del buf187  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf189, buf192, 9984, 121, grid=grid(9984), stream=stream0)
        buf193 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf192, buf193, 768, 13, grid=grid(768), stream=stream0)
        buf194 = reinterpret_tensor(buf190, (8, 196, 3072), (602112, 3072, 1), 0); del buf190  # reuse
        # Source Nodes: [x_111], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_18.run(buf194, addmm_22, 4816896, grid=grid(4816896), stream=stream0)
        del addmm_22
        buf195 = buf182; del buf182  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf194, (1568, 3072), (3072, 1), 0), permute_244, out=buf195)
        del permute_244
        buf196 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf194, (3072, 1568), (1, 3072), 0), view_205, out=buf196)
        del view_205
        buf197 = buf144; del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf194, buf197, 39936, 121, grid=grid(39936), stream=stream0)
        buf198 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf197, buf198, 3072, 13, grid=grid(3072), stream=stream0)
        buf205 = buf189; del buf189  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_33.run(buf205, buf195, primals_41, mul_75, div_47, 1568, 768, grid=grid(1568), stream=stream0)
        del div_47
        del primals_41
        buf201 = reinterpret_tensor(buf192, (768, 13), (1, 768), 0); del buf192  # reuse
        buf203 = buf185; del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf195, mul_75, buf201, buf203, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_75
        buf202 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf201, buf202, 768, 13, grid=grid(768), stream=stream0)
        buf204 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf203, buf204, 768, 13, grid=grid(768), stream=stream0)
        buf206 = buf195; del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf205, (1568, 768), (768, 1), 0), permute_248, out=buf206)
        del permute_248
        buf207 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf205, (768, 1568), (1, 768), 0), view_203, out=buf207)
        del view_203
        buf208 = reinterpret_tensor(buf203, (1, 768, 13), (9984, 1, 768), 0); del buf203  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf205, buf208, 9984, 121, grid=grid(9984), stream=stream0)
        buf209 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf208, buf209, 768, 13, grid=grid(768), stream=stream0)
        buf210 = reinterpret_tensor(buf162, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf162  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_23.run(buf206, buf210, 1204224, grid=grid(1204224), stream=stream0)
        buf211 = reinterpret_tensor(buf206, (128, 196, 48), (9408, 48, 1), 0); del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_253, reinterpret_tensor(buf210, (128, 196, 48), (9408, 48, 1), 0), out=buf211)
        del permute_253
        buf212 = reinterpret_tensor(buf178, (128, 196, 196), (38416, 196, 1), 0); del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf210, (128, 196, 48), (9408, 48, 1), 0), permute_254, out=buf212)
        del permute_254
        buf213 = reinterpret_tensor(buf210, (1568, 768), (768, 1), 0); del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_24.run(buf211, buf213, 1204224, grid=grid(1204224), stream=stream0)
        buf214 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf213, (768, 1568), (1, 768), 0), view_187, out=buf214)
        buf215 = reinterpret_tensor(buf211, (1568, 768), (768, 1), 0); del buf211  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf213, permute_258, out=buf215)
        del permute_258
        buf216 = reinterpret_tensor(buf170, (8, 16, 196, 1), (3136, 196, 1, 1), 0); del buf170  # reuse
        buf222 = buf169; del buf169  # reuse
        buf223 = reinterpret_tensor(buf163, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf163  # reuse
        buf226 = reinterpret_tensor(buf159, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf159  # reuse
        # Source Nodes: [attn_28, mul_22, mul_23, sigmoid_14, sub_15], Original ATen: [aten._softmax_backward_data, aten.add, aten.div, aten.mul, aten.neg, aten.rsub, aten.sigmoid, aten.sum]
        triton_per_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sum_25.run(buf212, primals_40, div_21, div_22, unsqueeze_47, buf216, buf222, buf223, buf226, 25088, 196, grid=grid(25088), stream=stream0)
        buf217 = reinterpret_tensor(buf176, (1, 16, 1, 1, 38), (608, 38, 608, 608, 1), 0); del buf176  # reuse
        buf219 = buf166; del buf166  # reuse
        buf229 = reinterpret_tensor(buf164, (1, 1, 1, 16, 38), (608, 608, 608, 38, 1), 0); del buf164  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_26.run(buf212, unsqueeze_47, buf216, div_22, div_21, buf222, buf223, buf217, buf219, buf229, 608, 8088, grid=grid(608), stream=stream0)
        del div_21
        del unsqueeze_47
        buf218 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf221 = reinterpret_tensor(buf218, (16, ), (1, ), 0); del buf218  # reuse
        # Source Nodes: [sigmoid_14], Original ATen: [aten.add, aten.div, aten.mul, aten.neg, aten.sigmoid, aten.sigmoid_backward, aten.sum, aten.view]
        triton_per_fused_add_div_mul_neg_sigmoid_sigmoid_backward_sum_view_27.run(buf221, buf217, buf219, primals_40, 16, 38, grid=grid(16), stream=stream0)
        del primals_40
        buf227 = reinterpret_tensor(buf213, (128, 48, 196), (9408, 196, 1), 0); del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_260, reinterpret_tensor(buf226, (128, 196, 196), (38416, 196, 1), 0), out=buf227)
        del permute_260
        buf228 = reinterpret_tensor(buf174, (128, 196, 48), (9408, 48, 1), 0); del buf174  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf226, (128, 196, 196), (38416, 196, 1), 0), permute_261, out=buf228)
        del permute_261
        buf230 = empty((1, 1, 1, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_28.run(buf229, buf230, 16, 38, grid=grid(16), stream=stream0)
        buf231 = reinterpret_tensor(buf226, (8, 196, 196, 16), (614656, 3136, 16, 1), 0); del buf226  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf222, div_22, buf223, buf231, 128, 38416, grid=grid(128, 38416), stream=stream0)
        del div_22
        buf232 = empty((16, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf231, (16, 307328), (1, 16), 0), view_8, out=buf232)
        buf233 = buf180; del buf180  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_30.run(buf227, buf228, buf233, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        buf234 = empty((1536, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf233, (1536, 1568), (1, 1536), 0), view_187, out=buf234)
        del view_187
        buf235 = reinterpret_tensor(buf228, (1568, 768), (768, 1), 0); del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf233, (1568, 1536), (1536, 1), 0), permute_270, out=buf235)
        del permute_270
        buf242 = buf205; del buf205  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_31.run(buf242, buf215, buf235, primals_38, mul_70, div_51, 1568, 768, grid=grid(1568), stream=stream0)
        del div_51
        del primals_38
        buf238 = reinterpret_tensor(buf208, (768, 13), (1, 768), 0); del buf208  # reuse
        buf240 = buf201; del buf201  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_32.run(buf215, buf235, mul_70, buf238, buf240, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_70
        buf239 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf238, buf239, 768, 13, grid=grid(768), stream=stream0)
        buf241 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf240, buf241, 768, 13, grid=grid(768), stream=stream0)
        buf243 = reinterpret_tensor(buf194, (1568, 3072), (3072, 1), 0); del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf242, (1568, 768), (768, 1), 0), permute_272, out=buf243)
        del permute_272
        buf244 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf242, (768, 1568), (1, 768), 0), view_181, out=buf244)
        del view_181
        buf245 = reinterpret_tensor(buf240, (1, 768, 13), (9984, 1, 768), 0); del buf240  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf242, buf245, 9984, 121, grid=grid(9984), stream=stream0)
        buf246 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf245, buf246, 768, 13, grid=grid(768), stream=stream0)
        buf247 = reinterpret_tensor(buf243, (8, 196, 3072), (602112, 3072, 1), 0); del buf243  # reuse
        # Source Nodes: [x_97], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_18.run(buf247, addmm_19, 4816896, grid=grid(4816896), stream=stream0)
        del addmm_19
        buf248 = buf235; del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf247, (1568, 3072), (3072, 1), 0), permute_276, out=buf248)
        del permute_276
        buf249 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf247, (3072, 1568), (1, 3072), 0), view_179, out=buf249)
        del view_179
        buf250 = buf197; del buf197  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf247, buf250, 39936, 121, grid=grid(39936), stream=stream0)
        buf251 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf250, buf251, 3072, 13, grid=grid(3072), stream=stream0)
        buf258 = buf242; del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_33.run(buf258, buf248, primals_36, mul_65, div_52, 1568, 768, grid=grid(1568), stream=stream0)
        del div_52
        del primals_36
        buf254 = reinterpret_tensor(buf245, (768, 13), (1, 768), 0); del buf245  # reuse
        buf256 = buf238; del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf248, mul_65, buf254, buf256, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_65
        buf255 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf254, buf255, 768, 13, grid=grid(768), stream=stream0)
        buf257 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf256, buf257, 768, 13, grid=grid(768), stream=stream0)
        buf259 = buf248; del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf258, (1568, 768), (768, 1), 0), permute_280, out=buf259)
        del permute_280
        buf260 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf258, (768, 1568), (1, 768), 0), view_177, out=buf260)
        del view_177
        buf261 = reinterpret_tensor(buf256, (1, 768, 13), (9984, 1, 768), 0); del buf256  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf258, buf261, 9984, 121, grid=grid(9984), stream=stream0)
        buf262 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf261, buf262, 768, 13, grid=grid(768), stream=stream0)
        buf263 = reinterpret_tensor(buf215, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_23.run(buf259, buf263, 1204224, grid=grid(1204224), stream=stream0)
        buf264 = reinterpret_tensor(buf259, (128, 196, 48), (9408, 48, 1), 0); del buf259  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_285, reinterpret_tensor(buf263, (128, 196, 48), (9408, 48, 1), 0), out=buf264)
        del permute_285
        buf265 = reinterpret_tensor(buf231, (128, 196, 196), (38416, 196, 1), 0); del buf231  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf263, (128, 196, 48), (9408, 48, 1), 0), permute_286, out=buf265)
        del permute_286
        buf266 = reinterpret_tensor(buf263, (1568, 768), (768, 1), 0); del buf263  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_24.run(buf264, buf266, 1204224, grid=grid(1204224), stream=stream0)
        buf267 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf266, (768, 1568), (1, 768), 0), view_161, out=buf267)
        buf268 = reinterpret_tensor(buf264, (1568, 768), (768, 1), 0); del buf264  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf266, permute_290, out=buf268)
        del permute_290
        buf269 = reinterpret_tensor(buf223, (8, 16, 196, 1), (3136, 196, 1, 1), 0); del buf223  # reuse
        buf275 = buf222; del buf222  # reuse
        buf276 = reinterpret_tensor(buf216, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf216  # reuse
        buf279 = reinterpret_tensor(buf212, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf212  # reuse
        # Source Nodes: [attn_24, mul_19, mul_20, sigmoid_12, sub_13], Original ATen: [aten._softmax_backward_data, aten.add, aten.div, aten.mul, aten.neg, aten.rsub, aten.sigmoid, aten.sum]
        triton_per_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sum_25.run(buf265, primals_35, div_18, div_19, unsqueeze_41, buf269, buf275, buf276, buf279, 25088, 196, grid=grid(25088), stream=stream0)
        buf270 = reinterpret_tensor(buf229, (1, 16, 1, 1, 38), (608, 38, 608, 608, 1), 0); del buf229  # reuse
        buf272 = buf219; del buf219  # reuse
        buf282 = reinterpret_tensor(buf217, (1, 1, 1, 16, 38), (608, 608, 608, 38, 1), 0); del buf217  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_26.run(buf265, unsqueeze_41, buf269, div_19, div_18, buf275, buf276, buf270, buf272, buf282, 608, 8088, grid=grid(608), stream=stream0)
        del div_18
        del unsqueeze_41
        buf271 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf274 = reinterpret_tensor(buf271, (16, ), (1, ), 0); del buf271  # reuse
        # Source Nodes: [sigmoid_12], Original ATen: [aten.add, aten.div, aten.mul, aten.neg, aten.sigmoid, aten.sigmoid_backward, aten.sum, aten.view]
        triton_per_fused_add_div_mul_neg_sigmoid_sigmoid_backward_sum_view_27.run(buf274, buf270, buf272, primals_35, 16, 38, grid=grid(16), stream=stream0)
        del primals_35
        buf280 = reinterpret_tensor(buf266, (128, 48, 196), (9408, 196, 1), 0); del buf266  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_292, reinterpret_tensor(buf279, (128, 196, 196), (38416, 196, 1), 0), out=buf280)
        del permute_292
        buf281 = reinterpret_tensor(buf227, (128, 196, 48), (9408, 48, 1), 0); del buf227  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf279, (128, 196, 196), (38416, 196, 1), 0), permute_293, out=buf281)
        del permute_293
        buf283 = empty((1, 1, 1, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_28.run(buf282, buf283, 16, 38, grid=grid(16), stream=stream0)
        buf284 = reinterpret_tensor(buf279, (8, 196, 196, 16), (614656, 3136, 16, 1), 0); del buf279  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf275, div_19, buf276, buf284, 128, 38416, grid=grid(128, 38416), stream=stream0)
        del div_19
        buf285 = empty((16, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf284, (16, 307328), (1, 16), 0), view_8, out=buf285)
        buf286 = buf233; del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_30.run(buf280, buf281, buf286, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        buf287 = empty((1536, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf286, (1536, 1568), (1, 1536), 0), view_161, out=buf287)
        del view_161
        buf288 = reinterpret_tensor(buf281, (1568, 768), (768, 1), 0); del buf281  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf286, (1568, 1536), (1536, 1), 0), permute_302, out=buf288)
        del permute_302
        buf295 = buf258; del buf258  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_31.run(buf295, buf268, buf288, primals_33, mul_60, div_56, 1568, 768, grid=grid(1568), stream=stream0)
        del div_56
        del primals_33
        buf291 = reinterpret_tensor(buf261, (768, 13), (1, 768), 0); del buf261  # reuse
        buf293 = buf254; del buf254  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_32.run(buf268, buf288, mul_60, buf291, buf293, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_60
        buf292 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf291, buf292, 768, 13, grid=grid(768), stream=stream0)
        buf294 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf293, buf294, 768, 13, grid=grid(768), stream=stream0)
        buf296 = reinterpret_tensor(buf247, (1568, 3072), (3072, 1), 0); del buf247  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf295, (1568, 768), (768, 1), 0), permute_304, out=buf296)
        del permute_304
        buf297 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf295, (768, 1568), (1, 768), 0), view_155, out=buf297)
        del view_155
        buf298 = reinterpret_tensor(buf293, (1, 768, 13), (9984, 1, 768), 0); del buf293  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf295, buf298, 9984, 121, grid=grid(9984), stream=stream0)
        buf299 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf298, buf299, 768, 13, grid=grid(768), stream=stream0)
        buf300 = reinterpret_tensor(buf296, (8, 196, 3072), (602112, 3072, 1), 0); del buf296  # reuse
        # Source Nodes: [x_83], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_18.run(buf300, addmm_16, 4816896, grid=grid(4816896), stream=stream0)
        del addmm_16
        buf301 = buf288; del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf300, (1568, 3072), (3072, 1), 0), permute_308, out=buf301)
        del permute_308
        buf302 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf300, (3072, 1568), (1, 3072), 0), view_153, out=buf302)
        del view_153
        buf303 = buf250; del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf300, buf303, 39936, 121, grid=grid(39936), stream=stream0)
        buf304 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf303, buf304, 3072, 13, grid=grid(3072), stream=stream0)
        buf311 = buf295; del buf295  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_33.run(buf311, buf301, primals_31, mul_55, div_57, 1568, 768, grid=grid(1568), stream=stream0)
        del div_57
        del primals_31
        buf307 = reinterpret_tensor(buf298, (768, 13), (1, 768), 0); del buf298  # reuse
        buf309 = buf291; del buf291  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf301, mul_55, buf307, buf309, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_55
        buf308 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf307, buf308, 768, 13, grid=grid(768), stream=stream0)
        buf310 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf309, buf310, 768, 13, grid=grid(768), stream=stream0)
        buf312 = buf301; del buf301  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf311, (1568, 768), (768, 1), 0), permute_312, out=buf312)
        del permute_312
        buf313 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf311, (768, 1568), (1, 768), 0), view_151, out=buf313)
        del view_151
        buf314 = reinterpret_tensor(buf309, (1, 768, 13), (9984, 1, 768), 0); del buf309  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf311, buf314, 9984, 121, grid=grid(9984), stream=stream0)
        buf315 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf314, buf315, 768, 13, grid=grid(768), stream=stream0)
        buf316 = reinterpret_tensor(buf268, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf268  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_23.run(buf312, buf316, 1204224, grid=grid(1204224), stream=stream0)
        buf317 = reinterpret_tensor(buf312, (128, 196, 48), (9408, 48, 1), 0); del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_317, reinterpret_tensor(buf316, (128, 196, 48), (9408, 48, 1), 0), out=buf317)
        del permute_317
        buf318 = reinterpret_tensor(buf284, (128, 196, 196), (38416, 196, 1), 0); del buf284  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf316, (128, 196, 48), (9408, 48, 1), 0), permute_318, out=buf318)
        del permute_318
        buf319 = reinterpret_tensor(buf316, (1568, 768), (768, 1), 0); del buf316  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_24.run(buf317, buf319, 1204224, grid=grid(1204224), stream=stream0)
        buf320 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf319, (768, 1568), (1, 768), 0), view_135, out=buf320)
        buf321 = reinterpret_tensor(buf317, (1568, 768), (768, 1), 0); del buf317  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf319, permute_322, out=buf321)
        del permute_322
        buf322 = reinterpret_tensor(buf276, (8, 16, 196, 1), (3136, 196, 1, 1), 0); del buf276  # reuse
        buf328 = buf275; del buf275  # reuse
        buf329 = reinterpret_tensor(buf269, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf269  # reuse
        buf332 = reinterpret_tensor(buf265, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf265  # reuse
        # Source Nodes: [attn_20, mul_16, mul_17, sigmoid_10, sub_11], Original ATen: [aten._softmax_backward_data, aten.add, aten.div, aten.mul, aten.neg, aten.rsub, aten.sigmoid, aten.sum]
        triton_per_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sum_25.run(buf318, primals_30, div_15, div_16, unsqueeze_35, buf322, buf328, buf329, buf332, 25088, 196, grid=grid(25088), stream=stream0)
        buf323 = reinterpret_tensor(buf282, (1, 16, 1, 1, 38), (608, 38, 608, 608, 1), 0); del buf282  # reuse
        buf325 = buf272; del buf272  # reuse
        buf335 = reinterpret_tensor(buf270, (1, 1, 1, 16, 38), (608, 608, 608, 38, 1), 0); del buf270  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_26.run(buf318, unsqueeze_35, buf322, div_16, div_15, buf328, buf329, buf323, buf325, buf335, 608, 8088, grid=grid(608), stream=stream0)
        del div_15
        del unsqueeze_35
        buf324 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf327 = reinterpret_tensor(buf324, (16, ), (1, ), 0); del buf324  # reuse
        # Source Nodes: [sigmoid_10], Original ATen: [aten.add, aten.div, aten.mul, aten.neg, aten.sigmoid, aten.sigmoid_backward, aten.sum, aten.view]
        triton_per_fused_add_div_mul_neg_sigmoid_sigmoid_backward_sum_view_27.run(buf327, buf323, buf325, primals_30, 16, 38, grid=grid(16), stream=stream0)
        del primals_30
        buf333 = reinterpret_tensor(buf319, (128, 48, 196), (9408, 196, 1), 0); del buf319  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_324, reinterpret_tensor(buf332, (128, 196, 196), (38416, 196, 1), 0), out=buf333)
        del permute_324
        buf334 = reinterpret_tensor(buf280, (128, 196, 48), (9408, 48, 1), 0); del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf332, (128, 196, 196), (38416, 196, 1), 0), permute_325, out=buf334)
        del permute_325
        buf336 = empty((1, 1, 1, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_28.run(buf335, buf336, 16, 38, grid=grid(16), stream=stream0)
        buf337 = reinterpret_tensor(buf332, (8, 196, 196, 16), (614656, 3136, 16, 1), 0); del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf328, div_16, buf329, buf337, 128, 38416, grid=grid(128, 38416), stream=stream0)
        del div_16
        buf338 = empty((16, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf337, (16, 307328), (1, 16), 0), view_8, out=buf338)
        buf339 = buf286; del buf286  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_30.run(buf333, buf334, buf339, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        buf340 = empty((1536, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf339, (1536, 1568), (1, 1536), 0), view_135, out=buf340)
        del view_135
        buf341 = reinterpret_tensor(buf334, (1568, 768), (768, 1), 0); del buf334  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf339, (1568, 1536), (1536, 1), 0), permute_334, out=buf341)
        del permute_334
        buf348 = buf311; del buf311  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_31.run(buf348, buf321, buf341, primals_28, mul_50, div_61, 1568, 768, grid=grid(1568), stream=stream0)
        del div_61
        del primals_28
        buf344 = reinterpret_tensor(buf314, (768, 13), (1, 768), 0); del buf314  # reuse
        buf346 = buf307; del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_32.run(buf321, buf341, mul_50, buf344, buf346, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_50
        buf345 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf344, buf345, 768, 13, grid=grid(768), stream=stream0)
        buf347 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf346, buf347, 768, 13, grid=grid(768), stream=stream0)
        buf349 = reinterpret_tensor(buf300, (1568, 3072), (3072, 1), 0); del buf300  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf348, (1568, 768), (768, 1), 0), permute_336, out=buf349)
        del permute_336
        buf350 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf348, (768, 1568), (1, 768), 0), view_129, out=buf350)
        del view_129
        buf351 = reinterpret_tensor(buf346, (1, 768, 13), (9984, 1, 768), 0); del buf346  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf348, buf351, 9984, 121, grid=grid(9984), stream=stream0)
        buf352 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf351, buf352, 768, 13, grid=grid(768), stream=stream0)
        buf353 = reinterpret_tensor(buf349, (8, 196, 3072), (602112, 3072, 1), 0); del buf349  # reuse
        # Source Nodes: [x_69], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_18.run(buf353, addmm_13, 4816896, grid=grid(4816896), stream=stream0)
        del addmm_13
        buf354 = buf341; del buf341  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf353, (1568, 3072), (3072, 1), 0), permute_340, out=buf354)
        del permute_340
        buf355 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf353, (3072, 1568), (1, 3072), 0), view_127, out=buf355)
        del view_127
        buf356 = buf303; del buf303  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf353, buf356, 39936, 121, grid=grid(39936), stream=stream0)
        buf357 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf356, buf357, 3072, 13, grid=grid(3072), stream=stream0)
        buf364 = buf348; del buf348  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_33.run(buf364, buf354, primals_26, mul_45, div_62, 1568, 768, grid=grid(1568), stream=stream0)
        del div_62
        del primals_26
        buf360 = reinterpret_tensor(buf351, (768, 13), (1, 768), 0); del buf351  # reuse
        buf362 = buf344; del buf344  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf354, mul_45, buf360, buf362, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_45
        buf361 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf360, buf361, 768, 13, grid=grid(768), stream=stream0)
        buf363 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf362, buf363, 768, 13, grid=grid(768), stream=stream0)
        buf365 = buf354; del buf354  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf364, (1568, 768), (768, 1), 0), permute_344, out=buf365)
        del permute_344
        buf366 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf364, (768, 1568), (1, 768), 0), view_125, out=buf366)
        del view_125
        buf367 = reinterpret_tensor(buf362, (1, 768, 13), (9984, 1, 768), 0); del buf362  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf364, buf367, 9984, 121, grid=grid(9984), stream=stream0)
        buf368 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf367, buf368, 768, 13, grid=grid(768), stream=stream0)
        buf369 = reinterpret_tensor(buf321, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf321  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_23.run(buf365, buf369, 1204224, grid=grid(1204224), stream=stream0)
        buf370 = reinterpret_tensor(buf365, (128, 196, 48), (9408, 48, 1), 0); del buf365  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_349, reinterpret_tensor(buf369, (128, 196, 48), (9408, 48, 1), 0), out=buf370)
        del permute_349
        buf371 = reinterpret_tensor(buf337, (128, 196, 196), (38416, 196, 1), 0); del buf337  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf369, (128, 196, 48), (9408, 48, 1), 0), permute_350, out=buf371)
        del permute_350
        buf372 = reinterpret_tensor(buf369, (1568, 768), (768, 1), 0); del buf369  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_24.run(buf370, buf372, 1204224, grid=grid(1204224), stream=stream0)
        buf373 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf372, (768, 1568), (1, 768), 0), view_109, out=buf373)
        buf374 = reinterpret_tensor(buf370, (1568, 768), (768, 1), 0); del buf370  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf372, permute_354, out=buf374)
        del permute_354
        buf375 = reinterpret_tensor(buf329, (8, 16, 196, 1), (3136, 196, 1, 1), 0); del buf329  # reuse
        buf381 = buf328; del buf328  # reuse
        buf382 = reinterpret_tensor(buf322, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf322  # reuse
        buf385 = reinterpret_tensor(buf318, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf318  # reuse
        # Source Nodes: [attn_16, mul_13, mul_14, sigmoid_8, sub_9], Original ATen: [aten._softmax_backward_data, aten.add, aten.div, aten.mul, aten.neg, aten.rsub, aten.sigmoid, aten.sum]
        triton_per_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sum_25.run(buf371, primals_25, div_12, div_13, unsqueeze_29, buf375, buf381, buf382, buf385, 25088, 196, grid=grid(25088), stream=stream0)
        buf376 = reinterpret_tensor(buf335, (1, 16, 1, 1, 38), (608, 38, 608, 608, 1), 0); del buf335  # reuse
        buf378 = buf325; del buf325  # reuse
        buf388 = reinterpret_tensor(buf323, (1, 1, 1, 16, 38), (608, 608, 608, 38, 1), 0); del buf323  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_26.run(buf371, unsqueeze_29, buf375, div_13, div_12, buf381, buf382, buf376, buf378, buf388, 608, 8088, grid=grid(608), stream=stream0)
        del div_12
        del unsqueeze_29
        buf377 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf380 = reinterpret_tensor(buf377, (16, ), (1, ), 0); del buf377  # reuse
        # Source Nodes: [sigmoid_8], Original ATen: [aten.add, aten.div, aten.mul, aten.neg, aten.sigmoid, aten.sigmoid_backward, aten.sum, aten.view]
        triton_per_fused_add_div_mul_neg_sigmoid_sigmoid_backward_sum_view_27.run(buf380, buf376, buf378, primals_25, 16, 38, grid=grid(16), stream=stream0)
        del primals_25
        buf386 = reinterpret_tensor(buf372, (128, 48, 196), (9408, 196, 1), 0); del buf372  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_356, reinterpret_tensor(buf385, (128, 196, 196), (38416, 196, 1), 0), out=buf386)
        del permute_356
        buf387 = reinterpret_tensor(buf333, (128, 196, 48), (9408, 48, 1), 0); del buf333  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf385, (128, 196, 196), (38416, 196, 1), 0), permute_357, out=buf387)
        del permute_357
        buf389 = empty((1, 1, 1, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_28.run(buf388, buf389, 16, 38, grid=grid(16), stream=stream0)
        buf390 = reinterpret_tensor(buf385, (8, 196, 196, 16), (614656, 3136, 16, 1), 0); del buf385  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf381, div_13, buf382, buf390, 128, 38416, grid=grid(128, 38416), stream=stream0)
        del div_13
        buf391 = empty((16, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf390, (16, 307328), (1, 16), 0), view_8, out=buf391)
        buf392 = buf339; del buf339  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_30.run(buf386, buf387, buf392, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        buf393 = empty((1536, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf392, (1536, 1568), (1, 1536), 0), view_109, out=buf393)
        del view_109
        buf394 = reinterpret_tensor(buf387, (1568, 768), (768, 1), 0); del buf387  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf392, (1568, 1536), (1536, 1), 0), permute_366, out=buf394)
        del permute_366
        buf401 = buf364; del buf364  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_31.run(buf401, buf374, buf394, primals_23, mul_40, div_66, 1568, 768, grid=grid(1568), stream=stream0)
        del div_66
        del primals_23
        buf397 = reinterpret_tensor(buf367, (768, 13), (1, 768), 0); del buf367  # reuse
        buf399 = buf360; del buf360  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_32.run(buf374, buf394, mul_40, buf397, buf399, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_40
        buf398 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf397, buf398, 768, 13, grid=grid(768), stream=stream0)
        buf400 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf399, buf400, 768, 13, grid=grid(768), stream=stream0)
        buf402 = reinterpret_tensor(buf353, (1568, 3072), (3072, 1), 0); del buf353  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf401, (1568, 768), (768, 1), 0), permute_368, out=buf402)
        del permute_368
        buf403 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf401, (768, 1568), (1, 768), 0), view_103, out=buf403)
        del view_103
        buf404 = reinterpret_tensor(buf399, (1, 768, 13), (9984, 1, 768), 0); del buf399  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf401, buf404, 9984, 121, grid=grid(9984), stream=stream0)
        buf405 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf404, buf405, 768, 13, grid=grid(768), stream=stream0)
        buf406 = reinterpret_tensor(buf402, (8, 196, 3072), (602112, 3072, 1), 0); del buf402  # reuse
        # Source Nodes: [x_55], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_18.run(buf406, addmm_10, 4816896, grid=grid(4816896), stream=stream0)
        del addmm_10
        buf407 = buf394; del buf394  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf406, (1568, 3072), (3072, 1), 0), permute_372, out=buf407)
        del permute_372
        buf408 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf406, (3072, 1568), (1, 3072), 0), view_101, out=buf408)
        del view_101
        buf409 = buf356; del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf406, buf409, 39936, 121, grid=grid(39936), stream=stream0)
        buf410 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf409, buf410, 3072, 13, grid=grid(3072), stream=stream0)
        buf417 = buf401; del buf401  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_33.run(buf417, buf407, primals_21, mul_35, div_67, 1568, 768, grid=grid(1568), stream=stream0)
        del div_67
        del primals_21
        buf413 = reinterpret_tensor(buf404, (768, 13), (1, 768), 0); del buf404  # reuse
        buf415 = buf397; del buf397  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf407, mul_35, buf413, buf415, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_35
        buf414 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf413, buf414, 768, 13, grid=grid(768), stream=stream0)
        buf416 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf415, buf416, 768, 13, grid=grid(768), stream=stream0)
        buf418 = buf407; del buf407  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf417, (1568, 768), (768, 1), 0), permute_376, out=buf418)
        del permute_376
        buf419 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf417, (768, 1568), (1, 768), 0), view_99, out=buf419)
        del view_99
        buf420 = reinterpret_tensor(buf415, (1, 768, 13), (9984, 1, 768), 0); del buf415  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf417, buf420, 9984, 121, grid=grid(9984), stream=stream0)
        buf421 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf420, buf421, 768, 13, grid=grid(768), stream=stream0)
        buf422 = reinterpret_tensor(buf374, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf374  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_23.run(buf418, buf422, 1204224, grid=grid(1204224), stream=stream0)
        buf423 = reinterpret_tensor(buf418, (128, 196, 48), (9408, 48, 1), 0); del buf418  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_381, reinterpret_tensor(buf422, (128, 196, 48), (9408, 48, 1), 0), out=buf423)
        del permute_381
        buf424 = reinterpret_tensor(buf390, (128, 196, 196), (38416, 196, 1), 0); del buf390  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf422, (128, 196, 48), (9408, 48, 1), 0), permute_382, out=buf424)
        del permute_382
        buf425 = reinterpret_tensor(buf422, (1568, 768), (768, 1), 0); del buf422  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_24.run(buf423, buf425, 1204224, grid=grid(1204224), stream=stream0)
        buf426 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf425, (768, 1568), (1, 768), 0), view_83, out=buf426)
        buf427 = reinterpret_tensor(buf423, (1568, 768), (768, 1), 0); del buf423  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf425, permute_386, out=buf427)
        del permute_386
        buf428 = reinterpret_tensor(buf382, (8, 16, 196, 1), (3136, 196, 1, 1), 0); del buf382  # reuse
        buf434 = buf381; del buf381  # reuse
        buf435 = reinterpret_tensor(buf375, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf375  # reuse
        buf438 = reinterpret_tensor(buf371, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf371  # reuse
        # Source Nodes: [attn_12, mul_10, mul_11, sigmoid_6, sub_7], Original ATen: [aten._softmax_backward_data, aten.add, aten.div, aten.mul, aten.neg, aten.rsub, aten.sigmoid, aten.sum]
        triton_per_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sum_25.run(buf424, primals_20, div_9, div_10, unsqueeze_23, buf428, buf434, buf435, buf438, 25088, 196, grid=grid(25088), stream=stream0)
        buf429 = reinterpret_tensor(buf388, (1, 16, 1, 1, 38), (608, 38, 608, 608, 1), 0); del buf388  # reuse
        buf431 = buf378; del buf378  # reuse
        buf441 = reinterpret_tensor(buf376, (1, 1, 1, 16, 38), (608, 608, 608, 38, 1), 0); del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_26.run(buf424, unsqueeze_23, buf428, div_10, div_9, buf434, buf435, buf429, buf431, buf441, 608, 8088, grid=grid(608), stream=stream0)
        del div_9
        del unsqueeze_23
        buf430 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf433 = reinterpret_tensor(buf430, (16, ), (1, ), 0); del buf430  # reuse
        # Source Nodes: [sigmoid_6], Original ATen: [aten.add, aten.div, aten.mul, aten.neg, aten.sigmoid, aten.sigmoid_backward, aten.sum, aten.view]
        triton_per_fused_add_div_mul_neg_sigmoid_sigmoid_backward_sum_view_27.run(buf433, buf429, buf431, primals_20, 16, 38, grid=grid(16), stream=stream0)
        del primals_20
        buf439 = reinterpret_tensor(buf425, (128, 48, 196), (9408, 196, 1), 0); del buf425  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_388, reinterpret_tensor(buf438, (128, 196, 196), (38416, 196, 1), 0), out=buf439)
        del permute_388
        buf440 = reinterpret_tensor(buf386, (128, 196, 48), (9408, 48, 1), 0); del buf386  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf438, (128, 196, 196), (38416, 196, 1), 0), permute_389, out=buf440)
        del permute_389
        buf442 = empty((1, 1, 1, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_28.run(buf441, buf442, 16, 38, grid=grid(16), stream=stream0)
        buf443 = reinterpret_tensor(buf438, (8, 196, 196, 16), (614656, 3136, 16, 1), 0); del buf438  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf434, div_10, buf435, buf443, 128, 38416, grid=grid(128, 38416), stream=stream0)
        del div_10
        buf444 = empty((16, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf443, (16, 307328), (1, 16), 0), view_8, out=buf444)
        buf445 = buf392; del buf392  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_30.run(buf439, buf440, buf445, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        buf446 = empty((1536, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf445, (1536, 1568), (1, 1536), 0), view_83, out=buf446)
        del view_83
        buf447 = reinterpret_tensor(buf440, (1568, 768), (768, 1), 0); del buf440  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf445, (1568, 1536), (1536, 1), 0), permute_398, out=buf447)
        del permute_398
        buf454 = buf417; del buf417  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_31.run(buf454, buf427, buf447, primals_18, mul_30, div_71, 1568, 768, grid=grid(1568), stream=stream0)
        del div_71
        del primals_18
        buf450 = reinterpret_tensor(buf420, (768, 13), (1, 768), 0); del buf420  # reuse
        buf452 = buf413; del buf413  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_32.run(buf427, buf447, mul_30, buf450, buf452, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_30
        buf451 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf450, buf451, 768, 13, grid=grid(768), stream=stream0)
        buf453 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf452, buf453, 768, 13, grid=grid(768), stream=stream0)
        buf455 = reinterpret_tensor(buf406, (1568, 3072), (3072, 1), 0); del buf406  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf454, (1568, 768), (768, 1), 0), permute_400, out=buf455)
        del permute_400
        buf456 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf454, (768, 1568), (1, 768), 0), view_77, out=buf456)
        del view_77
        buf457 = reinterpret_tensor(buf452, (1, 768, 13), (9984, 1, 768), 0); del buf452  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf454, buf457, 9984, 121, grid=grid(9984), stream=stream0)
        buf458 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf457, buf458, 768, 13, grid=grid(768), stream=stream0)
        buf459 = reinterpret_tensor(buf455, (8, 196, 3072), (602112, 3072, 1), 0); del buf455  # reuse
        # Source Nodes: [x_41], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_18.run(buf459, addmm_7, 4816896, grid=grid(4816896), stream=stream0)
        del addmm_7
        buf460 = buf447; del buf447  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf459, (1568, 3072), (3072, 1), 0), permute_404, out=buf460)
        del permute_404
        buf461 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf459, (3072, 1568), (1, 3072), 0), view_75, out=buf461)
        del view_75
        buf462 = buf409; del buf409  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf459, buf462, 39936, 121, grid=grid(39936), stream=stream0)
        buf463 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf462, buf463, 3072, 13, grid=grid(3072), stream=stream0)
        buf470 = buf454; del buf454  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_33.run(buf470, buf460, primals_16, mul_25, div_72, 1568, 768, grid=grid(1568), stream=stream0)
        del div_72
        del primals_16
        buf466 = reinterpret_tensor(buf457, (768, 13), (1, 768), 0); del buf457  # reuse
        buf468 = buf450; del buf450  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf460, mul_25, buf466, buf468, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_25
        buf467 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf466, buf467, 768, 13, grid=grid(768), stream=stream0)
        buf469 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf468, buf469, 768, 13, grid=grid(768), stream=stream0)
        buf471 = buf460; del buf460  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf470, (1568, 768), (768, 1), 0), permute_408, out=buf471)
        del permute_408
        buf472 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf470, (768, 1568), (1, 768), 0), view_73, out=buf472)
        del view_73
        buf473 = reinterpret_tensor(buf468, (1, 768, 13), (9984, 1, 768), 0); del buf468  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf470, buf473, 9984, 121, grid=grid(9984), stream=stream0)
        buf474 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf473, buf474, 768, 13, grid=grid(768), stream=stream0)
        buf475 = reinterpret_tensor(buf427, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf427  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_23.run(buf471, buf475, 1204224, grid=grid(1204224), stream=stream0)
        buf476 = reinterpret_tensor(buf471, (128, 196, 48), (9408, 48, 1), 0); del buf471  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_413, reinterpret_tensor(buf475, (128, 196, 48), (9408, 48, 1), 0), out=buf476)
        del permute_413
        buf477 = reinterpret_tensor(buf443, (128, 196, 196), (38416, 196, 1), 0); del buf443  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf475, (128, 196, 48), (9408, 48, 1), 0), permute_414, out=buf477)
        del permute_414
        buf478 = reinterpret_tensor(buf475, (1568, 768), (768, 1), 0); del buf475  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_24.run(buf476, buf478, 1204224, grid=grid(1204224), stream=stream0)
        buf479 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf478, (768, 1568), (1, 768), 0), view_57, out=buf479)
        buf480 = reinterpret_tensor(buf476, (1568, 768), (768, 1), 0); del buf476  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf478, permute_418, out=buf480)
        del permute_418
        buf481 = reinterpret_tensor(buf435, (8, 16, 196, 1), (3136, 196, 1, 1), 0); del buf435  # reuse
        buf487 = buf434; del buf434  # reuse
        buf488 = reinterpret_tensor(buf428, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf428  # reuse
        buf491 = reinterpret_tensor(buf424, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf424  # reuse
        # Source Nodes: [attn_8, mul_7, mul_8, sigmoid_4, sub_5], Original ATen: [aten._softmax_backward_data, aten.add, aten.div, aten.mul, aten.neg, aten.rsub, aten.sigmoid, aten.sum]
        triton_per_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sum_25.run(buf477, primals_15, div_6, div_7, unsqueeze_17, buf481, buf487, buf488, buf491, 25088, 196, grid=grid(25088), stream=stream0)
        buf482 = reinterpret_tensor(buf441, (1, 16, 1, 1, 38), (608, 38, 608, 608, 1), 0); del buf441  # reuse
        buf484 = buf431; del buf431  # reuse
        buf494 = reinterpret_tensor(buf429, (1, 1, 1, 16, 38), (608, 608, 608, 38, 1), 0); del buf429  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_26.run(buf477, unsqueeze_17, buf481, div_7, div_6, buf487, buf488, buf482, buf484, buf494, 608, 8088, grid=grid(608), stream=stream0)
        del div_6
        del unsqueeze_17
        buf483 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf486 = reinterpret_tensor(buf483, (16, ), (1, ), 0); del buf483  # reuse
        # Source Nodes: [sigmoid_4], Original ATen: [aten.add, aten.div, aten.mul, aten.neg, aten.sigmoid, aten.sigmoid_backward, aten.sum, aten.view]
        triton_per_fused_add_div_mul_neg_sigmoid_sigmoid_backward_sum_view_27.run(buf486, buf482, buf484, primals_15, 16, 38, grid=grid(16), stream=stream0)
        del primals_15
        buf492 = reinterpret_tensor(buf478, (128, 48, 196), (9408, 196, 1), 0); del buf478  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_420, reinterpret_tensor(buf491, (128, 196, 196), (38416, 196, 1), 0), out=buf492)
        del permute_420
        buf493 = reinterpret_tensor(buf439, (128, 196, 48), (9408, 48, 1), 0); del buf439  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf491, (128, 196, 196), (38416, 196, 1), 0), permute_421, out=buf493)
        del permute_421
        buf495 = empty((1, 1, 1, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_28.run(buf494, buf495, 16, 38, grid=grid(16), stream=stream0)
        buf496 = reinterpret_tensor(buf491, (8, 196, 196, 16), (614656, 3136, 16, 1), 0); del buf491  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf487, div_7, buf488, buf496, 128, 38416, grid=grid(128, 38416), stream=stream0)
        del div_7
        buf497 = empty((16, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf496, (16, 307328), (1, 16), 0), view_8, out=buf497)
        buf498 = buf445; del buf445  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_30.run(buf492, buf493, buf498, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        buf499 = empty((1536, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf498, (1536, 1568), (1, 1536), 0), view_57, out=buf499)
        del view_57
        buf500 = reinterpret_tensor(buf493, (1568, 768), (768, 1), 0); del buf493  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf498, (1568, 1536), (1536, 1), 0), permute_430, out=buf500)
        del permute_430
        buf507 = buf470; del buf470  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_31.run(buf507, buf480, buf500, primals_13, mul_20, div_76, 1568, 768, grid=grid(1568), stream=stream0)
        del div_76
        del primals_13
        buf503 = reinterpret_tensor(buf473, (768, 13), (1, 768), 0); del buf473  # reuse
        buf505 = buf466; del buf466  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_32.run(buf480, buf500, mul_20, buf503, buf505, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_20
        buf504 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf503, buf504, 768, 13, grid=grid(768), stream=stream0)
        buf506 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf505, buf506, 768, 13, grid=grid(768), stream=stream0)
        buf508 = reinterpret_tensor(buf459, (1568, 3072), (3072, 1), 0); del buf459  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf507, (1568, 768), (768, 1), 0), permute_432, out=buf508)
        del permute_432
        buf509 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf507, (768, 1568), (1, 768), 0), view_51, out=buf509)
        del view_51
        buf510 = reinterpret_tensor(buf505, (1, 768, 13), (9984, 1, 768), 0); del buf505  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf507, buf510, 9984, 121, grid=grid(9984), stream=stream0)
        buf511 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf510, buf511, 768, 13, grid=grid(768), stream=stream0)
        buf512 = reinterpret_tensor(buf508, (8, 196, 3072), (602112, 3072, 1), 0); del buf508  # reuse
        # Source Nodes: [x_27], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_18.run(buf512, addmm_4, 4816896, grid=grid(4816896), stream=stream0)
        del addmm_4
        buf513 = buf500; del buf500  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf512, (1568, 3072), (3072, 1), 0), permute_436, out=buf513)
        del permute_436
        buf514 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf512, (3072, 1568), (1, 3072), 0), view_49, out=buf514)
        del view_49
        buf515 = buf462; del buf462  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf512, buf515, 39936, 121, grid=grid(39936), stream=stream0)
        buf516 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf515, buf516, 3072, 13, grid=grid(3072), stream=stream0)
        buf523 = buf507; del buf507  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_33.run(buf523, buf513, primals_11, mul_15, div_77, 1568, 768, grid=grid(1568), stream=stream0)
        del div_77
        del primals_11
        buf519 = reinterpret_tensor(buf510, (768, 13), (1, 768), 0); del buf510  # reuse
        buf521 = buf503; del buf503  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf513, mul_15, buf519, buf521, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_15
        buf520 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf519, buf520, 768, 13, grid=grid(768), stream=stream0)
        buf522 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf521, buf522, 768, 13, grid=grid(768), stream=stream0)
        buf524 = buf513; del buf513  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf523, (1568, 768), (768, 1), 0), permute_440, out=buf524)
        del permute_440
        buf525 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf523, (768, 1568), (1, 768), 0), view_47, out=buf525)
        del view_47
        buf526 = reinterpret_tensor(buf521, (1, 768, 13), (9984, 1, 768), 0); del buf521  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf523, buf526, 9984, 121, grid=grid(9984), stream=stream0)
        buf527 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf526, buf527, 768, 13, grid=grid(768), stream=stream0)
        buf528 = reinterpret_tensor(buf480, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_23.run(buf524, buf528, 1204224, grid=grid(1204224), stream=stream0)
        buf529 = reinterpret_tensor(buf524, (128, 196, 48), (9408, 48, 1), 0); del buf524  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_445, reinterpret_tensor(buf528, (128, 196, 48), (9408, 48, 1), 0), out=buf529)
        del permute_445
        buf530 = reinterpret_tensor(buf496, (128, 196, 196), (38416, 196, 1), 0); del buf496  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf528, (128, 196, 48), (9408, 48, 1), 0), permute_446, out=buf530)
        del permute_446
        buf531 = reinterpret_tensor(buf528, (1568, 768), (768, 1), 0); del buf528  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_24.run(buf529, buf531, 1204224, grid=grid(1204224), stream=stream0)
        buf532 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf531, (768, 1568), (1, 768), 0), view_31, out=buf532)
        buf533 = reinterpret_tensor(buf529, (1568, 768), (768, 1), 0); del buf529  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf531, permute_450, out=buf533)
        del permute_450
        buf534 = reinterpret_tensor(buf488, (8, 16, 196, 1), (3136, 196, 1, 1), 0); del buf488  # reuse
        buf540 = buf487; del buf487  # reuse
        buf541 = reinterpret_tensor(buf481, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf481  # reuse
        buf544 = reinterpret_tensor(buf477, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf477  # reuse
        # Source Nodes: [attn_4, mul_4, mul_5, sigmoid_2, sub_3], Original ATen: [aten._softmax_backward_data, aten.add, aten.div, aten.mul, aten.neg, aten.rsub, aten.sigmoid, aten.sum]
        triton_per_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sum_25.run(buf530, primals_10, div_3, div_4, unsqueeze_11, buf534, buf540, buf541, buf544, 25088, 196, grid=grid(25088), stream=stream0)
        buf535 = reinterpret_tensor(buf494, (1, 16, 1, 1, 38), (608, 38, 608, 608, 1), 0); del buf494  # reuse
        buf537 = buf484; del buf484  # reuse
        buf547 = reinterpret_tensor(buf482, (1, 1, 1, 16, 38), (608, 608, 608, 38, 1), 0); del buf482  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_26.run(buf530, unsqueeze_11, buf534, div_4, div_3, buf540, buf541, buf535, buf537, buf547, 608, 8088, grid=grid(608), stream=stream0)
        del div_3
        del unsqueeze_11
        buf536 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf539 = reinterpret_tensor(buf536, (16, ), (1, ), 0); del buf536  # reuse
        # Source Nodes: [sigmoid_2], Original ATen: [aten.add, aten.div, aten.mul, aten.neg, aten.sigmoid, aten.sigmoid_backward, aten.sum, aten.view]
        triton_per_fused_add_div_mul_neg_sigmoid_sigmoid_backward_sum_view_27.run(buf539, buf535, buf537, primals_10, 16, 38, grid=grid(16), stream=stream0)
        del primals_10
        buf545 = reinterpret_tensor(buf531, (128, 48, 196), (9408, 196, 1), 0); del buf531  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_452, reinterpret_tensor(buf544, (128, 196, 196), (38416, 196, 1), 0), out=buf545)
        del permute_452
        buf546 = reinterpret_tensor(buf492, (128, 196, 48), (9408, 48, 1), 0); del buf492  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf544, (128, 196, 196), (38416, 196, 1), 0), permute_453, out=buf546)
        del permute_453
        buf548 = empty((1, 1, 1, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_28.run(buf547, buf548, 16, 38, grid=grid(16), stream=stream0)
        buf549 = reinterpret_tensor(buf544, (8, 196, 196, 16), (614656, 3136, 16, 1), 0); del buf544  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf540, div_4, buf541, buf549, 128, 38416, grid=grid(128, 38416), stream=stream0)
        del div_4
        buf550 = empty((16, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf549, (16, 307328), (1, 16), 0), view_8, out=buf550)
        buf551 = buf498; del buf498  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_30.run(buf545, buf546, buf551, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        buf552 = empty((1536, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf551, (1536, 1568), (1, 1536), 0), view_31, out=buf552)
        del view_31
        buf553 = reinterpret_tensor(buf546, (1568, 768), (768, 1), 0); del buf546  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf551, (1568, 1536), (1536, 1), 0), permute_462, out=buf553)
        del permute_462
        buf560 = buf523; del buf523  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_31.run(buf560, buf533, buf553, primals_8, mul_10, div_81, 1568, 768, grid=grid(1568), stream=stream0)
        del div_81
        del primals_8
        buf556 = reinterpret_tensor(buf526, (768, 13), (1, 768), 0); del buf526  # reuse
        buf558 = buf519; del buf519  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_32.run(buf533, buf553, mul_10, buf556, buf558, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_10
        buf557 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf556, buf557, 768, 13, grid=grid(768), stream=stream0)
        buf559 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf558, buf559, 768, 13, grid=grid(768), stream=stream0)
        buf561 = reinterpret_tensor(buf512, (1568, 3072), (3072, 1), 0); del buf512  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf560, (1568, 768), (768, 1), 0), permute_464, out=buf561)
        del permute_464
        buf562 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf560, (768, 1568), (1, 768), 0), view_25, out=buf562)
        del view_25
        buf563 = reinterpret_tensor(buf558, (1, 768, 13), (9984, 1, 768), 0); del buf558  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf560, buf563, 9984, 121, grid=grid(9984), stream=stream0)
        buf564 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf563, buf564, 768, 13, grid=grid(768), stream=stream0)
        buf565 = reinterpret_tensor(buf561, (8, 196, 3072), (602112, 3072, 1), 0); del buf561  # reuse
        # Source Nodes: [x_13], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_18.run(buf565, addmm_1, 4816896, grid=grid(4816896), stream=stream0)
        del addmm_1
        buf566 = buf553; del buf553  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf565, (1568, 3072), (3072, 1), 0), permute_468, out=buf566)
        del permute_468
        buf567 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf565, (3072, 1568), (1, 3072), 0), view_23, out=buf567)
        del view_23
        buf568 = buf515; del buf515  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_19.run(buf565, buf568, 39936, 121, grid=grid(39936), stream=stream0)
        del buf565
        buf569 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf568, buf569, 3072, 13, grid=grid(3072), stream=stream0)
        del buf568
        buf576 = buf560; del buf560  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_33.run(buf576, buf566, primals_6, mul_5, div_82, 1568, 768, grid=grid(1568), stream=stream0)
        del div_82
        del primals_6
        buf572 = reinterpret_tensor(buf563, (768, 13), (1, 768), 0); del buf563  # reuse
        buf574 = buf556; del buf556  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf566, mul_5, buf572, buf574, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_5
        buf573 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf572, buf573, 768, 13, grid=grid(768), stream=stream0)
        buf575 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf574, buf575, 768, 13, grid=grid(768), stream=stream0)
        buf577 = buf566; del buf566  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf576, (1568, 768), (768, 1), 0), permute_472, out=buf577)
        del permute_472
        buf578 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf576, (768, 1568), (1, 768), 0), view_21, out=buf578)
        del view_21
        buf579 = reinterpret_tensor(buf574, (1, 768, 13), (9984, 1, 768), 0); del buf574  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf576, buf579, 9984, 121, grid=grid(9984), stream=stream0)
        buf580 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf579, buf580, 768, 13, grid=grid(768), stream=stream0)
        buf581 = reinterpret_tensor(buf533, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf533  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_23.run(buf577, buf581, 1204224, grid=grid(1204224), stream=stream0)
        buf582 = reinterpret_tensor(buf577, (128, 196, 48), (9408, 48, 1), 0); del buf577  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_477, reinterpret_tensor(buf581, (128, 196, 48), (9408, 48, 1), 0), out=buf582)
        del permute_477
        buf583 = reinterpret_tensor(buf549, (128, 196, 196), (38416, 196, 1), 0); del buf549  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf581, (128, 196, 48), (9408, 48, 1), 0), permute_478, out=buf583)
        del permute_478
        buf584 = reinterpret_tensor(buf581, (1568, 768), (768, 1), 0); del buf581  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_24.run(buf582, buf584, 1204224, grid=grid(1204224), stream=stream0)
        buf585 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf584, (768, 1568), (1, 768), 0), view_5, out=buf585)
        buf586 = reinterpret_tensor(buf582, (1568, 768), (768, 1), 0); del buf582  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf584, permute_482, out=buf586)
        del permute_482
        buf587 = reinterpret_tensor(buf541, (8, 16, 196, 1), (3136, 196, 1, 1), 0); del buf541  # reuse
        buf593 = buf540; del buf540  # reuse
        buf594 = reinterpret_tensor(buf534, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf534  # reuse
        buf597 = reinterpret_tensor(buf530, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf530  # reuse
        # Source Nodes: [attn, mul_1, mul_2, sigmoid, sub_1], Original ATen: [aten._softmax_backward_data, aten.add, aten.div, aten.mul, aten.neg, aten.rsub, aten.sigmoid, aten.sum]
        triton_per_fused__softmax_backward_data_add_div_mul_neg_rsub_sigmoid_sum_25.run(buf583, primals_5, div, div_1, unsqueeze_5, buf587, buf593, buf594, buf597, 25088, 196, grid=grid(25088), stream=stream0)
        buf588 = reinterpret_tensor(buf547, (1, 16, 1, 1, 38), (608, 38, 608, 608, 1), 0); del buf547  # reuse
        buf590 = buf537; del buf537  # reuse
        buf600 = reinterpret_tensor(buf535, (1, 1, 1, 16, 38), (608, 608, 608, 38, 1), 0); del buf535  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused_add_div_mul_sum_26.run(buf583, unsqueeze_5, buf587, div_1, div, buf593, buf594, buf588, buf590, buf600, 608, 8088, grid=grid(608), stream=stream0)
        del buf583
        del buf587
        del div
        del unsqueeze_5
        buf589 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf592 = reinterpret_tensor(buf589, (16, ), (1, ), 0); del buf589  # reuse
        # Source Nodes: [sigmoid], Original ATen: [aten.add, aten.div, aten.mul, aten.neg, aten.sigmoid, aten.sigmoid_backward, aten.sum, aten.view]
        triton_per_fused_add_div_mul_neg_sigmoid_sigmoid_backward_sum_view_27.run(buf592, buf588, buf590, primals_5, 16, 38, grid=grid(16), stream=stream0)
        del buf588
        del buf590
        del primals_5
        buf598 = reinterpret_tensor(buf584, (128, 48, 196), (9408, 196, 1), 0); del buf584  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_484, reinterpret_tensor(buf597, (128, 196, 196), (38416, 196, 1), 0), out=buf598)
        del permute_484
        buf599 = reinterpret_tensor(buf545, (128, 196, 48), (9408, 48, 1), 0); del buf545  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf597, (128, 196, 196), (38416, 196, 1), 0), permute_485, out=buf599)
        del permute_485
        buf601 = empty((1, 1, 1, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_28.run(buf600, buf601, 16, 38, grid=grid(16), stream=stream0)
        del buf600
        buf602 = reinterpret_tensor(buf597, (8, 196, 196, 16), (614656, 3136, 16, 1), 0); del buf597  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf593, div_1, buf594, buf602, 128, 38416, grid=grid(128, 38416), stream=stream0)
        del buf593
        del buf594
        del div_1
        buf603 = empty((16, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf602, (16, 307328), (1, 16), 0), view_8, out=buf603)
        del buf602
        del view_8
        buf604 = buf551; del buf551  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_30.run(buf598, buf599, buf604, 1568, 1536, grid=grid(1568, 1536), stream=stream0)
        del buf598
        buf605 = empty((1536, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf604, (1536, 1568), (1, 1536), 0), view_5, out=buf605)
        del view_5
        buf606 = reinterpret_tensor(buf599, (1568, 768), (768, 1), 0); del buf599  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf604, (1568, 1536), (1536, 1), 0), permute_494, out=buf606)
        del buf604
        del permute_494
        buf613 = buf576; del buf576  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_31.run(buf613, buf586, buf606, primals_3, mul, div_86, 1568, 768, grid=grid(1568), stream=stream0)
        del div_86
        del primals_3
        buf609 = reinterpret_tensor(buf579, (768, 13), (1, 768), 0); del buf579  # reuse
        buf611 = buf572; del buf572  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_32.run(buf586, buf606, mul, buf609, buf611, 9984, 121, grid=grid(9984), stream=stream0)
        del buf586
        del buf606
        del mul
        buf610 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf609, buf610, 768, 13, grid=grid(768), stream=stream0)
        del buf609
        buf612 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf611, buf612, 768, 13, grid=grid(768), stream=stream0)
        buf614 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_34.run(buf82, buf614, 768, 8, grid=grid(768), stream=stream0)
        del buf82
        buf615 = empty((1, 196, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_35.run(buf613, buf615, 150528, 8, grid=grid(150528), stream=stream0)
        buf616 = buf611; del buf611  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_36.run(buf613, buf616, 9984, 121, grid=grid(9984), stream=stream0)
        buf617 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf616, buf617, 768, 13, grid=grid(768), stream=stream0)
        del buf616
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf618 = aten.convolution_backward(reinterpret_tensor(buf613, (8, 768, 14, 14), (150528, 1, 10752, 768), 0), primals_181, primals_63, [768], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf613
        del primals_181
        del primals_63
        buf619 = buf618[1]
        return (buf615, buf614, buf610, buf612, buf592, buf573, buf575, buf557, buf559, buf539, buf520, buf522, buf504, buf506, buf486, buf467, buf469, buf451, buf453, buf433, buf414, buf416, buf398, buf400, buf380, buf361, buf363, buf345, buf347, buf327, buf308, buf310, buf292, buf294, buf274, buf255, buf257, buf239, buf241, buf221, buf202, buf204, buf186, buf188, buf168, buf149, buf151, buf133, buf135, buf115, buf96, buf98, buf79, buf81, buf58, buf60, buf42, buf44, buf21, buf23, buf7, buf8, buf619, buf617, reinterpret_tensor(buf605, (1536, 768), (768, 1), 0), reinterpret_tensor(buf603, (16, 3), (3, 1), 0), reinterpret_tensor(buf601, (16, ), (1, ), 0), reinterpret_tensor(buf585, (768, 768), (768, 1), 0), reinterpret_tensor(buf578, (768, 768), (768, 1), 0), reinterpret_tensor(buf580, (768, ), (1, ), 0), reinterpret_tensor(buf567, (3072, 768), (768, 1), 0), reinterpret_tensor(buf569, (3072, ), (1, ), 0), reinterpret_tensor(buf562, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf564, (768, ), (1, ), 0), reinterpret_tensor(buf552, (1536, 768), (768, 1), 0), reinterpret_tensor(buf550, (16, 3), (3, 1), 0), reinterpret_tensor(buf548, (16, ), (1, ), 0), reinterpret_tensor(buf532, (768, 768), (768, 1), 0), reinterpret_tensor(buf525, (768, 768), (768, 1), 0), reinterpret_tensor(buf527, (768, ), (1, ), 0), reinterpret_tensor(buf514, (3072, 768), (768, 1), 0), reinterpret_tensor(buf516, (3072, ), (1, ), 0), reinterpret_tensor(buf509, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf511, (768, ), (1, ), 0), reinterpret_tensor(buf499, (1536, 768), (768, 1), 0), reinterpret_tensor(buf497, (16, 3), (3, 1), 0), reinterpret_tensor(buf495, (16, ), (1, ), 0), reinterpret_tensor(buf479, (768, 768), (768, 1), 0), reinterpret_tensor(buf472, (768, 768), (768, 1), 0), reinterpret_tensor(buf474, (768, ), (1, ), 0), reinterpret_tensor(buf461, (3072, 768), (768, 1), 0), reinterpret_tensor(buf463, (3072, ), (1, ), 0), reinterpret_tensor(buf456, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf458, (768, ), (1, ), 0), reinterpret_tensor(buf446, (1536, 768), (768, 1), 0), reinterpret_tensor(buf444, (16, 3), (3, 1), 0), reinterpret_tensor(buf442, (16, ), (1, ), 0), reinterpret_tensor(buf426, (768, 768), (768, 1), 0), reinterpret_tensor(buf419, (768, 768), (768, 1), 0), reinterpret_tensor(buf421, (768, ), (1, ), 0), reinterpret_tensor(buf408, (3072, 768), (768, 1), 0), reinterpret_tensor(buf410, (3072, ), (1, ), 0), reinterpret_tensor(buf403, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf405, (768, ), (1, ), 0), reinterpret_tensor(buf393, (1536, 768), (768, 1), 0), reinterpret_tensor(buf391, (16, 3), (3, 1), 0), reinterpret_tensor(buf389, (16, ), (1, ), 0), reinterpret_tensor(buf373, (768, 768), (768, 1), 0), reinterpret_tensor(buf366, (768, 768), (768, 1), 0), reinterpret_tensor(buf368, (768, ), (1, ), 0), reinterpret_tensor(buf355, (3072, 768), (768, 1), 0), reinterpret_tensor(buf357, (3072, ), (1, ), 0), reinterpret_tensor(buf350, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf352, (768, ), (1, ), 0), reinterpret_tensor(buf340, (1536, 768), (768, 1), 0), reinterpret_tensor(buf338, (16, 3), (3, 1), 0), reinterpret_tensor(buf336, (16, ), (1, ), 0), reinterpret_tensor(buf320, (768, 768), (768, 1), 0), reinterpret_tensor(buf313, (768, 768), (768, 1), 0), reinterpret_tensor(buf315, (768, ), (1, ), 0), reinterpret_tensor(buf302, (3072, 768), (768, 1), 0), reinterpret_tensor(buf304, (3072, ), (1, ), 0), reinterpret_tensor(buf297, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf299, (768, ), (1, ), 0), reinterpret_tensor(buf287, (1536, 768), (768, 1), 0), reinterpret_tensor(buf285, (16, 3), (3, 1), 0), reinterpret_tensor(buf283, (16, ), (1, ), 0), reinterpret_tensor(buf267, (768, 768), (768, 1), 0), reinterpret_tensor(buf260, (768, 768), (768, 1), 0), reinterpret_tensor(buf262, (768, ), (1, ), 0), reinterpret_tensor(buf249, (3072, 768), (768, 1), 0), reinterpret_tensor(buf251, (3072, ), (1, ), 0), reinterpret_tensor(buf244, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf246, (768, ), (1, ), 0), reinterpret_tensor(buf234, (1536, 768), (768, 1), 0), reinterpret_tensor(buf232, (16, 3), (3, 1), 0), reinterpret_tensor(buf230, (16, ), (1, ), 0), reinterpret_tensor(buf214, (768, 768), (768, 1), 0), reinterpret_tensor(buf207, (768, 768), (768, 1), 0), reinterpret_tensor(buf209, (768, ), (1, ), 0), reinterpret_tensor(buf196, (3072, 768), (768, 1), 0), reinterpret_tensor(buf198, (3072, ), (1, ), 0), reinterpret_tensor(buf191, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf193, (768, ), (1, ), 0), reinterpret_tensor(buf181, (1536, 768), (768, 1), 0), reinterpret_tensor(buf179, (16, 3), (3, 1), 0), reinterpret_tensor(buf177, (16, ), (1, ), 0), reinterpret_tensor(buf161, (768, 768), (768, 1), 0), reinterpret_tensor(buf154, (768, 768), (768, 1), 0), reinterpret_tensor(buf156, (768, ), (1, ), 0), reinterpret_tensor(buf143, (3072, 768), (768, 1), 0), reinterpret_tensor(buf145, (3072, ), (1, ), 0), reinterpret_tensor(buf138, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf140, (768, ), (1, ), 0), reinterpret_tensor(buf128, (1536, 768), (768, 1), 0), reinterpret_tensor(buf126, (16, 3), (3, 1), 0), reinterpret_tensor(buf124, (16, ), (1, ), 0), reinterpret_tensor(buf108, (768, 768), (768, 1), 0), reinterpret_tensor(buf101, (768, 768), (768, 1), 0), reinterpret_tensor(buf103, (768, ), (1, ), 0), reinterpret_tensor(buf90, (3072, 768), (768, 1), 0), reinterpret_tensor(buf92, (3072, ), (1, ), 0), reinterpret_tensor(buf85, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf87, (768, ), (1, ), 0), reinterpret_tensor(buf74, (2304, 768), (768, 1), 0), reinterpret_tensor(buf63, (768, 768), (768, 1), 0), reinterpret_tensor(buf65, (768, ), (1, ), 0), reinterpret_tensor(buf52, (3072, 768), (768, 1), 0), reinterpret_tensor(buf54, (3072, ), (1, ), 0), reinterpret_tensor(buf47, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf49, (768, ), (1, ), 0), reinterpret_tensor(buf37, (2304, 768), (768, 1), 0), reinterpret_tensor(buf26, (768, 768), (768, 1), 0), reinterpret_tensor(buf28, (768, ), (1, ), 0), reinterpret_tensor(buf15, (3072, 768), (768, 1), 0), reinterpret_tensor(buf17, (3072, ), (1, ), 0), reinterpret_tensor(buf10, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf12, (768, ), (1, ), 0), reinterpret_tensor(buf1, (1000, 768), (768, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_3 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((768, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    mul = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_5 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_8 = rand_strided((307328, 3), (3, 1), device='cuda:0', dtype=torch.float32)
    div = rand_strided((8, 16, 196, 196), (614656, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    div_1 = rand_strided((8, 16, 196, 196), (614656, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_5 = rand_strided((8, 16, 196, 1), (3136, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    view_21 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_5 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_23 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_1 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_25 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_10 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_31 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((8, 16, 196, 196), (614656, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    div_4 = rand_strided((8, 16, 196, 196), (614656, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_11 = rand_strided((8, 16, 196, 1), (3136, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    view_47 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_15 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_49 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_4 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_51 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_20 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_57 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_6 = rand_strided((8, 16, 196, 196), (614656, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    div_7 = rand_strided((8, 16, 196, 196), (614656, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_17 = rand_strided((8, 16, 196, 1), (3136, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    view_73 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_25 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_75 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_7 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_77 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_30 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_83 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_9 = rand_strided((8, 16, 196, 196), (614656, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    div_10 = rand_strided((8, 16, 196, 196), (614656, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_23 = rand_strided((8, 16, 196, 1), (3136, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    view_99 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_35 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_101 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_103 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_40 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_109 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_12 = rand_strided((8, 16, 196, 196), (614656, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    div_13 = rand_strided((8, 16, 196, 196), (614656, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_29 = rand_strided((8, 16, 196, 1), (3136, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    view_125 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_45 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_127 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_13 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_129 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_50 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_135 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_15 = rand_strided((8, 16, 196, 196), (614656, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    div_16 = rand_strided((8, 16, 196, 196), (614656, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_35 = rand_strided((8, 16, 196, 1), (3136, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    view_151 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_55 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_153 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_16 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_155 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_60 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_161 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_18 = rand_strided((8, 16, 196, 196), (614656, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    div_19 = rand_strided((8, 16, 196, 196), (614656, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_41 = rand_strided((8, 16, 196, 1), (3136, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    view_177 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_65 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_179 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_19 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_181 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_70 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_187 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_21 = rand_strided((8, 16, 196, 196), (614656, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    div_22 = rand_strided((8, 16, 196, 196), (614656, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_47 = rand_strided((8, 16, 196, 1), (3136, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    view_203 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_75 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_205 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_207 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_80 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_213 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_24 = rand_strided((8, 16, 196, 196), (614656, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    div_25 = rand_strided((8, 16, 196, 196), (614656, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_53 = rand_strided((8, 16, 196, 1), (3136, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    view_229 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_85 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_231 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_25 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_233 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_90 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_239 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_27 = rand_strided((8, 16, 196, 196), (614656, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    div_28 = rand_strided((8, 16, 196, 196), (614656, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_59 = rand_strided((8, 16, 196, 1), (3136, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    view_255 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_95 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_257 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_28 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_259 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    cat = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_41 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_20 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_261 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_271 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_103 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_273 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_31 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_275 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_108 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_277 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_287 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_111 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_289 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_291 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_116 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    clone_167 = rand_strided((8, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_126 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_32 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_130 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_134 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_33 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_138 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_143 = rand_strided((128, 197, 197), (38809, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_144 = rand_strided((128, 48, 197), (9456, 1, 48), device='cuda:0', dtype=torch.float32)
    alias_42 = rand_strided((8, 16, 197, 197), (620944, 38809, 197, 1), device='cuda:0', dtype=torch.float32)
    permute_145 = rand_strided((128, 48, 197), (9456, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_146 = rand_strided((128, 197, 48), (9456, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_151 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_34 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_153 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_157 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_35 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_161 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_166 = rand_strided((128, 197, 197), (38809, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_167 = rand_strided((128, 48, 197), (9456, 1, 48), device='cuda:0', dtype=torch.float32)
    alias_43 = rand_strided((8, 16, 197, 197), (620944, 38809, 197, 1), device='cuda:0', dtype=torch.float32)
    permute_168 = rand_strided((128, 48, 197), (9456, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_169 = rand_strided((128, 197, 48), (9456, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_174 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_176 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_180 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_37 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_184 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_189 = rand_strided((128, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_190 = rand_strided((128, 48, 196), (9408, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_194 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_196 = rand_strided((128, 48, 196), (9408, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_197 = rand_strided((128, 196, 48), (9408, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_206 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_41 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_208 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_212 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_42 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_216 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_221 = rand_strided((128, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_222 = rand_strided((128, 48, 196), (9408, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_226 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_228 = rand_strided((128, 48, 196), (9408, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_229 = rand_strided((128, 196, 48), (9408, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_238 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_46 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_240 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_244 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_47 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_248 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_253 = rand_strided((128, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_254 = rand_strided((128, 48, 196), (9408, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_258 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_260 = rand_strided((128, 48, 196), (9408, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_261 = rand_strided((128, 196, 48), (9408, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_270 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_51 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_272 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_276 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_52 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_280 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_285 = rand_strided((128, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_286 = rand_strided((128, 48, 196), (9408, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_290 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_292 = rand_strided((128, 48, 196), (9408, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_293 = rand_strided((128, 196, 48), (9408, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_302 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_56 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_304 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_308 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_57 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_312 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_317 = rand_strided((128, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_318 = rand_strided((128, 48, 196), (9408, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_322 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_324 = rand_strided((128, 48, 196), (9408, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_325 = rand_strided((128, 196, 48), (9408, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_334 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_61 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_336 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_340 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_62 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_344 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_349 = rand_strided((128, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_350 = rand_strided((128, 48, 196), (9408, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_354 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_356 = rand_strided((128, 48, 196), (9408, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_357 = rand_strided((128, 196, 48), (9408, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_366 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_66 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_368 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_372 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_67 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_376 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_381 = rand_strided((128, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_382 = rand_strided((128, 48, 196), (9408, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_386 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_388 = rand_strided((128, 48, 196), (9408, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_389 = rand_strided((128, 196, 48), (9408, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_398 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_71 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_400 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_404 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_72 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_408 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_413 = rand_strided((128, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_414 = rand_strided((128, 48, 196), (9408, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_418 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_420 = rand_strided((128, 48, 196), (9408, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_421 = rand_strided((128, 196, 48), (9408, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_430 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_76 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_432 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_436 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_77 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_440 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_445 = rand_strided((128, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_446 = rand_strided((128, 48, 196), (9408, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_450 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_452 = rand_strided((128, 48, 196), (9408, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_453 = rand_strided((128, 196, 48), (9408, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_462 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_81 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_464 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_468 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_82 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_472 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_477 = rand_strided((128, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_478 = rand_strided((128, 48, 196), (9408, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_482 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_484 = rand_strided((128, 48, 196), (9408, 1, 48), device='cuda:0', dtype=torch.float32)
    permute_485 = rand_strided((128, 196, 48), (9408, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_494 = rand_strided((1536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_86 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_3, primals_5, primals_6, primals_8, primals_10, primals_11, primals_13, primals_15, primals_16, primals_18, primals_20, primals_21, primals_23, primals_25, primals_26, primals_28, primals_30, primals_31, primals_33, primals_35, primals_36, primals_38, primals_40, primals_41, primals_43, primals_45, primals_46, primals_48, primals_50, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_181, mul, view_5, view_8, div, div_1, unsqueeze_5, view_21, mul_5, view_23, addmm_1, view_25, mul_10, view_31, div_3, div_4, unsqueeze_11, view_47, mul_15, view_49, addmm_4, view_51, mul_20, view_57, div_6, div_7, unsqueeze_17, view_73, mul_25, view_75, addmm_7, view_77, mul_30, view_83, div_9, div_10, unsqueeze_23, view_99, mul_35, view_101, addmm_10, view_103, mul_40, view_109, div_12, div_13, unsqueeze_29, view_125, mul_45, view_127, addmm_13, view_129, mul_50, view_135, div_15, div_16, unsqueeze_35, view_151, mul_55, view_153, addmm_16, view_155, mul_60, view_161, div_18, div_19, unsqueeze_41, view_177, mul_65, view_179, addmm_19, view_181, mul_70, view_187, div_21, div_22, unsqueeze_47, view_203, mul_75, view_205, addmm_22, view_207, mul_80, view_213, div_24, div_25, unsqueeze_53, view_229, mul_85, view_231, addmm_25, view_233, mul_90, view_239, div_27, div_28, unsqueeze_59, view_255, mul_95, view_257, addmm_28, view_259, cat, getitem_41, rsqrt_20, view_261, view_271, mul_103, view_273, addmm_31, view_275, mul_108, view_277, view_287, mul_111, view_289, addmm_34, view_291, mul_116, clone_167, permute_126, div_32, permute_130, permute_134, div_33, permute_138, permute_143, permute_144, alias_42, permute_145, permute_146, permute_151, div_34, permute_153, permute_157, div_35, permute_161, permute_166, permute_167, alias_43, permute_168, permute_169, permute_174, permute_176, permute_180, div_37, permute_184, permute_189, permute_190, permute_194, permute_196, permute_197, permute_206, div_41, permute_208, permute_212, div_42, permute_216, permute_221, permute_222, permute_226, permute_228, permute_229, permute_238, div_46, permute_240, permute_244, div_47, permute_248, permute_253, permute_254, permute_258, permute_260, permute_261, permute_270, div_51, permute_272, permute_276, div_52, permute_280, permute_285, permute_286, permute_290, permute_292, permute_293, permute_302, div_56, permute_304, permute_308, div_57, permute_312, permute_317, permute_318, permute_322, permute_324, permute_325, permute_334, div_61, permute_336, permute_340, div_62, permute_344, permute_349, permute_350, permute_354, permute_356, permute_357, permute_366, div_66, permute_368, permute_372, div_67, permute_376, permute_381, permute_382, permute_386, permute_388, permute_389, permute_398, div_71, permute_400, permute_404, div_72, permute_408, permute_413, permute_414, permute_418, permute_420, permute_421, permute_430, div_76, permute_432, permute_436, div_77, permute_440, permute_445, permute_446, permute_450, permute_452, permute_453, permute_462, div_81, permute_464, permute_468, div_82, permute_472, permute_477, permute_478, permute_482, permute_484, permute_485, permute_494, div_86, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('convit_base', benchmark_compiled_module)
