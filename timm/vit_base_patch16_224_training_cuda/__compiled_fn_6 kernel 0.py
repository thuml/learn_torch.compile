
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
# Source Nodes: [x_147], Original ATen: [aten.gelu, aten.gelu_backward]
# x_147 => add_83, erf_11, mul_82
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


# kernel path: /tmp/torchinductor_youkaichao/eg/cegnlotd52ae7vcllsnplinap6nxguasxvmof5rw6skzqlfwrnx2.py
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
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3631104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x5 = (xindex // 151296)
    x6 = xindex
    x0 = xindex % 768
    x3 = (xindex // 1210368)
    x7 = (xindex // 768) % 1576
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
    tmp12 = tl.load(in_ptr1 + ((-1210368) + x6), tmp11, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 24, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-2420736) + x6), tmp15, other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tl.where(tmp11, tmp14, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tl.store(out_ptr0 + (x0 + (768*x3) + (2304*x7)), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bu/cbux3bmw4uykmuw2ynbuaegn4ev3uq47tydz2l557ojbt5m7d3cd.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 29952
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 2304)
    x0 = xindex % 2304
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 1576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (2304*r2) + (281088*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/we/cwevxsfggk42t4irrnkjx3l2bf4lriavwolc63u3fyfo7az2quii.py
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
    size_hints=[4096, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2304
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2304*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qq/cqqxb2ek3uwumyhsr3kjtrzaewpuf3rdsx5qbcf5ej6ivhhwtw4w.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 151296
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


# kernel path: /tmp/torchinductor_youkaichao/j4/cj43zo7f72npmled6y4z4iqi7m3ca3jaiet6sw2z5h2r4zdrpajc.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_15', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/p7/cp746vxl4ht7g5rlih6rjvnbtfrqqt4np24krf5utgli7hyytkvy.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_16', 'mutated_arg_names': []}
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
        tmp3 = tl.load(in_ptr0 + (768 + x0 + (768*((r2 + (121*x1)) % 196)) + (151296*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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
    primals_3, primals_5, primals_11, primals_17, primals_23, primals_29, primals_35, primals_41, primals_47, primals_53, primals_59, primals_65, primals_71, primals_77, primals_83, primals_89, primals_95, primals_101, primals_107, primals_113, primals_119, primals_125, primals_131, primals_137, primals_143, primals_149, primals_153, mul, view_1, getitem_2, getitem_3, getitem_4, getitem_6, getitem_7, getitem_8, view_5, mul_2, view_7, addmm_2, view_9, mul_7, view_11, getitem_13, getitem_14, getitem_15, getitem_17, getitem_18, getitem_19, view_15, mul_9, view_17, addmm_6, view_19, mul_14, view_21, getitem_24, getitem_25, getitem_26, getitem_28, getitem_29, getitem_30, view_25, mul_16, view_27, addmm_10, view_29, mul_21, view_31, getitem_35, getitem_36, getitem_37, getitem_39, getitem_40, getitem_41, view_35, mul_23, view_37, addmm_14, view_39, mul_28, view_41, getitem_46, getitem_47, getitem_48, getitem_50, getitem_51, getitem_52, view_45, mul_30, view_47, addmm_18, view_49, mul_35, view_51, getitem_57, getitem_58, getitem_59, getitem_61, getitem_62, getitem_63, view_55, mul_37, view_57, addmm_22, view_59, mul_42, view_61, getitem_68, getitem_69, getitem_70, getitem_72, getitem_73, getitem_74, view_65, mul_44, view_67, addmm_26, view_69, mul_49, view_71, getitem_79, getitem_80, getitem_81, getitem_83, getitem_84, getitem_85, view_75, mul_51, view_77, addmm_30, view_79, mul_56, view_81, getitem_90, getitem_91, getitem_92, getitem_94, getitem_95, getitem_96, view_85, mul_58, view_87, addmm_34, view_89, mul_63, view_91, getitem_101, getitem_102, getitem_103, getitem_105, getitem_106, getitem_107, view_95, mul_65, view_97, addmm_38, view_99, mul_70, view_101, getitem_112, getitem_113, getitem_114, getitem_116, getitem_117, getitem_118, view_105, mul_72, view_107, addmm_42, view_109, mul_77, view_111, getitem_123, getitem_124, getitem_125, getitem_127, getitem_128, getitem_129, view_115, mul_79, view_117, addmm_46, view_119, mul_84, clone_37, permute_74, div, permute_78, permute_82, div_1, permute_86, alias_12, permute_92, div_2, permute_96, permute_100, div_3, permute_104, alias_13, permute_110, div_4, permute_114, permute_118, div_5, permute_122, alias_14, permute_128, div_6, permute_132, permute_136, div_7, permute_140, alias_15, permute_146, div_8, permute_150, permute_154, div_9, permute_158, alias_16, permute_164, div_10, permute_168, permute_172, div_11, permute_176, alias_17, permute_182, div_12, permute_186, permute_190, div_13, permute_194, alias_18, permute_200, div_14, permute_204, permute_208, div_15, permute_212, alias_19, permute_218, div_16, permute_222, permute_226, div_17, permute_230, alias_20, permute_236, div_18, permute_240, permute_244, div_19, permute_248, alias_21, permute_254, div_20, permute_258, permute_262, div_21, permute_266, alias_22, permute_272, div_22, permute_276, permute_280, div_23, permute_284, alias_23, permute_290, div_24, tangents_1 = args
    args.clear()
    assert_size_stride(primals_3, (768, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(primals_5, (768, ), (1, ))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_17, (768, ), (1, ))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_65, (768, ), (1, ))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_77, (768, ), (1, ))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_89, (768, ), (1, ))
    assert_size_stride(primals_95, (768, ), (1, ))
    assert_size_stride(primals_101, (768, ), (1, ))
    assert_size_stride(primals_107, (768, ), (1, ))
    assert_size_stride(primals_113, (768, ), (1, ))
    assert_size_stride(primals_119, (768, ), (1, ))
    assert_size_stride(primals_125, (768, ), (1, ))
    assert_size_stride(primals_131, (768, ), (1, ))
    assert_size_stride(primals_137, (768, ), (1, ))
    assert_size_stride(primals_143, (768, ), (1, ))
    assert_size_stride(primals_149, (768, ), (1, ))
    assert_size_stride(primals_153, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(mul, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_1, (1576, 768), (768, 1))
    assert_size_stride(getitem_2, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_3, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_4, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_6, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_7, (), ())
    assert_size_stride(getitem_8, (), ())
    assert_size_stride(view_5, (1576, 768), (768, 1))
    assert_size_stride(mul_2, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_7, (1576, 768), (768, 1))
    assert_size_stride(addmm_2, (1576, 3072), (3072, 1))
    assert_size_stride(view_9, (1576, 3072), (3072, 1))
    assert_size_stride(mul_7, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_11, (1576, 768), (768, 1))
    assert_size_stride(getitem_13, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_14, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_15, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_17, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_18, (), ())
    assert_size_stride(getitem_19, (), ())
    assert_size_stride(view_15, (1576, 768), (768, 1))
    assert_size_stride(mul_9, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_17, (1576, 768), (768, 1))
    assert_size_stride(addmm_6, (1576, 3072), (3072, 1))
    assert_size_stride(view_19, (1576, 3072), (3072, 1))
    assert_size_stride(mul_14, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_21, (1576, 768), (768, 1))
    assert_size_stride(getitem_24, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_25, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_26, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_28, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_29, (), ())
    assert_size_stride(getitem_30, (), ())
    assert_size_stride(view_25, (1576, 768), (768, 1))
    assert_size_stride(mul_16, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_27, (1576, 768), (768, 1))
    assert_size_stride(addmm_10, (1576, 3072), (3072, 1))
    assert_size_stride(view_29, (1576, 3072), (3072, 1))
    assert_size_stride(mul_21, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_31, (1576, 768), (768, 1))
    assert_size_stride(getitem_35, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_36, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_37, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_39, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_40, (), ())
    assert_size_stride(getitem_41, (), ())
    assert_size_stride(view_35, (1576, 768), (768, 1))
    assert_size_stride(mul_23, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_37, (1576, 768), (768, 1))
    assert_size_stride(addmm_14, (1576, 3072), (3072, 1))
    assert_size_stride(view_39, (1576, 3072), (3072, 1))
    assert_size_stride(mul_28, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_41, (1576, 768), (768, 1))
    assert_size_stride(getitem_46, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_47, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_48, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_50, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_51, (), ())
    assert_size_stride(getitem_52, (), ())
    assert_size_stride(view_45, (1576, 768), (768, 1))
    assert_size_stride(mul_30, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_47, (1576, 768), (768, 1))
    assert_size_stride(addmm_18, (1576, 3072), (3072, 1))
    assert_size_stride(view_49, (1576, 3072), (3072, 1))
    assert_size_stride(mul_35, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_51, (1576, 768), (768, 1))
    assert_size_stride(getitem_57, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_58, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_59, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_61, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_62, (), ())
    assert_size_stride(getitem_63, (), ())
    assert_size_stride(view_55, (1576, 768), (768, 1))
    assert_size_stride(mul_37, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_57, (1576, 768), (768, 1))
    assert_size_stride(addmm_22, (1576, 3072), (3072, 1))
    assert_size_stride(view_59, (1576, 3072), (3072, 1))
    assert_size_stride(mul_42, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_61, (1576, 768), (768, 1))
    assert_size_stride(getitem_68, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_69, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_70, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_72, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_73, (), ())
    assert_size_stride(getitem_74, (), ())
    assert_size_stride(view_65, (1576, 768), (768, 1))
    assert_size_stride(mul_44, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_67, (1576, 768), (768, 1))
    assert_size_stride(addmm_26, (1576, 3072), (3072, 1))
    assert_size_stride(view_69, (1576, 3072), (3072, 1))
    assert_size_stride(mul_49, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_71, (1576, 768), (768, 1))
    assert_size_stride(getitem_79, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_80, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_81, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_83, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_84, (), ())
    assert_size_stride(getitem_85, (), ())
    assert_size_stride(view_75, (1576, 768), (768, 1))
    assert_size_stride(mul_51, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_77, (1576, 768), (768, 1))
    assert_size_stride(addmm_30, (1576, 3072), (3072, 1))
    assert_size_stride(view_79, (1576, 3072), (3072, 1))
    assert_size_stride(mul_56, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_81, (1576, 768), (768, 1))
    assert_size_stride(getitem_90, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_91, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_92, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_94, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_95, (), ())
    assert_size_stride(getitem_96, (), ())
    assert_size_stride(view_85, (1576, 768), (768, 1))
    assert_size_stride(mul_58, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_87, (1576, 768), (768, 1))
    assert_size_stride(addmm_34, (1576, 3072), (3072, 1))
    assert_size_stride(view_89, (1576, 3072), (3072, 1))
    assert_size_stride(mul_63, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_91, (1576, 768), (768, 1))
    assert_size_stride(getitem_101, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_102, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_103, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_105, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_106, (), ())
    assert_size_stride(getitem_107, (), ())
    assert_size_stride(view_95, (1576, 768), (768, 1))
    assert_size_stride(mul_65, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_97, (1576, 768), (768, 1))
    assert_size_stride(addmm_38, (1576, 3072), (3072, 1))
    assert_size_stride(view_99, (1576, 3072), (3072, 1))
    assert_size_stride(mul_70, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_101, (1576, 768), (768, 1))
    assert_size_stride(getitem_112, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_113, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_114, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_116, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_117, (), ())
    assert_size_stride(getitem_118, (), ())
    assert_size_stride(view_105, (1576, 768), (768, 1))
    assert_size_stride(mul_72, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_107, (1576, 768), (768, 1))
    assert_size_stride(addmm_42, (1576, 3072), (3072, 1))
    assert_size_stride(view_109, (1576, 3072), (3072, 1))
    assert_size_stride(mul_77, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_111, (1576, 768), (768, 1))
    assert_size_stride(getitem_123, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_124, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_125, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_127, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_128, (), ())
    assert_size_stride(getitem_129, (), ())
    assert_size_stride(view_115, (1576, 768), (768, 1))
    assert_size_stride(mul_79, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_117, (1576, 768), (768, 1))
    assert_size_stride(addmm_46, (1576, 3072), (3072, 1))
    assert_size_stride(view_119, (1576, 3072), (3072, 1))
    assert_size_stride(mul_84, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(clone_37, (8, 768), (768, 1))
    assert_size_stride(permute_74, (1000, 768), (768, 1))
    assert_size_stride(div, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_78, (768, 3072), (3072, 1))
    assert_size_stride(permute_82, (3072, 768), (768, 1))
    assert_size_stride(div_1, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_86, (768, 768), (768, 1))
    assert_size_stride(alias_12, (8, 12, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(permute_92, (2304, 768), (768, 1))
    assert_size_stride(div_2, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_96, (768, 3072), (3072, 1))
    assert_size_stride(permute_100, (3072, 768), (768, 1))
    assert_size_stride(div_3, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_104, (768, 768), (768, 1))
    assert_size_stride(alias_13, (8, 12, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(permute_110, (2304, 768), (768, 1))
    assert_size_stride(div_4, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_114, (768, 3072), (3072, 1))
    assert_size_stride(permute_118, (3072, 768), (768, 1))
    assert_size_stride(div_5, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_122, (768, 768), (768, 1))
    assert_size_stride(alias_14, (8, 12, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(permute_128, (2304, 768), (768, 1))
    assert_size_stride(div_6, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_132, (768, 3072), (3072, 1))
    assert_size_stride(permute_136, (3072, 768), (768, 1))
    assert_size_stride(div_7, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_140, (768, 768), (768, 1))
    assert_size_stride(alias_15, (8, 12, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(permute_146, (2304, 768), (768, 1))
    assert_size_stride(div_8, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_150, (768, 3072), (3072, 1))
    assert_size_stride(permute_154, (3072, 768), (768, 1))
    assert_size_stride(div_9, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_158, (768, 768), (768, 1))
    assert_size_stride(alias_16, (8, 12, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(permute_164, (2304, 768), (768, 1))
    assert_size_stride(div_10, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_168, (768, 3072), (3072, 1))
    assert_size_stride(permute_172, (3072, 768), (768, 1))
    assert_size_stride(div_11, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_176, (768, 768), (768, 1))
    assert_size_stride(alias_17, (8, 12, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(permute_182, (2304, 768), (768, 1))
    assert_size_stride(div_12, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_186, (768, 3072), (3072, 1))
    assert_size_stride(permute_190, (3072, 768), (768, 1))
    assert_size_stride(div_13, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_194, (768, 768), (768, 1))
    assert_size_stride(alias_18, (8, 12, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(permute_200, (2304, 768), (768, 1))
    assert_size_stride(div_14, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_204, (768, 3072), (3072, 1))
    assert_size_stride(permute_208, (3072, 768), (768, 1))
    assert_size_stride(div_15, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_212, (768, 768), (768, 1))
    assert_size_stride(alias_19, (8, 12, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(permute_218, (2304, 768), (768, 1))
    assert_size_stride(div_16, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_222, (768, 3072), (3072, 1))
    assert_size_stride(permute_226, (3072, 768), (768, 1))
    assert_size_stride(div_17, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_230, (768, 768), (768, 1))
    assert_size_stride(alias_20, (8, 12, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(permute_236, (2304, 768), (768, 1))
    assert_size_stride(div_18, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_240, (768, 3072), (3072, 1))
    assert_size_stride(permute_244, (3072, 768), (768, 1))
    assert_size_stride(div_19, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_248, (768, 768), (768, 1))
    assert_size_stride(alias_21, (8, 12, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(permute_254, (2304, 768), (768, 1))
    assert_size_stride(div_20, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_258, (768, 3072), (3072, 1))
    assert_size_stride(permute_262, (3072, 768), (768, 1))
    assert_size_stride(div_21, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_266, (768, 768), (768, 1))
    assert_size_stride(alias_22, (8, 12, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(permute_272, (2304, 768), (768, 1))
    assert_size_stride(div_22, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_276, (768, 3072), (3072, 1))
    assert_size_stride(permute_280, (3072, 768), (768, 1))
    assert_size_stride(div_23, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_284, (768, 768), (768, 1))
    assert_size_stride(alias_23, (8, 12, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(permute_290, (2304, 768), (768, 1))
    assert_size_stride(div_24, (8, 197, 1), (197, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_74, out=buf0)
        del permute_74
        buf1 = empty((1000, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_37, out=buf1)
        del clone_37
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf5 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]
        triton_red_fused_native_layer_norm_backward_select_backward_1.run(buf0, primals_149, mul_84, div, buf5, 1576, 768, grid=grid(1576), stream=stream0)
        del div
        del primals_149
        buf6 = empty_strided((768, 13), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]
        triton_red_fused_native_layer_norm_backward_select_backward_2.run(buf0, mul_84, buf6, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_84
        buf7 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf6, buf7, 768, 13, grid=grid(768), stream=stream0)
        buf8 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]
        triton_red_fused_native_layer_norm_backward_select_backward_4.run(buf0, buf8, 768, 1576, grid=grid(768), stream=stream0)
        del buf0
        buf9 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (1576, 768), (768, 1), 0), permute_78, out=buf9)
        del permute_78
        buf10 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (768, 1576), (1, 768), 0), view_119, out=buf10)
        del view_119
        buf11 = reinterpret_tensor(buf6, (1, 768, 13), (9984, 1, 768), 0); del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf5, buf11, 9984, 122, grid=grid(9984), stream=stream0)
        buf12 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf11, buf12, 768, 13, grid=grid(768), stream=stream0)
        buf13 = reinterpret_tensor(buf9, (8, 197, 3072), (605184, 3072, 1), 0); del buf9  # reuse
        # Source Nodes: [x_147], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf13, addmm_46, 4841472, grid=grid(4841472), stream=stream0)
        del addmm_46
        buf14 = empty((1576, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (1576, 3072), (3072, 1), 0), permute_82, out=buf14)
        del permute_82
        buf15 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (3072, 1576), (1, 3072), 0), view_117, out=buf15)
        del view_117
        buf16 = empty_strided((1, 3072, 13), (39936, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf13, buf16, 39936, 122, grid=grid(39936), stream=stream0)
        buf17 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf16, buf17, 3072, 13, grid=grid(3072), stream=stream0)
        buf24 = buf5; del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf24, buf14, primals_143, mul_79, div_1, 1576, 768, grid=grid(1576), stream=stream0)
        del div_1
        del primals_143
        buf20 = reinterpret_tensor(buf11, (768, 13), (1, 768), 0); del buf11  # reuse
        buf22 = empty_strided((768, 13), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf14, mul_79, buf20, buf22, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_79
        buf21 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf20, buf21, 768, 13, grid=grid(768), stream=stream0)
        buf23 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf22, buf23, 768, 13, grid=grid(768), stream=stream0)
        buf25 = buf14; del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf24, (1576, 768), (768, 1), 0), permute_86, out=buf25)
        del permute_86
        buf26 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf24, (768, 1576), (1, 768), 0), view_115, out=buf26)
        del view_115
        buf27 = reinterpret_tensor(buf22, (1, 768, 13), (9984, 1, 768), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf24, buf27, 9984, 122, grid=grid(9984), stream=stream0)
        buf28 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf27, buf28, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf29 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf25, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_123, getitem_124, getitem_125, None, alias_12, getitem_127, getitem_128, getitem_129, 0.0, [True, True, True, False])
        del alias_12
        del buf25
        del getitem_123
        del getitem_124
        del getitem_125
        del getitem_127
        del getitem_128
        del getitem_129
        buf30 = buf29[0]
        buf31 = buf29[1]
        buf32 = buf29[2]
        del buf29
        buf33 = empty((8, 197, 3, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf30, buf31, buf32, buf33, 3631104, grid=grid(3631104), stream=stream0)
        del buf30
        del buf31
        buf34 = reinterpret_tensor(buf32, (1576, 768), (768, 1), 0); del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf33, (1576, 2304), (2304, 1), 0), permute_92, out=buf34)
        del permute_92
        buf35 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf33, (2304, 1576), (1, 2304), 0), view_111, out=buf35)
        del view_111
        buf36 = empty_strided((1, 2304, 13), (29952, 1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf33, buf36, 29952, 122, grid=grid(29952), stream=stream0)
        buf37 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf36, buf37, 2304, 13, grid=grid(2304), stream=stream0)
        buf44 = buf24; del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf44, buf34, primals_137, mul_77, div_2, 1576, 768, grid=grid(1576), stream=stream0)
        del div_2
        del primals_137
        buf40 = reinterpret_tensor(buf27, (768, 13), (1, 768), 0); del buf27  # reuse
        buf42 = buf20; del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf34, mul_77, buf40, buf42, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_77
        buf41 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf40, buf41, 768, 13, grid=grid(768), stream=stream0)
        buf43 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf42, buf43, 768, 13, grid=grid(768), stream=stream0)
        buf45 = reinterpret_tensor(buf13, (1576, 3072), (3072, 1), 0); del buf13  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf44, (1576, 768), (768, 1), 0), permute_96, out=buf45)
        del permute_96
        buf46 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf44, (768, 1576), (1, 768), 0), view_109, out=buf46)
        del view_109
        buf47 = reinterpret_tensor(buf42, (1, 768, 13), (9984, 1, 768), 0); del buf42  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf44, buf47, 9984, 122, grid=grid(9984), stream=stream0)
        buf48 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf47, buf48, 768, 13, grid=grid(768), stream=stream0)
        buf49 = reinterpret_tensor(buf45, (8, 197, 3072), (605184, 3072, 1), 0); del buf45  # reuse
        # Source Nodes: [x_135], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf49, addmm_42, 4841472, grid=grid(4841472), stream=stream0)
        del addmm_42
        buf50 = buf34; del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf49, (1576, 3072), (3072, 1), 0), permute_100, out=buf50)
        del permute_100
        buf51 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf49, (3072, 1576), (1, 3072), 0), view_107, out=buf51)
        del view_107
        buf52 = buf16; del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf49, buf52, 39936, 122, grid=grid(39936), stream=stream0)
        buf53 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf52, buf53, 3072, 13, grid=grid(3072), stream=stream0)
        buf60 = buf44; del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf60, buf50, primals_131, mul_72, div_3, 1576, 768, grid=grid(1576), stream=stream0)
        del div_3
        del primals_131
        buf56 = reinterpret_tensor(buf47, (768, 13), (1, 768), 0); del buf47  # reuse
        buf58 = buf40; del buf40  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf50, mul_72, buf56, buf58, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_72
        buf57 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf56, buf57, 768, 13, grid=grid(768), stream=stream0)
        buf59 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf58, buf59, 768, 13, grid=grid(768), stream=stream0)
        buf61 = buf50; del buf50  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf60, (1576, 768), (768, 1), 0), permute_104, out=buf61)
        del permute_104
        buf62 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf60, (768, 1576), (1, 768), 0), view_105, out=buf62)
        del view_105
        buf63 = reinterpret_tensor(buf58, (1, 768, 13), (9984, 1, 768), 0); del buf58  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf60, buf63, 9984, 122, grid=grid(9984), stream=stream0)
        buf64 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf63, buf64, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf65 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf61, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_112, getitem_113, getitem_114, None, alias_13, getitem_116, getitem_117, getitem_118, 0.0, [True, True, True, False])
        del alias_13
        del buf61
        del getitem_112
        del getitem_113
        del getitem_114
        del getitem_116
        del getitem_117
        del getitem_118
        buf66 = buf65[0]
        buf67 = buf65[1]
        buf68 = buf65[2]
        del buf65
        buf69 = buf33; del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf66, buf67, buf68, buf69, 3631104, grid=grid(3631104), stream=stream0)
        del buf66
        del buf67
        buf70 = reinterpret_tensor(buf68, (1576, 768), (768, 1), 0); del buf68  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (1576, 2304), (2304, 1), 0), permute_110, out=buf70)
        del permute_110
        buf71 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (2304, 1576), (1, 2304), 0), view_101, out=buf71)
        del view_101
        buf72 = buf36; del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf69, buf72, 29952, 122, grid=grid(29952), stream=stream0)
        buf73 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf72, buf73, 2304, 13, grid=grid(2304), stream=stream0)
        buf80 = buf60; del buf60  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf80, buf70, primals_125, mul_70, div_4, 1576, 768, grid=grid(1576), stream=stream0)
        del div_4
        del primals_125
        buf76 = reinterpret_tensor(buf63, (768, 13), (1, 768), 0); del buf63  # reuse
        buf78 = buf56; del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf70, mul_70, buf76, buf78, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_70
        buf77 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf76, buf77, 768, 13, grid=grid(768), stream=stream0)
        buf79 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf78, buf79, 768, 13, grid=grid(768), stream=stream0)
        buf81 = reinterpret_tensor(buf49, (1576, 3072), (3072, 1), 0); del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf80, (1576, 768), (768, 1), 0), permute_114, out=buf81)
        del permute_114
        buf82 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf80, (768, 1576), (1, 768), 0), view_99, out=buf82)
        del view_99
        buf83 = reinterpret_tensor(buf78, (1, 768, 13), (9984, 1, 768), 0); del buf78  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf80, buf83, 9984, 122, grid=grid(9984), stream=stream0)
        buf84 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf83, buf84, 768, 13, grid=grid(768), stream=stream0)
        buf85 = reinterpret_tensor(buf81, (8, 197, 3072), (605184, 3072, 1), 0); del buf81  # reuse
        # Source Nodes: [x_123], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf85, addmm_38, 4841472, grid=grid(4841472), stream=stream0)
        del addmm_38
        buf86 = buf70; del buf70  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (1576, 3072), (3072, 1), 0), permute_118, out=buf86)
        del permute_118
        buf87 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (3072, 1576), (1, 3072), 0), view_97, out=buf87)
        del view_97
        buf88 = buf52; del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf85, buf88, 39936, 122, grid=grid(39936), stream=stream0)
        buf89 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf88, buf89, 3072, 13, grid=grid(3072), stream=stream0)
        buf96 = buf80; del buf80  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf96, buf86, primals_119, mul_65, div_5, 1576, 768, grid=grid(1576), stream=stream0)
        del div_5
        del primals_119
        buf92 = reinterpret_tensor(buf83, (768, 13), (1, 768), 0); del buf83  # reuse
        buf94 = buf76; del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf86, mul_65, buf92, buf94, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_65
        buf93 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf92, buf93, 768, 13, grid=grid(768), stream=stream0)
        buf95 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf94, buf95, 768, 13, grid=grid(768), stream=stream0)
        buf97 = buf86; del buf86  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf96, (1576, 768), (768, 1), 0), permute_122, out=buf97)
        del permute_122
        buf98 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf96, (768, 1576), (1, 768), 0), view_95, out=buf98)
        del view_95
        buf99 = reinterpret_tensor(buf94, (1, 768, 13), (9984, 1, 768), 0); del buf94  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf96, buf99, 9984, 122, grid=grid(9984), stream=stream0)
        buf100 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf99, buf100, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf101 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf97, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_101, getitem_102, getitem_103, None, alias_14, getitem_105, getitem_106, getitem_107, 0.0, [True, True, True, False])
        del alias_14
        del buf97
        del getitem_101
        del getitem_102
        del getitem_103
        del getitem_105
        del getitem_106
        del getitem_107
        buf102 = buf101[0]
        buf103 = buf101[1]
        buf104 = buf101[2]
        del buf101
        buf105 = buf69; del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf102, buf103, buf104, buf105, 3631104, grid=grid(3631104), stream=stream0)
        del buf102
        del buf103
        buf106 = reinterpret_tensor(buf104, (1576, 768), (768, 1), 0); del buf104  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf105, (1576, 2304), (2304, 1), 0), permute_128, out=buf106)
        del permute_128
        buf107 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf105, (2304, 1576), (1, 2304), 0), view_91, out=buf107)
        del view_91
        buf108 = buf72; del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf105, buf108, 29952, 122, grid=grid(29952), stream=stream0)
        buf109 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf108, buf109, 2304, 13, grid=grid(2304), stream=stream0)
        buf116 = buf96; del buf96  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf116, buf106, primals_113, mul_63, div_6, 1576, 768, grid=grid(1576), stream=stream0)
        del div_6
        del primals_113
        buf112 = reinterpret_tensor(buf99, (768, 13), (1, 768), 0); del buf99  # reuse
        buf114 = buf92; del buf92  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf106, mul_63, buf112, buf114, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_63
        buf113 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf112, buf113, 768, 13, grid=grid(768), stream=stream0)
        buf115 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf114, buf115, 768, 13, grid=grid(768), stream=stream0)
        buf117 = reinterpret_tensor(buf85, (1576, 3072), (3072, 1), 0); del buf85  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (1576, 768), (768, 1), 0), permute_132, out=buf117)
        del permute_132
        buf118 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (768, 1576), (1, 768), 0), view_89, out=buf118)
        del view_89
        buf119 = reinterpret_tensor(buf114, (1, 768, 13), (9984, 1, 768), 0); del buf114  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf116, buf119, 9984, 122, grid=grid(9984), stream=stream0)
        buf120 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf119, buf120, 768, 13, grid=grid(768), stream=stream0)
        buf121 = reinterpret_tensor(buf117, (8, 197, 3072), (605184, 3072, 1), 0); del buf117  # reuse
        # Source Nodes: [x_111], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf121, addmm_34, 4841472, grid=grid(4841472), stream=stream0)
        del addmm_34
        buf122 = buf106; del buf106  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf121, (1576, 3072), (3072, 1), 0), permute_136, out=buf122)
        del permute_136
        buf123 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf121, (3072, 1576), (1, 3072), 0), view_87, out=buf123)
        del view_87
        buf124 = buf88; del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf121, buf124, 39936, 122, grid=grid(39936), stream=stream0)
        buf125 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf124, buf125, 3072, 13, grid=grid(3072), stream=stream0)
        buf132 = buf116; del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf132, buf122, primals_107, mul_58, div_7, 1576, 768, grid=grid(1576), stream=stream0)
        del div_7
        del primals_107
        buf128 = reinterpret_tensor(buf119, (768, 13), (1, 768), 0); del buf119  # reuse
        buf130 = buf112; del buf112  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf122, mul_58, buf128, buf130, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_58
        buf129 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf128, buf129, 768, 13, grid=grid(768), stream=stream0)
        buf131 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf130, buf131, 768, 13, grid=grid(768), stream=stream0)
        buf133 = buf122; del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf132, (1576, 768), (768, 1), 0), permute_140, out=buf133)
        del permute_140
        buf134 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf132, (768, 1576), (1, 768), 0), view_85, out=buf134)
        del view_85
        buf135 = reinterpret_tensor(buf130, (1, 768, 13), (9984, 1, 768), 0); del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf132, buf135, 9984, 122, grid=grid(9984), stream=stream0)
        buf136 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf135, buf136, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf137 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf133, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_90, getitem_91, getitem_92, None, alias_15, getitem_94, getitem_95, getitem_96, 0.0, [True, True, True, False])
        del alias_15
        del buf133
        del getitem_90
        del getitem_91
        del getitem_92
        del getitem_94
        del getitem_95
        del getitem_96
        buf138 = buf137[0]
        buf139 = buf137[1]
        buf140 = buf137[2]
        del buf137
        buf141 = buf105; del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf138, buf139, buf140, buf141, 3631104, grid=grid(3631104), stream=stream0)
        del buf138
        del buf139
        buf142 = reinterpret_tensor(buf140, (1576, 768), (768, 1), 0); del buf140  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf141, (1576, 2304), (2304, 1), 0), permute_146, out=buf142)
        del permute_146
        buf143 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf141, (2304, 1576), (1, 2304), 0), view_81, out=buf143)
        del view_81
        buf144 = buf108; del buf108  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf141, buf144, 29952, 122, grid=grid(29952), stream=stream0)
        buf145 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf144, buf145, 2304, 13, grid=grid(2304), stream=stream0)
        buf152 = buf132; del buf132  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf152, buf142, primals_101, mul_56, div_8, 1576, 768, grid=grid(1576), stream=stream0)
        del div_8
        del primals_101
        buf148 = reinterpret_tensor(buf135, (768, 13), (1, 768), 0); del buf135  # reuse
        buf150 = buf128; del buf128  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf142, mul_56, buf148, buf150, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_56
        buf149 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf148, buf149, 768, 13, grid=grid(768), stream=stream0)
        buf151 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf150, buf151, 768, 13, grid=grid(768), stream=stream0)
        buf153 = reinterpret_tensor(buf121, (1576, 3072), (3072, 1), 0); del buf121  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf152, (1576, 768), (768, 1), 0), permute_150, out=buf153)
        del permute_150
        buf154 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf152, (768, 1576), (1, 768), 0), view_79, out=buf154)
        del view_79
        buf155 = reinterpret_tensor(buf150, (1, 768, 13), (9984, 1, 768), 0); del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf152, buf155, 9984, 122, grid=grid(9984), stream=stream0)
        buf156 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf155, buf156, 768, 13, grid=grid(768), stream=stream0)
        buf157 = reinterpret_tensor(buf153, (8, 197, 3072), (605184, 3072, 1), 0); del buf153  # reuse
        # Source Nodes: [x_99], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf157, addmm_30, 4841472, grid=grid(4841472), stream=stream0)
        del addmm_30
        buf158 = buf142; del buf142  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf157, (1576, 3072), (3072, 1), 0), permute_154, out=buf158)
        del permute_154
        buf159 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf157, (3072, 1576), (1, 3072), 0), view_77, out=buf159)
        del view_77
        buf160 = buf124; del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf157, buf160, 39936, 122, grid=grid(39936), stream=stream0)
        buf161 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf160, buf161, 3072, 13, grid=grid(3072), stream=stream0)
        buf168 = buf152; del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf168, buf158, primals_95, mul_51, div_9, 1576, 768, grid=grid(1576), stream=stream0)
        del div_9
        del primals_95
        buf164 = reinterpret_tensor(buf155, (768, 13), (1, 768), 0); del buf155  # reuse
        buf166 = buf148; del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf158, mul_51, buf164, buf166, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_51
        buf165 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf164, buf165, 768, 13, grid=grid(768), stream=stream0)
        buf167 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf166, buf167, 768, 13, grid=grid(768), stream=stream0)
        buf169 = buf158; del buf158  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf168, (1576, 768), (768, 1), 0), permute_158, out=buf169)
        del permute_158
        buf170 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf168, (768, 1576), (1, 768), 0), view_75, out=buf170)
        del view_75
        buf171 = reinterpret_tensor(buf166, (1, 768, 13), (9984, 1, 768), 0); del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf168, buf171, 9984, 122, grid=grid(9984), stream=stream0)
        buf172 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf171, buf172, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf173 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf169, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_79, getitem_80, getitem_81, None, alias_16, getitem_83, getitem_84, getitem_85, 0.0, [True, True, True, False])
        del alias_16
        del buf169
        del getitem_79
        del getitem_80
        del getitem_81
        del getitem_83
        del getitem_84
        del getitem_85
        buf174 = buf173[0]
        buf175 = buf173[1]
        buf176 = buf173[2]
        del buf173
        buf177 = buf141; del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf174, buf175, buf176, buf177, 3631104, grid=grid(3631104), stream=stream0)
        del buf174
        del buf175
        buf178 = reinterpret_tensor(buf176, (1576, 768), (768, 1), 0); del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf177, (1576, 2304), (2304, 1), 0), permute_164, out=buf178)
        del permute_164
        buf179 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf177, (2304, 1576), (1, 2304), 0), view_71, out=buf179)
        del view_71
        buf180 = buf144; del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf177, buf180, 29952, 122, grid=grid(29952), stream=stream0)
        buf181 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf180, buf181, 2304, 13, grid=grid(2304), stream=stream0)
        buf188 = buf168; del buf168  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf188, buf178, primals_89, mul_49, div_10, 1576, 768, grid=grid(1576), stream=stream0)
        del div_10
        del primals_89
        buf184 = reinterpret_tensor(buf171, (768, 13), (1, 768), 0); del buf171  # reuse
        buf186 = buf164; del buf164  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf178, mul_49, buf184, buf186, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_49
        buf185 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf184, buf185, 768, 13, grid=grid(768), stream=stream0)
        buf187 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf186, buf187, 768, 13, grid=grid(768), stream=stream0)
        buf189 = reinterpret_tensor(buf157, (1576, 3072), (3072, 1), 0); del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf188, (1576, 768), (768, 1), 0), permute_168, out=buf189)
        del permute_168
        buf190 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf188, (768, 1576), (1, 768), 0), view_69, out=buf190)
        del view_69
        buf191 = reinterpret_tensor(buf186, (1, 768, 13), (9984, 1, 768), 0); del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf188, buf191, 9984, 122, grid=grid(9984), stream=stream0)
        buf192 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf191, buf192, 768, 13, grid=grid(768), stream=stream0)
        buf193 = reinterpret_tensor(buf189, (8, 197, 3072), (605184, 3072, 1), 0); del buf189  # reuse
        # Source Nodes: [x_87], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf193, addmm_26, 4841472, grid=grid(4841472), stream=stream0)
        del addmm_26
        buf194 = buf178; del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf193, (1576, 3072), (3072, 1), 0), permute_172, out=buf194)
        del permute_172
        buf195 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf193, (3072, 1576), (1, 3072), 0), view_67, out=buf195)
        del view_67
        buf196 = buf160; del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf193, buf196, 39936, 122, grid=grid(39936), stream=stream0)
        buf197 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf196, buf197, 3072, 13, grid=grid(3072), stream=stream0)
        buf204 = buf188; del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf204, buf194, primals_83, mul_44, div_11, 1576, 768, grid=grid(1576), stream=stream0)
        del div_11
        del primals_83
        buf200 = reinterpret_tensor(buf191, (768, 13), (1, 768), 0); del buf191  # reuse
        buf202 = buf184; del buf184  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf194, mul_44, buf200, buf202, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_44
        buf201 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf200, buf201, 768, 13, grid=grid(768), stream=stream0)
        buf203 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf202, buf203, 768, 13, grid=grid(768), stream=stream0)
        buf205 = buf194; del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf204, (1576, 768), (768, 1), 0), permute_176, out=buf205)
        del permute_176
        buf206 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf204, (768, 1576), (1, 768), 0), view_65, out=buf206)
        del view_65
        buf207 = reinterpret_tensor(buf202, (1, 768, 13), (9984, 1, 768), 0); del buf202  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf204, buf207, 9984, 122, grid=grid(9984), stream=stream0)
        buf208 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf207, buf208, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf209 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf205, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_68, getitem_69, getitem_70, None, alias_17, getitem_72, getitem_73, getitem_74, 0.0, [True, True, True, False])
        del alias_17
        del buf205
        del getitem_68
        del getitem_69
        del getitem_70
        del getitem_72
        del getitem_73
        del getitem_74
        buf210 = buf209[0]
        buf211 = buf209[1]
        buf212 = buf209[2]
        del buf209
        buf213 = buf177; del buf177  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf210, buf211, buf212, buf213, 3631104, grid=grid(3631104), stream=stream0)
        del buf210
        del buf211
        buf214 = reinterpret_tensor(buf212, (1576, 768), (768, 1), 0); del buf212  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf213, (1576, 2304), (2304, 1), 0), permute_182, out=buf214)
        del permute_182
        buf215 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf213, (2304, 1576), (1, 2304), 0), view_61, out=buf215)
        del view_61
        buf216 = buf180; del buf180  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf213, buf216, 29952, 122, grid=grid(29952), stream=stream0)
        buf217 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf216, buf217, 2304, 13, grid=grid(2304), stream=stream0)
        buf224 = buf204; del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf224, buf214, primals_77, mul_42, div_12, 1576, 768, grid=grid(1576), stream=stream0)
        del div_12
        del primals_77
        buf220 = reinterpret_tensor(buf207, (768, 13), (1, 768), 0); del buf207  # reuse
        buf222 = buf200; del buf200  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf214, mul_42, buf220, buf222, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_42
        buf221 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf220, buf221, 768, 13, grid=grid(768), stream=stream0)
        buf223 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf222, buf223, 768, 13, grid=grid(768), stream=stream0)
        buf225 = reinterpret_tensor(buf193, (1576, 3072), (3072, 1), 0); del buf193  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf224, (1576, 768), (768, 1), 0), permute_186, out=buf225)
        del permute_186
        buf226 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf224, (768, 1576), (1, 768), 0), view_59, out=buf226)
        del view_59
        buf227 = reinterpret_tensor(buf222, (1, 768, 13), (9984, 1, 768), 0); del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf224, buf227, 9984, 122, grid=grid(9984), stream=stream0)
        buf228 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf227, buf228, 768, 13, grid=grid(768), stream=stream0)
        buf229 = reinterpret_tensor(buf225, (8, 197, 3072), (605184, 3072, 1), 0); del buf225  # reuse
        # Source Nodes: [x_75], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf229, addmm_22, 4841472, grid=grid(4841472), stream=stream0)
        del addmm_22
        buf230 = buf214; del buf214  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf229, (1576, 3072), (3072, 1), 0), permute_190, out=buf230)
        del permute_190
        buf231 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf229, (3072, 1576), (1, 3072), 0), view_57, out=buf231)
        del view_57
        buf232 = buf196; del buf196  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf229, buf232, 39936, 122, grid=grid(39936), stream=stream0)
        buf233 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf232, buf233, 3072, 13, grid=grid(3072), stream=stream0)
        buf240 = buf224; del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf240, buf230, primals_71, mul_37, div_13, 1576, 768, grid=grid(1576), stream=stream0)
        del div_13
        del primals_71
        buf236 = reinterpret_tensor(buf227, (768, 13), (1, 768), 0); del buf227  # reuse
        buf238 = buf220; del buf220  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf230, mul_37, buf236, buf238, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_37
        buf237 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf236, buf237, 768, 13, grid=grid(768), stream=stream0)
        buf239 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf238, buf239, 768, 13, grid=grid(768), stream=stream0)
        buf241 = buf230; del buf230  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf240, (1576, 768), (768, 1), 0), permute_194, out=buf241)
        del permute_194
        buf242 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf240, (768, 1576), (1, 768), 0), view_55, out=buf242)
        del view_55
        buf243 = reinterpret_tensor(buf238, (1, 768, 13), (9984, 1, 768), 0); del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf240, buf243, 9984, 122, grid=grid(9984), stream=stream0)
        buf244 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf243, buf244, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf245 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf241, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_57, getitem_58, getitem_59, None, alias_18, getitem_61, getitem_62, getitem_63, 0.0, [True, True, True, False])
        del alias_18
        del buf241
        del getitem_57
        del getitem_58
        del getitem_59
        del getitem_61
        del getitem_62
        del getitem_63
        buf246 = buf245[0]
        buf247 = buf245[1]
        buf248 = buf245[2]
        del buf245
        buf249 = buf213; del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf246, buf247, buf248, buf249, 3631104, grid=grid(3631104), stream=stream0)
        del buf246
        del buf247
        buf250 = reinterpret_tensor(buf248, (1576, 768), (768, 1), 0); del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf249, (1576, 2304), (2304, 1), 0), permute_200, out=buf250)
        del permute_200
        buf251 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf249, (2304, 1576), (1, 2304), 0), view_51, out=buf251)
        del view_51
        buf252 = buf216; del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf249, buf252, 29952, 122, grid=grid(29952), stream=stream0)
        buf253 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf252, buf253, 2304, 13, grid=grid(2304), stream=stream0)
        buf260 = buf240; del buf240  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf260, buf250, primals_65, mul_35, div_14, 1576, 768, grid=grid(1576), stream=stream0)
        del div_14
        del primals_65
        buf256 = reinterpret_tensor(buf243, (768, 13), (1, 768), 0); del buf243  # reuse
        buf258 = buf236; del buf236  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf250, mul_35, buf256, buf258, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_35
        buf257 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf256, buf257, 768, 13, grid=grid(768), stream=stream0)
        buf259 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf258, buf259, 768, 13, grid=grid(768), stream=stream0)
        buf261 = reinterpret_tensor(buf229, (1576, 3072), (3072, 1), 0); del buf229  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf260, (1576, 768), (768, 1), 0), permute_204, out=buf261)
        del permute_204
        buf262 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf260, (768, 1576), (1, 768), 0), view_49, out=buf262)
        del view_49
        buf263 = reinterpret_tensor(buf258, (1, 768, 13), (9984, 1, 768), 0); del buf258  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf260, buf263, 9984, 122, grid=grid(9984), stream=stream0)
        buf264 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf263, buf264, 768, 13, grid=grid(768), stream=stream0)
        buf265 = reinterpret_tensor(buf261, (8, 197, 3072), (605184, 3072, 1), 0); del buf261  # reuse
        # Source Nodes: [x_63], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf265, addmm_18, 4841472, grid=grid(4841472), stream=stream0)
        del addmm_18
        buf266 = buf250; del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf265, (1576, 3072), (3072, 1), 0), permute_208, out=buf266)
        del permute_208
        buf267 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf265, (3072, 1576), (1, 3072), 0), view_47, out=buf267)
        del view_47
        buf268 = buf232; del buf232  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf265, buf268, 39936, 122, grid=grid(39936), stream=stream0)
        buf269 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf268, buf269, 3072, 13, grid=grid(3072), stream=stream0)
        buf276 = buf260; del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf276, buf266, primals_59, mul_30, div_15, 1576, 768, grid=grid(1576), stream=stream0)
        del div_15
        del primals_59
        buf272 = reinterpret_tensor(buf263, (768, 13), (1, 768), 0); del buf263  # reuse
        buf274 = buf256; del buf256  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf266, mul_30, buf272, buf274, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_30
        buf273 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf272, buf273, 768, 13, grid=grid(768), stream=stream0)
        buf275 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf274, buf275, 768, 13, grid=grid(768), stream=stream0)
        buf277 = buf266; del buf266  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf276, (1576, 768), (768, 1), 0), permute_212, out=buf277)
        del permute_212
        buf278 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf276, (768, 1576), (1, 768), 0), view_45, out=buf278)
        del view_45
        buf279 = reinterpret_tensor(buf274, (1, 768, 13), (9984, 1, 768), 0); del buf274  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf276, buf279, 9984, 122, grid=grid(9984), stream=stream0)
        buf280 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf279, buf280, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf281 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf277, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_46, getitem_47, getitem_48, None, alias_19, getitem_50, getitem_51, getitem_52, 0.0, [True, True, True, False])
        del alias_19
        del buf277
        del getitem_46
        del getitem_47
        del getitem_48
        del getitem_50
        del getitem_51
        del getitem_52
        buf282 = buf281[0]
        buf283 = buf281[1]
        buf284 = buf281[2]
        del buf281
        buf285 = buf249; del buf249  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf282, buf283, buf284, buf285, 3631104, grid=grid(3631104), stream=stream0)
        del buf282
        del buf283
        buf286 = reinterpret_tensor(buf284, (1576, 768), (768, 1), 0); del buf284  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf285, (1576, 2304), (2304, 1), 0), permute_218, out=buf286)
        del permute_218
        buf287 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf285, (2304, 1576), (1, 2304), 0), view_41, out=buf287)
        del view_41
        buf288 = buf252; del buf252  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf285, buf288, 29952, 122, grid=grid(29952), stream=stream0)
        buf289 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf288, buf289, 2304, 13, grid=grid(2304), stream=stream0)
        buf296 = buf276; del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf296, buf286, primals_53, mul_28, div_16, 1576, 768, grid=grid(1576), stream=stream0)
        del div_16
        del primals_53
        buf292 = reinterpret_tensor(buf279, (768, 13), (1, 768), 0); del buf279  # reuse
        buf294 = buf272; del buf272  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf286, mul_28, buf292, buf294, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_28
        buf293 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf292, buf293, 768, 13, grid=grid(768), stream=stream0)
        buf295 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf294, buf295, 768, 13, grid=grid(768), stream=stream0)
        buf297 = reinterpret_tensor(buf265, (1576, 3072), (3072, 1), 0); del buf265  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf296, (1576, 768), (768, 1), 0), permute_222, out=buf297)
        del permute_222
        buf298 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf296, (768, 1576), (1, 768), 0), view_39, out=buf298)
        del view_39
        buf299 = reinterpret_tensor(buf294, (1, 768, 13), (9984, 1, 768), 0); del buf294  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf296, buf299, 9984, 122, grid=grid(9984), stream=stream0)
        buf300 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf299, buf300, 768, 13, grid=grid(768), stream=stream0)
        buf301 = reinterpret_tensor(buf297, (8, 197, 3072), (605184, 3072, 1), 0); del buf297  # reuse
        # Source Nodes: [x_51], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf301, addmm_14, 4841472, grid=grid(4841472), stream=stream0)
        del addmm_14
        buf302 = buf286; del buf286  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf301, (1576, 3072), (3072, 1), 0), permute_226, out=buf302)
        del permute_226
        buf303 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf301, (3072, 1576), (1, 3072), 0), view_37, out=buf303)
        del view_37
        buf304 = buf268; del buf268  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf301, buf304, 39936, 122, grid=grid(39936), stream=stream0)
        buf305 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf304, buf305, 3072, 13, grid=grid(3072), stream=stream0)
        buf312 = buf296; del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf312, buf302, primals_47, mul_23, div_17, 1576, 768, grid=grid(1576), stream=stream0)
        del div_17
        del primals_47
        buf308 = reinterpret_tensor(buf299, (768, 13), (1, 768), 0); del buf299  # reuse
        buf310 = buf292; del buf292  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf302, mul_23, buf308, buf310, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_23
        buf309 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf308, buf309, 768, 13, grid=grid(768), stream=stream0)
        buf311 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf310, buf311, 768, 13, grid=grid(768), stream=stream0)
        buf313 = buf302; del buf302  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf312, (1576, 768), (768, 1), 0), permute_230, out=buf313)
        del permute_230
        buf314 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf312, (768, 1576), (1, 768), 0), view_35, out=buf314)
        del view_35
        buf315 = reinterpret_tensor(buf310, (1, 768, 13), (9984, 1, 768), 0); del buf310  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf312, buf315, 9984, 122, grid=grid(9984), stream=stream0)
        buf316 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf315, buf316, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf317 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf313, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_35, getitem_36, getitem_37, None, alias_20, getitem_39, getitem_40, getitem_41, 0.0, [True, True, True, False])
        del alias_20
        del buf313
        del getitem_35
        del getitem_36
        del getitem_37
        del getitem_39
        del getitem_40
        del getitem_41
        buf318 = buf317[0]
        buf319 = buf317[1]
        buf320 = buf317[2]
        del buf317
        buf321 = buf285; del buf285  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf318, buf319, buf320, buf321, 3631104, grid=grid(3631104), stream=stream0)
        del buf318
        del buf319
        buf322 = reinterpret_tensor(buf320, (1576, 768), (768, 1), 0); del buf320  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf321, (1576, 2304), (2304, 1), 0), permute_236, out=buf322)
        del permute_236
        buf323 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf321, (2304, 1576), (1, 2304), 0), view_31, out=buf323)
        del view_31
        buf324 = buf288; del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf321, buf324, 29952, 122, grid=grid(29952), stream=stream0)
        buf325 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf324, buf325, 2304, 13, grid=grid(2304), stream=stream0)
        buf332 = buf312; del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf332, buf322, primals_41, mul_21, div_18, 1576, 768, grid=grid(1576), stream=stream0)
        del div_18
        del primals_41
        buf328 = reinterpret_tensor(buf315, (768, 13), (1, 768), 0); del buf315  # reuse
        buf330 = buf308; del buf308  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf322, mul_21, buf328, buf330, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_21
        buf329 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf328, buf329, 768, 13, grid=grid(768), stream=stream0)
        buf331 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf330, buf331, 768, 13, grid=grid(768), stream=stream0)
        buf333 = reinterpret_tensor(buf301, (1576, 3072), (3072, 1), 0); del buf301  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf332, (1576, 768), (768, 1), 0), permute_240, out=buf333)
        del permute_240
        buf334 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf332, (768, 1576), (1, 768), 0), view_29, out=buf334)
        del view_29
        buf335 = reinterpret_tensor(buf330, (1, 768, 13), (9984, 1, 768), 0); del buf330  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf332, buf335, 9984, 122, grid=grid(9984), stream=stream0)
        buf336 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf335, buf336, 768, 13, grid=grid(768), stream=stream0)
        buf337 = reinterpret_tensor(buf333, (8, 197, 3072), (605184, 3072, 1), 0); del buf333  # reuse
        # Source Nodes: [x_39], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf337, addmm_10, 4841472, grid=grid(4841472), stream=stream0)
        del addmm_10
        buf338 = buf322; del buf322  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf337, (1576, 3072), (3072, 1), 0), permute_244, out=buf338)
        del permute_244
        buf339 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf337, (3072, 1576), (1, 3072), 0), view_27, out=buf339)
        del view_27
        buf340 = buf304; del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf337, buf340, 39936, 122, grid=grid(39936), stream=stream0)
        buf341 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf340, buf341, 3072, 13, grid=grid(3072), stream=stream0)
        buf348 = buf332; del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf348, buf338, primals_35, mul_16, div_19, 1576, 768, grid=grid(1576), stream=stream0)
        del div_19
        del primals_35
        buf344 = reinterpret_tensor(buf335, (768, 13), (1, 768), 0); del buf335  # reuse
        buf346 = buf328; del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf338, mul_16, buf344, buf346, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_16
        buf345 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf344, buf345, 768, 13, grid=grid(768), stream=stream0)
        buf347 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf346, buf347, 768, 13, grid=grid(768), stream=stream0)
        buf349 = buf338; del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf348, (1576, 768), (768, 1), 0), permute_248, out=buf349)
        del permute_248
        buf350 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf348, (768, 1576), (1, 768), 0), view_25, out=buf350)
        del view_25
        buf351 = reinterpret_tensor(buf346, (1, 768, 13), (9984, 1, 768), 0); del buf346  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf348, buf351, 9984, 122, grid=grid(9984), stream=stream0)
        buf352 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf351, buf352, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf353 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf349, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_24, getitem_25, getitem_26, None, alias_21, getitem_28, getitem_29, getitem_30, 0.0, [True, True, True, False])
        del alias_21
        del buf349
        del getitem_24
        del getitem_25
        del getitem_26
        del getitem_28
        del getitem_29
        del getitem_30
        buf354 = buf353[0]
        buf355 = buf353[1]
        buf356 = buf353[2]
        del buf353
        buf357 = buf321; del buf321  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf354, buf355, buf356, buf357, 3631104, grid=grid(3631104), stream=stream0)
        del buf354
        del buf355
        buf358 = reinterpret_tensor(buf356, (1576, 768), (768, 1), 0); del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf357, (1576, 2304), (2304, 1), 0), permute_254, out=buf358)
        del permute_254
        buf359 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf357, (2304, 1576), (1, 2304), 0), view_21, out=buf359)
        del view_21
        buf360 = buf324; del buf324  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf357, buf360, 29952, 122, grid=grid(29952), stream=stream0)
        buf361 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf360, buf361, 2304, 13, grid=grid(2304), stream=stream0)
        buf368 = buf348; del buf348  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf368, buf358, primals_29, mul_14, div_20, 1576, 768, grid=grid(1576), stream=stream0)
        del div_20
        del primals_29
        buf364 = reinterpret_tensor(buf351, (768, 13), (1, 768), 0); del buf351  # reuse
        buf366 = buf344; del buf344  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf358, mul_14, buf364, buf366, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_14
        buf365 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf364, buf365, 768, 13, grid=grid(768), stream=stream0)
        buf367 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf366, buf367, 768, 13, grid=grid(768), stream=stream0)
        buf369 = reinterpret_tensor(buf337, (1576, 3072), (3072, 1), 0); del buf337  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf368, (1576, 768), (768, 1), 0), permute_258, out=buf369)
        del permute_258
        buf370 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf368, (768, 1576), (1, 768), 0), view_19, out=buf370)
        del view_19
        buf371 = reinterpret_tensor(buf366, (1, 768, 13), (9984, 1, 768), 0); del buf366  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf368, buf371, 9984, 122, grid=grid(9984), stream=stream0)
        buf372 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf371, buf372, 768, 13, grid=grid(768), stream=stream0)
        buf373 = reinterpret_tensor(buf369, (8, 197, 3072), (605184, 3072, 1), 0); del buf369  # reuse
        # Source Nodes: [x_27], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf373, addmm_6, 4841472, grid=grid(4841472), stream=stream0)
        del addmm_6
        buf374 = buf358; del buf358  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf373, (1576, 3072), (3072, 1), 0), permute_262, out=buf374)
        del permute_262
        buf375 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf373, (3072, 1576), (1, 3072), 0), view_17, out=buf375)
        del view_17
        buf376 = buf340; del buf340  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf373, buf376, 39936, 122, grid=grid(39936), stream=stream0)
        buf377 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf376, buf377, 3072, 13, grid=grid(3072), stream=stream0)
        buf384 = buf368; del buf368  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf384, buf374, primals_23, mul_9, div_21, 1576, 768, grid=grid(1576), stream=stream0)
        del div_21
        del primals_23
        buf380 = reinterpret_tensor(buf371, (768, 13), (1, 768), 0); del buf371  # reuse
        buf382 = buf364; del buf364  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf374, mul_9, buf380, buf382, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_9
        buf381 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf380, buf381, 768, 13, grid=grid(768), stream=stream0)
        buf383 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf382, buf383, 768, 13, grid=grid(768), stream=stream0)
        buf385 = buf374; del buf374  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf384, (1576, 768), (768, 1), 0), permute_266, out=buf385)
        del permute_266
        buf386 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf384, (768, 1576), (1, 768), 0), view_15, out=buf386)
        del view_15
        buf387 = reinterpret_tensor(buf382, (1, 768, 13), (9984, 1, 768), 0); del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf384, buf387, 9984, 122, grid=grid(9984), stream=stream0)
        buf388 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf387, buf388, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf389 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf385, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_13, getitem_14, getitem_15, None, alias_22, getitem_17, getitem_18, getitem_19, 0.0, [True, True, True, False])
        del alias_22
        del buf385
        del getitem_13
        del getitem_14
        del getitem_15
        del getitem_17
        del getitem_18
        del getitem_19
        buf390 = buf389[0]
        buf391 = buf389[1]
        buf392 = buf389[2]
        del buf389
        buf393 = buf357; del buf357  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf390, buf391, buf392, buf393, 3631104, grid=grid(3631104), stream=stream0)
        del buf390
        del buf391
        buf394 = reinterpret_tensor(buf392, (1576, 768), (768, 1), 0); del buf392  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf393, (1576, 2304), (2304, 1), 0), permute_272, out=buf394)
        del permute_272
        buf395 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf393, (2304, 1576), (1, 2304), 0), view_11, out=buf395)
        del view_11
        buf396 = buf360; del buf360  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf393, buf396, 29952, 122, grid=grid(29952), stream=stream0)
        buf397 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf396, buf397, 2304, 13, grid=grid(2304), stream=stream0)
        buf404 = buf384; del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf404, buf394, primals_17, mul_7, div_22, 1576, 768, grid=grid(1576), stream=stream0)
        del div_22
        del primals_17
        buf400 = reinterpret_tensor(buf387, (768, 13), (1, 768), 0); del buf387  # reuse
        buf402 = buf380; del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf394, mul_7, buf400, buf402, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_7
        buf401 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf400, buf401, 768, 13, grid=grid(768), stream=stream0)
        buf403 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf402, buf403, 768, 13, grid=grid(768), stream=stream0)
        buf405 = reinterpret_tensor(buf373, (1576, 3072), (3072, 1), 0); del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf404, (1576, 768), (768, 1), 0), permute_276, out=buf405)
        del permute_276
        buf406 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf404, (768, 1576), (1, 768), 0), view_9, out=buf406)
        del view_9
        buf407 = reinterpret_tensor(buf402, (1, 768, 13), (9984, 1, 768), 0); del buf402  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf404, buf407, 9984, 122, grid=grid(9984), stream=stream0)
        buf408 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf407, buf408, 768, 13, grid=grid(768), stream=stream0)
        buf409 = reinterpret_tensor(buf405, (8, 197, 3072), (605184, 3072, 1), 0); del buf405  # reuse
        # Source Nodes: [x_15], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf409, addmm_2, 4841472, grid=grid(4841472), stream=stream0)
        del addmm_2
        buf410 = buf394; del buf394  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf409, (1576, 3072), (3072, 1), 0), permute_280, out=buf410)
        del permute_280
        buf411 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf409, (3072, 1576), (1, 3072), 0), view_7, out=buf411)
        del view_7
        buf412 = buf376; del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf409, buf412, 39936, 122, grid=grid(39936), stream=stream0)
        del buf409
        buf413 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf412, buf413, 3072, 13, grid=grid(3072), stream=stream0)
        del buf412
        buf420 = buf404; del buf404  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf420, buf410, primals_11, mul_2, div_23, 1576, 768, grid=grid(1576), stream=stream0)
        del div_23
        del primals_11
        buf416 = reinterpret_tensor(buf407, (768, 13), (1, 768), 0); del buf407  # reuse
        buf418 = buf400; del buf400  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf410, mul_2, buf416, buf418, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_2
        buf417 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf416, buf417, 768, 13, grid=grid(768), stream=stream0)
        buf419 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf418, buf419, 768, 13, grid=grid(768), stream=stream0)
        buf421 = buf410; del buf410  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf420, (1576, 768), (768, 1), 0), permute_284, out=buf421)
        del permute_284
        buf422 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf420, (768, 1576), (1, 768), 0), view_5, out=buf422)
        del view_5
        buf423 = reinterpret_tensor(buf418, (1, 768, 13), (9984, 1, 768), 0); del buf418  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf420, buf423, 9984, 122, grid=grid(9984), stream=stream0)
        buf424 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf423, buf424, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf425 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf421, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_2, getitem_3, getitem_4, None, alias_23, getitem_6, getitem_7, getitem_8, 0.0, [True, True, True, False])
        del alias_23
        del buf421
        del getitem_2
        del getitem_3
        del getitem_4
        del getitem_6
        del getitem_7
        del getitem_8
        buf426 = buf425[0]
        buf427 = buf425[1]
        buf428 = buf425[2]
        del buf425
        buf429 = buf393; del buf393  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf426, buf427, buf428, buf429, 3631104, grid=grid(3631104), stream=stream0)
        del buf426
        del buf427
        buf430 = reinterpret_tensor(buf428, (1576, 768), (768, 1), 0); del buf428  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf429, (1576, 2304), (2304, 1), 0), permute_290, out=buf430)
        del permute_290
        buf431 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf429, (2304, 1576), (1, 2304), 0), view_1, out=buf431)
        del view_1
        buf432 = buf396; del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf429, buf432, 29952, 122, grid=grid(29952), stream=stream0)
        del buf429
        buf433 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf432, buf433, 2304, 13, grid=grid(2304), stream=stream0)
        del buf432
        buf440 = buf420; del buf420  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf440, buf430, primals_5, mul, div_24, 1576, 768, grid=grid(1576), stream=stream0)
        del div_24
        del primals_5
        buf436 = reinterpret_tensor(buf423, (768, 13), (1, 768), 0); del buf423  # reuse
        buf438 = buf416; del buf416  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf430, mul, buf436, buf438, 9984, 122, grid=grid(9984), stream=stream0)
        del buf430
        del mul
        buf437 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf436, buf437, 768, 13, grid=grid(768), stream=stream0)
        del buf436
        buf439 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf438, buf439, 768, 13, grid=grid(768), stream=stream0)
        buf441 = empty((1, 197, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf440, buf441, 151296, 8, grid=grid(151296), stream=stream0)
        buf442 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_15.run(buf440, buf442, 768, 8, grid=grid(768), stream=stream0)
        buf443 = buf438; del buf438  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_16.run(buf440, buf443, 9984, 121, grid=grid(9984), stream=stream0)
        buf444 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf443, buf444, 768, 13, grid=grid(768), stream=stream0)
        del buf443
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf445 = aten.convolution_backward(reinterpret_tensor(buf440, (8, 768, 14, 14), (151296, 1, 10752, 768), 768), primals_153, primals_3, [768], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf440
        del primals_153
        del primals_3
        buf446 = buf445[1]
        return (buf441, buf442, buf446, buf444, buf437, buf439, reinterpret_tensor(buf431, (2304, 768), (768, 1), 0), reinterpret_tensor(buf433, (2304, ), (1, ), 0), reinterpret_tensor(buf422, (768, 768), (768, 1), 0), reinterpret_tensor(buf424, (768, ), (1, ), 0), buf417, buf419, reinterpret_tensor(buf411, (3072, 768), (768, 1), 0), reinterpret_tensor(buf413, (3072, ), (1, ), 0), reinterpret_tensor(buf406, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf408, (768, ), (1, ), 0), buf401, buf403, reinterpret_tensor(buf395, (2304, 768), (768, 1), 0), reinterpret_tensor(buf397, (2304, ), (1, ), 0), reinterpret_tensor(buf386, (768, 768), (768, 1), 0), reinterpret_tensor(buf388, (768, ), (1, ), 0), buf381, buf383, reinterpret_tensor(buf375, (3072, 768), (768, 1), 0), reinterpret_tensor(buf377, (3072, ), (1, ), 0), reinterpret_tensor(buf370, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf372, (768, ), (1, ), 0), buf365, buf367, reinterpret_tensor(buf359, (2304, 768), (768, 1), 0), reinterpret_tensor(buf361, (2304, ), (1, ), 0), reinterpret_tensor(buf350, (768, 768), (768, 1), 0), reinterpret_tensor(buf352, (768, ), (1, ), 0), buf345, buf347, reinterpret_tensor(buf339, (3072, 768), (768, 1), 0), reinterpret_tensor(buf341, (3072, ), (1, ), 0), reinterpret_tensor(buf334, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf336, (768, ), (1, ), 0), buf329, buf331, reinterpret_tensor(buf323, (2304, 768), (768, 1), 0), reinterpret_tensor(buf325, (2304, ), (1, ), 0), reinterpret_tensor(buf314, (768, 768), (768, 1), 0), reinterpret_tensor(buf316, (768, ), (1, ), 0), buf309, buf311, reinterpret_tensor(buf303, (3072, 768), (768, 1), 0), reinterpret_tensor(buf305, (3072, ), (1, ), 0), reinterpret_tensor(buf298, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf300, (768, ), (1, ), 0), buf293, buf295, reinterpret_tensor(buf287, (2304, 768), (768, 1), 0), reinterpret_tensor(buf289, (2304, ), (1, ), 0), reinterpret_tensor(buf278, (768, 768), (768, 1), 0), reinterpret_tensor(buf280, (768, ), (1, ), 0), buf273, buf275, reinterpret_tensor(buf267, (3072, 768), (768, 1), 0), reinterpret_tensor(buf269, (3072, ), (1, ), 0), reinterpret_tensor(buf262, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf264, (768, ), (1, ), 0), buf257, buf259, reinterpret_tensor(buf251, (2304, 768), (768, 1), 0), reinterpret_tensor(buf253, (2304, ), (1, ), 0), reinterpret_tensor(buf242, (768, 768), (768, 1), 0), reinterpret_tensor(buf244, (768, ), (1, ), 0), buf237, buf239, reinterpret_tensor(buf231, (3072, 768), (768, 1), 0), reinterpret_tensor(buf233, (3072, ), (1, ), 0), reinterpret_tensor(buf226, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf228, (768, ), (1, ), 0), buf221, buf223, reinterpret_tensor(buf215, (2304, 768), (768, 1), 0), reinterpret_tensor(buf217, (2304, ), (1, ), 0), reinterpret_tensor(buf206, (768, 768), (768, 1), 0), reinterpret_tensor(buf208, (768, ), (1, ), 0), buf201, buf203, reinterpret_tensor(buf195, (3072, 768), (768, 1), 0), reinterpret_tensor(buf197, (3072, ), (1, ), 0), reinterpret_tensor(buf190, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf192, (768, ), (1, ), 0), buf185, buf187, reinterpret_tensor(buf179, (2304, 768), (768, 1), 0), reinterpret_tensor(buf181, (2304, ), (1, ), 0), reinterpret_tensor(buf170, (768, 768), (768, 1), 0), reinterpret_tensor(buf172, (768, ), (1, ), 0), buf165, buf167, reinterpret_tensor(buf159, (3072, 768), (768, 1), 0), reinterpret_tensor(buf161, (3072, ), (1, ), 0), reinterpret_tensor(buf154, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf156, (768, ), (1, ), 0), buf149, buf151, reinterpret_tensor(buf143, (2304, 768), (768, 1), 0), reinterpret_tensor(buf145, (2304, ), (1, ), 0), reinterpret_tensor(buf134, (768, 768), (768, 1), 0), reinterpret_tensor(buf136, (768, ), (1, ), 0), buf129, buf131, reinterpret_tensor(buf123, (3072, 768), (768, 1), 0), reinterpret_tensor(buf125, (3072, ), (1, ), 0), reinterpret_tensor(buf118, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf120, (768, ), (1, ), 0), buf113, buf115, reinterpret_tensor(buf107, (2304, 768), (768, 1), 0), reinterpret_tensor(buf109, (2304, ), (1, ), 0), reinterpret_tensor(buf98, (768, 768), (768, 1), 0), reinterpret_tensor(buf100, (768, ), (1, ), 0), buf93, buf95, reinterpret_tensor(buf87, (3072, 768), (768, 1), 0), reinterpret_tensor(buf89, (3072, ), (1, ), 0), reinterpret_tensor(buf82, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf84, (768, ), (1, ), 0), buf77, buf79, reinterpret_tensor(buf71, (2304, 768), (768, 1), 0), reinterpret_tensor(buf73, (2304, ), (1, ), 0), reinterpret_tensor(buf62, (768, 768), (768, 1), 0), reinterpret_tensor(buf64, (768, ), (1, ), 0), buf57, buf59, reinterpret_tensor(buf51, (3072, 768), (768, 1), 0), reinterpret_tensor(buf53, (3072, ), (1, ), 0), reinterpret_tensor(buf46, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf48, (768, ), (1, ), 0), buf41, buf43, reinterpret_tensor(buf35, (2304, 768), (768, 1), 0), reinterpret_tensor(buf37, (2304, ), (1, ), 0), reinterpret_tensor(buf26, (768, 768), (768, 1), 0), reinterpret_tensor(buf28, (768, ), (1, ), 0), buf21, buf23, reinterpret_tensor(buf15, (3072, 768), (768, 1), 0), reinterpret_tensor(buf17, (3072, ), (1, ), 0), reinterpret_tensor(buf10, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf12, (768, ), (1, ), 0), buf7, buf8, reinterpret_tensor(buf1, (1000, 768), (768, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_3 = rand_strided((768, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    mul = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_1 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_2 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_4 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_6 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_8 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_5 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_2 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_7 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_2 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_9 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_7 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_11 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_13 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_14 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_15 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_17 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_18 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_19 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_15 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_9 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_17 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_6 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_19 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_14 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_21 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_24 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_25 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_26 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_28 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_29 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_30 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_25 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_16 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_27 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_29 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_21 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_31 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_35 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_36 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_37 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_39 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_40 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_41 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_35 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_23 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_37 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_14 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_39 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_28 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_41 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_46 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_47 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_48 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_50 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_51 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_52 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_45 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_30 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_47 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_18 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_49 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_35 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_51 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_57 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_58 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_59 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_61 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_62 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_63 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_55 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_37 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_57 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_59 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_42 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_61 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_68 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_69 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_70 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_72 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_73 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_74 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_65 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_44 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_67 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_26 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_69 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_49 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_71 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_79 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_80 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_81 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_83 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_84 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_85 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_75 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_51 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_77 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_30 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_79 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_56 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_81 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_90 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_91 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_92 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_94 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_95 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_96 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_85 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_58 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_87 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_89 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_63 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_91 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_101 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_102 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_103 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_105 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_106 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_107 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_95 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_65 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_97 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_38 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_99 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_70 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_101 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_112 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_113 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_114 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_116 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_117 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_118 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_105 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_72 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_107 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_42 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_109 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_77 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_111 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_123 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_124 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_125 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_127 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_128 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_129 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_115 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_79 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_117 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_46 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_119 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_84 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    clone_37 = rand_strided((8, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_74 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_78 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_82 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_1 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_86 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_12 = rand_strided((8, 12, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_92 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_2 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_96 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_100 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_104 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_13 = rand_strided((8, 12, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_110 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_4 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_114 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_118 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_5 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_122 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_14 = rand_strided((8, 12, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_128 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_6 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_132 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_136 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_7 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_140 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_15 = rand_strided((8, 12, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_146 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_8 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_150 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_154 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_9 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_158 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_16 = rand_strided((8, 12, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_164 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_10 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_168 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_172 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_11 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_176 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_17 = rand_strided((8, 12, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_182 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_12 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_186 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_190 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_13 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_194 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_18 = rand_strided((8, 12, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_200 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_14 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_204 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_208 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_15 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_212 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_19 = rand_strided((8, 12, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_218 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_16 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_222 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_226 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_17 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_230 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_20 = rand_strided((8, 12, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_236 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_18 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_240 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_244 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_19 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_248 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_21 = rand_strided((8, 12, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_254 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_20 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_258 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_262 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_21 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_266 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_22 = rand_strided((8, 12, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_272 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_22 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_276 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_280 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_23 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_284 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_23 = rand_strided((8, 12, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_290 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_24 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_3, primals_5, primals_11, primals_17, primals_23, primals_29, primals_35, primals_41, primals_47, primals_53, primals_59, primals_65, primals_71, primals_77, primals_83, primals_89, primals_95, primals_101, primals_107, primals_113, primals_119, primals_125, primals_131, primals_137, primals_143, primals_149, primals_153, mul, view_1, getitem_2, getitem_3, getitem_4, getitem_6, getitem_7, getitem_8, view_5, mul_2, view_7, addmm_2, view_9, mul_7, view_11, getitem_13, getitem_14, getitem_15, getitem_17, getitem_18, getitem_19, view_15, mul_9, view_17, addmm_6, view_19, mul_14, view_21, getitem_24, getitem_25, getitem_26, getitem_28, getitem_29, getitem_30, view_25, mul_16, view_27, addmm_10, view_29, mul_21, view_31, getitem_35, getitem_36, getitem_37, getitem_39, getitem_40, getitem_41, view_35, mul_23, view_37, addmm_14, view_39, mul_28, view_41, getitem_46, getitem_47, getitem_48, getitem_50, getitem_51, getitem_52, view_45, mul_30, view_47, addmm_18, view_49, mul_35, view_51, getitem_57, getitem_58, getitem_59, getitem_61, getitem_62, getitem_63, view_55, mul_37, view_57, addmm_22, view_59, mul_42, view_61, getitem_68, getitem_69, getitem_70, getitem_72, getitem_73, getitem_74, view_65, mul_44, view_67, addmm_26, view_69, mul_49, view_71, getitem_79, getitem_80, getitem_81, getitem_83, getitem_84, getitem_85, view_75, mul_51, view_77, addmm_30, view_79, mul_56, view_81, getitem_90, getitem_91, getitem_92, getitem_94, getitem_95, getitem_96, view_85, mul_58, view_87, addmm_34, view_89, mul_63, view_91, getitem_101, getitem_102, getitem_103, getitem_105, getitem_106, getitem_107, view_95, mul_65, view_97, addmm_38, view_99, mul_70, view_101, getitem_112, getitem_113, getitem_114, getitem_116, getitem_117, getitem_118, view_105, mul_72, view_107, addmm_42, view_109, mul_77, view_111, getitem_123, getitem_124, getitem_125, getitem_127, getitem_128, getitem_129, view_115, mul_79, view_117, addmm_46, view_119, mul_84, clone_37, permute_74, div, permute_78, permute_82, div_1, permute_86, alias_12, permute_92, div_2, permute_96, permute_100, div_3, permute_104, alias_13, permute_110, div_4, permute_114, permute_118, div_5, permute_122, alias_14, permute_128, div_6, permute_132, permute_136, div_7, permute_140, alias_15, permute_146, div_8, permute_150, permute_154, div_9, permute_158, alias_16, permute_164, div_10, permute_168, permute_172, div_11, permute_176, alias_17, permute_182, div_12, permute_186, permute_190, div_13, permute_194, alias_18, permute_200, div_14, permute_204, permute_208, div_15, permute_212, alias_19, permute_218, div_16, permute_222, permute_226, div_17, permute_230, alias_20, permute_236, div_18, permute_240, permute_244, div_19, permute_248, alias_21, permute_254, div_20, permute_258, permute_262, div_21, permute_266, alias_22, permute_272, div_22, permute_276, permute_280, div_23, permute_284, alias_23, permute_290, div_24, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('vit_base_patch16_224', benchmark_compiled_module)
