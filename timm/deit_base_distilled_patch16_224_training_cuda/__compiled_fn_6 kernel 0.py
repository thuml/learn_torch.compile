
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


# kernel path: /tmp/torchinductor_youkaichao/3n/c3ntxglseeqcqxfguqqwuhehgwi7qhqk2n3zdiguoi63i2gbbcpr.py
# Source Nodes: [], Original ATen: [aten.div]

triton_poi_fused_div_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 2.0
    tmp2 = tmp0 / tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
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


# kernel path: /tmp/torchinductor_youkaichao/3c/c3c7qgok7qi7uzzxhrenaummsvhmf6n4tf7nzz5vpzxbe7twgv3f.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.select_backward]

triton_red_fused_add_native_layer_norm_backward_select_backward_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_select_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1584
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 198
    x1 = (xindex // 198)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp3 = tl.load(in_ptr0 + (r2 + (768*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr1 + (r2 + (768*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr3 + (r2 + (768*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 1, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp4 = 0.0
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.full([1, 1], 0, tl.int32)
        tmp7 = tmp0 == tmp6
        tmp9 = tl.where(tmp7, tmp8, tmp4)
        tmp10 = tmp5 + tmp9
        tmp12 = tmp10 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
        tmp17 = tmp12 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tmp21 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp25 = tl.load(in_ptr0 + (r2 + (768*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp30 = tl.load(in_ptr1 + (r2 + (768*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp33 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp38 = tl.load(in_ptr3 + (r2 + (768*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp22 = x0
        tmp23 = tl.full([1, 1], 1, tl.int32)
        tmp24 = tmp22 == tmp23
        tmp26 = 0.0
        tmp27 = tl.where(tmp24, tmp25, tmp26)
        tmp28 = tl.full([1, 1], 0, tl.int32)
        tmp29 = tmp22 == tmp28
        tmp31 = tl.where(tmp29, tmp30, tmp26)
        tmp32 = tmp27 + tmp31
        tmp34 = tmp32 * tmp33
        tmp35 = 768.0
        tmp36 = tmp34 * tmp35
        tmp37 = tmp36 - tmp14
        tmp39 = tmp38 * tmp19
        tmp40 = tmp37 - tmp39
        tmp41 = tmp21 * tmp40
        tl.store(out_ptr2 + (r2 + (768*x3)), tmp41, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o5/co5lcl23xfzodgryrmsh4dexblrs676lj3fyd5u4gxjrxoa55nfn.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.select_backward]

triton_red_fused_add_native_layer_norm_backward_select_backward_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_select_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9984
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 768)
    x0 = xindex % 768
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 1584, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = (r2 + (122*x1)) % 198
        tmp4 = tl.full([1, 1], 1, tl.int32)
        tmp5 = tmp3 == tmp4
        tmp6 = tl.load(in_ptr0 + (x0 + (768*(((r2 + (122*x1)) // 198) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = 0.0
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = tl.full([1, 1], 0, tl.int32)
        tmp10 = tmp3 == tmp9
        tmp11 = tl.load(in_ptr1 + (x0 + (768*(((r2 + (122*x1)) // 198) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.where(tmp10, tmp11, tmp7)
        tmp13 = tmp8 + tmp12
        tmp14 = tl.load(in_ptr2 + (x0 + (768*((r2 + (122*x1)) % 1584))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tmp13 * tmp14
        tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
        tmp17 = tl.where(tmp2, tmp15, tmp16)
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xk/cxkkk4lu75wtqayevhkf264xnzodjetbqrad25z3jcminx3lwsuu.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.select_backward]

triton_per_fused_add_native_layer_norm_backward_select_backward_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_select_backward_4', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/o6/co6jol56xrmw24xwtifpcuj5dwek272ctwudcs6uiazlqtx7tr62.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.select_backward]

triton_red_fused_add_native_layer_norm_backward_select_backward_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_select_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 1584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 198
        r2 = (rindex // 198)
        tmp3 = tl.load(in_ptr0 + (x0 + (768*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr1 + (x0 + (768*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = r1
        tmp1 = tl.full([1, 1], 1, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp4 = 0.0
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.full([1, 1], 0, tl.int32)
        tmp7 = tmp0 == tmp6
        tmp9 = tl.where(tmp7, tmp8, tmp4)
        tmp10 = tmp5 + tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ze/czeqtdplunisvfjq3aoa57lb2s3dyp2nxxqauu2kxn2cqij6t7nu.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_6', 'mutated_arg_names': []}
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
        tmp1 = tl.full([1, 1], 1584, tl.int32)
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


# kernel path: /tmp/torchinductor_youkaichao/3h/c3h4ppetclza3tc2xdj2t53abcwbgtq7muukaudoo77y7i2tv4nl.py
# Source Nodes: [x_147], Original ATen: [aten.gelu, aten.gelu_backward]
# x_147 => add_83, erf_11, mul_82
triton_poi_fused_gelu_gelu_backward_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4866048
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


# kernel path: /tmp/torchinductor_youkaichao/c5/cc5kbbbmuqr4s4bsjjpazrrwf6iymyiwqqnyqb6wsqdqzznvxjhj.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_8', 'mutated_arg_names': []}
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
        tmp1 = tl.full([1, 1], 1584, tl.int32)
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


# kernel path: /tmp/torchinductor_youkaichao/c3/cc3vakdtz566hv3l33fag5lbktfdzvvt3ioxfbqi37ho6q4h4h4x.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_9', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/mq/cmqn7z5a5z7dggiixcfsyrzrvprsacmxr6lrq7htwijj5doyk7ih.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_10', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 1584
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


# kernel path: /tmp/torchinductor_youkaichao/4f/c4ftwx2oukeqjxzi4f7ypgmcsltcszztn6togcjzea4akwuwd6zh.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_11', 'mutated_arg_names': []}
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
        tmp1 = tl.full([1, 1], 1584, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (768*((r2 + (122*x1)) % 1584))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (768*((r2 + (122*x1)) % 1584))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/oq/coqvyv4vusz45ks5vi2g5varnqa6jbwtrrtdnb7rc336kmuo52cw.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3649536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x5 = (xindex // 152064)
    x6 = xindex
    x0 = xindex % 768
    x3 = (xindex // 1216512)
    x7 = (xindex // 768) % 1584
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
    tmp12 = tl.load(in_ptr1 + ((-1216512) + x6), tmp11, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 24, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-2433024) + x6), tmp15, other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tl.where(tmp11, tmp14, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tl.store(out_ptr0 + (x0 + (768*x3) + (2304*x7)), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/oq/coqr7gtaaopbygzpascorxrfqtnbrqs43qhfy3mfq6xlsmyv72uq.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_13', 'mutated_arg_names': []}
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
        tmp1 = tl.full([1, 1], 1584, tl.int32)
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


# kernel path: /tmp/torchinductor_youkaichao/cn/ccnz462rbuntcyexgrfrlmnzwzhajsqbnul5vej64egpbnioscgw.py
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
    size_hints=[4096, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_14', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/qi/cqiu2xykjaweib4gbj4wkn7d3gwkzhyfjkoa6wovpbaoirfzhqju.py
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
    size_hints=[262144, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 152064
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (152064*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sj/csjyxxn6qoqeshid5x4gfqovmvobhzx4obac3z6h3pte2abhcmwj.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_16', 'mutated_arg_names': []}
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
    tmp0 = tl.load(in_ptr0 + (768 + x0 + (152064*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/te/cte25tlst76owzx63q5b2imewjrady4ascbi7jzzob4pks6xry3r.py
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
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_17', 'mutated_arg_names': []}
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
    tmp0 = tl.load(in_ptr0 + (x0 + (152064*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wn/cwnb6522dr3jeva26lvmh5sctsv76zpeoro3hmsjzndz3jxxs3pq.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_18', 'mutated_arg_names': []}
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
        tmp3 = tl.load(in_ptr0 + (1536 + x0 + (768*((r2 + (121*x1)) % 196)) + (152064*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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
    primals_4, primals_6, primals_12, primals_18, primals_24, primals_30, primals_36, primals_42, primals_48, primals_54, primals_60, primals_66, primals_72, primals_78, primals_84, primals_90, primals_96, primals_102, primals_108, primals_114, primals_120, primals_126, primals_132, primals_138, primals_144, primals_150, primals_156, mul, view_1, getitem_2, getitem_3, getitem_4, getitem_6, getitem_7, getitem_8, view_5, mul_2, view_7, addmm_2, view_9, mul_7, view_11, getitem_13, getitem_14, getitem_15, getitem_17, getitem_18, getitem_19, view_15, mul_9, view_17, addmm_6, view_19, mul_14, view_21, getitem_24, getitem_25, getitem_26, getitem_28, getitem_29, getitem_30, view_25, mul_16, view_27, addmm_10, view_29, mul_21, view_31, getitem_35, getitem_36, getitem_37, getitem_39, getitem_40, getitem_41, view_35, mul_23, view_37, addmm_14, view_39, mul_28, view_41, getitem_46, getitem_47, getitem_48, getitem_50, getitem_51, getitem_52, view_45, mul_30, view_47, addmm_18, view_49, mul_35, view_51, getitem_57, getitem_58, getitem_59, getitem_61, getitem_62, getitem_63, view_55, mul_37, view_57, addmm_22, view_59, mul_42, view_61, getitem_68, getitem_69, getitem_70, getitem_72, getitem_73, getitem_74, view_65, mul_44, view_67, addmm_26, view_69, mul_49, view_71, getitem_79, getitem_80, getitem_81, getitem_83, getitem_84, getitem_85, view_75, mul_51, view_77, addmm_30, view_79, mul_56, view_81, getitem_90, getitem_91, getitem_92, getitem_94, getitem_95, getitem_96, view_85, mul_58, view_87, addmm_34, view_89, mul_63, view_91, getitem_101, getitem_102, getitem_103, getitem_105, getitem_106, getitem_107, view_95, mul_65, view_97, addmm_38, view_99, mul_70, view_101, getitem_112, getitem_113, getitem_114, getitem_116, getitem_117, getitem_118, view_105, mul_72, view_107, addmm_42, view_109, mul_77, view_111, getitem_123, getitem_124, getitem_125, getitem_127, getitem_128, getitem_129, view_115, mul_79, view_117, addmm_46, view_119, mul_84, select, select_1, permute_75, permute_79, div_2, permute_83, permute_87, div_3, permute_91, alias_12, permute_97, div_4, permute_101, permute_105, div_5, permute_109, alias_13, permute_115, div_6, permute_119, permute_123, div_7, permute_127, alias_14, permute_133, div_8, permute_137, permute_141, div_9, permute_145, alias_15, permute_151, div_10, permute_155, permute_159, div_11, permute_163, alias_16, permute_169, div_12, permute_173, permute_177, div_13, permute_181, alias_17, permute_187, div_14, permute_191, permute_195, div_15, permute_199, alias_18, permute_205, div_16, permute_209, permute_213, div_17, permute_217, alias_19, permute_223, div_18, permute_227, permute_231, div_19, permute_235, alias_20, permute_241, div_20, permute_245, permute_249, div_21, permute_253, alias_21, permute_259, div_22, permute_263, permute_267, div_23, permute_271, alias_22, permute_277, div_24, permute_281, permute_285, div_25, permute_289, alias_23, permute_295, div_26, tangents_1 = args
    args.clear()
    assert_size_stride(primals_4, (768, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(primals_6, (768, ), (1, ))
    assert_size_stride(primals_12, (768, ), (1, ))
    assert_size_stride(primals_18, (768, ), (1, ))
    assert_size_stride(primals_24, (768, ), (1, ))
    assert_size_stride(primals_30, (768, ), (1, ))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_42, (768, ), (1, ))
    assert_size_stride(primals_48, (768, ), (1, ))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_60, (768, ), (1, ))
    assert_size_stride(primals_66, (768, ), (1, ))
    assert_size_stride(primals_72, (768, ), (1, ))
    assert_size_stride(primals_78, (768, ), (1, ))
    assert_size_stride(primals_84, (768, ), (1, ))
    assert_size_stride(primals_90, (768, ), (1, ))
    assert_size_stride(primals_96, (768, ), (1, ))
    assert_size_stride(primals_102, (768, ), (1, ))
    assert_size_stride(primals_108, (768, ), (1, ))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_120, (768, ), (1, ))
    assert_size_stride(primals_126, (768, ), (1, ))
    assert_size_stride(primals_132, (768, ), (1, ))
    assert_size_stride(primals_138, (768, ), (1, ))
    assert_size_stride(primals_144, (768, ), (1, ))
    assert_size_stride(primals_150, (768, ), (1, ))
    assert_size_stride(primals_156, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(mul, (8, 198, 768), (152064, 768, 1))
    assert_size_stride(view_1, (1584, 768), (768, 1))
    assert_size_stride(getitem_2, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_3, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_4, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_6, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_7, (), ())
    assert_size_stride(getitem_8, (), ())
    assert_size_stride(view_5, (1584, 768), (768, 1))
    assert_size_stride(mul_2, (8, 198, 768), (152064, 768, 1))
    assert_size_stride(view_7, (1584, 768), (768, 1))
    assert_size_stride(addmm_2, (1584, 3072), (3072, 1))
    assert_size_stride(view_9, (1584, 3072), (3072, 1))
    assert_size_stride(mul_7, (8, 198, 768), (152064, 768, 1))
    assert_size_stride(view_11, (1584, 768), (768, 1))
    assert_size_stride(getitem_13, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_14, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_15, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_17, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_18, (), ())
    assert_size_stride(getitem_19, (), ())
    assert_size_stride(view_15, (1584, 768), (768, 1))
    assert_size_stride(mul_9, (8, 198, 768), (152064, 768, 1))
    assert_size_stride(view_17, (1584, 768), (768, 1))
    assert_size_stride(addmm_6, (1584, 3072), (3072, 1))
    assert_size_stride(view_19, (1584, 3072), (3072, 1))
    assert_size_stride(mul_14, (8, 198, 768), (152064, 768, 1))
    assert_size_stride(view_21, (1584, 768), (768, 1))
    assert_size_stride(getitem_24, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_25, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_26, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_28, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_29, (), ())
    assert_size_stride(getitem_30, (), ())
    assert_size_stride(view_25, (1584, 768), (768, 1))
    assert_size_stride(mul_16, (8, 198, 768), (152064, 768, 1))
    assert_size_stride(view_27, (1584, 768), (768, 1))
    assert_size_stride(addmm_10, (1584, 3072), (3072, 1))
    assert_size_stride(view_29, (1584, 3072), (3072, 1))
    assert_size_stride(mul_21, (8, 198, 768), (152064, 768, 1))
    assert_size_stride(view_31, (1584, 768), (768, 1))
    assert_size_stride(getitem_35, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_36, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_37, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_39, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_40, (), ())
    assert_size_stride(getitem_41, (), ())
    assert_size_stride(view_35, (1584, 768), (768, 1))
    assert_size_stride(mul_23, (8, 198, 768), (152064, 768, 1))
    assert_size_stride(view_37, (1584, 768), (768, 1))
    assert_size_stride(addmm_14, (1584, 3072), (3072, 1))
    assert_size_stride(view_39, (1584, 3072), (3072, 1))
    assert_size_stride(mul_28, (8, 198, 768), (152064, 768, 1))
    assert_size_stride(view_41, (1584, 768), (768, 1))
    assert_size_stride(getitem_46, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_47, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_48, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_50, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_51, (), ())
    assert_size_stride(getitem_52, (), ())
    assert_size_stride(view_45, (1584, 768), (768, 1))
    assert_size_stride(mul_30, (8, 198, 768), (152064, 768, 1))
    assert_size_stride(view_47, (1584, 768), (768, 1))
    assert_size_stride(addmm_18, (1584, 3072), (3072, 1))
    assert_size_stride(view_49, (1584, 3072), (3072, 1))
    assert_size_stride(mul_35, (8, 198, 768), (152064, 768, 1))
    assert_size_stride(view_51, (1584, 768), (768, 1))
    assert_size_stride(getitem_57, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_58, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_59, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_61, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_62, (), ())
    assert_size_stride(getitem_63, (), ())
    assert_size_stride(view_55, (1584, 768), (768, 1))
    assert_size_stride(mul_37, (8, 198, 768), (152064, 768, 1))
    assert_size_stride(view_57, (1584, 768), (768, 1))
    assert_size_stride(addmm_22, (1584, 3072), (3072, 1))
    assert_size_stride(view_59, (1584, 3072), (3072, 1))
    assert_size_stride(mul_42, (8, 198, 768), (152064, 768, 1))
    assert_size_stride(view_61, (1584, 768), (768, 1))
    assert_size_stride(getitem_68, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_69, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_70, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_72, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_73, (), ())
    assert_size_stride(getitem_74, (), ())
    assert_size_stride(view_65, (1584, 768), (768, 1))
    assert_size_stride(mul_44, (8, 198, 768), (152064, 768, 1))
    assert_size_stride(view_67, (1584, 768), (768, 1))
    assert_size_stride(addmm_26, (1584, 3072), (3072, 1))
    assert_size_stride(view_69, (1584, 3072), (3072, 1))
    assert_size_stride(mul_49, (8, 198, 768), (152064, 768, 1))
    assert_size_stride(view_71, (1584, 768), (768, 1))
    assert_size_stride(getitem_79, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_80, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_81, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_83, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_84, (), ())
    assert_size_stride(getitem_85, (), ())
    assert_size_stride(view_75, (1584, 768), (768, 1))
    assert_size_stride(mul_51, (8, 198, 768), (152064, 768, 1))
    assert_size_stride(view_77, (1584, 768), (768, 1))
    assert_size_stride(addmm_30, (1584, 3072), (3072, 1))
    assert_size_stride(view_79, (1584, 3072), (3072, 1))
    assert_size_stride(mul_56, (8, 198, 768), (152064, 768, 1))
    assert_size_stride(view_81, (1584, 768), (768, 1))
    assert_size_stride(getitem_90, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_91, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_92, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_94, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_95, (), ())
    assert_size_stride(getitem_96, (), ())
    assert_size_stride(view_85, (1584, 768), (768, 1))
    assert_size_stride(mul_58, (8, 198, 768), (152064, 768, 1))
    assert_size_stride(view_87, (1584, 768), (768, 1))
    assert_size_stride(addmm_34, (1584, 3072), (3072, 1))
    assert_size_stride(view_89, (1584, 3072), (3072, 1))
    assert_size_stride(mul_63, (8, 198, 768), (152064, 768, 1))
    assert_size_stride(view_91, (1584, 768), (768, 1))
    assert_size_stride(getitem_101, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_102, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_103, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_105, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_106, (), ())
    assert_size_stride(getitem_107, (), ())
    assert_size_stride(view_95, (1584, 768), (768, 1))
    assert_size_stride(mul_65, (8, 198, 768), (152064, 768, 1))
    assert_size_stride(view_97, (1584, 768), (768, 1))
    assert_size_stride(addmm_38, (1584, 3072), (3072, 1))
    assert_size_stride(view_99, (1584, 3072), (3072, 1))
    assert_size_stride(mul_70, (8, 198, 768), (152064, 768, 1))
    assert_size_stride(view_101, (1584, 768), (768, 1))
    assert_size_stride(getitem_112, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_113, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_114, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_116, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_117, (), ())
    assert_size_stride(getitem_118, (), ())
    assert_size_stride(view_105, (1584, 768), (768, 1))
    assert_size_stride(mul_72, (8, 198, 768), (152064, 768, 1))
    assert_size_stride(view_107, (1584, 768), (768, 1))
    assert_size_stride(addmm_42, (1584, 3072), (3072, 1))
    assert_size_stride(view_109, (1584, 3072), (3072, 1))
    assert_size_stride(mul_77, (8, 198, 768), (152064, 768, 1))
    assert_size_stride(view_111, (1584, 768), (768, 1))
    assert_size_stride(getitem_123, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_124, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_125, (8, 12, 198, 64), (456192, 64, 2304, 1))
    assert_size_stride(getitem_127, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_128, (), ())
    assert_size_stride(getitem_129, (), ())
    assert_size_stride(view_115, (1584, 768), (768, 1))
    assert_size_stride(mul_79, (8, 198, 768), (152064, 768, 1))
    assert_size_stride(view_117, (1584, 768), (768, 1))
    assert_size_stride(addmm_46, (1584, 3072), (3072, 1))
    assert_size_stride(view_119, (1584, 3072), (3072, 1))
    assert_size_stride(mul_84, (8, 198, 768), (152064, 768, 1))
    assert_size_stride(select, (8, 768), (152064, 1))
    assert_size_stride(select_1, (8, 768), (152064, 1))
    assert_size_stride(permute_75, (1000, 768), (768, 1))
    assert_size_stride(permute_79, (1000, 768), (768, 1))
    assert_size_stride(div_2, (8, 198, 1), (198, 1, 1))
    assert_size_stride(permute_83, (768, 3072), (3072, 1))
    assert_size_stride(permute_87, (3072, 768), (768, 1))
    assert_size_stride(div_3, (8, 198, 1), (198, 1, 1))
    assert_size_stride(permute_91, (768, 768), (768, 1))
    assert_size_stride(alias_12, (8, 12, 198, 64), (152064, 64, 768, 1))
    assert_size_stride(permute_97, (2304, 768), (768, 1))
    assert_size_stride(div_4, (8, 198, 1), (198, 1, 1))
    assert_size_stride(permute_101, (768, 3072), (3072, 1))
    assert_size_stride(permute_105, (3072, 768), (768, 1))
    assert_size_stride(div_5, (8, 198, 1), (198, 1, 1))
    assert_size_stride(permute_109, (768, 768), (768, 1))
    assert_size_stride(alias_13, (8, 12, 198, 64), (152064, 64, 768, 1))
    assert_size_stride(permute_115, (2304, 768), (768, 1))
    assert_size_stride(div_6, (8, 198, 1), (198, 1, 1))
    assert_size_stride(permute_119, (768, 3072), (3072, 1))
    assert_size_stride(permute_123, (3072, 768), (768, 1))
    assert_size_stride(div_7, (8, 198, 1), (198, 1, 1))
    assert_size_stride(permute_127, (768, 768), (768, 1))
    assert_size_stride(alias_14, (8, 12, 198, 64), (152064, 64, 768, 1))
    assert_size_stride(permute_133, (2304, 768), (768, 1))
    assert_size_stride(div_8, (8, 198, 1), (198, 1, 1))
    assert_size_stride(permute_137, (768, 3072), (3072, 1))
    assert_size_stride(permute_141, (3072, 768), (768, 1))
    assert_size_stride(div_9, (8, 198, 1), (198, 1, 1))
    assert_size_stride(permute_145, (768, 768), (768, 1))
    assert_size_stride(alias_15, (8, 12, 198, 64), (152064, 64, 768, 1))
    assert_size_stride(permute_151, (2304, 768), (768, 1))
    assert_size_stride(div_10, (8, 198, 1), (198, 1, 1))
    assert_size_stride(permute_155, (768, 3072), (3072, 1))
    assert_size_stride(permute_159, (3072, 768), (768, 1))
    assert_size_stride(div_11, (8, 198, 1), (198, 1, 1))
    assert_size_stride(permute_163, (768, 768), (768, 1))
    assert_size_stride(alias_16, (8, 12, 198, 64), (152064, 64, 768, 1))
    assert_size_stride(permute_169, (2304, 768), (768, 1))
    assert_size_stride(div_12, (8, 198, 1), (198, 1, 1))
    assert_size_stride(permute_173, (768, 3072), (3072, 1))
    assert_size_stride(permute_177, (3072, 768), (768, 1))
    assert_size_stride(div_13, (8, 198, 1), (198, 1, 1))
    assert_size_stride(permute_181, (768, 768), (768, 1))
    assert_size_stride(alias_17, (8, 12, 198, 64), (152064, 64, 768, 1))
    assert_size_stride(permute_187, (2304, 768), (768, 1))
    assert_size_stride(div_14, (8, 198, 1), (198, 1, 1))
    assert_size_stride(permute_191, (768, 3072), (3072, 1))
    assert_size_stride(permute_195, (3072, 768), (768, 1))
    assert_size_stride(div_15, (8, 198, 1), (198, 1, 1))
    assert_size_stride(permute_199, (768, 768), (768, 1))
    assert_size_stride(alias_18, (8, 12, 198, 64), (152064, 64, 768, 1))
    assert_size_stride(permute_205, (2304, 768), (768, 1))
    assert_size_stride(div_16, (8, 198, 1), (198, 1, 1))
    assert_size_stride(permute_209, (768, 3072), (3072, 1))
    assert_size_stride(permute_213, (3072, 768), (768, 1))
    assert_size_stride(div_17, (8, 198, 1), (198, 1, 1))
    assert_size_stride(permute_217, (768, 768), (768, 1))
    assert_size_stride(alias_19, (8, 12, 198, 64), (152064, 64, 768, 1))
    assert_size_stride(permute_223, (2304, 768), (768, 1))
    assert_size_stride(div_18, (8, 198, 1), (198, 1, 1))
    assert_size_stride(permute_227, (768, 3072), (3072, 1))
    assert_size_stride(permute_231, (3072, 768), (768, 1))
    assert_size_stride(div_19, (8, 198, 1), (198, 1, 1))
    assert_size_stride(permute_235, (768, 768), (768, 1))
    assert_size_stride(alias_20, (8, 12, 198, 64), (152064, 64, 768, 1))
    assert_size_stride(permute_241, (2304, 768), (768, 1))
    assert_size_stride(div_20, (8, 198, 1), (198, 1, 1))
    assert_size_stride(permute_245, (768, 3072), (3072, 1))
    assert_size_stride(permute_249, (3072, 768), (768, 1))
    assert_size_stride(div_21, (8, 198, 1), (198, 1, 1))
    assert_size_stride(permute_253, (768, 768), (768, 1))
    assert_size_stride(alias_21, (8, 12, 198, 64), (152064, 64, 768, 1))
    assert_size_stride(permute_259, (2304, 768), (768, 1))
    assert_size_stride(div_22, (8, 198, 1), (198, 1, 1))
    assert_size_stride(permute_263, (768, 3072), (3072, 1))
    assert_size_stride(permute_267, (3072, 768), (768, 1))
    assert_size_stride(div_23, (8, 198, 1), (198, 1, 1))
    assert_size_stride(permute_271, (768, 768), (768, 1))
    assert_size_stride(alias_22, (8, 12, 198, 64), (152064, 64, 768, 1))
    assert_size_stride(permute_277, (2304, 768), (768, 1))
    assert_size_stride(div_24, (8, 198, 1), (198, 1, 1))
    assert_size_stride(permute_281, (768, 3072), (3072, 1))
    assert_size_stride(permute_285, (3072, 768), (768, 1))
    assert_size_stride(div_25, (8, 198, 1), (198, 1, 1))
    assert_size_stride(permute_289, (768, 768), (768, 1))
    assert_size_stride(alias_23, (8, 12, 198, 64), (152064, 64, 768, 1))
    assert_size_stride(permute_295, (2304, 768), (768, 1))
    assert_size_stride(div_26, (8, 198, 1), (198, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_div_0.run(tangents_1, buf0, 8000, grid=grid(8000), stream=stream0)
        del tangents_1
        buf1 = empty((8, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf0, permute_75, out=buf1)
        del permute_75
        buf2 = empty((1000, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (1000, 8), (1, 1000), 0), select_1, out=buf2)
        del select_1
        buf3 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_1.run(buf0, buf3, 1000, 8, grid=grid(1000), stream=stream0)
        buf4 = empty((8, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf0, permute_79, out=buf4)
        del permute_79
        buf5 = empty((1000, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (1000, 8), (1, 1000), 0), select, out=buf5)
        del buf0
        del select
        buf8 = empty((8, 198, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.select_backward]
        triton_red_fused_add_native_layer_norm_backward_select_backward_2.run(buf1, buf4, primals_150, mul_84, div_2, buf8, 1584, 768, grid=grid(1584), stream=stream0)
        del div_2
        del primals_150
        buf9 = empty_strided((768, 13), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.select_backward]
        triton_red_fused_add_native_layer_norm_backward_select_backward_3.run(buf1, buf4, mul_84, buf9, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_84
        buf10 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.select_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf9, buf10, 768, 13, grid=grid(768), stream=stream0)
        buf11 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.select_backward]
        triton_red_fused_add_native_layer_norm_backward_select_backward_5.run(buf1, buf4, buf11, 768, 1584, grid=grid(768), stream=stream0)
        del buf1
        del buf4
        buf12 = empty((1584, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf8, (1584, 768), (768, 1), 0), permute_83, out=buf12)
        del permute_83
        buf13 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf8, (768, 1584), (1, 768), 0), view_119, out=buf13)
        del view_119
        buf14 = reinterpret_tensor(buf9, (1, 768, 13), (9984, 1, 768), 0); del buf9  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf8, buf14, 9984, 122, grid=grid(9984), stream=stream0)
        buf15 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf14, buf15, 768, 13, grid=grid(768), stream=stream0)
        buf16 = reinterpret_tensor(buf12, (8, 198, 3072), (608256, 3072, 1), 0); del buf12  # reuse
        # Source Nodes: [x_147], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_7.run(buf16, addmm_46, 4866048, grid=grid(4866048), stream=stream0)
        del addmm_46
        buf17 = empty((1584, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf16, (1584, 3072), (3072, 1), 0), permute_87, out=buf17)
        del permute_87
        buf18 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf16, (3072, 1584), (1, 3072), 0), view_117, out=buf18)
        del view_117
        buf19 = empty_strided((1, 3072, 13), (39936, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf16, buf19, 39936, 122, grid=grid(39936), stream=stream0)
        buf20 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf19, buf20, 3072, 13, grid=grid(3072), stream=stream0)
        buf27 = buf8; del buf8  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf27, buf17, primals_144, mul_79, div_3, 1584, 768, grid=grid(1584), stream=stream0)
        del div_3
        del primals_144
        buf23 = reinterpret_tensor(buf14, (768, 13), (1, 768), 0); del buf14  # reuse
        buf25 = empty_strided((768, 13), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf17, mul_79, buf23, buf25, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_79
        buf24 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf23, buf24, 768, 13, grid=grid(768), stream=stream0)
        buf26 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf25, buf26, 768, 13, grid=grid(768), stream=stream0)
        buf28 = buf17; del buf17  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (1584, 768), (768, 1), 0), permute_91, out=buf28)
        del permute_91
        buf29 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (768, 1584), (1, 768), 0), view_115, out=buf29)
        del view_115
        buf30 = reinterpret_tensor(buf25, (1, 768, 13), (9984, 1, 768), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf27, buf30, 9984, 122, grid=grid(9984), stream=stream0)
        buf31 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf30, buf31, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf32 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf28, (8, 12, 198, 64), (152064, 64, 768, 1), 0), getitem_123, getitem_124, getitem_125, None, alias_12, getitem_127, getitem_128, getitem_129, 0.0, [True, True, True, False])
        del alias_12
        del buf28
        del getitem_123
        del getitem_124
        del getitem_125
        del getitem_127
        del getitem_128
        del getitem_129
        buf33 = buf32[0]
        buf34 = buf32[1]
        buf35 = buf32[2]
        del buf32
        buf36 = empty((8, 198, 3, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf33, buf34, buf35, buf36, 3649536, grid=grid(3649536), stream=stream0)
        del buf33
        del buf34
        buf37 = reinterpret_tensor(buf35, (1584, 768), (768, 1), 0); del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (1584, 2304), (2304, 1), 0), permute_97, out=buf37)
        del permute_97
        buf38 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (2304, 1584), (1, 2304), 0), view_111, out=buf38)
        del view_111
        buf39 = empty_strided((1, 2304, 13), (29952, 1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf36, buf39, 29952, 122, grid=grid(29952), stream=stream0)
        buf40 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf39, buf40, 2304, 13, grid=grid(2304), stream=stream0)
        buf47 = buf27; del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf47, buf37, primals_138, mul_77, div_4, 1584, 768, grid=grid(1584), stream=stream0)
        del div_4
        del primals_138
        buf43 = reinterpret_tensor(buf30, (768, 13), (1, 768), 0); del buf30  # reuse
        buf45 = buf23; del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf37, mul_77, buf43, buf45, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_77
        buf44 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf43, buf44, 768, 13, grid=grid(768), stream=stream0)
        buf46 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf45, buf46, 768, 13, grid=grid(768), stream=stream0)
        buf48 = reinterpret_tensor(buf16, (1584, 3072), (3072, 1), 0); del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf47, (1584, 768), (768, 1), 0), permute_101, out=buf48)
        del permute_101
        buf49 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf47, (768, 1584), (1, 768), 0), view_109, out=buf49)
        del view_109
        buf50 = reinterpret_tensor(buf45, (1, 768, 13), (9984, 1, 768), 0); del buf45  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf47, buf50, 9984, 122, grid=grid(9984), stream=stream0)
        buf51 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf50, buf51, 768, 13, grid=grid(768), stream=stream0)
        buf52 = reinterpret_tensor(buf48, (8, 198, 3072), (608256, 3072, 1), 0); del buf48  # reuse
        # Source Nodes: [x_135], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_7.run(buf52, addmm_42, 4866048, grid=grid(4866048), stream=stream0)
        del addmm_42
        buf53 = buf37; del buf37  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf52, (1584, 3072), (3072, 1), 0), permute_105, out=buf53)
        del permute_105
        buf54 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf52, (3072, 1584), (1, 3072), 0), view_107, out=buf54)
        del view_107
        buf55 = buf19; del buf19  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf52, buf55, 39936, 122, grid=grid(39936), stream=stream0)
        buf56 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf55, buf56, 3072, 13, grid=grid(3072), stream=stream0)
        buf63 = buf47; del buf47  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf63, buf53, primals_132, mul_72, div_5, 1584, 768, grid=grid(1584), stream=stream0)
        del div_5
        del primals_132
        buf59 = reinterpret_tensor(buf50, (768, 13), (1, 768), 0); del buf50  # reuse
        buf61 = buf43; del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf53, mul_72, buf59, buf61, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_72
        buf60 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf59, buf60, 768, 13, grid=grid(768), stream=stream0)
        buf62 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf61, buf62, 768, 13, grid=grid(768), stream=stream0)
        buf64 = buf53; del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf63, (1584, 768), (768, 1), 0), permute_109, out=buf64)
        del permute_109
        buf65 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf63, (768, 1584), (1, 768), 0), view_105, out=buf65)
        del view_105
        buf66 = reinterpret_tensor(buf61, (1, 768, 13), (9984, 1, 768), 0); del buf61  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf63, buf66, 9984, 122, grid=grid(9984), stream=stream0)
        buf67 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf66, buf67, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf68 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf64, (8, 12, 198, 64), (152064, 64, 768, 1), 0), getitem_112, getitem_113, getitem_114, None, alias_13, getitem_116, getitem_117, getitem_118, 0.0, [True, True, True, False])
        del alias_13
        del buf64
        del getitem_112
        del getitem_113
        del getitem_114
        del getitem_116
        del getitem_117
        del getitem_118
        buf69 = buf68[0]
        buf70 = buf68[1]
        buf71 = buf68[2]
        del buf68
        buf72 = buf36; del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf69, buf70, buf71, buf72, 3649536, grid=grid(3649536), stream=stream0)
        del buf69
        del buf70
        buf73 = reinterpret_tensor(buf71, (1584, 768), (768, 1), 0); del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf72, (1584, 2304), (2304, 1), 0), permute_115, out=buf73)
        del permute_115
        buf74 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf72, (2304, 1584), (1, 2304), 0), view_101, out=buf74)
        del view_101
        buf75 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf72, buf75, 29952, 122, grid=grid(29952), stream=stream0)
        buf76 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf75, buf76, 2304, 13, grid=grid(2304), stream=stream0)
        buf83 = buf63; del buf63  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf83, buf73, primals_126, mul_70, div_6, 1584, 768, grid=grid(1584), stream=stream0)
        del div_6
        del primals_126
        buf79 = reinterpret_tensor(buf66, (768, 13), (1, 768), 0); del buf66  # reuse
        buf81 = buf59; del buf59  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf73, mul_70, buf79, buf81, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_70
        buf80 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf79, buf80, 768, 13, grid=grid(768), stream=stream0)
        buf82 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf81, buf82, 768, 13, grid=grid(768), stream=stream0)
        buf84 = reinterpret_tensor(buf52, (1584, 3072), (3072, 1), 0); del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf83, (1584, 768), (768, 1), 0), permute_119, out=buf84)
        del permute_119
        buf85 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf83, (768, 1584), (1, 768), 0), view_99, out=buf85)
        del view_99
        buf86 = reinterpret_tensor(buf81, (1, 768, 13), (9984, 1, 768), 0); del buf81  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf83, buf86, 9984, 122, grid=grid(9984), stream=stream0)
        buf87 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf86, buf87, 768, 13, grid=grid(768), stream=stream0)
        buf88 = reinterpret_tensor(buf84, (8, 198, 3072), (608256, 3072, 1), 0); del buf84  # reuse
        # Source Nodes: [x_123], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_7.run(buf88, addmm_38, 4866048, grid=grid(4866048), stream=stream0)
        del addmm_38
        buf89 = buf73; del buf73  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf88, (1584, 3072), (3072, 1), 0), permute_123, out=buf89)
        del permute_123
        buf90 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf88, (3072, 1584), (1, 3072), 0), view_97, out=buf90)
        del view_97
        buf91 = buf55; del buf55  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf88, buf91, 39936, 122, grid=grid(39936), stream=stream0)
        buf92 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf91, buf92, 3072, 13, grid=grid(3072), stream=stream0)
        buf99 = buf83; del buf83  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf99, buf89, primals_120, mul_65, div_7, 1584, 768, grid=grid(1584), stream=stream0)
        del div_7
        del primals_120
        buf95 = reinterpret_tensor(buf86, (768, 13), (1, 768), 0); del buf86  # reuse
        buf97 = buf79; del buf79  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf89, mul_65, buf95, buf97, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_65
        buf96 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf95, buf96, 768, 13, grid=grid(768), stream=stream0)
        buf98 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf97, buf98, 768, 13, grid=grid(768), stream=stream0)
        buf100 = buf89; del buf89  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (1584, 768), (768, 1), 0), permute_127, out=buf100)
        del permute_127
        buf101 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (768, 1584), (1, 768), 0), view_95, out=buf101)
        del view_95
        buf102 = reinterpret_tensor(buf97, (1, 768, 13), (9984, 1, 768), 0); del buf97  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf99, buf102, 9984, 122, grid=grid(9984), stream=stream0)
        buf103 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf102, buf103, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf104 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf100, (8, 12, 198, 64), (152064, 64, 768, 1), 0), getitem_101, getitem_102, getitem_103, None, alias_14, getitem_105, getitem_106, getitem_107, 0.0, [True, True, True, False])
        del alias_14
        del buf100
        del getitem_101
        del getitem_102
        del getitem_103
        del getitem_105
        del getitem_106
        del getitem_107
        buf105 = buf104[0]
        buf106 = buf104[1]
        buf107 = buf104[2]
        del buf104
        buf108 = buf72; del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf105, buf106, buf107, buf108, 3649536, grid=grid(3649536), stream=stream0)
        del buf105
        del buf106
        buf109 = reinterpret_tensor(buf107, (1584, 768), (768, 1), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf108, (1584, 2304), (2304, 1), 0), permute_133, out=buf109)
        del permute_133
        buf110 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf108, (2304, 1584), (1, 2304), 0), view_91, out=buf110)
        del view_91
        buf111 = buf75; del buf75  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf108, buf111, 29952, 122, grid=grid(29952), stream=stream0)
        buf112 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf111, buf112, 2304, 13, grid=grid(2304), stream=stream0)
        buf119 = buf99; del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf119, buf109, primals_114, mul_63, div_8, 1584, 768, grid=grid(1584), stream=stream0)
        del div_8
        del primals_114
        buf115 = reinterpret_tensor(buf102, (768, 13), (1, 768), 0); del buf102  # reuse
        buf117 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf109, mul_63, buf115, buf117, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_63
        buf116 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf115, buf116, 768, 13, grid=grid(768), stream=stream0)
        buf118 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf117, buf118, 768, 13, grid=grid(768), stream=stream0)
        buf120 = reinterpret_tensor(buf88, (1584, 3072), (3072, 1), 0); del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf119, (1584, 768), (768, 1), 0), permute_137, out=buf120)
        del permute_137
        buf121 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf119, (768, 1584), (1, 768), 0), view_89, out=buf121)
        del view_89
        buf122 = reinterpret_tensor(buf117, (1, 768, 13), (9984, 1, 768), 0); del buf117  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf119, buf122, 9984, 122, grid=grid(9984), stream=stream0)
        buf123 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf122, buf123, 768, 13, grid=grid(768), stream=stream0)
        buf124 = reinterpret_tensor(buf120, (8, 198, 3072), (608256, 3072, 1), 0); del buf120  # reuse
        # Source Nodes: [x_111], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_7.run(buf124, addmm_34, 4866048, grid=grid(4866048), stream=stream0)
        del addmm_34
        buf125 = buf109; del buf109  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf124, (1584, 3072), (3072, 1), 0), permute_141, out=buf125)
        del permute_141
        buf126 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf124, (3072, 1584), (1, 3072), 0), view_87, out=buf126)
        del view_87
        buf127 = buf91; del buf91  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf124, buf127, 39936, 122, grid=grid(39936), stream=stream0)
        buf128 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf127, buf128, 3072, 13, grid=grid(3072), stream=stream0)
        buf135 = buf119; del buf119  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf135, buf125, primals_108, mul_58, div_9, 1584, 768, grid=grid(1584), stream=stream0)
        del div_9
        del primals_108
        buf131 = reinterpret_tensor(buf122, (768, 13), (1, 768), 0); del buf122  # reuse
        buf133 = buf115; del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf125, mul_58, buf131, buf133, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_58
        buf132 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf131, buf132, 768, 13, grid=grid(768), stream=stream0)
        buf134 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf133, buf134, 768, 13, grid=grid(768), stream=stream0)
        buf136 = buf125; del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf135, (1584, 768), (768, 1), 0), permute_145, out=buf136)
        del permute_145
        buf137 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf135, (768, 1584), (1, 768), 0), view_85, out=buf137)
        del view_85
        buf138 = reinterpret_tensor(buf133, (1, 768, 13), (9984, 1, 768), 0); del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf135, buf138, 9984, 122, grid=grid(9984), stream=stream0)
        buf139 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf138, buf139, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf140 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf136, (8, 12, 198, 64), (152064, 64, 768, 1), 0), getitem_90, getitem_91, getitem_92, None, alias_15, getitem_94, getitem_95, getitem_96, 0.0, [True, True, True, False])
        del alias_15
        del buf136
        del getitem_90
        del getitem_91
        del getitem_92
        del getitem_94
        del getitem_95
        del getitem_96
        buf141 = buf140[0]
        buf142 = buf140[1]
        buf143 = buf140[2]
        del buf140
        buf144 = buf108; del buf108  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf141, buf142, buf143, buf144, 3649536, grid=grid(3649536), stream=stream0)
        del buf141
        del buf142
        buf145 = reinterpret_tensor(buf143, (1584, 768), (768, 1), 0); del buf143  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf144, (1584, 2304), (2304, 1), 0), permute_151, out=buf145)
        del permute_151
        buf146 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf144, (2304, 1584), (1, 2304), 0), view_81, out=buf146)
        del view_81
        buf147 = buf111; del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf144, buf147, 29952, 122, grid=grid(29952), stream=stream0)
        buf148 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf147, buf148, 2304, 13, grid=grid(2304), stream=stream0)
        buf155 = buf135; del buf135  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf155, buf145, primals_102, mul_56, div_10, 1584, 768, grid=grid(1584), stream=stream0)
        del div_10
        del primals_102
        buf151 = reinterpret_tensor(buf138, (768, 13), (1, 768), 0); del buf138  # reuse
        buf153 = buf131; del buf131  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf145, mul_56, buf151, buf153, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_56
        buf152 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf151, buf152, 768, 13, grid=grid(768), stream=stream0)
        buf154 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf153, buf154, 768, 13, grid=grid(768), stream=stream0)
        buf156 = reinterpret_tensor(buf124, (1584, 3072), (3072, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf155, (1584, 768), (768, 1), 0), permute_155, out=buf156)
        del permute_155
        buf157 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf155, (768, 1584), (1, 768), 0), view_79, out=buf157)
        del view_79
        buf158 = reinterpret_tensor(buf153, (1, 768, 13), (9984, 1, 768), 0); del buf153  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf155, buf158, 9984, 122, grid=grid(9984), stream=stream0)
        buf159 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf158, buf159, 768, 13, grid=grid(768), stream=stream0)
        buf160 = reinterpret_tensor(buf156, (8, 198, 3072), (608256, 3072, 1), 0); del buf156  # reuse
        # Source Nodes: [x_99], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_7.run(buf160, addmm_30, 4866048, grid=grid(4866048), stream=stream0)
        del addmm_30
        buf161 = buf145; del buf145  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf160, (1584, 3072), (3072, 1), 0), permute_159, out=buf161)
        del permute_159
        buf162 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf160, (3072, 1584), (1, 3072), 0), view_77, out=buf162)
        del view_77
        buf163 = buf127; del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf160, buf163, 39936, 122, grid=grid(39936), stream=stream0)
        buf164 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf163, buf164, 3072, 13, grid=grid(3072), stream=stream0)
        buf171 = buf155; del buf155  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf171, buf161, primals_96, mul_51, div_11, 1584, 768, grid=grid(1584), stream=stream0)
        del div_11
        del primals_96
        buf167 = reinterpret_tensor(buf158, (768, 13), (1, 768), 0); del buf158  # reuse
        buf169 = buf151; del buf151  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf161, mul_51, buf167, buf169, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_51
        buf168 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf167, buf168, 768, 13, grid=grid(768), stream=stream0)
        buf170 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf169, buf170, 768, 13, grid=grid(768), stream=stream0)
        buf172 = buf161; del buf161  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf171, (1584, 768), (768, 1), 0), permute_163, out=buf172)
        del permute_163
        buf173 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf171, (768, 1584), (1, 768), 0), view_75, out=buf173)
        del view_75
        buf174 = reinterpret_tensor(buf169, (1, 768, 13), (9984, 1, 768), 0); del buf169  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf171, buf174, 9984, 122, grid=grid(9984), stream=stream0)
        buf175 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf174, buf175, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf176 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf172, (8, 12, 198, 64), (152064, 64, 768, 1), 0), getitem_79, getitem_80, getitem_81, None, alias_16, getitem_83, getitem_84, getitem_85, 0.0, [True, True, True, False])
        del alias_16
        del buf172
        del getitem_79
        del getitem_80
        del getitem_81
        del getitem_83
        del getitem_84
        del getitem_85
        buf177 = buf176[0]
        buf178 = buf176[1]
        buf179 = buf176[2]
        del buf176
        buf180 = buf144; del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf177, buf178, buf179, buf180, 3649536, grid=grid(3649536), stream=stream0)
        del buf177
        del buf178
        buf181 = reinterpret_tensor(buf179, (1584, 768), (768, 1), 0); del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf180, (1584, 2304), (2304, 1), 0), permute_169, out=buf181)
        del permute_169
        buf182 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf180, (2304, 1584), (1, 2304), 0), view_71, out=buf182)
        del view_71
        buf183 = buf147; del buf147  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf180, buf183, 29952, 122, grid=grid(29952), stream=stream0)
        buf184 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf183, buf184, 2304, 13, grid=grid(2304), stream=stream0)
        buf191 = buf171; del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf191, buf181, primals_90, mul_49, div_12, 1584, 768, grid=grid(1584), stream=stream0)
        del div_12
        del primals_90
        buf187 = reinterpret_tensor(buf174, (768, 13), (1, 768), 0); del buf174  # reuse
        buf189 = buf167; del buf167  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf181, mul_49, buf187, buf189, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_49
        buf188 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf187, buf188, 768, 13, grid=grid(768), stream=stream0)
        buf190 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf189, buf190, 768, 13, grid=grid(768), stream=stream0)
        buf192 = reinterpret_tensor(buf160, (1584, 3072), (3072, 1), 0); del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf191, (1584, 768), (768, 1), 0), permute_173, out=buf192)
        del permute_173
        buf193 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf191, (768, 1584), (1, 768), 0), view_69, out=buf193)
        del view_69
        buf194 = reinterpret_tensor(buf189, (1, 768, 13), (9984, 1, 768), 0); del buf189  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf191, buf194, 9984, 122, grid=grid(9984), stream=stream0)
        buf195 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf194, buf195, 768, 13, grid=grid(768), stream=stream0)
        buf196 = reinterpret_tensor(buf192, (8, 198, 3072), (608256, 3072, 1), 0); del buf192  # reuse
        # Source Nodes: [x_87], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_7.run(buf196, addmm_26, 4866048, grid=grid(4866048), stream=stream0)
        del addmm_26
        buf197 = buf181; del buf181  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf196, (1584, 3072), (3072, 1), 0), permute_177, out=buf197)
        del permute_177
        buf198 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf196, (3072, 1584), (1, 3072), 0), view_67, out=buf198)
        del view_67
        buf199 = buf163; del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf196, buf199, 39936, 122, grid=grid(39936), stream=stream0)
        buf200 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf199, buf200, 3072, 13, grid=grid(3072), stream=stream0)
        buf207 = buf191; del buf191  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf207, buf197, primals_84, mul_44, div_13, 1584, 768, grid=grid(1584), stream=stream0)
        del div_13
        del primals_84
        buf203 = reinterpret_tensor(buf194, (768, 13), (1, 768), 0); del buf194  # reuse
        buf205 = buf187; del buf187  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf197, mul_44, buf203, buf205, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_44
        buf204 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf203, buf204, 768, 13, grid=grid(768), stream=stream0)
        buf206 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf205, buf206, 768, 13, grid=grid(768), stream=stream0)
        buf208 = buf197; del buf197  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf207, (1584, 768), (768, 1), 0), permute_181, out=buf208)
        del permute_181
        buf209 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf207, (768, 1584), (1, 768), 0), view_65, out=buf209)
        del view_65
        buf210 = reinterpret_tensor(buf205, (1, 768, 13), (9984, 1, 768), 0); del buf205  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf207, buf210, 9984, 122, grid=grid(9984), stream=stream0)
        buf211 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf210, buf211, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf212 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf208, (8, 12, 198, 64), (152064, 64, 768, 1), 0), getitem_68, getitem_69, getitem_70, None, alias_17, getitem_72, getitem_73, getitem_74, 0.0, [True, True, True, False])
        del alias_17
        del buf208
        del getitem_68
        del getitem_69
        del getitem_70
        del getitem_72
        del getitem_73
        del getitem_74
        buf213 = buf212[0]
        buf214 = buf212[1]
        buf215 = buf212[2]
        del buf212
        buf216 = buf180; del buf180  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf213, buf214, buf215, buf216, 3649536, grid=grid(3649536), stream=stream0)
        del buf213
        del buf214
        buf217 = reinterpret_tensor(buf215, (1584, 768), (768, 1), 0); del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf216, (1584, 2304), (2304, 1), 0), permute_187, out=buf217)
        del permute_187
        buf218 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf216, (2304, 1584), (1, 2304), 0), view_61, out=buf218)
        del view_61
        buf219 = buf183; del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf216, buf219, 29952, 122, grid=grid(29952), stream=stream0)
        buf220 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf219, buf220, 2304, 13, grid=grid(2304), stream=stream0)
        buf227 = buf207; del buf207  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf227, buf217, primals_78, mul_42, div_14, 1584, 768, grid=grid(1584), stream=stream0)
        del div_14
        del primals_78
        buf223 = reinterpret_tensor(buf210, (768, 13), (1, 768), 0); del buf210  # reuse
        buf225 = buf203; del buf203  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf217, mul_42, buf223, buf225, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_42
        buf224 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf223, buf224, 768, 13, grid=grid(768), stream=stream0)
        buf226 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf225, buf226, 768, 13, grid=grid(768), stream=stream0)
        buf228 = reinterpret_tensor(buf196, (1584, 3072), (3072, 1), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf227, (1584, 768), (768, 1), 0), permute_191, out=buf228)
        del permute_191
        buf229 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf227, (768, 1584), (1, 768), 0), view_59, out=buf229)
        del view_59
        buf230 = reinterpret_tensor(buf225, (1, 768, 13), (9984, 1, 768), 0); del buf225  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf227, buf230, 9984, 122, grid=grid(9984), stream=stream0)
        buf231 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf230, buf231, 768, 13, grid=grid(768), stream=stream0)
        buf232 = reinterpret_tensor(buf228, (8, 198, 3072), (608256, 3072, 1), 0); del buf228  # reuse
        # Source Nodes: [x_75], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_7.run(buf232, addmm_22, 4866048, grid=grid(4866048), stream=stream0)
        del addmm_22
        buf233 = buf217; del buf217  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf232, (1584, 3072), (3072, 1), 0), permute_195, out=buf233)
        del permute_195
        buf234 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf232, (3072, 1584), (1, 3072), 0), view_57, out=buf234)
        del view_57
        buf235 = buf199; del buf199  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf232, buf235, 39936, 122, grid=grid(39936), stream=stream0)
        buf236 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf235, buf236, 3072, 13, grid=grid(3072), stream=stream0)
        buf243 = buf227; del buf227  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf243, buf233, primals_72, mul_37, div_15, 1584, 768, grid=grid(1584), stream=stream0)
        del div_15
        del primals_72
        buf239 = reinterpret_tensor(buf230, (768, 13), (1, 768), 0); del buf230  # reuse
        buf241 = buf223; del buf223  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf233, mul_37, buf239, buf241, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_37
        buf240 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf239, buf240, 768, 13, grid=grid(768), stream=stream0)
        buf242 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf241, buf242, 768, 13, grid=grid(768), stream=stream0)
        buf244 = buf233; del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf243, (1584, 768), (768, 1), 0), permute_199, out=buf244)
        del permute_199
        buf245 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf243, (768, 1584), (1, 768), 0), view_55, out=buf245)
        del view_55
        buf246 = reinterpret_tensor(buf241, (1, 768, 13), (9984, 1, 768), 0); del buf241  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf243, buf246, 9984, 122, grid=grid(9984), stream=stream0)
        buf247 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf246, buf247, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf248 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf244, (8, 12, 198, 64), (152064, 64, 768, 1), 0), getitem_57, getitem_58, getitem_59, None, alias_18, getitem_61, getitem_62, getitem_63, 0.0, [True, True, True, False])
        del alias_18
        del buf244
        del getitem_57
        del getitem_58
        del getitem_59
        del getitem_61
        del getitem_62
        del getitem_63
        buf249 = buf248[0]
        buf250 = buf248[1]
        buf251 = buf248[2]
        del buf248
        buf252 = buf216; del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf249, buf250, buf251, buf252, 3649536, grid=grid(3649536), stream=stream0)
        del buf249
        del buf250
        buf253 = reinterpret_tensor(buf251, (1584, 768), (768, 1), 0); del buf251  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf252, (1584, 2304), (2304, 1), 0), permute_205, out=buf253)
        del permute_205
        buf254 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf252, (2304, 1584), (1, 2304), 0), view_51, out=buf254)
        del view_51
        buf255 = buf219; del buf219  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf252, buf255, 29952, 122, grid=grid(29952), stream=stream0)
        buf256 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf255, buf256, 2304, 13, grid=grid(2304), stream=stream0)
        buf263 = buf243; del buf243  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf263, buf253, primals_66, mul_35, div_16, 1584, 768, grid=grid(1584), stream=stream0)
        del div_16
        del primals_66
        buf259 = reinterpret_tensor(buf246, (768, 13), (1, 768), 0); del buf246  # reuse
        buf261 = buf239; del buf239  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf253, mul_35, buf259, buf261, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_35
        buf260 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf259, buf260, 768, 13, grid=grid(768), stream=stream0)
        buf262 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf261, buf262, 768, 13, grid=grid(768), stream=stream0)
        buf264 = reinterpret_tensor(buf232, (1584, 3072), (3072, 1), 0); del buf232  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf263, (1584, 768), (768, 1), 0), permute_209, out=buf264)
        del permute_209
        buf265 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf263, (768, 1584), (1, 768), 0), view_49, out=buf265)
        del view_49
        buf266 = reinterpret_tensor(buf261, (1, 768, 13), (9984, 1, 768), 0); del buf261  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf263, buf266, 9984, 122, grid=grid(9984), stream=stream0)
        buf267 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf266, buf267, 768, 13, grid=grid(768), stream=stream0)
        buf268 = reinterpret_tensor(buf264, (8, 198, 3072), (608256, 3072, 1), 0); del buf264  # reuse
        # Source Nodes: [x_63], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_7.run(buf268, addmm_18, 4866048, grid=grid(4866048), stream=stream0)
        del addmm_18
        buf269 = buf253; del buf253  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf268, (1584, 3072), (3072, 1), 0), permute_213, out=buf269)
        del permute_213
        buf270 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf268, (3072, 1584), (1, 3072), 0), view_47, out=buf270)
        del view_47
        buf271 = buf235; del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf268, buf271, 39936, 122, grid=grid(39936), stream=stream0)
        buf272 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf271, buf272, 3072, 13, grid=grid(3072), stream=stream0)
        buf279 = buf263; del buf263  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf279, buf269, primals_60, mul_30, div_17, 1584, 768, grid=grid(1584), stream=stream0)
        del div_17
        del primals_60
        buf275 = reinterpret_tensor(buf266, (768, 13), (1, 768), 0); del buf266  # reuse
        buf277 = buf259; del buf259  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf269, mul_30, buf275, buf277, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_30
        buf276 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf275, buf276, 768, 13, grid=grid(768), stream=stream0)
        buf278 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf277, buf278, 768, 13, grid=grid(768), stream=stream0)
        buf280 = buf269; del buf269  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf279, (1584, 768), (768, 1), 0), permute_217, out=buf280)
        del permute_217
        buf281 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf279, (768, 1584), (1, 768), 0), view_45, out=buf281)
        del view_45
        buf282 = reinterpret_tensor(buf277, (1, 768, 13), (9984, 1, 768), 0); del buf277  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf279, buf282, 9984, 122, grid=grid(9984), stream=stream0)
        buf283 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf282, buf283, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf284 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf280, (8, 12, 198, 64), (152064, 64, 768, 1), 0), getitem_46, getitem_47, getitem_48, None, alias_19, getitem_50, getitem_51, getitem_52, 0.0, [True, True, True, False])
        del alias_19
        del buf280
        del getitem_46
        del getitem_47
        del getitem_48
        del getitem_50
        del getitem_51
        del getitem_52
        buf285 = buf284[0]
        buf286 = buf284[1]
        buf287 = buf284[2]
        del buf284
        buf288 = buf252; del buf252  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf285, buf286, buf287, buf288, 3649536, grid=grid(3649536), stream=stream0)
        del buf285
        del buf286
        buf289 = reinterpret_tensor(buf287, (1584, 768), (768, 1), 0); del buf287  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf288, (1584, 2304), (2304, 1), 0), permute_223, out=buf289)
        del permute_223
        buf290 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf288, (2304, 1584), (1, 2304), 0), view_41, out=buf290)
        del view_41
        buf291 = buf255; del buf255  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf288, buf291, 29952, 122, grid=grid(29952), stream=stream0)
        buf292 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf291, buf292, 2304, 13, grid=grid(2304), stream=stream0)
        buf299 = buf279; del buf279  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf299, buf289, primals_54, mul_28, div_18, 1584, 768, grid=grid(1584), stream=stream0)
        del div_18
        del primals_54
        buf295 = reinterpret_tensor(buf282, (768, 13), (1, 768), 0); del buf282  # reuse
        buf297 = buf275; del buf275  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf289, mul_28, buf295, buf297, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_28
        buf296 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf295, buf296, 768, 13, grid=grid(768), stream=stream0)
        buf298 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf297, buf298, 768, 13, grid=grid(768), stream=stream0)
        buf300 = reinterpret_tensor(buf268, (1584, 3072), (3072, 1), 0); del buf268  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf299, (1584, 768), (768, 1), 0), permute_227, out=buf300)
        del permute_227
        buf301 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf299, (768, 1584), (1, 768), 0), view_39, out=buf301)
        del view_39
        buf302 = reinterpret_tensor(buf297, (1, 768, 13), (9984, 1, 768), 0); del buf297  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf299, buf302, 9984, 122, grid=grid(9984), stream=stream0)
        buf303 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf302, buf303, 768, 13, grid=grid(768), stream=stream0)
        buf304 = reinterpret_tensor(buf300, (8, 198, 3072), (608256, 3072, 1), 0); del buf300  # reuse
        # Source Nodes: [x_51], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_7.run(buf304, addmm_14, 4866048, grid=grid(4866048), stream=stream0)
        del addmm_14
        buf305 = buf289; del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (1584, 3072), (3072, 1), 0), permute_231, out=buf305)
        del permute_231
        buf306 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (3072, 1584), (1, 3072), 0), view_37, out=buf306)
        del view_37
        buf307 = buf271; del buf271  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf304, buf307, 39936, 122, grid=grid(39936), stream=stream0)
        buf308 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf307, buf308, 3072, 13, grid=grid(3072), stream=stream0)
        buf315 = buf299; del buf299  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf315, buf305, primals_48, mul_23, div_19, 1584, 768, grid=grid(1584), stream=stream0)
        del div_19
        del primals_48
        buf311 = reinterpret_tensor(buf302, (768, 13), (1, 768), 0); del buf302  # reuse
        buf313 = buf295; del buf295  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf305, mul_23, buf311, buf313, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_23
        buf312 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf311, buf312, 768, 13, grid=grid(768), stream=stream0)
        buf314 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf313, buf314, 768, 13, grid=grid(768), stream=stream0)
        buf316 = buf305; del buf305  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf315, (1584, 768), (768, 1), 0), permute_235, out=buf316)
        del permute_235
        buf317 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf315, (768, 1584), (1, 768), 0), view_35, out=buf317)
        del view_35
        buf318 = reinterpret_tensor(buf313, (1, 768, 13), (9984, 1, 768), 0); del buf313  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf315, buf318, 9984, 122, grid=grid(9984), stream=stream0)
        buf319 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf318, buf319, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf320 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf316, (8, 12, 198, 64), (152064, 64, 768, 1), 0), getitem_35, getitem_36, getitem_37, None, alias_20, getitem_39, getitem_40, getitem_41, 0.0, [True, True, True, False])
        del alias_20
        del buf316
        del getitem_35
        del getitem_36
        del getitem_37
        del getitem_39
        del getitem_40
        del getitem_41
        buf321 = buf320[0]
        buf322 = buf320[1]
        buf323 = buf320[2]
        del buf320
        buf324 = buf288; del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf321, buf322, buf323, buf324, 3649536, grid=grid(3649536), stream=stream0)
        del buf321
        del buf322
        buf325 = reinterpret_tensor(buf323, (1584, 768), (768, 1), 0); del buf323  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf324, (1584, 2304), (2304, 1), 0), permute_241, out=buf325)
        del permute_241
        buf326 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf324, (2304, 1584), (1, 2304), 0), view_31, out=buf326)
        del view_31
        buf327 = buf291; del buf291  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf324, buf327, 29952, 122, grid=grid(29952), stream=stream0)
        buf328 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf327, buf328, 2304, 13, grid=grid(2304), stream=stream0)
        buf335 = buf315; del buf315  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf335, buf325, primals_42, mul_21, div_20, 1584, 768, grid=grid(1584), stream=stream0)
        del div_20
        del primals_42
        buf331 = reinterpret_tensor(buf318, (768, 13), (1, 768), 0); del buf318  # reuse
        buf333 = buf311; del buf311  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf325, mul_21, buf331, buf333, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_21
        buf332 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf331, buf332, 768, 13, grid=grid(768), stream=stream0)
        buf334 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf333, buf334, 768, 13, grid=grid(768), stream=stream0)
        buf336 = reinterpret_tensor(buf304, (1584, 3072), (3072, 1), 0); del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf335, (1584, 768), (768, 1), 0), permute_245, out=buf336)
        del permute_245
        buf337 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf335, (768, 1584), (1, 768), 0), view_29, out=buf337)
        del view_29
        buf338 = reinterpret_tensor(buf333, (1, 768, 13), (9984, 1, 768), 0); del buf333  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf335, buf338, 9984, 122, grid=grid(9984), stream=stream0)
        buf339 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf338, buf339, 768, 13, grid=grid(768), stream=stream0)
        buf340 = reinterpret_tensor(buf336, (8, 198, 3072), (608256, 3072, 1), 0); del buf336  # reuse
        # Source Nodes: [x_39], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_7.run(buf340, addmm_10, 4866048, grid=grid(4866048), stream=stream0)
        del addmm_10
        buf341 = buf325; del buf325  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf340, (1584, 3072), (3072, 1), 0), permute_249, out=buf341)
        del permute_249
        buf342 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf340, (3072, 1584), (1, 3072), 0), view_27, out=buf342)
        del view_27
        buf343 = buf307; del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf340, buf343, 39936, 122, grid=grid(39936), stream=stream0)
        buf344 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf343, buf344, 3072, 13, grid=grid(3072), stream=stream0)
        buf351 = buf335; del buf335  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf351, buf341, primals_36, mul_16, div_21, 1584, 768, grid=grid(1584), stream=stream0)
        del div_21
        del primals_36
        buf347 = reinterpret_tensor(buf338, (768, 13), (1, 768), 0); del buf338  # reuse
        buf349 = buf331; del buf331  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf341, mul_16, buf347, buf349, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_16
        buf348 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf347, buf348, 768, 13, grid=grid(768), stream=stream0)
        buf350 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf349, buf350, 768, 13, grid=grid(768), stream=stream0)
        buf352 = buf341; del buf341  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf351, (1584, 768), (768, 1), 0), permute_253, out=buf352)
        del permute_253
        buf353 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf351, (768, 1584), (1, 768), 0), view_25, out=buf353)
        del view_25
        buf354 = reinterpret_tensor(buf349, (1, 768, 13), (9984, 1, 768), 0); del buf349  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf351, buf354, 9984, 122, grid=grid(9984), stream=stream0)
        buf355 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf354, buf355, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf356 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf352, (8, 12, 198, 64), (152064, 64, 768, 1), 0), getitem_24, getitem_25, getitem_26, None, alias_21, getitem_28, getitem_29, getitem_30, 0.0, [True, True, True, False])
        del alias_21
        del buf352
        del getitem_24
        del getitem_25
        del getitem_26
        del getitem_28
        del getitem_29
        del getitem_30
        buf357 = buf356[0]
        buf358 = buf356[1]
        buf359 = buf356[2]
        del buf356
        buf360 = buf324; del buf324  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf357, buf358, buf359, buf360, 3649536, grid=grid(3649536), stream=stream0)
        del buf357
        del buf358
        buf361 = reinterpret_tensor(buf359, (1584, 768), (768, 1), 0); del buf359  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf360, (1584, 2304), (2304, 1), 0), permute_259, out=buf361)
        del permute_259
        buf362 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf360, (2304, 1584), (1, 2304), 0), view_21, out=buf362)
        del view_21
        buf363 = buf327; del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf360, buf363, 29952, 122, grid=grid(29952), stream=stream0)
        buf364 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf363, buf364, 2304, 13, grid=grid(2304), stream=stream0)
        buf371 = buf351; del buf351  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf371, buf361, primals_30, mul_14, div_22, 1584, 768, grid=grid(1584), stream=stream0)
        del div_22
        del primals_30
        buf367 = reinterpret_tensor(buf354, (768, 13), (1, 768), 0); del buf354  # reuse
        buf369 = buf347; del buf347  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf361, mul_14, buf367, buf369, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_14
        buf368 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf367, buf368, 768, 13, grid=grid(768), stream=stream0)
        buf370 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf369, buf370, 768, 13, grid=grid(768), stream=stream0)
        buf372 = reinterpret_tensor(buf340, (1584, 3072), (3072, 1), 0); del buf340  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf371, (1584, 768), (768, 1), 0), permute_263, out=buf372)
        del permute_263
        buf373 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf371, (768, 1584), (1, 768), 0), view_19, out=buf373)
        del view_19
        buf374 = reinterpret_tensor(buf369, (1, 768, 13), (9984, 1, 768), 0); del buf369  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf371, buf374, 9984, 122, grid=grid(9984), stream=stream0)
        buf375 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf374, buf375, 768, 13, grid=grid(768), stream=stream0)
        buf376 = reinterpret_tensor(buf372, (8, 198, 3072), (608256, 3072, 1), 0); del buf372  # reuse
        # Source Nodes: [x_27], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_7.run(buf376, addmm_6, 4866048, grid=grid(4866048), stream=stream0)
        del addmm_6
        buf377 = buf361; del buf361  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf376, (1584, 3072), (3072, 1), 0), permute_267, out=buf377)
        del permute_267
        buf378 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf376, (3072, 1584), (1, 3072), 0), view_17, out=buf378)
        del view_17
        buf379 = buf343; del buf343  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf376, buf379, 39936, 122, grid=grid(39936), stream=stream0)
        buf380 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf379, buf380, 3072, 13, grid=grid(3072), stream=stream0)
        buf387 = buf371; del buf371  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf387, buf377, primals_24, mul_9, div_23, 1584, 768, grid=grid(1584), stream=stream0)
        del div_23
        del primals_24
        buf383 = reinterpret_tensor(buf374, (768, 13), (1, 768), 0); del buf374  # reuse
        buf385 = buf367; del buf367  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf377, mul_9, buf383, buf385, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_9
        buf384 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf383, buf384, 768, 13, grid=grid(768), stream=stream0)
        buf386 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf385, buf386, 768, 13, grid=grid(768), stream=stream0)
        buf388 = buf377; del buf377  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf387, (1584, 768), (768, 1), 0), permute_271, out=buf388)
        del permute_271
        buf389 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf387, (768, 1584), (1, 768), 0), view_15, out=buf389)
        del view_15
        buf390 = reinterpret_tensor(buf385, (1, 768, 13), (9984, 1, 768), 0); del buf385  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf387, buf390, 9984, 122, grid=grid(9984), stream=stream0)
        buf391 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf390, buf391, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf392 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf388, (8, 12, 198, 64), (152064, 64, 768, 1), 0), getitem_13, getitem_14, getitem_15, None, alias_22, getitem_17, getitem_18, getitem_19, 0.0, [True, True, True, False])
        del alias_22
        del buf388
        del getitem_13
        del getitem_14
        del getitem_15
        del getitem_17
        del getitem_18
        del getitem_19
        buf393 = buf392[0]
        buf394 = buf392[1]
        buf395 = buf392[2]
        del buf392
        buf396 = buf360; del buf360  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf393, buf394, buf395, buf396, 3649536, grid=grid(3649536), stream=stream0)
        del buf393
        del buf394
        buf397 = reinterpret_tensor(buf395, (1584, 768), (768, 1), 0); del buf395  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf396, (1584, 2304), (2304, 1), 0), permute_277, out=buf397)
        del permute_277
        buf398 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf396, (2304, 1584), (1, 2304), 0), view_11, out=buf398)
        del view_11
        buf399 = buf363; del buf363  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf396, buf399, 29952, 122, grid=grid(29952), stream=stream0)
        buf400 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf399, buf400, 2304, 13, grid=grid(2304), stream=stream0)
        buf407 = buf387; del buf387  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf407, buf397, primals_18, mul_7, div_24, 1584, 768, grid=grid(1584), stream=stream0)
        del div_24
        del primals_18
        buf403 = reinterpret_tensor(buf390, (768, 13), (1, 768), 0); del buf390  # reuse
        buf405 = buf383; del buf383  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf397, mul_7, buf403, buf405, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_7
        buf404 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf403, buf404, 768, 13, grid=grid(768), stream=stream0)
        buf406 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf405, buf406, 768, 13, grid=grid(768), stream=stream0)
        buf408 = reinterpret_tensor(buf376, (1584, 3072), (3072, 1), 0); del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf407, (1584, 768), (768, 1), 0), permute_281, out=buf408)
        del permute_281
        buf409 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf407, (768, 1584), (1, 768), 0), view_9, out=buf409)
        del view_9
        buf410 = reinterpret_tensor(buf405, (1, 768, 13), (9984, 1, 768), 0); del buf405  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf407, buf410, 9984, 122, grid=grid(9984), stream=stream0)
        buf411 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf410, buf411, 768, 13, grid=grid(768), stream=stream0)
        buf412 = reinterpret_tensor(buf408, (8, 198, 3072), (608256, 3072, 1), 0); del buf408  # reuse
        # Source Nodes: [x_15], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_7.run(buf412, addmm_2, 4866048, grid=grid(4866048), stream=stream0)
        del addmm_2
        buf413 = buf397; del buf397  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf412, (1584, 3072), (3072, 1), 0), permute_285, out=buf413)
        del permute_285
        buf414 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf412, (3072, 1584), (1, 3072), 0), view_7, out=buf414)
        del view_7
        buf415 = buf379; del buf379  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf412, buf415, 39936, 122, grid=grid(39936), stream=stream0)
        del buf412
        buf416 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf415, buf416, 3072, 13, grid=grid(3072), stream=stream0)
        del buf415
        buf423 = buf407; del buf407  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf423, buf413, primals_12, mul_2, div_25, 1584, 768, grid=grid(1584), stream=stream0)
        del div_25
        del primals_12
        buf419 = reinterpret_tensor(buf410, (768, 13), (1, 768), 0); del buf410  # reuse
        buf421 = buf403; del buf403  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf413, mul_2, buf419, buf421, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_2
        buf420 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf419, buf420, 768, 13, grid=grid(768), stream=stream0)
        buf422 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf421, buf422, 768, 13, grid=grid(768), stream=stream0)
        buf424 = buf413; del buf413  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf423, (1584, 768), (768, 1), 0), permute_289, out=buf424)
        del permute_289
        buf425 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf423, (768, 1584), (1, 768), 0), view_5, out=buf425)
        del view_5
        buf426 = reinterpret_tensor(buf421, (1, 768, 13), (9984, 1, 768), 0); del buf421  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf423, buf426, 9984, 122, grid=grid(9984), stream=stream0)
        buf427 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf426, buf427, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf428 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf424, (8, 12, 198, 64), (152064, 64, 768, 1), 0), getitem_2, getitem_3, getitem_4, None, alias_23, getitem_6, getitem_7, getitem_8, 0.0, [True, True, True, False])
        del alias_23
        del buf424
        del getitem_2
        del getitem_3
        del getitem_4
        del getitem_6
        del getitem_7
        del getitem_8
        buf429 = buf428[0]
        buf430 = buf428[1]
        buf431 = buf428[2]
        del buf428
        buf432 = buf396; del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf429, buf430, buf431, buf432, 3649536, grid=grid(3649536), stream=stream0)
        del buf429
        del buf430
        buf433 = reinterpret_tensor(buf431, (1584, 768), (768, 1), 0); del buf431  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf432, (1584, 2304), (2304, 1), 0), permute_295, out=buf433)
        del permute_295
        buf434 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf432, (2304, 1584), (1, 2304), 0), view_1, out=buf434)
        del view_1
        buf435 = buf399; del buf399  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_13.run(buf432, buf435, 29952, 122, grid=grid(29952), stream=stream0)
        del buf432
        buf436 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_14.run(buf435, buf436, 2304, 13, grid=grid(2304), stream=stream0)
        del buf435
        buf443 = buf423; del buf423  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf443, buf433, primals_6, mul, div_26, 1584, 768, grid=grid(1584), stream=stream0)
        del div_26
        del primals_6
        buf439 = reinterpret_tensor(buf426, (768, 13), (1, 768), 0); del buf426  # reuse
        buf441 = buf419; del buf419  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf433, mul, buf439, buf441, 9984, 122, grid=grid(9984), stream=stream0)
        del buf433
        del mul
        buf440 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf439, buf440, 768, 13, grid=grid(768), stream=stream0)
        del buf439
        buf442 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf441, buf442, 768, 13, grid=grid(768), stream=stream0)
        buf444 = empty((1, 198, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_15.run(buf443, buf444, 152064, 8, grid=grid(152064), stream=stream0)
        buf445 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf443, buf445, 768, 8, grid=grid(768), stream=stream0)
        buf446 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf443, buf446, 768, 8, grid=grid(768), stream=stream0)
        buf447 = buf441; del buf441  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_18.run(buf443, buf447, 9984, 121, grid=grid(9984), stream=stream0)
        buf448 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_4.run(buf447, buf448, 768, 13, grid=grid(768), stream=stream0)
        del buf447
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf449 = aten.convolution_backward(reinterpret_tensor(buf443, (8, 768, 14, 14), (152064, 1, 10752, 768), 1536), primals_156, primals_4, [768], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf443
        del primals_156
        del primals_4
        buf450 = buf449[1]
        return (buf444, buf446, buf445, buf450, buf448, buf440, buf442, reinterpret_tensor(buf434, (2304, 768), (768, 1), 0), reinterpret_tensor(buf436, (2304, ), (1, ), 0), reinterpret_tensor(buf425, (768, 768), (768, 1), 0), reinterpret_tensor(buf427, (768, ), (1, ), 0), buf420, buf422, reinterpret_tensor(buf414, (3072, 768), (768, 1), 0), reinterpret_tensor(buf416, (3072, ), (1, ), 0), reinterpret_tensor(buf409, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf411, (768, ), (1, ), 0), buf404, buf406, reinterpret_tensor(buf398, (2304, 768), (768, 1), 0), reinterpret_tensor(buf400, (2304, ), (1, ), 0), reinterpret_tensor(buf389, (768, 768), (768, 1), 0), reinterpret_tensor(buf391, (768, ), (1, ), 0), buf384, buf386, reinterpret_tensor(buf378, (3072, 768), (768, 1), 0), reinterpret_tensor(buf380, (3072, ), (1, ), 0), reinterpret_tensor(buf373, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf375, (768, ), (1, ), 0), buf368, buf370, reinterpret_tensor(buf362, (2304, 768), (768, 1), 0), reinterpret_tensor(buf364, (2304, ), (1, ), 0), reinterpret_tensor(buf353, (768, 768), (768, 1), 0), reinterpret_tensor(buf355, (768, ), (1, ), 0), buf348, buf350, reinterpret_tensor(buf342, (3072, 768), (768, 1), 0), reinterpret_tensor(buf344, (3072, ), (1, ), 0), reinterpret_tensor(buf337, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf339, (768, ), (1, ), 0), buf332, buf334, reinterpret_tensor(buf326, (2304, 768), (768, 1), 0), reinterpret_tensor(buf328, (2304, ), (1, ), 0), reinterpret_tensor(buf317, (768, 768), (768, 1), 0), reinterpret_tensor(buf319, (768, ), (1, ), 0), buf312, buf314, reinterpret_tensor(buf306, (3072, 768), (768, 1), 0), reinterpret_tensor(buf308, (3072, ), (1, ), 0), reinterpret_tensor(buf301, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf303, (768, ), (1, ), 0), buf296, buf298, reinterpret_tensor(buf290, (2304, 768), (768, 1), 0), reinterpret_tensor(buf292, (2304, ), (1, ), 0), reinterpret_tensor(buf281, (768, 768), (768, 1), 0), reinterpret_tensor(buf283, (768, ), (1, ), 0), buf276, buf278, reinterpret_tensor(buf270, (3072, 768), (768, 1), 0), reinterpret_tensor(buf272, (3072, ), (1, ), 0), reinterpret_tensor(buf265, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf267, (768, ), (1, ), 0), buf260, buf262, reinterpret_tensor(buf254, (2304, 768), (768, 1), 0), reinterpret_tensor(buf256, (2304, ), (1, ), 0), reinterpret_tensor(buf245, (768, 768), (768, 1), 0), reinterpret_tensor(buf247, (768, ), (1, ), 0), buf240, buf242, reinterpret_tensor(buf234, (3072, 768), (768, 1), 0), reinterpret_tensor(buf236, (3072, ), (1, ), 0), reinterpret_tensor(buf229, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf231, (768, ), (1, ), 0), buf224, buf226, reinterpret_tensor(buf218, (2304, 768), (768, 1), 0), reinterpret_tensor(buf220, (2304, ), (1, ), 0), reinterpret_tensor(buf209, (768, 768), (768, 1), 0), reinterpret_tensor(buf211, (768, ), (1, ), 0), buf204, buf206, reinterpret_tensor(buf198, (3072, 768), (768, 1), 0), reinterpret_tensor(buf200, (3072, ), (1, ), 0), reinterpret_tensor(buf193, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf195, (768, ), (1, ), 0), buf188, buf190, reinterpret_tensor(buf182, (2304, 768), (768, 1), 0), reinterpret_tensor(buf184, (2304, ), (1, ), 0), reinterpret_tensor(buf173, (768, 768), (768, 1), 0), reinterpret_tensor(buf175, (768, ), (1, ), 0), buf168, buf170, reinterpret_tensor(buf162, (3072, 768), (768, 1), 0), reinterpret_tensor(buf164, (3072, ), (1, ), 0), reinterpret_tensor(buf157, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf159, (768, ), (1, ), 0), buf152, buf154, reinterpret_tensor(buf146, (2304, 768), (768, 1), 0), reinterpret_tensor(buf148, (2304, ), (1, ), 0), reinterpret_tensor(buf137, (768, 768), (768, 1), 0), reinterpret_tensor(buf139, (768, ), (1, ), 0), buf132, buf134, reinterpret_tensor(buf126, (3072, 768), (768, 1), 0), reinterpret_tensor(buf128, (3072, ), (1, ), 0), reinterpret_tensor(buf121, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf123, (768, ), (1, ), 0), buf116, buf118, reinterpret_tensor(buf110, (2304, 768), (768, 1), 0), reinterpret_tensor(buf112, (2304, ), (1, ), 0), reinterpret_tensor(buf101, (768, 768), (768, 1), 0), reinterpret_tensor(buf103, (768, ), (1, ), 0), buf96, buf98, reinterpret_tensor(buf90, (3072, 768), (768, 1), 0), reinterpret_tensor(buf92, (3072, ), (1, ), 0), reinterpret_tensor(buf85, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf87, (768, ), (1, ), 0), buf80, buf82, reinterpret_tensor(buf74, (2304, 768), (768, 1), 0), reinterpret_tensor(buf76, (2304, ), (1, ), 0), reinterpret_tensor(buf65, (768, 768), (768, 1), 0), reinterpret_tensor(buf67, (768, ), (1, ), 0), buf60, buf62, reinterpret_tensor(buf54, (3072, 768), (768, 1), 0), reinterpret_tensor(buf56, (3072, ), (1, ), 0), reinterpret_tensor(buf49, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf51, (768, ), (1, ), 0), buf44, buf46, reinterpret_tensor(buf38, (2304, 768), (768, 1), 0), reinterpret_tensor(buf40, (2304, ), (1, ), 0), reinterpret_tensor(buf29, (768, 768), (768, 1), 0), reinterpret_tensor(buf31, (768, ), (1, ), 0), buf24, buf26, reinterpret_tensor(buf18, (3072, 768), (768, 1), 0), reinterpret_tensor(buf20, (3072, ), (1, ), 0), reinterpret_tensor(buf13, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf15, (768, ), (1, ), 0), buf10, buf11, reinterpret_tensor(buf5, (1000, 768), (768, 1), 0), reinterpret_tensor(buf3, (1000, ), (1, ), 0), reinterpret_tensor(buf2, (1000, 768), (768, 1), 0), reinterpret_tensor(buf3, (1000, ), (1, ), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((768, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    mul = rand_strided((8, 198, 768), (152064, 768, 1), device='cuda:0', dtype=torch.float32)
    view_1 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_2 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_4 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_6 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_8 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_5 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_2 = rand_strided((8, 198, 768), (152064, 768, 1), device='cuda:0', dtype=torch.float32)
    view_7 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_2 = rand_strided((1584, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_9 = rand_strided((1584, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_7 = rand_strided((8, 198, 768), (152064, 768, 1), device='cuda:0', dtype=torch.float32)
    view_11 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_13 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_14 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_15 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_17 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_18 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_19 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_15 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_9 = rand_strided((8, 198, 768), (152064, 768, 1), device='cuda:0', dtype=torch.float32)
    view_17 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_6 = rand_strided((1584, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_19 = rand_strided((1584, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_14 = rand_strided((8, 198, 768), (152064, 768, 1), device='cuda:0', dtype=torch.float32)
    view_21 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_24 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_25 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_26 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_28 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_29 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_30 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_25 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_16 = rand_strided((8, 198, 768), (152064, 768, 1), device='cuda:0', dtype=torch.float32)
    view_27 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((1584, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_29 = rand_strided((1584, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_21 = rand_strided((8, 198, 768), (152064, 768, 1), device='cuda:0', dtype=torch.float32)
    view_31 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_35 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_36 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_37 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_39 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_40 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_41 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_35 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_23 = rand_strided((8, 198, 768), (152064, 768, 1), device='cuda:0', dtype=torch.float32)
    view_37 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_14 = rand_strided((1584, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_39 = rand_strided((1584, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_28 = rand_strided((8, 198, 768), (152064, 768, 1), device='cuda:0', dtype=torch.float32)
    view_41 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_46 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_47 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_48 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_50 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_51 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_52 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_45 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_30 = rand_strided((8, 198, 768), (152064, 768, 1), device='cuda:0', dtype=torch.float32)
    view_47 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_18 = rand_strided((1584, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_49 = rand_strided((1584, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_35 = rand_strided((8, 198, 768), (152064, 768, 1), device='cuda:0', dtype=torch.float32)
    view_51 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_57 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_58 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_59 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_61 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_62 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_63 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_55 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_37 = rand_strided((8, 198, 768), (152064, 768, 1), device='cuda:0', dtype=torch.float32)
    view_57 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((1584, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_59 = rand_strided((1584, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_42 = rand_strided((8, 198, 768), (152064, 768, 1), device='cuda:0', dtype=torch.float32)
    view_61 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_68 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_69 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_70 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_72 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_73 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_74 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_65 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_44 = rand_strided((8, 198, 768), (152064, 768, 1), device='cuda:0', dtype=torch.float32)
    view_67 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_26 = rand_strided((1584, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_69 = rand_strided((1584, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_49 = rand_strided((8, 198, 768), (152064, 768, 1), device='cuda:0', dtype=torch.float32)
    view_71 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_79 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_80 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_81 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_83 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_84 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_85 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_75 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_51 = rand_strided((8, 198, 768), (152064, 768, 1), device='cuda:0', dtype=torch.float32)
    view_77 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_30 = rand_strided((1584, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_79 = rand_strided((1584, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_56 = rand_strided((8, 198, 768), (152064, 768, 1), device='cuda:0', dtype=torch.float32)
    view_81 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_90 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_91 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_92 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_94 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_95 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_96 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_85 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_58 = rand_strided((8, 198, 768), (152064, 768, 1), device='cuda:0', dtype=torch.float32)
    view_87 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((1584, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_89 = rand_strided((1584, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_63 = rand_strided((8, 198, 768), (152064, 768, 1), device='cuda:0', dtype=torch.float32)
    view_91 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_101 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_102 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_103 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_105 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_106 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_107 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_95 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_65 = rand_strided((8, 198, 768), (152064, 768, 1), device='cuda:0', dtype=torch.float32)
    view_97 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_38 = rand_strided((1584, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_99 = rand_strided((1584, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_70 = rand_strided((8, 198, 768), (152064, 768, 1), device='cuda:0', dtype=torch.float32)
    view_101 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_112 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_113 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_114 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_116 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_117 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_118 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_105 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_72 = rand_strided((8, 198, 768), (152064, 768, 1), device='cuda:0', dtype=torch.float32)
    view_107 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_42 = rand_strided((1584, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_109 = rand_strided((1584, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_77 = rand_strided((8, 198, 768), (152064, 768, 1), device='cuda:0', dtype=torch.float32)
    view_111 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_123 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_124 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_125 = rand_strided((8, 12, 198, 64), (456192, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_127 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_128 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_129 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_115 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_79 = rand_strided((8, 198, 768), (152064, 768, 1), device='cuda:0', dtype=torch.float32)
    view_117 = rand_strided((1584, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_46 = rand_strided((1584, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_119 = rand_strided((1584, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_84 = rand_strided((8, 198, 768), (152064, 768, 1), device='cuda:0', dtype=torch.float32)
    select = rand_strided((8, 768), (152064, 1), device='cuda:0', dtype=torch.float32)
    select_1 = rand_strided((8, 768), (152064, 1), device='cuda:0', dtype=torch.float32)
    permute_75 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_79 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_2 = rand_strided((8, 198, 1), (198, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_83 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_87 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((8, 198, 1), (198, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_91 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_12 = rand_strided((8, 12, 198, 64), (152064, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_97 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_4 = rand_strided((8, 198, 1), (198, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_101 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_105 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_5 = rand_strided((8, 198, 1), (198, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_109 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_13 = rand_strided((8, 12, 198, 64), (152064, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_115 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_6 = rand_strided((8, 198, 1), (198, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_119 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_123 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_7 = rand_strided((8, 198, 1), (198, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_127 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_14 = rand_strided((8, 12, 198, 64), (152064, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_133 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_8 = rand_strided((8, 198, 1), (198, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_137 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_141 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_9 = rand_strided((8, 198, 1), (198, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_145 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_15 = rand_strided((8, 12, 198, 64), (152064, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_151 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_10 = rand_strided((8, 198, 1), (198, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_155 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_159 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_11 = rand_strided((8, 198, 1), (198, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_163 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_16 = rand_strided((8, 12, 198, 64), (152064, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_169 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_12 = rand_strided((8, 198, 1), (198, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_173 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_177 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_13 = rand_strided((8, 198, 1), (198, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_181 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_17 = rand_strided((8, 12, 198, 64), (152064, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_187 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_14 = rand_strided((8, 198, 1), (198, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_191 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_195 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_15 = rand_strided((8, 198, 1), (198, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_199 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_18 = rand_strided((8, 12, 198, 64), (152064, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_205 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_16 = rand_strided((8, 198, 1), (198, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_209 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_213 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_17 = rand_strided((8, 198, 1), (198, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_217 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_19 = rand_strided((8, 12, 198, 64), (152064, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_223 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_18 = rand_strided((8, 198, 1), (198, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_227 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_231 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_19 = rand_strided((8, 198, 1), (198, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_235 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_20 = rand_strided((8, 12, 198, 64), (152064, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_241 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_20 = rand_strided((8, 198, 1), (198, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_245 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_249 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_21 = rand_strided((8, 198, 1), (198, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_253 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_21 = rand_strided((8, 12, 198, 64), (152064, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_259 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_22 = rand_strided((8, 198, 1), (198, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_263 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_267 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_23 = rand_strided((8, 198, 1), (198, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_271 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_22 = rand_strided((8, 12, 198, 64), (152064, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_277 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_24 = rand_strided((8, 198, 1), (198, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_281 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_285 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_25 = rand_strided((8, 198, 1), (198, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_289 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_23 = rand_strided((8, 12, 198, 64), (152064, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_295 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_26 = rand_strided((8, 198, 1), (198, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_4, primals_6, primals_12, primals_18, primals_24, primals_30, primals_36, primals_42, primals_48, primals_54, primals_60, primals_66, primals_72, primals_78, primals_84, primals_90, primals_96, primals_102, primals_108, primals_114, primals_120, primals_126, primals_132, primals_138, primals_144, primals_150, primals_156, mul, view_1, getitem_2, getitem_3, getitem_4, getitem_6, getitem_7, getitem_8, view_5, mul_2, view_7, addmm_2, view_9, mul_7, view_11, getitem_13, getitem_14, getitem_15, getitem_17, getitem_18, getitem_19, view_15, mul_9, view_17, addmm_6, view_19, mul_14, view_21, getitem_24, getitem_25, getitem_26, getitem_28, getitem_29, getitem_30, view_25, mul_16, view_27, addmm_10, view_29, mul_21, view_31, getitem_35, getitem_36, getitem_37, getitem_39, getitem_40, getitem_41, view_35, mul_23, view_37, addmm_14, view_39, mul_28, view_41, getitem_46, getitem_47, getitem_48, getitem_50, getitem_51, getitem_52, view_45, mul_30, view_47, addmm_18, view_49, mul_35, view_51, getitem_57, getitem_58, getitem_59, getitem_61, getitem_62, getitem_63, view_55, mul_37, view_57, addmm_22, view_59, mul_42, view_61, getitem_68, getitem_69, getitem_70, getitem_72, getitem_73, getitem_74, view_65, mul_44, view_67, addmm_26, view_69, mul_49, view_71, getitem_79, getitem_80, getitem_81, getitem_83, getitem_84, getitem_85, view_75, mul_51, view_77, addmm_30, view_79, mul_56, view_81, getitem_90, getitem_91, getitem_92, getitem_94, getitem_95, getitem_96, view_85, mul_58, view_87, addmm_34, view_89, mul_63, view_91, getitem_101, getitem_102, getitem_103, getitem_105, getitem_106, getitem_107, view_95, mul_65, view_97, addmm_38, view_99, mul_70, view_101, getitem_112, getitem_113, getitem_114, getitem_116, getitem_117, getitem_118, view_105, mul_72, view_107, addmm_42, view_109, mul_77, view_111, getitem_123, getitem_124, getitem_125, getitem_127, getitem_128, getitem_129, view_115, mul_79, view_117, addmm_46, view_119, mul_84, select, select_1, permute_75, permute_79, div_2, permute_83, permute_87, div_3, permute_91, alias_12, permute_97, div_4, permute_101, permute_105, div_5, permute_109, alias_13, permute_115, div_6, permute_119, permute_123, div_7, permute_127, alias_14, permute_133, div_8, permute_137, permute_141, div_9, permute_145, alias_15, permute_151, div_10, permute_155, permute_159, div_11, permute_163, alias_16, permute_169, div_12, permute_173, permute_177, div_13, permute_181, alias_17, permute_187, div_14, permute_191, permute_195, div_15, permute_199, alias_18, permute_205, div_16, permute_209, permute_213, div_17, permute_217, alias_19, permute_223, div_18, permute_227, permute_231, div_19, permute_235, alias_20, permute_241, div_20, permute_245, permute_249, div_21, permute_253, alias_21, permute_259, div_22, permute_263, permute_267, div_23, permute_271, alias_22, permute_277, div_24, permute_281, permute_285, div_25, permute_289, alias_23, permute_295, div_26, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('deit_base_distilled_patch16_224', benchmark_compiled_module)
