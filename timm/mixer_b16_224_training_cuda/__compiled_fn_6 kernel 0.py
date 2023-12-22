
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


# kernel path: /tmp/torchinductor_youkaichao/5a/c5aqsxs2owd6rvdsyyq5lphgwmoyw3bghnv4kvqfy3gdnkgawhtw.py
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
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_native_layer_norm_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 768
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
        tmp0 = tl.load(in_ptr0 + (r2 + (768*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r2 + (768*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
        tmp14 = tl.load(in_ptr0 + (r2 + (768*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tl.load(in_ptr2 + (r2 + (768*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = 196.0
        tmp16 = tmp14 / tmp15
        tmp18 = tmp16 * tmp17
        tmp19 = 768.0
        tmp20 = tmp18 * tmp19
        tmp21 = tmp20 - tmp6
        tmp23 = tmp22 * tmp11
        tmp24 = tmp21 - tmp23
        tmp25 = tmp13 * tmp24
        tl.store(out_ptr2 + (r2 + (768*x3)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fk/cfk5v7zxpmzxiq2y42yvfzy6eppjcicpmwbrrcmmrtxv5cpfrb4c.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_native_layer_norm_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (768*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 196.0
        tmp5 = tmp3 / tmp4
        tmp6 = tl.load(in_ptr1 + (x0 + (768*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tmp5 * tmp6
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ms/cmsrfto3pwnkdfv2zmge2rqamtykvzb47ynm2fxuxwvfijzwgnb7.py
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
    size_hints=[1024, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_layer_norm_backward_3', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ja/cja35vgymb6v7mesgp2bsadnchgtse7eso3n2xztd42f7bnqqq6f.py
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
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_native_layer_norm_backward_4', 'mutated_arg_names': []}
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
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 196.0
        tmp2 = tmp0 / tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zw/czw32ftymhdbzmnaxwjl3evdcfv55hqkyzgfnzr7k46u7flaq4qe.py
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


# kernel path: /tmp/torchinductor_youkaichao/cu/ccupnkbltezpzlbah6pq2p7nw6gdwfifneqi4rp6gke7o5rteafo.py
# Source Nodes: [x_166], Original ATen: [aten.gelu, aten.gelu_backward]
# x_166 => add_106, erf_23, mul_118
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


# kernel path: /tmp/torchinductor_youkaichao/jl/cjl6dhiy5yrfemaekm7ovry34aysrugg3gt35cjiagaow5wzijx7.py
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


# kernel path: /tmp/torchinductor_youkaichao/lf/clfbqb7jnn7fbzn653tyqokvhas6bybfpwpbpynyu2iqv6nx6mzr.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_9', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/rk/crkosb53qp7kjok3lbjicgrh4p5nakdna5hm5xkm4gzrr4gnprw7.py
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


# kernel path: /tmp/torchinductor_youkaichao/e6/ce64d4x45sqhx2rrwjebzno4smyukikjkvppcuey5hjs5xvhyfpz.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]

triton_poi_fused__unsafe_view_clone_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + ((768*x1) + (150528*(y0 // 768)) + (y0 % 768)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + (196*y0)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vc/cvcrix7zllsbzsdf6u7rfw6lv3h3uox7ljq4dscpusazcvv7tubn.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/42/c426nqzlqv7j5j7fcm6xlvxok5i2ftqwo5dczmigy3vxsixqfolz.py
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
    size_hints=[256, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 196
    rnumel = 48
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uy/cuy2caafhpmafvsgouomicpax3ypt3jov7t4deoyicsgozqb5riv.py
# Source Nodes: [x_158, x_159], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
# x_158 => add_101
# x_159 => add_102, erf_22, mul_113
triton_red_fused_add_gelu_gelu_backward_sum_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_gelu_gelu_backward_sum_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18432
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    x1 = (xindex // 384)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (49152*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (384*r2) + (49152*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = 0.7071067811865476
        tmp5 = tmp3 * tmp4
        tmp6 = tl.math.erf(tmp5)
        tmp7 = 1.0
        tmp8 = tmp6 + tmp7
        tmp9 = 0.5
        tmp10 = tmp8 * tmp9
        tmp11 = tmp3 * tmp3
        tmp12 = -0.5
        tmp13 = tmp11 * tmp12
        tmp14 = tl.exp(tmp13)
        tmp15 = 0.3989422804014327
        tmp16 = tmp14 * tmp15
        tmp17 = tmp3 * tmp16
        tmp18 = tmp10 + tmp17
        tmp19 = tmp0 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask, tmp22, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fz/cfzsippb6bxxo2yn7amvbboconzhbpyq2ivxyymdwgn7lsk3ucfy.py
# Source Nodes: [x_158, x_159], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
# x_158 => add_101
# x_159 => add_102, erf_22, mul_113
triton_per_fused_add_gelu_gelu_backward_sum_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_gelu_gelu_backward_sum_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 48
    RBLOCK: tl.constexpr = 64
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


# kernel path: /tmp/torchinductor_youkaichao/g2/cg2rjvwkmdxsadhzxmgu5ccqwgkfek3ygvhwkew536li3pa7wden.py
# Source Nodes: [x_158, x_159], Original ATen: [aten.add, aten.gelu, aten.gelu_backward]
# x_158 => add_101
# x_159 => add_102, erf_22, mul_113
triton_poi_fused_add_gelu_gelu_backward_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_gelu_backward_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 0.7071067811865476
    tmp5 = tmp3 * tmp4
    tmp6 = tl.math.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp3 * tmp3
    tmp12 = -0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tl.exp(tmp13)
    tmp15 = 0.3989422804014327
    tmp16 = tmp14 * tmp15
    tmp17 = tmp3 * tmp16
    tmp18 = tmp10 + tmp17
    tmp19 = tmp0 * tmp18
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cq/ccqitzv4tjqmm3kgbc64ha6tmh5geiqteptkedtiahg5hbvw2twt.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_17', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/5f/c5f4ljvmg4tj5n4hji3x3dk6kb2hicaozde6pnnrxafk33zrpcpy.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_18', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ip/cipxvn75eegi5yb2jtdm4xwjm6rmjt4krue37bxhwrr2sipkug25.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_19', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/t2/ct25k6irqc64ollcln2cevbtqjrndqyhfvmwssezxmmr7c2wkbtr.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_20', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/oq/coqjz47vzpf4ppnzklbljszecblzoeriyj3ldyh53x6ck6ebtbid.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_21', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/fo/cfotrm633ipxonjiaegeygpkpljvnpp4c2ewtyj6oskmfs3c45ee.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_22', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/zp/czphxhtfbzfgt6dwed6z4vqebv2jr7luhwx3mfh4a23mxxfr6zy3.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_23', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/sj/csjh6yj3vmphyjmxyokurgxo6qijmmj3titfeysfpn6hfvuicn5n.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_poi_fused_add_native_layer_norm_backward_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_backward_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y0 + (196*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x2 + (768*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y3), ymask, eviction_policy='evict_last')
    tmp4 = tmp2 * tmp3
    tmp5 = 768.0
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 - tmp7
    tmp11 = tmp9 * tmp10
    tmp12 = tmp8 - tmp11
    tmp13 = tmp1 * tmp12
    tmp14 = tmp0 + tmp13
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (768*y3)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7s/c7sipyf2yivs4kpcoxhrghe5xxe66ownqdyqreddouo5cqxuonz2.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_25', 'mutated_arg_names': []}
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
    primals_1, primals_3, primals_6, primals_9, primals_15, primals_18, primals_21, primals_27, primals_30, primals_33, primals_39, primals_42, primals_45, primals_51, primals_54, primals_57, primals_63, primals_66, primals_69, primals_75, primals_78, primals_81, primals_87, primals_90, primals_93, primals_99, primals_102, primals_105, primals_111, primals_114, primals_117, primals_123, primals_126, primals_129, primals_135, primals_138, primals_141, primals_147, primals_151, mul, view_1, mm, view_3, mul_5, view_5, addmm_1, view_7, mul_10, view_9, mm_1, view_11, mul_15, view_13, addmm_4, view_15, mul_20, view_17, mm_2, view_19, mul_25, view_21, addmm_7, view_23, mul_30, view_25, mm_3, view_27, mul_35, view_29, addmm_10, view_31, mul_40, view_33, mm_4, view_35, mul_45, view_37, addmm_13, view_39, mul_50, view_41, mm_5, view_43, mul_55, view_45, addmm_16, view_47, mul_60, view_49, mm_6, view_51, mul_65, view_53, addmm_19, view_55, mul_70, view_57, mm_7, view_59, mul_75, view_61, addmm_22, view_63, mul_80, view_65, mm_8, view_67, mul_85, view_69, addmm_25, view_71, mul_90, view_73, mm_9, view_75, mul_95, view_77, addmm_28, view_79, mul_100, view_81, mm_10, view_83, mul_105, view_85, addmm_31, view_87, mul_110, view_89, mm_11, view_91, mul_115, view_93, addmm_34, view_95, mul_120, clone_85, permute_74, div_1, permute_78, permute_82, div_2, permute_87, permute_93, div_3, permute_96, permute_100, div_4, permute_105, permute_111, div_5, permute_114, permute_118, div_6, permute_123, permute_129, div_7, permute_132, permute_136, div_8, permute_141, permute_147, div_9, permute_150, permute_154, div_10, permute_159, permute_165, div_11, permute_168, permute_172, div_12, permute_177, permute_183, div_13, permute_186, permute_190, div_14, permute_195, permute_201, div_15, permute_204, permute_208, div_16, permute_213, permute_219, div_17, permute_222, permute_226, div_18, permute_231, permute_237, div_19, permute_240, permute_244, div_20, permute_249, permute_255, div_21, permute_258, permute_262, div_22, permute_267, permute_273, div_23, permute_276, permute_280, div_24, permute_285, permute_291, div_25, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (768, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_6, (384, ), (1, ))
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_18, (384, ), (1, ))
    assert_size_stride(primals_21, (768, ), (1, ))
    assert_size_stride(primals_27, (768, ), (1, ))
    assert_size_stride(primals_30, (384, ), (1, ))
    assert_size_stride(primals_33, (768, ), (1, ))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_42, (384, ), (1, ))
    assert_size_stride(primals_45, (768, ), (1, ))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_54, (384, ), (1, ))
    assert_size_stride(primals_57, (768, ), (1, ))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_66, (384, ), (1, ))
    assert_size_stride(primals_69, (768, ), (1, ))
    assert_size_stride(primals_75, (768, ), (1, ))
    assert_size_stride(primals_78, (384, ), (1, ))
    assert_size_stride(primals_81, (768, ), (1, ))
    assert_size_stride(primals_87, (768, ), (1, ))
    assert_size_stride(primals_90, (384, ), (1, ))
    assert_size_stride(primals_93, (768, ), (1, ))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_102, (384, ), (1, ))
    assert_size_stride(primals_105, (768, ), (1, ))
    assert_size_stride(primals_111, (768, ), (1, ))
    assert_size_stride(primals_114, (384, ), (1, ))
    assert_size_stride(primals_117, (768, ), (1, ))
    assert_size_stride(primals_123, (768, ), (1, ))
    assert_size_stride(primals_126, (384, ), (1, ))
    assert_size_stride(primals_129, (768, ), (1, ))
    assert_size_stride(primals_135, (768, ), (1, ))
    assert_size_stride(primals_138, (384, ), (1, ))
    assert_size_stride(primals_141, (768, ), (1, ))
    assert_size_stride(primals_147, (768, ), (1, ))
    assert_size_stride(primals_151, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(mul, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_1, (6144, 196), (196, 1))
    assert_size_stride(mm, (6144, 384), (384, 1))
    assert_size_stride(view_3, (6144, 384), (384, 1))
    assert_size_stride(mul_5, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_5, (1568, 768), (768, 1))
    assert_size_stride(addmm_1, (1568, 3072), (3072, 1))
    assert_size_stride(view_7, (1568, 3072), (3072, 1))
    assert_size_stride(mul_10, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_9, (6144, 196), (196, 1))
    assert_size_stride(mm_1, (6144, 384), (384, 1))
    assert_size_stride(view_11, (6144, 384), (384, 1))
    assert_size_stride(mul_15, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_13, (1568, 768), (768, 1))
    assert_size_stride(addmm_4, (1568, 3072), (3072, 1))
    assert_size_stride(view_15, (1568, 3072), (3072, 1))
    assert_size_stride(mul_20, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_17, (6144, 196), (196, 1))
    assert_size_stride(mm_2, (6144, 384), (384, 1))
    assert_size_stride(view_19, (6144, 384), (384, 1))
    assert_size_stride(mul_25, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_21, (1568, 768), (768, 1))
    assert_size_stride(addmm_7, (1568, 3072), (3072, 1))
    assert_size_stride(view_23, (1568, 3072), (3072, 1))
    assert_size_stride(mul_30, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_25, (6144, 196), (196, 1))
    assert_size_stride(mm_3, (6144, 384), (384, 1))
    assert_size_stride(view_27, (6144, 384), (384, 1))
    assert_size_stride(mul_35, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_29, (1568, 768), (768, 1))
    assert_size_stride(addmm_10, (1568, 3072), (3072, 1))
    assert_size_stride(view_31, (1568, 3072), (3072, 1))
    assert_size_stride(mul_40, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_33, (6144, 196), (196, 1))
    assert_size_stride(mm_4, (6144, 384), (384, 1))
    assert_size_stride(view_35, (6144, 384), (384, 1))
    assert_size_stride(mul_45, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_37, (1568, 768), (768, 1))
    assert_size_stride(addmm_13, (1568, 3072), (3072, 1))
    assert_size_stride(view_39, (1568, 3072), (3072, 1))
    assert_size_stride(mul_50, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_41, (6144, 196), (196, 1))
    assert_size_stride(mm_5, (6144, 384), (384, 1))
    assert_size_stride(view_43, (6144, 384), (384, 1))
    assert_size_stride(mul_55, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_45, (1568, 768), (768, 1))
    assert_size_stride(addmm_16, (1568, 3072), (3072, 1))
    assert_size_stride(view_47, (1568, 3072), (3072, 1))
    assert_size_stride(mul_60, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_49, (6144, 196), (196, 1))
    assert_size_stride(mm_6, (6144, 384), (384, 1))
    assert_size_stride(view_51, (6144, 384), (384, 1))
    assert_size_stride(mul_65, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_53, (1568, 768), (768, 1))
    assert_size_stride(addmm_19, (1568, 3072), (3072, 1))
    assert_size_stride(view_55, (1568, 3072), (3072, 1))
    assert_size_stride(mul_70, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_57, (6144, 196), (196, 1))
    assert_size_stride(mm_7, (6144, 384), (384, 1))
    assert_size_stride(view_59, (6144, 384), (384, 1))
    assert_size_stride(mul_75, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_61, (1568, 768), (768, 1))
    assert_size_stride(addmm_22, (1568, 3072), (3072, 1))
    assert_size_stride(view_63, (1568, 3072), (3072, 1))
    assert_size_stride(mul_80, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_65, (6144, 196), (196, 1))
    assert_size_stride(mm_8, (6144, 384), (384, 1))
    assert_size_stride(view_67, (6144, 384), (384, 1))
    assert_size_stride(mul_85, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_69, (1568, 768), (768, 1))
    assert_size_stride(addmm_25, (1568, 3072), (3072, 1))
    assert_size_stride(view_71, (1568, 3072), (3072, 1))
    assert_size_stride(mul_90, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_73, (6144, 196), (196, 1))
    assert_size_stride(mm_9, (6144, 384), (384, 1))
    assert_size_stride(view_75, (6144, 384), (384, 1))
    assert_size_stride(mul_95, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_77, (1568, 768), (768, 1))
    assert_size_stride(addmm_28, (1568, 3072), (3072, 1))
    assert_size_stride(view_79, (1568, 3072), (3072, 1))
    assert_size_stride(mul_100, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_81, (6144, 196), (196, 1))
    assert_size_stride(mm_10, (6144, 384), (384, 1))
    assert_size_stride(view_83, (6144, 384), (384, 1))
    assert_size_stride(mul_105, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_85, (1568, 768), (768, 1))
    assert_size_stride(addmm_31, (1568, 3072), (3072, 1))
    assert_size_stride(view_87, (1568, 3072), (3072, 1))
    assert_size_stride(mul_110, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_89, (6144, 196), (196, 1))
    assert_size_stride(mm_11, (6144, 384), (384, 1))
    assert_size_stride(view_91, (6144, 384), (384, 1))
    assert_size_stride(mul_115, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(view_93, (1568, 768), (768, 1))
    assert_size_stride(addmm_34, (1568, 3072), (3072, 1))
    assert_size_stride(view_95, (1568, 3072), (3072, 1))
    assert_size_stride(mul_120, (8, 196, 768), (150528, 768, 1))
    assert_size_stride(clone_85, (8, 768), (768, 1))
    assert_size_stride(permute_74, (1000, 768), (768, 1))
    assert_size_stride(div_1, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_78, (768, 3072), (3072, 1))
    assert_size_stride(permute_82, (3072, 768), (768, 1))
    assert_size_stride(div_2, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_87, (196, 384), (384, 1))
    assert_size_stride(permute_93, (384, 196), (196, 1))
    assert_size_stride(div_3, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_96, (768, 3072), (3072, 1))
    assert_size_stride(permute_100, (3072, 768), (768, 1))
    assert_size_stride(div_4, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_105, (196, 384), (384, 1))
    assert_size_stride(permute_111, (384, 196), (196, 1))
    assert_size_stride(div_5, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_114, (768, 3072), (3072, 1))
    assert_size_stride(permute_118, (3072, 768), (768, 1))
    assert_size_stride(div_6, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_123, (196, 384), (384, 1))
    assert_size_stride(permute_129, (384, 196), (196, 1))
    assert_size_stride(div_7, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_132, (768, 3072), (3072, 1))
    assert_size_stride(permute_136, (3072, 768), (768, 1))
    assert_size_stride(div_8, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_141, (196, 384), (384, 1))
    assert_size_stride(permute_147, (384, 196), (196, 1))
    assert_size_stride(div_9, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_150, (768, 3072), (3072, 1))
    assert_size_stride(permute_154, (3072, 768), (768, 1))
    assert_size_stride(div_10, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_159, (196, 384), (384, 1))
    assert_size_stride(permute_165, (384, 196), (196, 1))
    assert_size_stride(div_11, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_168, (768, 3072), (3072, 1))
    assert_size_stride(permute_172, (3072, 768), (768, 1))
    assert_size_stride(div_12, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_177, (196, 384), (384, 1))
    assert_size_stride(permute_183, (384, 196), (196, 1))
    assert_size_stride(div_13, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_186, (768, 3072), (3072, 1))
    assert_size_stride(permute_190, (3072, 768), (768, 1))
    assert_size_stride(div_14, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_195, (196, 384), (384, 1))
    assert_size_stride(permute_201, (384, 196), (196, 1))
    assert_size_stride(div_15, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_204, (768, 3072), (3072, 1))
    assert_size_stride(permute_208, (3072, 768), (768, 1))
    assert_size_stride(div_16, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_213, (196, 384), (384, 1))
    assert_size_stride(permute_219, (384, 196), (196, 1))
    assert_size_stride(div_17, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_222, (768, 3072), (3072, 1))
    assert_size_stride(permute_226, (3072, 768), (768, 1))
    assert_size_stride(div_18, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_231, (196, 384), (384, 1))
    assert_size_stride(permute_237, (384, 196), (196, 1))
    assert_size_stride(div_19, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_240, (768, 3072), (3072, 1))
    assert_size_stride(permute_244, (3072, 768), (768, 1))
    assert_size_stride(div_20, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_249, (196, 384), (384, 1))
    assert_size_stride(permute_255, (384, 196), (196, 1))
    assert_size_stride(div_21, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_258, (768, 3072), (3072, 1))
    assert_size_stride(permute_262, (3072, 768), (768, 1))
    assert_size_stride(div_22, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_267, (196, 384), (384, 1))
    assert_size_stride(permute_273, (384, 196), (196, 1))
    assert_size_stride(div_23, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_276, (768, 3072), (3072, 1))
    assert_size_stride(permute_280, (3072, 768), (768, 1))
    assert_size_stride(div_24, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_285, (196, 384), (384, 1))
    assert_size_stride(permute_291, (384, 196), (196, 1))
    assert_size_stride(div_25, (8, 196, 1), (196, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_74, out=buf0)
        del permute_74
        buf1 = empty((1000, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_85, out=buf1)
        del clone_85
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf5 = empty((8, 196, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_layer_norm_backward]
        triton_red_fused_div_native_layer_norm_backward_1.run(buf0, primals_147, mul_120, div_1, buf5, 1568, 768, grid=grid(1568), stream=stream0)
        del div_1
        del primals_147
        buf6 = empty_strided((768, 13), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_layer_norm_backward]
        triton_red_fused_div_native_layer_norm_backward_2.run(buf0, mul_120, buf6, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_120
        buf7 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf6, buf7, 768, 13, grid=grid(768), stream=stream0)
        buf8 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_layer_norm_backward]
        triton_red_fused_div_native_layer_norm_backward_4.run(buf0, buf8, 768, 1568, grid=grid(768), stream=stream0)
        del buf0
        buf9 = empty((1568, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (1568, 768), (768, 1), 0), permute_78, out=buf9)
        del permute_78
        buf10 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (768, 1568), (1, 768), 0), view_95, out=buf10)
        del view_95
        buf11 = reinterpret_tensor(buf6, (1, 768, 13), (9984, 1, 768), 0); del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf5, buf11, 9984, 121, grid=grid(9984), stream=stream0)
        buf12 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf11, buf12, 768, 13, grid=grid(768), stream=stream0)
        buf13 = reinterpret_tensor(buf9, (8, 196, 3072), (602112, 3072, 1), 0); del buf9  # reuse
        # Source Nodes: [x_166], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf13, addmm_34, 4816896, grid=grid(4816896), stream=stream0)
        del addmm_34
        buf14 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (1568, 3072), (3072, 1), 0), permute_82, out=buf14)
        del permute_82
        buf15 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (3072, 1568), (1, 3072), 0), view_93, out=buf15)
        del view_93
        buf16 = empty_strided((1, 3072, 13), (39936, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf13, buf16, 39936, 121, grid=grid(39936), stream=stream0)
        buf17 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf16, buf17, 3072, 13, grid=grid(3072), stream=stream0)
        buf24 = buf5; del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf24, buf14, primals_141, mul_115, div_2, 1568, 768, grid=grid(1568), stream=stream0)
        del div_2
        del primals_141
        buf20 = reinterpret_tensor(buf11, (768, 13), (1, 768), 0); del buf11  # reuse
        buf22 = empty_strided((768, 13), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf14, mul_115, buf20, buf22, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_115
        buf21 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf20, buf21, 768, 13, grid=grid(768), stream=stream0)
        buf23 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf22, buf23, 768, 13, grid=grid(768), stream=stream0)
        buf25 = reinterpret_tensor(buf14, (6144, 196), (196, 1), 0); del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf24, buf25, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf26 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf25, permute_87, out=buf26)
        del permute_87
        buf27 = empty((196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf25, (196, 6144), (1, 196), 0), view_91, out=buf27)
        del view_91
        buf28 = empty_strided((1, 196, 48), (9408, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf25, buf28, 9408, 128, grid=grid(9408), stream=stream0)
        buf29 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf28, buf29, 196, 48, grid=grid(196), stream=stream0)
        buf30 = empty_strided((1, 1, 384, 48), (18432, 18432, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_158, x_159], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_red_fused_add_gelu_gelu_backward_sum_14.run(buf26, mm_11, primals_138, buf30, 18432, 128, grid=grid(18432), stream=stream0)
        buf31 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_158, x_159], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_per_fused_add_gelu_gelu_backward_sum_15.run(buf30, buf31, 384, 48, grid=grid(384), stream=stream0)
        buf32 = reinterpret_tensor(buf26, (8, 768, 384), (294912, 384, 1), 0); del buf26  # reuse
        # Source Nodes: [x_158, x_159], Original ATen: [aten.add, aten.gelu, aten.gelu_backward]
        triton_poi_fused_add_gelu_gelu_backward_16.run(buf32, mm_11, primals_138, 2359296, grid=grid(2359296), stream=stream0)
        del mm_11
        del primals_138
        buf33 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf32, (384, 6144), (1, 384), 0), view_89, out=buf33)
        del view_89
        buf34 = buf25; del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf32, (6144, 384), (384, 1), 0), permute_93, out=buf34)
        del permute_93
        buf35 = reinterpret_tensor(buf28, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf34, primals_135, buf35, 9408, 128, grid=grid(9408), stream=stream0)
        buf36 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf35, buf36, 1568, 6, grid=grid(1568), stream=stream0)
        buf37 = reinterpret_tensor(buf35, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf34, primals_135, mul_110, buf37, 9408, 128, grid=grid(9408), stream=stream0)
        buf38 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf37, buf38, 1568, 6, grid=grid(1568), stream=stream0)
        buf39 = reinterpret_tensor(buf22, (768, 13), (13, 1), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf34, mul_110, buf39, 9984, 121, grid=grid(9984), stream=stream0)
        buf40 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf39, buf40, 768, 13, grid=grid(768), stream=stream0)
        buf41 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf34, buf41, 768, 1568, grid=grid(768), stream=stream0)
        buf42 = buf24; del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf42, div_3, buf34, primals_135, buf36, mul_110, buf38, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del div_3
        del mul_110
        del primals_135
        buf43 = reinterpret_tensor(buf13, (1568, 3072), (3072, 1), 0); del buf13  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf42, (1568, 768), (768, 1), 0), permute_96, out=buf43)
        del permute_96
        buf44 = reinterpret_tensor(buf32, (768, 3072), (3072, 1), 0); del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf42, (768, 1568), (1, 768), 0), view_87, out=buf44)
        del view_87
        buf45 = reinterpret_tensor(buf39, (1, 768, 13), (9984, 1, 768), 0); del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf42, buf45, 9984, 121, grid=grid(9984), stream=stream0)
        buf46 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf45, buf46, 768, 13, grid=grid(768), stream=stream0)
        buf47 = reinterpret_tensor(buf43, (8, 196, 3072), (602112, 3072, 1), 0); del buf43  # reuse
        # Source Nodes: [x_152], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf47, addmm_31, 4816896, grid=grid(4816896), stream=stream0)
        del addmm_31
        buf48 = reinterpret_tensor(buf34, (1568, 768), (768, 1), 0); del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf47, (1568, 3072), (3072, 1), 0), permute_100, out=buf48)
        del permute_100
        buf49 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf47, (3072, 1568), (1, 3072), 0), view_85, out=buf49)
        del view_85
        buf50 = buf16; del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf47, buf50, 39936, 121, grid=grid(39936), stream=stream0)
        buf51 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf50, buf51, 3072, 13, grid=grid(3072), stream=stream0)
        buf58 = buf42; del buf42  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf58, buf48, primals_129, mul_105, div_4, 1568, 768, grid=grid(1568), stream=stream0)
        del div_4
        del primals_129
        buf54 = reinterpret_tensor(buf45, (768, 13), (1, 768), 0); del buf45  # reuse
        buf56 = buf20; del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf48, mul_105, buf54, buf56, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_105
        buf55 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf54, buf55, 768, 13, grid=grid(768), stream=stream0)
        buf57 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf56, buf57, 768, 13, grid=grid(768), stream=stream0)
        buf59 = reinterpret_tensor(buf48, (6144, 196), (196, 1), 0); del buf48  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf58, buf59, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf60 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf59, permute_105, out=buf60)
        del permute_105
        buf61 = empty((196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf59, (196, 6144), (1, 196), 0), view_83, out=buf61)
        del view_83
        buf62 = reinterpret_tensor(buf37, (1, 196, 48), (9408, 1, 196), 0); del buf37  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf59, buf62, 9408, 128, grid=grid(9408), stream=stream0)
        buf63 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf62, buf63, 196, 48, grid=grid(196), stream=stream0)
        buf64 = buf30; del buf30  # reuse
        # Source Nodes: [x_144, x_145], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_red_fused_add_gelu_gelu_backward_sum_14.run(buf60, mm_10, primals_126, buf64, 18432, 128, grid=grid(18432), stream=stream0)
        buf65 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_144, x_145], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_per_fused_add_gelu_gelu_backward_sum_15.run(buf64, buf65, 384, 48, grid=grid(384), stream=stream0)
        buf66 = reinterpret_tensor(buf60, (8, 768, 384), (294912, 384, 1), 0); del buf60  # reuse
        # Source Nodes: [x_144, x_145], Original ATen: [aten.add, aten.gelu, aten.gelu_backward]
        triton_poi_fused_add_gelu_gelu_backward_16.run(buf66, mm_10, primals_126, 2359296, grid=grid(2359296), stream=stream0)
        del mm_10
        del primals_126
        buf67 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf66, (384, 6144), (1, 384), 0), view_81, out=buf67)
        del view_81
        buf68 = buf59; del buf59  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf66, (6144, 384), (384, 1), 0), permute_111, out=buf68)
        del permute_111
        buf69 = reinterpret_tensor(buf62, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf62  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf68, primals_123, buf69, 9408, 128, grid=grid(9408), stream=stream0)
        buf70 = buf38; del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf69, buf70, 1568, 6, grid=grid(1568), stream=stream0)
        buf71 = reinterpret_tensor(buf69, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf68, primals_123, mul_100, buf71, 9408, 128, grid=grid(9408), stream=stream0)
        buf72 = buf36; del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf71, buf72, 1568, 6, grid=grid(1568), stream=stream0)
        buf73 = reinterpret_tensor(buf56, (768, 13), (13, 1), 0); del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf68, mul_100, buf73, 9984, 121, grid=grid(9984), stream=stream0)
        buf74 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf73, buf74, 768, 13, grid=grid(768), stream=stream0)
        buf75 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf68, buf75, 768, 1568, grid=grid(768), stream=stream0)
        buf76 = buf58; del buf58  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf76, div_5, buf68, primals_123, buf70, mul_100, buf72, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del div_5
        del mul_100
        del primals_123
        buf77 = reinterpret_tensor(buf47, (1568, 3072), (3072, 1), 0); del buf47  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf76, (1568, 768), (768, 1), 0), permute_114, out=buf77)
        del permute_114
        buf78 = reinterpret_tensor(buf66, (768, 3072), (3072, 1), 0); del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf76, (768, 1568), (1, 768), 0), view_79, out=buf78)
        del view_79
        buf79 = reinterpret_tensor(buf73, (1, 768, 13), (9984, 1, 768), 0); del buf73  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf76, buf79, 9984, 121, grid=grid(9984), stream=stream0)
        buf80 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf79, buf80, 768, 13, grid=grid(768), stream=stream0)
        buf81 = reinterpret_tensor(buf77, (8, 196, 3072), (602112, 3072, 1), 0); del buf77  # reuse
        # Source Nodes: [x_138], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf81, addmm_28, 4816896, grid=grid(4816896), stream=stream0)
        del addmm_28
        buf82 = reinterpret_tensor(buf68, (1568, 768), (768, 1), 0); del buf68  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf81, (1568, 3072), (3072, 1), 0), permute_118, out=buf82)
        del permute_118
        buf83 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf81, (3072, 1568), (1, 3072), 0), view_77, out=buf83)
        del view_77
        buf84 = buf50; del buf50  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf81, buf84, 39936, 121, grid=grid(39936), stream=stream0)
        buf85 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf84, buf85, 3072, 13, grid=grid(3072), stream=stream0)
        buf92 = buf76; del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf92, buf82, primals_117, mul_95, div_6, 1568, 768, grid=grid(1568), stream=stream0)
        del div_6
        del primals_117
        buf88 = reinterpret_tensor(buf79, (768, 13), (1, 768), 0); del buf79  # reuse
        buf90 = buf54; del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf82, mul_95, buf88, buf90, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_95
        buf89 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf88, buf89, 768, 13, grid=grid(768), stream=stream0)
        buf91 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf90, buf91, 768, 13, grid=grid(768), stream=stream0)
        buf93 = reinterpret_tensor(buf82, (6144, 196), (196, 1), 0); del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf92, buf93, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf94 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf93, permute_123, out=buf94)
        del permute_123
        buf95 = empty((196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf93, (196, 6144), (1, 196), 0), view_75, out=buf95)
        del view_75
        buf96 = reinterpret_tensor(buf71, (1, 196, 48), (9408, 1, 196), 0); del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf93, buf96, 9408, 128, grid=grid(9408), stream=stream0)
        buf97 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf96, buf97, 196, 48, grid=grid(196), stream=stream0)
        buf98 = buf64; del buf64  # reuse
        # Source Nodes: [x_130, x_131], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_red_fused_add_gelu_gelu_backward_sum_14.run(buf94, mm_9, primals_114, buf98, 18432, 128, grid=grid(18432), stream=stream0)
        buf99 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_130, x_131], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_per_fused_add_gelu_gelu_backward_sum_15.run(buf98, buf99, 384, 48, grid=grid(384), stream=stream0)
        buf100 = reinterpret_tensor(buf94, (8, 768, 384), (294912, 384, 1), 0); del buf94  # reuse
        # Source Nodes: [x_130, x_131], Original ATen: [aten.add, aten.gelu, aten.gelu_backward]
        triton_poi_fused_add_gelu_gelu_backward_16.run(buf100, mm_9, primals_114, 2359296, grid=grid(2359296), stream=stream0)
        del mm_9
        del primals_114
        buf101 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf100, (384, 6144), (1, 384), 0), view_73, out=buf101)
        del view_73
        buf102 = buf93; del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf100, (6144, 384), (384, 1), 0), permute_129, out=buf102)
        del permute_129
        buf103 = reinterpret_tensor(buf96, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf96  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf102, primals_111, buf103, 9408, 128, grid=grid(9408), stream=stream0)
        buf104 = buf72; del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf103, buf104, 1568, 6, grid=grid(1568), stream=stream0)
        buf105 = reinterpret_tensor(buf103, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf102, primals_111, mul_90, buf105, 9408, 128, grid=grid(9408), stream=stream0)
        buf106 = buf70; del buf70  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf105, buf106, 1568, 6, grid=grid(1568), stream=stream0)
        buf107 = reinterpret_tensor(buf90, (768, 13), (13, 1), 0); del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf102, mul_90, buf107, 9984, 121, grid=grid(9984), stream=stream0)
        buf108 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf107, buf108, 768, 13, grid=grid(768), stream=stream0)
        buf109 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf102, buf109, 768, 1568, grid=grid(768), stream=stream0)
        buf110 = buf92; del buf92  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf110, div_7, buf102, primals_111, buf104, mul_90, buf106, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del div_7
        del mul_90
        del primals_111
        buf111 = reinterpret_tensor(buf81, (1568, 3072), (3072, 1), 0); del buf81  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf110, (1568, 768), (768, 1), 0), permute_132, out=buf111)
        del permute_132
        buf112 = reinterpret_tensor(buf100, (768, 3072), (3072, 1), 0); del buf100  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf110, (768, 1568), (1, 768), 0), view_71, out=buf112)
        del view_71
        buf113 = reinterpret_tensor(buf107, (1, 768, 13), (9984, 1, 768), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf110, buf113, 9984, 121, grid=grid(9984), stream=stream0)
        buf114 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf113, buf114, 768, 13, grid=grid(768), stream=stream0)
        buf115 = reinterpret_tensor(buf111, (8, 196, 3072), (602112, 3072, 1), 0); del buf111  # reuse
        # Source Nodes: [x_124], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf115, addmm_25, 4816896, grid=grid(4816896), stream=stream0)
        del addmm_25
        buf116 = reinterpret_tensor(buf102, (1568, 768), (768, 1), 0); del buf102  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf115, (1568, 3072), (3072, 1), 0), permute_136, out=buf116)
        del permute_136
        buf117 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf115, (3072, 1568), (1, 3072), 0), view_69, out=buf117)
        del view_69
        buf118 = buf84; del buf84  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf115, buf118, 39936, 121, grid=grid(39936), stream=stream0)
        buf119 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf118, buf119, 3072, 13, grid=grid(3072), stream=stream0)
        buf126 = buf110; del buf110  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf126, buf116, primals_105, mul_85, div_8, 1568, 768, grid=grid(1568), stream=stream0)
        del div_8
        del primals_105
        buf122 = reinterpret_tensor(buf113, (768, 13), (1, 768), 0); del buf113  # reuse
        buf124 = buf88; del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf116, mul_85, buf122, buf124, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_85
        buf123 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf122, buf123, 768, 13, grid=grid(768), stream=stream0)
        buf125 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf124, buf125, 768, 13, grid=grid(768), stream=stream0)
        buf127 = reinterpret_tensor(buf116, (6144, 196), (196, 1), 0); del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf126, buf127, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf128 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf127, permute_141, out=buf128)
        del permute_141
        buf129 = empty((196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf127, (196, 6144), (1, 196), 0), view_67, out=buf129)
        del view_67
        buf130 = reinterpret_tensor(buf105, (1, 196, 48), (9408, 1, 196), 0); del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf127, buf130, 9408, 128, grid=grid(9408), stream=stream0)
        buf131 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf130, buf131, 196, 48, grid=grid(196), stream=stream0)
        buf132 = buf98; del buf98  # reuse
        # Source Nodes: [x_116, x_117], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_red_fused_add_gelu_gelu_backward_sum_14.run(buf128, mm_8, primals_102, buf132, 18432, 128, grid=grid(18432), stream=stream0)
        buf133 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_116, x_117], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_per_fused_add_gelu_gelu_backward_sum_15.run(buf132, buf133, 384, 48, grid=grid(384), stream=stream0)
        buf134 = reinterpret_tensor(buf128, (8, 768, 384), (294912, 384, 1), 0); del buf128  # reuse
        # Source Nodes: [x_116, x_117], Original ATen: [aten.add, aten.gelu, aten.gelu_backward]
        triton_poi_fused_add_gelu_gelu_backward_16.run(buf134, mm_8, primals_102, 2359296, grid=grid(2359296), stream=stream0)
        del mm_8
        del primals_102
        buf135 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf134, (384, 6144), (1, 384), 0), view_65, out=buf135)
        del view_65
        buf136 = buf127; del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf134, (6144, 384), (384, 1), 0), permute_147, out=buf136)
        del permute_147
        buf137 = reinterpret_tensor(buf130, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf136, primals_99, buf137, 9408, 128, grid=grid(9408), stream=stream0)
        buf138 = buf106; del buf106  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf137, buf138, 1568, 6, grid=grid(1568), stream=stream0)
        buf139 = reinterpret_tensor(buf137, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf137  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf136, primals_99, mul_80, buf139, 9408, 128, grid=grid(9408), stream=stream0)
        buf140 = buf104; del buf104  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf139, buf140, 1568, 6, grid=grid(1568), stream=stream0)
        buf141 = reinterpret_tensor(buf124, (768, 13), (13, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf136, mul_80, buf141, 9984, 121, grid=grid(9984), stream=stream0)
        buf142 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf141, buf142, 768, 13, grid=grid(768), stream=stream0)
        buf143 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf136, buf143, 768, 1568, grid=grid(768), stream=stream0)
        buf144 = buf126; del buf126  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf144, div_9, buf136, primals_99, buf138, mul_80, buf140, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del div_9
        del mul_80
        del primals_99
        buf145 = reinterpret_tensor(buf115, (1568, 3072), (3072, 1), 0); del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf144, (1568, 768), (768, 1), 0), permute_150, out=buf145)
        del permute_150
        buf146 = reinterpret_tensor(buf134, (768, 3072), (3072, 1), 0); del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf144, (768, 1568), (1, 768), 0), view_63, out=buf146)
        del view_63
        buf147 = reinterpret_tensor(buf141, (1, 768, 13), (9984, 1, 768), 0); del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf144, buf147, 9984, 121, grid=grid(9984), stream=stream0)
        buf148 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf147, buf148, 768, 13, grid=grid(768), stream=stream0)
        buf149 = reinterpret_tensor(buf145, (8, 196, 3072), (602112, 3072, 1), 0); del buf145  # reuse
        # Source Nodes: [x_110], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf149, addmm_22, 4816896, grid=grid(4816896), stream=stream0)
        del addmm_22
        buf150 = reinterpret_tensor(buf136, (1568, 768), (768, 1), 0); del buf136  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (1568, 3072), (3072, 1), 0), permute_154, out=buf150)
        del permute_154
        buf151 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (3072, 1568), (1, 3072), 0), view_61, out=buf151)
        del view_61
        buf152 = buf118; del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf149, buf152, 39936, 121, grid=grid(39936), stream=stream0)
        buf153 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf152, buf153, 3072, 13, grid=grid(3072), stream=stream0)
        buf160 = buf144; del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf160, buf150, primals_93, mul_75, div_10, 1568, 768, grid=grid(1568), stream=stream0)
        del div_10
        del primals_93
        buf156 = reinterpret_tensor(buf147, (768, 13), (1, 768), 0); del buf147  # reuse
        buf158 = buf122; del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf150, mul_75, buf156, buf158, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_75
        buf157 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf156, buf157, 768, 13, grid=grid(768), stream=stream0)
        buf159 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf158, buf159, 768, 13, grid=grid(768), stream=stream0)
        buf161 = reinterpret_tensor(buf150, (6144, 196), (196, 1), 0); del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf160, buf161, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf162 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf161, permute_159, out=buf162)
        del permute_159
        buf163 = empty((196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf161, (196, 6144), (1, 196), 0), view_59, out=buf163)
        del view_59
        buf164 = reinterpret_tensor(buf139, (1, 196, 48), (9408, 1, 196), 0); del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf161, buf164, 9408, 128, grid=grid(9408), stream=stream0)
        buf165 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf164, buf165, 196, 48, grid=grid(196), stream=stream0)
        buf166 = buf132; del buf132  # reuse
        # Source Nodes: [x_102, x_103], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_red_fused_add_gelu_gelu_backward_sum_14.run(buf162, mm_7, primals_90, buf166, 18432, 128, grid=grid(18432), stream=stream0)
        buf167 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_102, x_103], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_per_fused_add_gelu_gelu_backward_sum_15.run(buf166, buf167, 384, 48, grid=grid(384), stream=stream0)
        buf168 = reinterpret_tensor(buf162, (8, 768, 384), (294912, 384, 1), 0); del buf162  # reuse
        # Source Nodes: [x_102, x_103], Original ATen: [aten.add, aten.gelu, aten.gelu_backward]
        triton_poi_fused_add_gelu_gelu_backward_16.run(buf168, mm_7, primals_90, 2359296, grid=grid(2359296), stream=stream0)
        del mm_7
        del primals_90
        buf169 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf168, (384, 6144), (1, 384), 0), view_57, out=buf169)
        del view_57
        buf170 = buf161; del buf161  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf168, (6144, 384), (384, 1), 0), permute_165, out=buf170)
        del permute_165
        buf171 = reinterpret_tensor(buf164, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf164  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf170, primals_87, buf171, 9408, 128, grid=grid(9408), stream=stream0)
        buf172 = buf140; del buf140  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf171, buf172, 1568, 6, grid=grid(1568), stream=stream0)
        buf173 = reinterpret_tensor(buf171, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf170, primals_87, mul_70, buf173, 9408, 128, grid=grid(9408), stream=stream0)
        buf174 = buf138; del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf173, buf174, 1568, 6, grid=grid(1568), stream=stream0)
        buf175 = reinterpret_tensor(buf158, (768, 13), (13, 1), 0); del buf158  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf170, mul_70, buf175, 9984, 121, grid=grid(9984), stream=stream0)
        buf176 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf175, buf176, 768, 13, grid=grid(768), stream=stream0)
        buf177 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf170, buf177, 768, 1568, grid=grid(768), stream=stream0)
        buf178 = buf160; del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf178, div_11, buf170, primals_87, buf172, mul_70, buf174, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del div_11
        del mul_70
        del primals_87
        buf179 = reinterpret_tensor(buf149, (1568, 3072), (3072, 1), 0); del buf149  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf178, (1568, 768), (768, 1), 0), permute_168, out=buf179)
        del permute_168
        buf180 = reinterpret_tensor(buf168, (768, 3072), (3072, 1), 0); del buf168  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf178, (768, 1568), (1, 768), 0), view_55, out=buf180)
        del view_55
        buf181 = reinterpret_tensor(buf175, (1, 768, 13), (9984, 1, 768), 0); del buf175  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf178, buf181, 9984, 121, grid=grid(9984), stream=stream0)
        buf182 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf181, buf182, 768, 13, grid=grid(768), stream=stream0)
        buf183 = reinterpret_tensor(buf179, (8, 196, 3072), (602112, 3072, 1), 0); del buf179  # reuse
        # Source Nodes: [x_96], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf183, addmm_19, 4816896, grid=grid(4816896), stream=stream0)
        del addmm_19
        buf184 = reinterpret_tensor(buf170, (1568, 768), (768, 1), 0); del buf170  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf183, (1568, 3072), (3072, 1), 0), permute_172, out=buf184)
        del permute_172
        buf185 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf183, (3072, 1568), (1, 3072), 0), view_53, out=buf185)
        del view_53
        buf186 = buf152; del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf183, buf186, 39936, 121, grid=grid(39936), stream=stream0)
        buf187 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf186, buf187, 3072, 13, grid=grid(3072), stream=stream0)
        buf194 = buf178; del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf194, buf184, primals_81, mul_65, div_12, 1568, 768, grid=grid(1568), stream=stream0)
        del div_12
        del primals_81
        buf190 = reinterpret_tensor(buf181, (768, 13), (1, 768), 0); del buf181  # reuse
        buf192 = buf156; del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf184, mul_65, buf190, buf192, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_65
        buf191 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf190, buf191, 768, 13, grid=grid(768), stream=stream0)
        buf193 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf192, buf193, 768, 13, grid=grid(768), stream=stream0)
        buf195 = reinterpret_tensor(buf184, (6144, 196), (196, 1), 0); del buf184  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf194, buf195, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf196 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf195, permute_177, out=buf196)
        del permute_177
        buf197 = empty((196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf195, (196, 6144), (1, 196), 0), view_51, out=buf197)
        del view_51
        buf198 = reinterpret_tensor(buf173, (1, 196, 48), (9408, 1, 196), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf195, buf198, 9408, 128, grid=grid(9408), stream=stream0)
        buf199 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf198, buf199, 196, 48, grid=grid(196), stream=stream0)
        buf200 = buf166; del buf166  # reuse
        # Source Nodes: [x_88, x_89], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_red_fused_add_gelu_gelu_backward_sum_14.run(buf196, mm_6, primals_78, buf200, 18432, 128, grid=grid(18432), stream=stream0)
        buf201 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_88, x_89], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_per_fused_add_gelu_gelu_backward_sum_15.run(buf200, buf201, 384, 48, grid=grid(384), stream=stream0)
        buf202 = reinterpret_tensor(buf196, (8, 768, 384), (294912, 384, 1), 0); del buf196  # reuse
        # Source Nodes: [x_88, x_89], Original ATen: [aten.add, aten.gelu, aten.gelu_backward]
        triton_poi_fused_add_gelu_gelu_backward_16.run(buf202, mm_6, primals_78, 2359296, grid=grid(2359296), stream=stream0)
        del mm_6
        del primals_78
        buf203 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf202, (384, 6144), (1, 384), 0), view_49, out=buf203)
        del view_49
        buf204 = buf195; del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf202, (6144, 384), (384, 1), 0), permute_183, out=buf204)
        del permute_183
        buf205 = reinterpret_tensor(buf198, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf204, primals_75, buf205, 9408, 128, grid=grid(9408), stream=stream0)
        buf206 = buf174; del buf174  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf205, buf206, 1568, 6, grid=grid(1568), stream=stream0)
        buf207 = reinterpret_tensor(buf205, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf205  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf204, primals_75, mul_60, buf207, 9408, 128, grid=grid(9408), stream=stream0)
        buf208 = buf172; del buf172  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf207, buf208, 1568, 6, grid=grid(1568), stream=stream0)
        buf209 = reinterpret_tensor(buf192, (768, 13), (13, 1), 0); del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf204, mul_60, buf209, 9984, 121, grid=grid(9984), stream=stream0)
        buf210 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf209, buf210, 768, 13, grid=grid(768), stream=stream0)
        buf211 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf204, buf211, 768, 1568, grid=grid(768), stream=stream0)
        buf212 = buf194; del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf212, div_13, buf204, primals_75, buf206, mul_60, buf208, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del div_13
        del mul_60
        del primals_75
        buf213 = reinterpret_tensor(buf183, (1568, 3072), (3072, 1), 0); del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf212, (1568, 768), (768, 1), 0), permute_186, out=buf213)
        del permute_186
        buf214 = reinterpret_tensor(buf202, (768, 3072), (3072, 1), 0); del buf202  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf212, (768, 1568), (1, 768), 0), view_47, out=buf214)
        del view_47
        buf215 = reinterpret_tensor(buf209, (1, 768, 13), (9984, 1, 768), 0); del buf209  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf212, buf215, 9984, 121, grid=grid(9984), stream=stream0)
        buf216 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf215, buf216, 768, 13, grid=grid(768), stream=stream0)
        buf217 = reinterpret_tensor(buf213, (8, 196, 3072), (602112, 3072, 1), 0); del buf213  # reuse
        # Source Nodes: [x_82], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf217, addmm_16, 4816896, grid=grid(4816896), stream=stream0)
        del addmm_16
        buf218 = reinterpret_tensor(buf204, (1568, 768), (768, 1), 0); del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf217, (1568, 3072), (3072, 1), 0), permute_190, out=buf218)
        del permute_190
        buf219 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf217, (3072, 1568), (1, 3072), 0), view_45, out=buf219)
        del view_45
        buf220 = buf186; del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf217, buf220, 39936, 121, grid=grid(39936), stream=stream0)
        buf221 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf220, buf221, 3072, 13, grid=grid(3072), stream=stream0)
        buf228 = buf212; del buf212  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf228, buf218, primals_69, mul_55, div_14, 1568, 768, grid=grid(1568), stream=stream0)
        del div_14
        del primals_69
        buf224 = reinterpret_tensor(buf215, (768, 13), (1, 768), 0); del buf215  # reuse
        buf226 = buf190; del buf190  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf218, mul_55, buf224, buf226, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_55
        buf225 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf224, buf225, 768, 13, grid=grid(768), stream=stream0)
        buf227 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf226, buf227, 768, 13, grid=grid(768), stream=stream0)
        buf229 = reinterpret_tensor(buf218, (6144, 196), (196, 1), 0); del buf218  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf228, buf229, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf230 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf229, permute_195, out=buf230)
        del permute_195
        buf231 = empty((196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf229, (196, 6144), (1, 196), 0), view_43, out=buf231)
        del view_43
        buf232 = reinterpret_tensor(buf207, (1, 196, 48), (9408, 1, 196), 0); del buf207  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf229, buf232, 9408, 128, grid=grid(9408), stream=stream0)
        buf233 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf232, buf233, 196, 48, grid=grid(196), stream=stream0)
        buf234 = buf200; del buf200  # reuse
        # Source Nodes: [x_74, x_75], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_red_fused_add_gelu_gelu_backward_sum_14.run(buf230, mm_5, primals_66, buf234, 18432, 128, grid=grid(18432), stream=stream0)
        buf235 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74, x_75], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_per_fused_add_gelu_gelu_backward_sum_15.run(buf234, buf235, 384, 48, grid=grid(384), stream=stream0)
        buf236 = reinterpret_tensor(buf230, (8, 768, 384), (294912, 384, 1), 0); del buf230  # reuse
        # Source Nodes: [x_74, x_75], Original ATen: [aten.add, aten.gelu, aten.gelu_backward]
        triton_poi_fused_add_gelu_gelu_backward_16.run(buf236, mm_5, primals_66, 2359296, grid=grid(2359296), stream=stream0)
        del mm_5
        del primals_66
        buf237 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf236, (384, 6144), (1, 384), 0), view_41, out=buf237)
        del view_41
        buf238 = buf229; del buf229  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf236, (6144, 384), (384, 1), 0), permute_201, out=buf238)
        del permute_201
        buf239 = reinterpret_tensor(buf232, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf232  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf238, primals_63, buf239, 9408, 128, grid=grid(9408), stream=stream0)
        buf240 = buf208; del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf239, buf240, 1568, 6, grid=grid(1568), stream=stream0)
        buf241 = reinterpret_tensor(buf239, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf239  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf238, primals_63, mul_50, buf241, 9408, 128, grid=grid(9408), stream=stream0)
        buf242 = buf206; del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf241, buf242, 1568, 6, grid=grid(1568), stream=stream0)
        buf243 = reinterpret_tensor(buf226, (768, 13), (13, 1), 0); del buf226  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf238, mul_50, buf243, 9984, 121, grid=grid(9984), stream=stream0)
        buf244 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf243, buf244, 768, 13, grid=grid(768), stream=stream0)
        buf245 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf238, buf245, 768, 1568, grid=grid(768), stream=stream0)
        buf246 = buf228; del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf246, div_15, buf238, primals_63, buf240, mul_50, buf242, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del div_15
        del mul_50
        del primals_63
        buf247 = reinterpret_tensor(buf217, (1568, 3072), (3072, 1), 0); del buf217  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf246, (1568, 768), (768, 1), 0), permute_204, out=buf247)
        del permute_204
        buf248 = reinterpret_tensor(buf236, (768, 3072), (3072, 1), 0); del buf236  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf246, (768, 1568), (1, 768), 0), view_39, out=buf248)
        del view_39
        buf249 = reinterpret_tensor(buf243, (1, 768, 13), (9984, 1, 768), 0); del buf243  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf246, buf249, 9984, 121, grid=grid(9984), stream=stream0)
        buf250 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf249, buf250, 768, 13, grid=grid(768), stream=stream0)
        buf251 = reinterpret_tensor(buf247, (8, 196, 3072), (602112, 3072, 1), 0); del buf247  # reuse
        # Source Nodes: [x_68], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf251, addmm_13, 4816896, grid=grid(4816896), stream=stream0)
        del addmm_13
        buf252 = reinterpret_tensor(buf238, (1568, 768), (768, 1), 0); del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf251, (1568, 3072), (3072, 1), 0), permute_208, out=buf252)
        del permute_208
        buf253 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf251, (3072, 1568), (1, 3072), 0), view_37, out=buf253)
        del view_37
        buf254 = buf220; del buf220  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf251, buf254, 39936, 121, grid=grid(39936), stream=stream0)
        buf255 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf254, buf255, 3072, 13, grid=grid(3072), stream=stream0)
        buf262 = buf246; del buf246  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf262, buf252, primals_57, mul_45, div_16, 1568, 768, grid=grid(1568), stream=stream0)
        del div_16
        del primals_57
        buf258 = reinterpret_tensor(buf249, (768, 13), (1, 768), 0); del buf249  # reuse
        buf260 = buf224; del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf252, mul_45, buf258, buf260, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_45
        buf259 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf258, buf259, 768, 13, grid=grid(768), stream=stream0)
        buf261 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf260, buf261, 768, 13, grid=grid(768), stream=stream0)
        buf263 = reinterpret_tensor(buf252, (6144, 196), (196, 1), 0); del buf252  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf262, buf263, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf264 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf263, permute_213, out=buf264)
        del permute_213
        buf265 = empty((196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf263, (196, 6144), (1, 196), 0), view_35, out=buf265)
        del view_35
        buf266 = reinterpret_tensor(buf241, (1, 196, 48), (9408, 1, 196), 0); del buf241  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf263, buf266, 9408, 128, grid=grid(9408), stream=stream0)
        buf267 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf266, buf267, 196, 48, grid=grid(196), stream=stream0)
        buf268 = buf234; del buf234  # reuse
        # Source Nodes: [x_60, x_61], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_red_fused_add_gelu_gelu_backward_sum_14.run(buf264, mm_4, primals_54, buf268, 18432, 128, grid=grid(18432), stream=stream0)
        buf269 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_60, x_61], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_per_fused_add_gelu_gelu_backward_sum_15.run(buf268, buf269, 384, 48, grid=grid(384), stream=stream0)
        buf270 = reinterpret_tensor(buf264, (8, 768, 384), (294912, 384, 1), 0); del buf264  # reuse
        # Source Nodes: [x_60, x_61], Original ATen: [aten.add, aten.gelu, aten.gelu_backward]
        triton_poi_fused_add_gelu_gelu_backward_16.run(buf270, mm_4, primals_54, 2359296, grid=grid(2359296), stream=stream0)
        del mm_4
        del primals_54
        buf271 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf270, (384, 6144), (1, 384), 0), view_33, out=buf271)
        del view_33
        buf272 = buf263; del buf263  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf270, (6144, 384), (384, 1), 0), permute_219, out=buf272)
        del permute_219
        buf273 = reinterpret_tensor(buf266, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf266  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf272, primals_51, buf273, 9408, 128, grid=grid(9408), stream=stream0)
        buf274 = buf242; del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf273, buf274, 1568, 6, grid=grid(1568), stream=stream0)
        buf275 = reinterpret_tensor(buf273, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf273  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf272, primals_51, mul_40, buf275, 9408, 128, grid=grid(9408), stream=stream0)
        buf276 = buf240; del buf240  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf275, buf276, 1568, 6, grid=grid(1568), stream=stream0)
        buf277 = reinterpret_tensor(buf260, (768, 13), (13, 1), 0); del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf272, mul_40, buf277, 9984, 121, grid=grid(9984), stream=stream0)
        buf278 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf277, buf278, 768, 13, grid=grid(768), stream=stream0)
        buf279 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf272, buf279, 768, 1568, grid=grid(768), stream=stream0)
        buf280 = buf262; del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf280, div_17, buf272, primals_51, buf274, mul_40, buf276, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del div_17
        del mul_40
        del primals_51
        buf281 = reinterpret_tensor(buf251, (1568, 3072), (3072, 1), 0); del buf251  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf280, (1568, 768), (768, 1), 0), permute_222, out=buf281)
        del permute_222
        buf282 = reinterpret_tensor(buf270, (768, 3072), (3072, 1), 0); del buf270  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf280, (768, 1568), (1, 768), 0), view_31, out=buf282)
        del view_31
        buf283 = reinterpret_tensor(buf277, (1, 768, 13), (9984, 1, 768), 0); del buf277  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf280, buf283, 9984, 121, grid=grid(9984), stream=stream0)
        buf284 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf283, buf284, 768, 13, grid=grid(768), stream=stream0)
        buf285 = reinterpret_tensor(buf281, (8, 196, 3072), (602112, 3072, 1), 0); del buf281  # reuse
        # Source Nodes: [x_54], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf285, addmm_10, 4816896, grid=grid(4816896), stream=stream0)
        del addmm_10
        buf286 = reinterpret_tensor(buf272, (1568, 768), (768, 1), 0); del buf272  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf285, (1568, 3072), (3072, 1), 0), permute_226, out=buf286)
        del permute_226
        buf287 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf285, (3072, 1568), (1, 3072), 0), view_29, out=buf287)
        del view_29
        buf288 = buf254; del buf254  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf285, buf288, 39936, 121, grid=grid(39936), stream=stream0)
        buf289 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf288, buf289, 3072, 13, grid=grid(3072), stream=stream0)
        buf296 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf296, buf286, primals_45, mul_35, div_18, 1568, 768, grid=grid(1568), stream=stream0)
        del div_18
        del primals_45
        buf292 = reinterpret_tensor(buf283, (768, 13), (1, 768), 0); del buf283  # reuse
        buf294 = buf258; del buf258  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf286, mul_35, buf292, buf294, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_35
        buf293 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf292, buf293, 768, 13, grid=grid(768), stream=stream0)
        buf295 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf294, buf295, 768, 13, grid=grid(768), stream=stream0)
        buf297 = reinterpret_tensor(buf286, (6144, 196), (196, 1), 0); del buf286  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf296, buf297, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf298 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf297, permute_231, out=buf298)
        del permute_231
        buf299 = empty((196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf297, (196, 6144), (1, 196), 0), view_27, out=buf299)
        del view_27
        buf300 = reinterpret_tensor(buf275, (1, 196, 48), (9408, 1, 196), 0); del buf275  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf297, buf300, 9408, 128, grid=grid(9408), stream=stream0)
        buf301 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf300, buf301, 196, 48, grid=grid(196), stream=stream0)
        buf302 = buf268; del buf268  # reuse
        # Source Nodes: [x_46, x_47], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_red_fused_add_gelu_gelu_backward_sum_14.run(buf298, mm_3, primals_42, buf302, 18432, 128, grid=grid(18432), stream=stream0)
        buf303 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_46, x_47], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_per_fused_add_gelu_gelu_backward_sum_15.run(buf302, buf303, 384, 48, grid=grid(384), stream=stream0)
        buf304 = reinterpret_tensor(buf298, (8, 768, 384), (294912, 384, 1), 0); del buf298  # reuse
        # Source Nodes: [x_46, x_47], Original ATen: [aten.add, aten.gelu, aten.gelu_backward]
        triton_poi_fused_add_gelu_gelu_backward_16.run(buf304, mm_3, primals_42, 2359296, grid=grid(2359296), stream=stream0)
        del mm_3
        del primals_42
        buf305 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (384, 6144), (1, 384), 0), view_25, out=buf305)
        del view_25
        buf306 = buf297; del buf297  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (6144, 384), (384, 1), 0), permute_237, out=buf306)
        del permute_237
        buf307 = reinterpret_tensor(buf300, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf300  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf306, primals_39, buf307, 9408, 128, grid=grid(9408), stream=stream0)
        buf308 = buf276; del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf307, buf308, 1568, 6, grid=grid(1568), stream=stream0)
        buf309 = reinterpret_tensor(buf307, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf306, primals_39, mul_30, buf309, 9408, 128, grid=grid(9408), stream=stream0)
        buf310 = buf274; del buf274  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf309, buf310, 1568, 6, grid=grid(1568), stream=stream0)
        buf311 = reinterpret_tensor(buf294, (768, 13), (13, 1), 0); del buf294  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf306, mul_30, buf311, 9984, 121, grid=grid(9984), stream=stream0)
        buf312 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf311, buf312, 768, 13, grid=grid(768), stream=stream0)
        buf313 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf306, buf313, 768, 1568, grid=grid(768), stream=stream0)
        buf314 = buf296; del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf314, div_19, buf306, primals_39, buf308, mul_30, buf310, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del div_19
        del mul_30
        del primals_39
        buf315 = reinterpret_tensor(buf285, (1568, 3072), (3072, 1), 0); del buf285  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf314, (1568, 768), (768, 1), 0), permute_240, out=buf315)
        del permute_240
        buf316 = reinterpret_tensor(buf304, (768, 3072), (3072, 1), 0); del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf314, (768, 1568), (1, 768), 0), view_23, out=buf316)
        del view_23
        buf317 = reinterpret_tensor(buf311, (1, 768, 13), (9984, 1, 768), 0); del buf311  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf314, buf317, 9984, 121, grid=grid(9984), stream=stream0)
        buf318 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf317, buf318, 768, 13, grid=grid(768), stream=stream0)
        buf319 = reinterpret_tensor(buf315, (8, 196, 3072), (602112, 3072, 1), 0); del buf315  # reuse
        # Source Nodes: [x_40], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf319, addmm_7, 4816896, grid=grid(4816896), stream=stream0)
        del addmm_7
        buf320 = reinterpret_tensor(buf306, (1568, 768), (768, 1), 0); del buf306  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf319, (1568, 3072), (3072, 1), 0), permute_244, out=buf320)
        del permute_244
        buf321 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf319, (3072, 1568), (1, 3072), 0), view_21, out=buf321)
        del view_21
        buf322 = buf288; del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf319, buf322, 39936, 121, grid=grid(39936), stream=stream0)
        buf323 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf322, buf323, 3072, 13, grid=grid(3072), stream=stream0)
        buf330 = buf314; del buf314  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf330, buf320, primals_33, mul_25, div_20, 1568, 768, grid=grid(1568), stream=stream0)
        del div_20
        del primals_33
        buf326 = reinterpret_tensor(buf317, (768, 13), (1, 768), 0); del buf317  # reuse
        buf328 = buf292; del buf292  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf320, mul_25, buf326, buf328, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_25
        buf327 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf326, buf327, 768, 13, grid=grid(768), stream=stream0)
        buf329 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf328, buf329, 768, 13, grid=grid(768), stream=stream0)
        buf331 = reinterpret_tensor(buf320, (6144, 196), (196, 1), 0); del buf320  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf330, buf331, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf332 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf331, permute_249, out=buf332)
        del permute_249
        buf333 = empty((196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf331, (196, 6144), (1, 196), 0), view_19, out=buf333)
        del view_19
        buf334 = reinterpret_tensor(buf309, (1, 196, 48), (9408, 1, 196), 0); del buf309  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf331, buf334, 9408, 128, grid=grid(9408), stream=stream0)
        buf335 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf334, buf335, 196, 48, grid=grid(196), stream=stream0)
        buf336 = buf302; del buf302  # reuse
        # Source Nodes: [x_32, x_33], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_red_fused_add_gelu_gelu_backward_sum_14.run(buf332, mm_2, primals_30, buf336, 18432, 128, grid=grid(18432), stream=stream0)
        buf337 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32, x_33], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_per_fused_add_gelu_gelu_backward_sum_15.run(buf336, buf337, 384, 48, grid=grid(384), stream=stream0)
        buf338 = reinterpret_tensor(buf332, (8, 768, 384), (294912, 384, 1), 0); del buf332  # reuse
        # Source Nodes: [x_32, x_33], Original ATen: [aten.add, aten.gelu, aten.gelu_backward]
        triton_poi_fused_add_gelu_gelu_backward_16.run(buf338, mm_2, primals_30, 2359296, grid=grid(2359296), stream=stream0)
        del mm_2
        del primals_30
        buf339 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf338, (384, 6144), (1, 384), 0), view_17, out=buf339)
        del view_17
        buf340 = buf331; del buf331  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf338, (6144, 384), (384, 1), 0), permute_255, out=buf340)
        del permute_255
        buf341 = reinterpret_tensor(buf334, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf334  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf340, primals_27, buf341, 9408, 128, grid=grid(9408), stream=stream0)
        buf342 = buf310; del buf310  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf341, buf342, 1568, 6, grid=grid(1568), stream=stream0)
        buf343 = reinterpret_tensor(buf341, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf341  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf340, primals_27, mul_20, buf343, 9408, 128, grid=grid(9408), stream=stream0)
        buf344 = buf308; del buf308  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf343, buf344, 1568, 6, grid=grid(1568), stream=stream0)
        buf345 = reinterpret_tensor(buf328, (768, 13), (13, 1), 0); del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf340, mul_20, buf345, 9984, 121, grid=grid(9984), stream=stream0)
        buf346 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf345, buf346, 768, 13, grid=grid(768), stream=stream0)
        buf347 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf340, buf347, 768, 1568, grid=grid(768), stream=stream0)
        buf348 = buf330; del buf330  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf348, div_21, buf340, primals_27, buf342, mul_20, buf344, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del div_21
        del mul_20
        del primals_27
        buf349 = reinterpret_tensor(buf319, (1568, 3072), (3072, 1), 0); del buf319  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf348, (1568, 768), (768, 1), 0), permute_258, out=buf349)
        del permute_258
        buf350 = reinterpret_tensor(buf338, (768, 3072), (3072, 1), 0); del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf348, (768, 1568), (1, 768), 0), view_15, out=buf350)
        del view_15
        buf351 = reinterpret_tensor(buf345, (1, 768, 13), (9984, 1, 768), 0); del buf345  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf348, buf351, 9984, 121, grid=grid(9984), stream=stream0)
        buf352 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf351, buf352, 768, 13, grid=grid(768), stream=stream0)
        buf353 = reinterpret_tensor(buf349, (8, 196, 3072), (602112, 3072, 1), 0); del buf349  # reuse
        # Source Nodes: [x_26], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf353, addmm_4, 4816896, grid=grid(4816896), stream=stream0)
        del addmm_4
        buf354 = reinterpret_tensor(buf340, (1568, 768), (768, 1), 0); del buf340  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf353, (1568, 3072), (3072, 1), 0), permute_262, out=buf354)
        del permute_262
        buf355 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf353, (3072, 1568), (1, 3072), 0), view_13, out=buf355)
        del view_13
        buf356 = buf322; del buf322  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf353, buf356, 39936, 121, grid=grid(39936), stream=stream0)
        buf357 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf356, buf357, 3072, 13, grid=grid(3072), stream=stream0)
        buf364 = buf348; del buf348  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf364, buf354, primals_21, mul_15, div_22, 1568, 768, grid=grid(1568), stream=stream0)
        del div_22
        del primals_21
        buf360 = reinterpret_tensor(buf351, (768, 13), (1, 768), 0); del buf351  # reuse
        buf362 = buf326; del buf326  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf354, mul_15, buf360, buf362, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_15
        buf361 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf360, buf361, 768, 13, grid=grid(768), stream=stream0)
        buf363 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf362, buf363, 768, 13, grid=grid(768), stream=stream0)
        buf365 = reinterpret_tensor(buf354, (6144, 196), (196, 1), 0); del buf354  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf364, buf365, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf366 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf365, permute_267, out=buf366)
        del permute_267
        buf367 = empty((196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf365, (196, 6144), (1, 196), 0), view_11, out=buf367)
        del view_11
        buf368 = reinterpret_tensor(buf343, (1, 196, 48), (9408, 1, 196), 0); del buf343  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf365, buf368, 9408, 128, grid=grid(9408), stream=stream0)
        buf369 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf368, buf369, 196, 48, grid=grid(196), stream=stream0)
        buf370 = buf336; del buf336  # reuse
        # Source Nodes: [x_18, x_19], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_red_fused_add_gelu_gelu_backward_sum_14.run(buf366, mm_1, primals_18, buf370, 18432, 128, grid=grid(18432), stream=stream0)
        buf371 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_18, x_19], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_per_fused_add_gelu_gelu_backward_sum_15.run(buf370, buf371, 384, 48, grid=grid(384), stream=stream0)
        buf372 = reinterpret_tensor(buf366, (8, 768, 384), (294912, 384, 1), 0); del buf366  # reuse
        # Source Nodes: [x_18, x_19], Original ATen: [aten.add, aten.gelu, aten.gelu_backward]
        triton_poi_fused_add_gelu_gelu_backward_16.run(buf372, mm_1, primals_18, 2359296, grid=grid(2359296), stream=stream0)
        del mm_1
        del primals_18
        buf373 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf372, (384, 6144), (1, 384), 0), view_9, out=buf373)
        del view_9
        buf374 = buf365; del buf365  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf372, (6144, 384), (384, 1), 0), permute_273, out=buf374)
        del permute_273
        buf375 = reinterpret_tensor(buf368, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf368  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf374, primals_15, buf375, 9408, 128, grid=grid(9408), stream=stream0)
        buf376 = buf344; del buf344  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf375, buf376, 1568, 6, grid=grid(1568), stream=stream0)
        buf377 = reinterpret_tensor(buf375, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf374, primals_15, mul_10, buf377, 9408, 128, grid=grid(9408), stream=stream0)
        buf378 = buf342; del buf342  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf377, buf378, 1568, 6, grid=grid(1568), stream=stream0)
        buf379 = reinterpret_tensor(buf362, (768, 13), (13, 1), 0); del buf362  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf374, mul_10, buf379, 9984, 121, grid=grid(9984), stream=stream0)
        buf380 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf379, buf380, 768, 13, grid=grid(768), stream=stream0)
        buf381 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf374, buf381, 768, 1568, grid=grid(768), stream=stream0)
        buf382 = buf364; del buf364  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf382, div_23, buf374, primals_15, buf376, mul_10, buf378, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del div_23
        del mul_10
        del primals_15
        buf383 = reinterpret_tensor(buf353, (1568, 3072), (3072, 1), 0); del buf353  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf382, (1568, 768), (768, 1), 0), permute_276, out=buf383)
        del permute_276
        buf384 = reinterpret_tensor(buf372, (768, 3072), (3072, 1), 0); del buf372  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf382, (768, 1568), (1, 768), 0), view_7, out=buf384)
        del view_7
        buf385 = reinterpret_tensor(buf379, (1, 768, 13), (9984, 1, 768), 0); del buf379  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf382, buf385, 9984, 121, grid=grid(9984), stream=stream0)
        buf386 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf385, buf386, 768, 13, grid=grid(768), stream=stream0)
        buf387 = reinterpret_tensor(buf383, (8, 196, 3072), (602112, 3072, 1), 0); del buf383  # reuse
        # Source Nodes: [x_12], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf387, addmm_1, 4816896, grid=grid(4816896), stream=stream0)
        del addmm_1
        buf388 = reinterpret_tensor(buf374, (1568, 768), (768, 1), 0); del buf374  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf387, (1568, 3072), (3072, 1), 0), permute_280, out=buf388)
        del permute_280
        buf389 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf387, (3072, 1568), (1, 3072), 0), view_5, out=buf389)
        del view_5
        buf390 = buf356; del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf387, buf390, 39936, 121, grid=grid(39936), stream=stream0)
        del buf387
        buf391 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf390, buf391, 3072, 13, grid=grid(3072), stream=stream0)
        del buf390
        buf398 = buf382; del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf398, buf388, primals_9, mul_5, div_24, 1568, 768, grid=grid(1568), stream=stream0)
        del div_24
        del primals_9
        buf394 = reinterpret_tensor(buf385, (768, 13), (1, 768), 0); del buf385  # reuse
        buf396 = buf360; del buf360  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf388, mul_5, buf394, buf396, 9984, 121, grid=grid(9984), stream=stream0)
        del mul_5
        buf395 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf394, buf395, 768, 13, grid=grid(768), stream=stream0)
        del buf394
        buf397 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf396, buf397, 768, 13, grid=grid(768), stream=stream0)
        buf399 = reinterpret_tensor(buf388, (6144, 196), (196, 1), 0); del buf388  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf398, buf399, 6144, 196, grid=grid(6144, 196), stream=stream0)
        buf400 = empty((6144, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf399, permute_285, out=buf400)
        del permute_285
        buf401 = empty((196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf399, (196, 6144), (1, 196), 0), view_3, out=buf401)
        del view_3
        buf402 = reinterpret_tensor(buf377, (1, 196, 48), (9408, 1, 196), 0); del buf377  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf399, buf402, 9408, 128, grid=grid(9408), stream=stream0)
        buf403 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf402, buf403, 196, 48, grid=grid(196), stream=stream0)
        buf404 = buf370; del buf370  # reuse
        # Source Nodes: [x_4, x_5], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_red_fused_add_gelu_gelu_backward_sum_14.run(buf400, mm, primals_6, buf404, 18432, 128, grid=grid(18432), stream=stream0)
        buf405 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_4, x_5], Original ATen: [aten.add, aten.gelu, aten.gelu_backward, aten.sum]
        triton_per_fused_add_gelu_gelu_backward_sum_15.run(buf404, buf405, 384, 48, grid=grid(384), stream=stream0)
        del buf404
        buf406 = reinterpret_tensor(buf400, (8, 768, 384), (294912, 384, 1), 0); del buf400  # reuse
        # Source Nodes: [x_4, x_5], Original ATen: [aten.add, aten.gelu, aten.gelu_backward]
        triton_poi_fused_add_gelu_gelu_backward_16.run(buf406, mm, primals_6, 2359296, grid=grid(2359296), stream=stream0)
        del mm
        del primals_6
        buf407 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf406, (384, 6144), (1, 384), 0), view_1, out=buf407)
        del view_1
        buf408 = buf399; del buf399  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf406, (6144, 384), (384, 1), 0), permute_291, out=buf408)
        del buf406
        del permute_291
        buf409 = reinterpret_tensor(buf402, (8, 196, 1, 6), (1176, 1, 9408, 196), 0); del buf402  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf408, primals_3, buf409, 9408, 128, grid=grid(9408), stream=stream0)
        buf410 = buf378; del buf378  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf409, buf410, 1568, 6, grid=grid(1568), stream=stream0)
        buf411 = reinterpret_tensor(buf409, (8, 196, 1, 6), (1176, 6, 9408, 1), 0); del buf409  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf408, primals_3, mul, buf411, 9408, 128, grid=grid(9408), stream=stream0)
        buf412 = buf376; del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf411, buf412, 1568, 6, grid=grid(1568), stream=stream0)
        del buf411
        buf413 = reinterpret_tensor(buf396, (768, 13), (13, 1), 0); del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf408, mul, buf413, 9984, 121, grid=grid(9984), stream=stream0)
        buf414 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf413, buf414, 768, 13, grid=grid(768), stream=stream0)
        buf415 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf408, buf415, 768, 1568, grid=grid(768), stream=stream0)
        buf416 = buf398; del buf398  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf416, div_25, buf408, primals_3, buf410, mul, buf412, 1568, 768, grid=grid(1568, 768), stream=stream0)
        del buf408
        del buf410
        del buf412
        del div_25
        del mul
        del primals_3
        buf417 = reinterpret_tensor(buf413, (768, 13), (1, 768), 0); del buf413  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_25.run(buf416, buf417, 9984, 121, grid=grid(9984), stream=stream0)
        buf418 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf417, buf418, 768, 13, grid=grid(768), stream=stream0)
        del buf417
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf419 = aten.convolution_backward(reinterpret_tensor(buf416, (8, 768, 14, 14), (150528, 1, 10752, 768), 0), primals_151, primals_1, [768], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf416
        del primals_1
        del primals_151
        buf420 = buf419[1]
        return (buf420, buf418, buf414, buf415, reinterpret_tensor(buf407, (384, 196), (196, 1), 0), reinterpret_tensor(buf405, (384, ), (1, ), 0), reinterpret_tensor(buf401, (196, 384), (384, 1), 0), reinterpret_tensor(buf403, (196, ), (1, ), 0), buf395, buf397, reinterpret_tensor(buf389, (3072, 768), (768, 1), 0), reinterpret_tensor(buf391, (3072, ), (1, ), 0), reinterpret_tensor(buf384, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf386, (768, ), (1, ), 0), buf380, buf381, reinterpret_tensor(buf373, (384, 196), (196, 1), 0), reinterpret_tensor(buf371, (384, ), (1, ), 0), reinterpret_tensor(buf367, (196, 384), (384, 1), 0), reinterpret_tensor(buf369, (196, ), (1, ), 0), buf361, buf363, reinterpret_tensor(buf355, (3072, 768), (768, 1), 0), reinterpret_tensor(buf357, (3072, ), (1, ), 0), reinterpret_tensor(buf350, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf352, (768, ), (1, ), 0), buf346, buf347, reinterpret_tensor(buf339, (384, 196), (196, 1), 0), reinterpret_tensor(buf337, (384, ), (1, ), 0), reinterpret_tensor(buf333, (196, 384), (384, 1), 0), reinterpret_tensor(buf335, (196, ), (1, ), 0), buf327, buf329, reinterpret_tensor(buf321, (3072, 768), (768, 1), 0), reinterpret_tensor(buf323, (3072, ), (1, ), 0), reinterpret_tensor(buf316, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf318, (768, ), (1, ), 0), buf312, buf313, reinterpret_tensor(buf305, (384, 196), (196, 1), 0), reinterpret_tensor(buf303, (384, ), (1, ), 0), reinterpret_tensor(buf299, (196, 384), (384, 1), 0), reinterpret_tensor(buf301, (196, ), (1, ), 0), buf293, buf295, reinterpret_tensor(buf287, (3072, 768), (768, 1), 0), reinterpret_tensor(buf289, (3072, ), (1, ), 0), reinterpret_tensor(buf282, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf284, (768, ), (1, ), 0), buf278, buf279, reinterpret_tensor(buf271, (384, 196), (196, 1), 0), reinterpret_tensor(buf269, (384, ), (1, ), 0), reinterpret_tensor(buf265, (196, 384), (384, 1), 0), reinterpret_tensor(buf267, (196, ), (1, ), 0), buf259, buf261, reinterpret_tensor(buf253, (3072, 768), (768, 1), 0), reinterpret_tensor(buf255, (3072, ), (1, ), 0), reinterpret_tensor(buf248, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf250, (768, ), (1, ), 0), buf244, buf245, reinterpret_tensor(buf237, (384, 196), (196, 1), 0), reinterpret_tensor(buf235, (384, ), (1, ), 0), reinterpret_tensor(buf231, (196, 384), (384, 1), 0), reinterpret_tensor(buf233, (196, ), (1, ), 0), buf225, buf227, reinterpret_tensor(buf219, (3072, 768), (768, 1), 0), reinterpret_tensor(buf221, (3072, ), (1, ), 0), reinterpret_tensor(buf214, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf216, (768, ), (1, ), 0), buf210, buf211, reinterpret_tensor(buf203, (384, 196), (196, 1), 0), reinterpret_tensor(buf201, (384, ), (1, ), 0), reinterpret_tensor(buf197, (196, 384), (384, 1), 0), reinterpret_tensor(buf199, (196, ), (1, ), 0), buf191, buf193, reinterpret_tensor(buf185, (3072, 768), (768, 1), 0), reinterpret_tensor(buf187, (3072, ), (1, ), 0), reinterpret_tensor(buf180, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf182, (768, ), (1, ), 0), buf176, buf177, reinterpret_tensor(buf169, (384, 196), (196, 1), 0), reinterpret_tensor(buf167, (384, ), (1, ), 0), reinterpret_tensor(buf163, (196, 384), (384, 1), 0), reinterpret_tensor(buf165, (196, ), (1, ), 0), buf157, buf159, reinterpret_tensor(buf151, (3072, 768), (768, 1), 0), reinterpret_tensor(buf153, (3072, ), (1, ), 0), reinterpret_tensor(buf146, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf148, (768, ), (1, ), 0), buf142, buf143, reinterpret_tensor(buf135, (384, 196), (196, 1), 0), reinterpret_tensor(buf133, (384, ), (1, ), 0), reinterpret_tensor(buf129, (196, 384), (384, 1), 0), reinterpret_tensor(buf131, (196, ), (1, ), 0), buf123, buf125, reinterpret_tensor(buf117, (3072, 768), (768, 1), 0), reinterpret_tensor(buf119, (3072, ), (1, ), 0), reinterpret_tensor(buf112, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf114, (768, ), (1, ), 0), buf108, buf109, reinterpret_tensor(buf101, (384, 196), (196, 1), 0), reinterpret_tensor(buf99, (384, ), (1, ), 0), reinterpret_tensor(buf95, (196, 384), (384, 1), 0), reinterpret_tensor(buf97, (196, ), (1, ), 0), buf89, buf91, reinterpret_tensor(buf83, (3072, 768), (768, 1), 0), reinterpret_tensor(buf85, (3072, ), (1, ), 0), reinterpret_tensor(buf78, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf80, (768, ), (1, ), 0), buf74, buf75, reinterpret_tensor(buf67, (384, 196), (196, 1), 0), reinterpret_tensor(buf65, (384, ), (1, ), 0), reinterpret_tensor(buf61, (196, 384), (384, 1), 0), reinterpret_tensor(buf63, (196, ), (1, ), 0), buf55, buf57, reinterpret_tensor(buf49, (3072, 768), (768, 1), 0), reinterpret_tensor(buf51, (3072, ), (1, ), 0), reinterpret_tensor(buf44, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf46, (768, ), (1, ), 0), buf40, buf41, reinterpret_tensor(buf33, (384, 196), (196, 1), 0), reinterpret_tensor(buf31, (384, ), (1, ), 0), reinterpret_tensor(buf27, (196, 384), (384, 1), 0), reinterpret_tensor(buf29, (196, ), (1, ), 0), buf21, buf23, reinterpret_tensor(buf15, (3072, 768), (768, 1), 0), reinterpret_tensor(buf17, (3072, ), (1, ), 0), reinterpret_tensor(buf10, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf12, (768, ), (1, ), 0), buf7, buf8, reinterpret_tensor(buf1, (1000, 768), (768, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((768, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    mul = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_1 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm = rand_strided((6144, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_3 = rand_strided((6144, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_5 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_5 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_1 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_7 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_10 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_9 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_1 = rand_strided((6144, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_11 = rand_strided((6144, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_15 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_13 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_4 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_15 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_20 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_17 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_2 = rand_strided((6144, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_19 = rand_strided((6144, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_25 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_21 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_7 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_23 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_30 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_25 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_3 = rand_strided((6144, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_27 = rand_strided((6144, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_35 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_29 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_31 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_40 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_33 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_4 = rand_strided((6144, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_35 = rand_strided((6144, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_45 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_37 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_13 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_39 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_50 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_41 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_5 = rand_strided((6144, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_43 = rand_strided((6144, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_55 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_45 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_16 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_47 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_60 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_49 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_6 = rand_strided((6144, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_51 = rand_strided((6144, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_65 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_53 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_19 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_55 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_70 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_57 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_7 = rand_strided((6144, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_59 = rand_strided((6144, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_75 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_61 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_63 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_80 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_65 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_8 = rand_strided((6144, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_67 = rand_strided((6144, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_85 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_69 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_25 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_71 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_90 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_73 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_9 = rand_strided((6144, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_75 = rand_strided((6144, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_95 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_77 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_28 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_79 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_100 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_81 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_10 = rand_strided((6144, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_83 = rand_strided((6144, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_105 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_85 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_31 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_87 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_110 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_89 = rand_strided((6144, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    mm_11 = rand_strided((6144, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_91 = rand_strided((6144, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_115 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    view_93 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_95 = rand_strided((1568, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    mul_120 = rand_strided((8, 196, 768), (150528, 768, 1), device='cuda:0', dtype=torch.float32)
    clone_85 = rand_strided((8, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_74 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_1 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_78 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_82 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_2 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_87 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_93 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_96 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_100 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_4 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_105 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_111 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_5 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_114 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_118 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_6 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_123 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_129 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_7 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_132 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_136 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_8 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_141 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_147 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_9 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_150 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_154 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_10 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_159 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_165 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_11 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_168 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_172 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_12 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_177 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_183 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_13 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_186 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_190 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_14 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_195 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_201 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_15 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_204 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_208 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_16 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_213 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_219 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_17 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_222 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_226 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_18 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_231 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_237 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_19 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_240 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_244 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_20 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_249 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_255 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_21 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_258 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_262 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_22 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_267 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_273 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_23 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_276 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_280 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_24 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_285 = rand_strided((196, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_291 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_25 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_6, primals_9, primals_15, primals_18, primals_21, primals_27, primals_30, primals_33, primals_39, primals_42, primals_45, primals_51, primals_54, primals_57, primals_63, primals_66, primals_69, primals_75, primals_78, primals_81, primals_87, primals_90, primals_93, primals_99, primals_102, primals_105, primals_111, primals_114, primals_117, primals_123, primals_126, primals_129, primals_135, primals_138, primals_141, primals_147, primals_151, mul, view_1, mm, view_3, mul_5, view_5, addmm_1, view_7, mul_10, view_9, mm_1, view_11, mul_15, view_13, addmm_4, view_15, mul_20, view_17, mm_2, view_19, mul_25, view_21, addmm_7, view_23, mul_30, view_25, mm_3, view_27, mul_35, view_29, addmm_10, view_31, mul_40, view_33, mm_4, view_35, mul_45, view_37, addmm_13, view_39, mul_50, view_41, mm_5, view_43, mul_55, view_45, addmm_16, view_47, mul_60, view_49, mm_6, view_51, mul_65, view_53, addmm_19, view_55, mul_70, view_57, mm_7, view_59, mul_75, view_61, addmm_22, view_63, mul_80, view_65, mm_8, view_67, mul_85, view_69, addmm_25, view_71, mul_90, view_73, mm_9, view_75, mul_95, view_77, addmm_28, view_79, mul_100, view_81, mm_10, view_83, mul_105, view_85, addmm_31, view_87, mul_110, view_89, mm_11, view_91, mul_115, view_93, addmm_34, view_95, mul_120, clone_85, permute_74, div_1, permute_78, permute_82, div_2, permute_87, permute_93, div_3, permute_96, permute_100, div_4, permute_105, permute_111, div_5, permute_114, permute_118, div_6, permute_123, permute_129, div_7, permute_132, permute_136, div_8, permute_141, permute_147, div_9, permute_150, permute_154, div_10, permute_159, permute_165, div_11, permute_168, permute_172, div_12, permute_177, permute_183, div_13, permute_186, permute_190, div_14, permute_195, permute_201, div_15, permute_204, permute_208, div_16, permute_213, permute_219, div_17, permute_222, permute_226, div_18, permute_231, permute_237, div_19, permute_240, permute_244, div_20, permute_249, permute_255, div_21, permute_258, permute_262, div_22, permute_267, permute_273, div_23, permute_276, permute_280, div_24, permute_285, permute_291, div_25, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mixer_b16_224', benchmark_compiled_module)
