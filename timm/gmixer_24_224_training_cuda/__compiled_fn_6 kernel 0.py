
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


# kernel path: /tmp/torchinductor_youkaichao/bi/cbiz56xfnw4dcftzjaxutrcllykl3derll7wlmrn6qechkgg7qji.py
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
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_native_layer_norm_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 384
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
        tmp0 = tl.load(in_ptr0 + (r2 + (384*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
        tmp14 = tl.load(in_ptr0 + (r2 + (384*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tl.load(in_ptr2 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = 196.0
        tmp16 = tmp14 / tmp15
        tmp18 = tmp16 * tmp17
        tmp19 = 384.0
        tmp20 = tmp18 * tmp19
        tmp21 = tmp20 - tmp6
        tmp23 = tmp22 * tmp11
        tmp24 = tmp21 - tmp23
        tmp25 = tmp13 * tmp24
        tl.store(out_ptr2 + (r2 + (384*x3)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/aw/caww2t5rzjzge23ldquh2ezybgkqpsm5cuazm25fxbae67skzrjt.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_native_layer_norm_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 196.0
        tmp5 = tmp3 / tmp4
        tmp6 = tl.load(in_ptr1 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tmp5 * tmp6
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/26/c26ujfoga3mzbgn65j4xbcij6unkuh6l6ou55hoc6sph7u5azkwj.py
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
    size_hints=[512, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_native_layer_norm_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 13
    RBLOCK: tl.constexpr = 16
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


# kernel path: /tmp/torchinductor_youkaichao/wa/cwaa2goulxqoupviav2jbim5fqesgo3c6crpcsnlzjtpwhalhbr6.py
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
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_native_layer_norm_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
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
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 196.0
        tmp2 = tmp0 / tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g4/cg4dywmcvcdedkownxpu2sxexamh7d6ow4qb54ysdftsodfov5ag.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
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
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*r2) + (46464*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/aq/caq2khzp7dibsbytdqorzvowvpc6krij2rgkl2zwsrdvumy4bqo2.py
# Source Nodes: [], Original ATen: [aten.cat]

triton_poi_fused_cat_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1536
    x1 = (xindex // 1536)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 768, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (768*x1)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x2), tmp4, other=0.0)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 1536, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr0 + ((-768) + x0 + (768*x1)), tmp12, other=0.0)
    tmp16 = tl.load(in_ptr2 + ((-768) + x2), tmp12, other=0.0)
    tmp17 = tmp15 * tmp16
    tmp18 = tl.load(in_ptr1 + ((-768) + x2), tmp12, other=0.0)
    tmp19 = tl.sigmoid(tmp18)
    tmp20 = 1.0
    tmp21 = tmp20 - tmp19
    tmp22 = tmp18 * tmp21
    tmp23 = tmp22 + tmp20
    tmp24 = tmp19 * tmp23
    tmp25 = tmp17 * tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp12, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp11, tmp27)
    tl.store(out_ptr0 + (x2), tmp28, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wc/cwc6z5ny66f7gw7dl3qql65wchu3p7gevbosnhlboym7jl3x7xd4.py
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_7', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/em/cempzdhbav2v6q5rzkzjfu4uosxdybqulbsjljo3il465b5efpcs.py
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
    size_hints=[2048, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_8', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/45/c45tatnebgb2abtjhcfmifx6n4sskrlcgbxtthgl5pvx3lxukzsj.py
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
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_9', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 384.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(in_out_ptr0 + (r1 + (384*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yp/cypkydf2zwokgayvrwktqpcp33jaor75hnsthhcoeg6f34tvonuu.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
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
        tmp3 = tl.load(in_ptr0 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/un/cunhfffr3izgejffma3pnfueua66nccltoy6gbe2snoic2swbmnp.py
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
    size_hints=[4096, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + ((384*x1) + (75264*(y0 // 384)) + (y0 % 384)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + (196*y0)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vu/cvu3b5gnhjp4h63jj5qk6aei4voucsxj7ukbwql5u5iakb7svv57.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
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


# kernel path: /tmp/torchinductor_youkaichao/yn/cyndilcytiyko5uaeh3ve3msn5p72ly7uvzlyick6shnuu66vw2i.py
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
    size_hints=[256, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 196
    rnumel = 24
    RBLOCK: tl.constexpr = 32
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


# kernel path: /tmp/torchinductor_youkaichao/4e/c4e63a7akdcvggktj6yzxpy26w5m7avadylsi4xllnxq4r5atvqg.py
# Source Nodes: [], Original ATen: [aten.cat]

triton_poi_fused_cat_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 384
    x1 = (xindex // 384)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 192, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (192*x1)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x2), tmp4, other=0.0)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 384, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr0 + ((-192) + x0 + (192*x1)), tmp12, other=0.0)
    tmp16 = tl.load(in_ptr2 + ((-192) + x2), tmp12, other=0.0)
    tmp17 = tmp15 * tmp16
    tmp18 = tl.load(in_ptr1 + ((-192) + x2), tmp12, other=0.0)
    tmp19 = tl.sigmoid(tmp18)
    tmp20 = 1.0
    tmp21 = tmp20 - tmp19
    tmp22 = tmp18 * tmp21
    tmp23 = tmp22 + tmp20
    tmp24 = tmp19 * tmp23
    tmp25 = tmp17 * tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp12, tmp25, tmp26)
    tmp28 = tl.where(tmp4, tmp11, tmp27)
    tl.store(out_ptr0 + (x2), tmp28, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7w/c7w3jnbfvoeceolzpanrtoxncep35otswux3u26r53srxu77inmm.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9216
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    x1 = (xindex // 384)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (49152*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ot/cotn2inccuhagkwnl7ozlq4mh4v4b2co6qvaornyaverkbuxsc4m.py
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
    size_hints=[512, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 24
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


# kernel path: /tmp/torchinductor_youkaichao/3t/c3tkkufwnzrukac3s77ra6p5tf7rg2qvfbownl2bslj7f4ch2qsy.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x4 = (xindex // 196)
    x1 = (xindex // 196) % 3
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


# kernel path: /tmp/torchinductor_youkaichao/tl/ctlsmahqcvd2gbtrectc5tuqwqxgt56rgpu2l6277jjnebm7zgrc.py
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
    size_hints=[2048, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 196
    x1 = (xindex // 196)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (588*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rr/crrza3shptdtszxvd4q4onqo3rjmnahtp6sajh33q2mok6a6ztgz.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3
    x1 = (xindex // 3) % 196
    x2 = (xindex // 588)
    x5 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (196*r3) + (25088*x0) + (75264*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ie/cienctmxhke2ea4nraup6zsgmvbcngdyazecasoqpofspmr6sfua.py
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
    size_hints=[2048, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (3*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oz/cozsmepqegokxs5s7hrpo6madaadkphdzs7lf5cmwgjsuidmv3ek.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
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
        tmp3 = tl.load(in_ptr0 + ((196*x1) + (75264*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (384*((r2 + (121*x0)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sp/cspkfu5vngmllr77gy6ymhb5sv77l5zu7bqjuhlx7hwr7ldhsbcz.py
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
    size_hints=[512, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
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


# kernel path: /tmp/torchinductor_youkaichao/qe/cqefo5ibnfkgkozx2f2rjp4ujhzovgmohhcos77cqebwrfgekvon.py
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
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (75264*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ox/coxgr4badhlrim5gk2vv6g52ui4uh4ia3eqpnfcbs7lzwghmywce.py
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
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_backward_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 384
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x2 + (384*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (y3), ymask, eviction_policy='evict_last')
    tmp4 = tmp2 * tmp3
    tmp5 = 384.0
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 - tmp7
    tmp11 = tmp9 * tmp10
    tmp12 = tmp8 - tmp11
    tmp13 = tmp1 * tmp12
    tmp14 = tmp0 + tmp13
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (384*y3)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hu/chuxr2fgshmte7dt4dgrqoh7y7msqpd6raumkrtkaehhnbmcw5zj.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
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
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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
    primals_1, primals_3, primals_9, primals_15, primals_21, primals_27, primals_33, primals_39, primals_45, primals_51, primals_57, primals_63, primals_69, primals_75, primals_81, primals_87, primals_93, primals_99, primals_105, primals_111, primals_117, primals_123, primals_129, primals_135, primals_141, primals_147, primals_153, primals_159, primals_165, primals_171, primals_177, primals_183, primals_189, primals_195, primals_201, primals_207, primals_213, primals_219, primals_225, primals_231, primals_237, primals_243, primals_249, primals_255, primals_261, primals_267, primals_273, primals_279, primals_285, primals_291, primals_295, mul, view_1, getitem_2, getitem_3, view_3, mul_4, view_5, getitem_6, getitem_7, view_7, mul_8, view_9, getitem_10, getitem_11, view_11, mul_12, view_13, getitem_14, getitem_15, view_15, mul_16, view_17, getitem_18, getitem_19, view_19, mul_20, view_21, getitem_22, getitem_23, view_23, mul_24, view_25, getitem_26, getitem_27, view_27, mul_28, view_29, getitem_30, getitem_31, view_31, mul_32, view_33, getitem_34, getitem_35, view_35, mul_36, view_37, getitem_38, getitem_39, view_39, mul_40, view_41, getitem_42, getitem_43, view_43, mul_44, view_45, getitem_46, getitem_47, view_47, mul_48, view_49, getitem_50, getitem_51, view_51, mul_52, view_53, getitem_54, getitem_55, view_55, mul_56, view_57, getitem_58, getitem_59, view_59, mul_60, view_61, getitem_62, getitem_63, view_63, mul_64, view_65, getitem_66, getitem_67, view_67, mul_68, view_69, getitem_70, getitem_71, view_71, mul_72, view_73, getitem_74, getitem_75, view_75, mul_76, view_77, getitem_78, getitem_79, view_79, mul_80, view_81, getitem_82, getitem_83, view_83, mul_84, view_85, getitem_86, getitem_87, view_87, mul_88, view_89, getitem_90, getitem_91, view_91, mul_92, view_93, getitem_94, getitem_95, view_95, mul_96, view_97, getitem_98, getitem_99, view_99, mul_100, view_101, getitem_102, getitem_103, view_103, mul_104, view_105, getitem_106, getitem_107, view_107, mul_108, view_109, getitem_110, getitem_111, view_111, mul_112, view_113, getitem_114, getitem_115, view_115, mul_116, view_117, getitem_118, getitem_119, view_119, mul_120, view_121, getitem_122, getitem_123, view_123, mul_124, view_125, getitem_126, getitem_127, view_127, mul_128, view_129, getitem_130, getitem_131, view_131, mul_132, view_133, getitem_134, getitem_135, view_135, mul_136, view_137, getitem_138, getitem_139, view_139, mul_140, view_141, getitem_142, getitem_143, view_143, mul_144, view_145, getitem_146, getitem_147, view_147, mul_148, view_149, getitem_150, getitem_151, view_151, mul_152, view_153, getitem_154, getitem_155, view_155, mul_156, view_157, getitem_158, getitem_159, view_159, mul_160, view_161, getitem_162, getitem_163, view_163, mul_164, view_165, getitem_166, getitem_167, view_167, mul_168, view_169, getitem_170, getitem_171, view_171, mul_172, view_173, getitem_174, getitem_175, view_175, mul_176, view_177, getitem_178, getitem_179, view_179, mul_180, view_181, getitem_182, getitem_183, view_183, mul_184, view_185, getitem_186, getitem_187, view_187, mul_188, view_189, getitem_190, getitem_191, view_191, mul_192, clone_169, permute_146, div_1, permute_150, permute_155, div_2, permute_160, permute_167, div_3, permute_170, permute_175, div_4, permute_180, permute_187, div_5, permute_190, permute_195, div_6, permute_200, permute_207, div_7, permute_210, permute_215, div_8, permute_220, permute_227, div_9, permute_230, permute_235, div_10, permute_240, permute_247, div_11, permute_250, permute_255, div_12, permute_260, permute_267, div_13, permute_270, permute_275, div_14, permute_280, permute_287, div_15, permute_290, permute_295, div_16, permute_300, permute_307, div_17, permute_310, permute_315, div_18, permute_320, permute_327, div_19, permute_330, permute_335, div_20, permute_340, permute_347, div_21, permute_350, permute_355, div_22, permute_360, permute_367, div_23, permute_370, permute_375, div_24, permute_380, permute_387, div_25, permute_390, permute_395, div_26, permute_400, permute_407, div_27, permute_410, permute_415, div_28, permute_420, permute_427, div_29, permute_430, permute_435, div_30, permute_440, permute_447, div_31, permute_450, permute_455, div_32, permute_460, permute_467, div_33, permute_470, permute_475, div_34, permute_480, permute_487, div_35, permute_490, permute_495, div_36, permute_500, permute_507, div_37, permute_510, permute_515, div_38, permute_520, permute_527, div_39, permute_530, permute_535, div_40, permute_540, permute_547, div_41, permute_550, permute_555, div_42, permute_560, permute_567, div_43, permute_570, permute_575, div_44, permute_580, permute_587, div_45, permute_590, permute_595, div_46, permute_600, permute_607, div_47, permute_610, permute_615, div_48, permute_620, permute_627, div_49, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (384, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(primals_3, (384, ), (1, ))
    assert_size_stride(primals_9, (384, ), (1, ))
    assert_size_stride(primals_15, (384, ), (1, ))
    assert_size_stride(primals_21, (384, ), (1, ))
    assert_size_stride(primals_27, (384, ), (1, ))
    assert_size_stride(primals_33, (384, ), (1, ))
    assert_size_stride(primals_39, (384, ), (1, ))
    assert_size_stride(primals_45, (384, ), (1, ))
    assert_size_stride(primals_51, (384, ), (1, ))
    assert_size_stride(primals_57, (384, ), (1, ))
    assert_size_stride(primals_63, (384, ), (1, ))
    assert_size_stride(primals_69, (384, ), (1, ))
    assert_size_stride(primals_75, (384, ), (1, ))
    assert_size_stride(primals_81, (384, ), (1, ))
    assert_size_stride(primals_87, (384, ), (1, ))
    assert_size_stride(primals_93, (384, ), (1, ))
    assert_size_stride(primals_99, (384, ), (1, ))
    assert_size_stride(primals_105, (384, ), (1, ))
    assert_size_stride(primals_111, (384, ), (1, ))
    assert_size_stride(primals_117, (384, ), (1, ))
    assert_size_stride(primals_123, (384, ), (1, ))
    assert_size_stride(primals_129, (384, ), (1, ))
    assert_size_stride(primals_135, (384, ), (1, ))
    assert_size_stride(primals_141, (384, ), (1, ))
    assert_size_stride(primals_147, (384, ), (1, ))
    assert_size_stride(primals_153, (384, ), (1, ))
    assert_size_stride(primals_159, (384, ), (1, ))
    assert_size_stride(primals_165, (384, ), (1, ))
    assert_size_stride(primals_171, (384, ), (1, ))
    assert_size_stride(primals_177, (384, ), (1, ))
    assert_size_stride(primals_183, (384, ), (1, ))
    assert_size_stride(primals_189, (384, ), (1, ))
    assert_size_stride(primals_195, (384, ), (1, ))
    assert_size_stride(primals_201, (384, ), (1, ))
    assert_size_stride(primals_207, (384, ), (1, ))
    assert_size_stride(primals_213, (384, ), (1, ))
    assert_size_stride(primals_219, (384, ), (1, ))
    assert_size_stride(primals_225, (384, ), (1, ))
    assert_size_stride(primals_231, (384, ), (1, ))
    assert_size_stride(primals_237, (384, ), (1, ))
    assert_size_stride(primals_243, (384, ), (1, ))
    assert_size_stride(primals_249, (384, ), (1, ))
    assert_size_stride(primals_255, (384, ), (1, ))
    assert_size_stride(primals_261, (384, ), (1, ))
    assert_size_stride(primals_267, (384, ), (1, ))
    assert_size_stride(primals_273, (384, ), (1, ))
    assert_size_stride(primals_279, (384, ), (1, ))
    assert_size_stride(primals_285, (384, ), (1, ))
    assert_size_stride(primals_291, (384, ), (1, ))
    assert_size_stride(primals_295, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(mul, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_1, (3072, 196), (196, 1))
    assert_size_stride(getitem_2, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_3, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_3, (3072, 192), (192, 1))
    assert_size_stride(mul_4, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_5, (1568, 384), (384, 1))
    assert_size_stride(getitem_6, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_7, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_7, (1568, 768), (768, 1))
    assert_size_stride(mul_8, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_9, (3072, 196), (196, 1))
    assert_size_stride(getitem_10, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_11, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_11, (3072, 192), (192, 1))
    assert_size_stride(mul_12, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_13, (1568, 384), (384, 1))
    assert_size_stride(getitem_14, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_15, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_15, (1568, 768), (768, 1))
    assert_size_stride(mul_16, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_17, (3072, 196), (196, 1))
    assert_size_stride(getitem_18, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_19, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_19, (3072, 192), (192, 1))
    assert_size_stride(mul_20, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_21, (1568, 384), (384, 1))
    assert_size_stride(getitem_22, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_23, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_23, (1568, 768), (768, 1))
    assert_size_stride(mul_24, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_25, (3072, 196), (196, 1))
    assert_size_stride(getitem_26, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_27, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_27, (3072, 192), (192, 1))
    assert_size_stride(mul_28, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_29, (1568, 384), (384, 1))
    assert_size_stride(getitem_30, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_31, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_31, (1568, 768), (768, 1))
    assert_size_stride(mul_32, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_33, (3072, 196), (196, 1))
    assert_size_stride(getitem_34, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_35, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_35, (3072, 192), (192, 1))
    assert_size_stride(mul_36, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_37, (1568, 384), (384, 1))
    assert_size_stride(getitem_38, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_39, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_39, (1568, 768), (768, 1))
    assert_size_stride(mul_40, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_41, (3072, 196), (196, 1))
    assert_size_stride(getitem_42, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_43, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_43, (3072, 192), (192, 1))
    assert_size_stride(mul_44, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_45, (1568, 384), (384, 1))
    assert_size_stride(getitem_46, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_47, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_47, (1568, 768), (768, 1))
    assert_size_stride(mul_48, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_49, (3072, 196), (196, 1))
    assert_size_stride(getitem_50, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_51, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_51, (3072, 192), (192, 1))
    assert_size_stride(mul_52, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_53, (1568, 384), (384, 1))
    assert_size_stride(getitem_54, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_55, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_55, (1568, 768), (768, 1))
    assert_size_stride(mul_56, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_57, (3072, 196), (196, 1))
    assert_size_stride(getitem_58, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_59, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_59, (3072, 192), (192, 1))
    assert_size_stride(mul_60, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_61, (1568, 384), (384, 1))
    assert_size_stride(getitem_62, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_63, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_63, (1568, 768), (768, 1))
    assert_size_stride(mul_64, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_65, (3072, 196), (196, 1))
    assert_size_stride(getitem_66, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_67, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_67, (3072, 192), (192, 1))
    assert_size_stride(mul_68, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_69, (1568, 384), (384, 1))
    assert_size_stride(getitem_70, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_71, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_71, (1568, 768), (768, 1))
    assert_size_stride(mul_72, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_73, (3072, 196), (196, 1))
    assert_size_stride(getitem_74, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_75, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_75, (3072, 192), (192, 1))
    assert_size_stride(mul_76, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_77, (1568, 384), (384, 1))
    assert_size_stride(getitem_78, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_79, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_79, (1568, 768), (768, 1))
    assert_size_stride(mul_80, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_81, (3072, 196), (196, 1))
    assert_size_stride(getitem_82, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_83, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_83, (3072, 192), (192, 1))
    assert_size_stride(mul_84, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_85, (1568, 384), (384, 1))
    assert_size_stride(getitem_86, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_87, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_87, (1568, 768), (768, 1))
    assert_size_stride(mul_88, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_89, (3072, 196), (196, 1))
    assert_size_stride(getitem_90, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_91, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_91, (3072, 192), (192, 1))
    assert_size_stride(mul_92, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_93, (1568, 384), (384, 1))
    assert_size_stride(getitem_94, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_95, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_95, (1568, 768), (768, 1))
    assert_size_stride(mul_96, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_97, (3072, 196), (196, 1))
    assert_size_stride(getitem_98, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_99, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_99, (3072, 192), (192, 1))
    assert_size_stride(mul_100, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_101, (1568, 384), (384, 1))
    assert_size_stride(getitem_102, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_103, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_103, (1568, 768), (768, 1))
    assert_size_stride(mul_104, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_105, (3072, 196), (196, 1))
    assert_size_stride(getitem_106, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_107, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_107, (3072, 192), (192, 1))
    assert_size_stride(mul_108, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_109, (1568, 384), (384, 1))
    assert_size_stride(getitem_110, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_111, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_111, (1568, 768), (768, 1))
    assert_size_stride(mul_112, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_113, (3072, 196), (196, 1))
    assert_size_stride(getitem_114, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_115, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_115, (3072, 192), (192, 1))
    assert_size_stride(mul_116, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_117, (1568, 384), (384, 1))
    assert_size_stride(getitem_118, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_119, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_119, (1568, 768), (768, 1))
    assert_size_stride(mul_120, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_121, (3072, 196), (196, 1))
    assert_size_stride(getitem_122, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_123, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_123, (3072, 192), (192, 1))
    assert_size_stride(mul_124, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_125, (1568, 384), (384, 1))
    assert_size_stride(getitem_126, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_127, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_127, (1568, 768), (768, 1))
    assert_size_stride(mul_128, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_129, (3072, 196), (196, 1))
    assert_size_stride(getitem_130, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_131, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_131, (3072, 192), (192, 1))
    assert_size_stride(mul_132, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_133, (1568, 384), (384, 1))
    assert_size_stride(getitem_134, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_135, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_135, (1568, 768), (768, 1))
    assert_size_stride(mul_136, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_137, (3072, 196), (196, 1))
    assert_size_stride(getitem_138, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_139, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_139, (3072, 192), (192, 1))
    assert_size_stride(mul_140, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_141, (1568, 384), (384, 1))
    assert_size_stride(getitem_142, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_143, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_143, (1568, 768), (768, 1))
    assert_size_stride(mul_144, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_145, (3072, 196), (196, 1))
    assert_size_stride(getitem_146, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_147, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_147, (3072, 192), (192, 1))
    assert_size_stride(mul_148, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_149, (1568, 384), (384, 1))
    assert_size_stride(getitem_150, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_151, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_151, (1568, 768), (768, 1))
    assert_size_stride(mul_152, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_153, (3072, 196), (196, 1))
    assert_size_stride(getitem_154, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_155, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_155, (3072, 192), (192, 1))
    assert_size_stride(mul_156, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_157, (1568, 384), (384, 1))
    assert_size_stride(getitem_158, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_159, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_159, (1568, 768), (768, 1))
    assert_size_stride(mul_160, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_161, (3072, 196), (196, 1))
    assert_size_stride(getitem_162, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_163, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_163, (3072, 192), (192, 1))
    assert_size_stride(mul_164, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_165, (1568, 384), (384, 1))
    assert_size_stride(getitem_166, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_167, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_167, (1568, 768), (768, 1))
    assert_size_stride(mul_168, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_169, (3072, 196), (196, 1))
    assert_size_stride(getitem_170, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_171, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_171, (3072, 192), (192, 1))
    assert_size_stride(mul_172, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_173, (1568, 384), (384, 1))
    assert_size_stride(getitem_174, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_175, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_175, (1568, 768), (768, 1))
    assert_size_stride(mul_176, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_177, (3072, 196), (196, 1))
    assert_size_stride(getitem_178, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_179, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_179, (3072, 192), (192, 1))
    assert_size_stride(mul_180, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_181, (1568, 384), (384, 1))
    assert_size_stride(getitem_182, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_183, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_183, (1568, 768), (768, 1))
    assert_size_stride(mul_184, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_185, (3072, 196), (196, 1))
    assert_size_stride(getitem_186, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(getitem_187, (8, 384, 192), (147456, 384, 1))
    assert_size_stride(view_187, (3072, 192), (192, 1))
    assert_size_stride(mul_188, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(view_189, (1568, 384), (384, 1))
    assert_size_stride(getitem_190, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(getitem_191, (8, 196, 768), (301056, 1536, 1))
    assert_size_stride(view_191, (1568, 768), (768, 1))
    assert_size_stride(mul_192, (8, 196, 384), (75264, 384, 1))
    assert_size_stride(clone_169, (8, 384), (384, 1))
    assert_size_stride(permute_146, (1000, 384), (384, 1))
    assert_size_stride(div_1, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_150, (384, 768), (768, 1))
    assert_size_stride(permute_155, (1536, 384), (384, 1))
    assert_size_stride(div_2, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_160, (196, 192), (192, 1))
    assert_size_stride(permute_167, (384, 196), (196, 1))
    assert_size_stride(div_3, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_170, (384, 768), (768, 1))
    assert_size_stride(permute_175, (1536, 384), (384, 1))
    assert_size_stride(div_4, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_180, (196, 192), (192, 1))
    assert_size_stride(permute_187, (384, 196), (196, 1))
    assert_size_stride(div_5, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_190, (384, 768), (768, 1))
    assert_size_stride(permute_195, (1536, 384), (384, 1))
    assert_size_stride(div_6, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_200, (196, 192), (192, 1))
    assert_size_stride(permute_207, (384, 196), (196, 1))
    assert_size_stride(div_7, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_210, (384, 768), (768, 1))
    assert_size_stride(permute_215, (1536, 384), (384, 1))
    assert_size_stride(div_8, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_220, (196, 192), (192, 1))
    assert_size_stride(permute_227, (384, 196), (196, 1))
    assert_size_stride(div_9, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_230, (384, 768), (768, 1))
    assert_size_stride(permute_235, (1536, 384), (384, 1))
    assert_size_stride(div_10, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_240, (196, 192), (192, 1))
    assert_size_stride(permute_247, (384, 196), (196, 1))
    assert_size_stride(div_11, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_250, (384, 768), (768, 1))
    assert_size_stride(permute_255, (1536, 384), (384, 1))
    assert_size_stride(div_12, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_260, (196, 192), (192, 1))
    assert_size_stride(permute_267, (384, 196), (196, 1))
    assert_size_stride(div_13, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_270, (384, 768), (768, 1))
    assert_size_stride(permute_275, (1536, 384), (384, 1))
    assert_size_stride(div_14, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_280, (196, 192), (192, 1))
    assert_size_stride(permute_287, (384, 196), (196, 1))
    assert_size_stride(div_15, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_290, (384, 768), (768, 1))
    assert_size_stride(permute_295, (1536, 384), (384, 1))
    assert_size_stride(div_16, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_300, (196, 192), (192, 1))
    assert_size_stride(permute_307, (384, 196), (196, 1))
    assert_size_stride(div_17, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_310, (384, 768), (768, 1))
    assert_size_stride(permute_315, (1536, 384), (384, 1))
    assert_size_stride(div_18, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_320, (196, 192), (192, 1))
    assert_size_stride(permute_327, (384, 196), (196, 1))
    assert_size_stride(div_19, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_330, (384, 768), (768, 1))
    assert_size_stride(permute_335, (1536, 384), (384, 1))
    assert_size_stride(div_20, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_340, (196, 192), (192, 1))
    assert_size_stride(permute_347, (384, 196), (196, 1))
    assert_size_stride(div_21, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_350, (384, 768), (768, 1))
    assert_size_stride(permute_355, (1536, 384), (384, 1))
    assert_size_stride(div_22, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_360, (196, 192), (192, 1))
    assert_size_stride(permute_367, (384, 196), (196, 1))
    assert_size_stride(div_23, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_370, (384, 768), (768, 1))
    assert_size_stride(permute_375, (1536, 384), (384, 1))
    assert_size_stride(div_24, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_380, (196, 192), (192, 1))
    assert_size_stride(permute_387, (384, 196), (196, 1))
    assert_size_stride(div_25, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_390, (384, 768), (768, 1))
    assert_size_stride(permute_395, (1536, 384), (384, 1))
    assert_size_stride(div_26, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_400, (196, 192), (192, 1))
    assert_size_stride(permute_407, (384, 196), (196, 1))
    assert_size_stride(div_27, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_410, (384, 768), (768, 1))
    assert_size_stride(permute_415, (1536, 384), (384, 1))
    assert_size_stride(div_28, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_420, (196, 192), (192, 1))
    assert_size_stride(permute_427, (384, 196), (196, 1))
    assert_size_stride(div_29, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_430, (384, 768), (768, 1))
    assert_size_stride(permute_435, (1536, 384), (384, 1))
    assert_size_stride(div_30, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_440, (196, 192), (192, 1))
    assert_size_stride(permute_447, (384, 196), (196, 1))
    assert_size_stride(div_31, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_450, (384, 768), (768, 1))
    assert_size_stride(permute_455, (1536, 384), (384, 1))
    assert_size_stride(div_32, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_460, (196, 192), (192, 1))
    assert_size_stride(permute_467, (384, 196), (196, 1))
    assert_size_stride(div_33, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_470, (384, 768), (768, 1))
    assert_size_stride(permute_475, (1536, 384), (384, 1))
    assert_size_stride(div_34, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_480, (196, 192), (192, 1))
    assert_size_stride(permute_487, (384, 196), (196, 1))
    assert_size_stride(div_35, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_490, (384, 768), (768, 1))
    assert_size_stride(permute_495, (1536, 384), (384, 1))
    assert_size_stride(div_36, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_500, (196, 192), (192, 1))
    assert_size_stride(permute_507, (384, 196), (196, 1))
    assert_size_stride(div_37, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_510, (384, 768), (768, 1))
    assert_size_stride(permute_515, (1536, 384), (384, 1))
    assert_size_stride(div_38, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_520, (196, 192), (192, 1))
    assert_size_stride(permute_527, (384, 196), (196, 1))
    assert_size_stride(div_39, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_530, (384, 768), (768, 1))
    assert_size_stride(permute_535, (1536, 384), (384, 1))
    assert_size_stride(div_40, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_540, (196, 192), (192, 1))
    assert_size_stride(permute_547, (384, 196), (196, 1))
    assert_size_stride(div_41, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_550, (384, 768), (768, 1))
    assert_size_stride(permute_555, (1536, 384), (384, 1))
    assert_size_stride(div_42, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_560, (196, 192), (192, 1))
    assert_size_stride(permute_567, (384, 196), (196, 1))
    assert_size_stride(div_43, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_570, (384, 768), (768, 1))
    assert_size_stride(permute_575, (1536, 384), (384, 1))
    assert_size_stride(div_44, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_580, (196, 192), (192, 1))
    assert_size_stride(permute_587, (384, 196), (196, 1))
    assert_size_stride(div_45, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_590, (384, 768), (768, 1))
    assert_size_stride(permute_595, (1536, 384), (384, 1))
    assert_size_stride(div_46, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_600, (196, 192), (192, 1))
    assert_size_stride(permute_607, (384, 196), (196, 1))
    assert_size_stride(div_47, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_610, (384, 768), (768, 1))
    assert_size_stride(permute_615, (1536, 384), (384, 1))
    assert_size_stride(div_48, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_620, (196, 192), (192, 1))
    assert_size_stride(permute_627, (384, 196), (196, 1))
    assert_size_stride(div_49, (8, 196, 1), (196, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_146, out=buf0)
        del permute_146
        buf1 = empty((1000, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_169, out=buf1)
        del clone_169
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf5 = empty((8, 196, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_layer_norm_backward]
        triton_red_fused_div_native_layer_norm_backward_1.run(buf0, primals_291, mul_192, div_1, buf5, 1568, 384, grid=grid(1568), stream=stream0)
        del div_1
        del primals_291
        buf6 = empty_strided((384, 13), (1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_layer_norm_backward]
        triton_red_fused_div_native_layer_norm_backward_2.run(buf0, mul_192, buf6, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_192
        buf7 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf6, buf7, 384, 13, grid=grid(384), stream=stream0)
        buf8 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.native_layer_norm_backward]
        triton_red_fused_div_native_layer_norm_backward_4.run(buf0, buf8, 384, 1568, grid=grid(384), stream=stream0)
        del buf0
        buf9 = empty((1568, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (1568, 384), (384, 1), 0), permute_150, out=buf9)
        del permute_150
        buf10 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (384, 1568), (1, 384), 0), view_191, out=buf10)
        del view_191
        buf11 = reinterpret_tensor(buf6, (1, 384, 13), (4992, 1, 384), 0); del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf5, buf11, 4992, 121, grid=grid(4992), stream=stream0)
        buf12 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf11, buf12, 384, 13, grid=grid(384), stream=stream0)
        buf13 = empty((8, 196, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf9, getitem_191, getitem_190, buf13, 2408448, grid=grid(2408448), stream=stream0)
        del getitem_190
        del getitem_191
        buf14 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (1568, 1536), (1536, 1), 0), permute_155, out=buf14)
        del permute_155
        buf15 = empty((1536, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (1536, 1568), (1, 1536), 0), view_189, out=buf15)
        del view_189
        buf16 = empty_strided((1, 1536, 13), (19968, 1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf13, buf16, 19968, 121, grid=grid(19968), stream=stream0)
        buf17 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf16, buf17, 1536, 13, grid=grid(1536), stream=stream0)
        buf24 = buf5; del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf24, buf14, primals_285, mul_188, div_2, 1568, 384, grid=grid(1568), stream=stream0)
        del div_2
        del primals_285
        buf20 = reinterpret_tensor(buf11, (384, 13), (1, 384), 0); del buf11  # reuse
        buf22 = empty_strided((384, 13), (1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf14, mul_188, buf20, buf22, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_188
        buf21 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf20, buf21, 384, 13, grid=grid(384), stream=stream0)
        buf23 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf22, buf23, 384, 13, grid=grid(384), stream=stream0)
        buf25 = reinterpret_tensor(buf14, (3072, 196), (196, 1), 0); del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf24, buf25, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf26 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf25, permute_160, out=buf26)
        del permute_160
        buf27 = empty((196, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf25, (196, 3072), (1, 196), 0), view_187, out=buf27)
        del view_187
        buf28 = empty_strided((1, 196, 24), (4704, 1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf25, buf28, 4704, 128, grid=grid(4704), stream=stream0)
        buf29 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf28, buf29, 196, 24, grid=grid(196), stream=stream0)
        buf30 = empty((8, 384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_14.run(buf26, getitem_187, getitem_186, buf30, 1179648, grid=grid(1179648), stream=stream0)
        del getitem_186
        del getitem_187
        buf31 = empty_strided((1, 1, 384, 24), (9216, 9216, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf30, buf31, 9216, 128, grid=grid(9216), stream=stream0)
        buf32 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf31, buf32, 384, 24, grid=grid(384), stream=stream0)
        buf33 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf30, (384, 3072), (1, 384), 0), view_185, out=buf33)
        del view_185
        buf34 = buf25; del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf30, (3072, 384), (384, 1), 0), permute_167, out=buf34)
        del permute_167
        buf35 = reinterpret_tensor(buf28, (8, 196, 1, 3), (588, 1, 4704, 196), 0); del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf34, primals_279, buf35, 4704, 128, grid=grid(4704), stream=stream0)
        buf36 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf35, buf36, 1568, 3, grid=grid(1568), stream=stream0)
        buf37 = reinterpret_tensor(buf35, (8, 196, 1, 3), (588, 3, 4704, 1), 0); del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf34, primals_279, mul_184, buf37, 4704, 128, grid=grid(4704), stream=stream0)
        buf38 = empty_strided((8, 196, 1), (196, 1, 1568), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf37, buf38, 1568, 3, grid=grid(1568), stream=stream0)
        buf39 = reinterpret_tensor(buf22, (384, 13), (13, 1), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf34, mul_184, buf39, 4992, 121, grid=grid(4992), stream=stream0)
        buf40 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf39, buf40, 384, 13, grid=grid(384), stream=stream0)
        buf41 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf34, buf41, 384, 1568, grid=grid(384), stream=stream0)
        buf42 = buf24; del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf42, div_3, buf34, primals_279, buf36, mul_184, buf38, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del div_3
        del mul_184
        del primals_279
        buf43 = buf9; del buf9  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf42, (1568, 384), (384, 1), 0), permute_170, out=buf43)
        del permute_170
        buf44 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf42, (384, 1568), (1, 384), 0), view_183, out=buf44)
        del view_183
        buf45 = reinterpret_tensor(buf39, (1, 384, 13), (4992, 1, 384), 0); del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf42, buf45, 4992, 121, grid=grid(4992), stream=stream0)
        buf46 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf45, buf46, 384, 13, grid=grid(384), stream=stream0)
        buf47 = buf13; del buf13  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf43, getitem_183, getitem_182, buf47, 2408448, grid=grid(2408448), stream=stream0)
        del getitem_182
        del getitem_183
        buf48 = reinterpret_tensor(buf34, (1568, 384), (384, 1), 0); del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf47, (1568, 1536), (1536, 1), 0), permute_175, out=buf48)
        del permute_175
        buf49 = reinterpret_tensor(buf26, (1536, 384), (384, 1), 0); del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf47, (1536, 1568), (1, 1536), 0), view_181, out=buf49)
        del view_181
        buf50 = buf16; del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf47, buf50, 19968, 121, grid=grid(19968), stream=stream0)
        buf51 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf50, buf51, 1536, 13, grid=grid(1536), stream=stream0)
        buf58 = buf42; del buf42  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf58, buf48, primals_273, mul_180, div_4, 1568, 384, grid=grid(1568), stream=stream0)
        del div_4
        del primals_273
        buf54 = reinterpret_tensor(buf45, (384, 13), (1, 384), 0); del buf45  # reuse
        buf56 = buf20; del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf48, mul_180, buf54, buf56, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_180
        buf55 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf54, buf55, 384, 13, grid=grid(384), stream=stream0)
        buf57 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf56, buf57, 384, 13, grid=grid(384), stream=stream0)
        buf59 = reinterpret_tensor(buf48, (3072, 196), (196, 1), 0); del buf48  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf58, buf59, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf60 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf59, permute_180, out=buf60)
        del permute_180
        buf61 = empty((196, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf59, (196, 3072), (1, 196), 0), view_179, out=buf61)
        del view_179
        buf62 = reinterpret_tensor(buf37, (1, 196, 24), (4704, 1, 196), 0); del buf37  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf59, buf62, 4704, 128, grid=grid(4704), stream=stream0)
        buf63 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf62, buf63, 196, 24, grid=grid(196), stream=stream0)
        buf64 = buf30; del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_14.run(buf60, getitem_179, getitem_178, buf64, 1179648, grid=grid(1179648), stream=stream0)
        del getitem_178
        del getitem_179
        buf65 = buf31; del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf64, buf65, 9216, 128, grid=grid(9216), stream=stream0)
        buf66 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf65, buf66, 384, 24, grid=grid(384), stream=stream0)
        buf67 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (384, 3072), (1, 384), 0), view_177, out=buf67)
        del view_177
        buf68 = buf59; del buf59  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (3072, 384), (384, 1), 0), permute_187, out=buf68)
        del permute_187
        buf69 = reinterpret_tensor(buf62, (8, 196, 1, 3), (588, 1, 4704, 196), 0); del buf62  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf68, primals_267, buf69, 4704, 128, grid=grid(4704), stream=stream0)
        buf70 = buf38; del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf69, buf70, 1568, 3, grid=grid(1568), stream=stream0)
        buf71 = reinterpret_tensor(buf69, (8, 196, 1, 3), (588, 3, 4704, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf68, primals_267, mul_176, buf71, 4704, 128, grid=grid(4704), stream=stream0)
        buf72 = buf36; del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf71, buf72, 1568, 3, grid=grid(1568), stream=stream0)
        buf73 = reinterpret_tensor(buf56, (384, 13), (13, 1), 0); del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf68, mul_176, buf73, 4992, 121, grid=grid(4992), stream=stream0)
        buf74 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf73, buf74, 384, 13, grid=grid(384), stream=stream0)
        buf75 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf68, buf75, 384, 1568, grid=grid(384), stream=stream0)
        buf76 = buf58; del buf58  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf76, div_5, buf68, primals_267, buf70, mul_176, buf72, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del div_5
        del mul_176
        del primals_267
        buf77 = buf43; del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf76, (1568, 384), (384, 1), 0), permute_190, out=buf77)
        del permute_190
        buf78 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf76, (384, 1568), (1, 384), 0), view_175, out=buf78)
        del view_175
        buf79 = reinterpret_tensor(buf73, (1, 384, 13), (4992, 1, 384), 0); del buf73  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf76, buf79, 4992, 121, grid=grid(4992), stream=stream0)
        buf80 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf79, buf80, 384, 13, grid=grid(384), stream=stream0)
        buf81 = buf47; del buf47  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf77, getitem_175, getitem_174, buf81, 2408448, grid=grid(2408448), stream=stream0)
        del getitem_174
        del getitem_175
        buf82 = reinterpret_tensor(buf68, (1568, 384), (384, 1), 0); del buf68  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf81, (1568, 1536), (1536, 1), 0), permute_195, out=buf82)
        del permute_195
        buf83 = reinterpret_tensor(buf60, (1536, 384), (384, 1), 0); del buf60  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf81, (1536, 1568), (1, 1536), 0), view_173, out=buf83)
        del view_173
        buf84 = buf50; del buf50  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf81, buf84, 19968, 121, grid=grid(19968), stream=stream0)
        buf85 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf84, buf85, 1536, 13, grid=grid(1536), stream=stream0)
        buf92 = buf76; del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf92, buf82, primals_261, mul_172, div_6, 1568, 384, grid=grid(1568), stream=stream0)
        del div_6
        del primals_261
        buf88 = reinterpret_tensor(buf79, (384, 13), (1, 384), 0); del buf79  # reuse
        buf90 = buf54; del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf82, mul_172, buf88, buf90, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_172
        buf89 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf88, buf89, 384, 13, grid=grid(384), stream=stream0)
        buf91 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf90, buf91, 384, 13, grid=grid(384), stream=stream0)
        buf93 = reinterpret_tensor(buf82, (3072, 196), (196, 1), 0); del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf92, buf93, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf94 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf93, permute_200, out=buf94)
        del permute_200
        buf95 = empty((196, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf93, (196, 3072), (1, 196), 0), view_171, out=buf95)
        del view_171
        buf96 = reinterpret_tensor(buf71, (1, 196, 24), (4704, 1, 196), 0); del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf93, buf96, 4704, 128, grid=grid(4704), stream=stream0)
        buf97 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf96, buf97, 196, 24, grid=grid(196), stream=stream0)
        buf98 = buf64; del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_14.run(buf94, getitem_171, getitem_170, buf98, 1179648, grid=grid(1179648), stream=stream0)
        del getitem_170
        del getitem_171
        buf99 = buf65; del buf65  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf98, buf99, 9216, 128, grid=grid(9216), stream=stream0)
        buf100 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf99, buf100, 384, 24, grid=grid(384), stream=stream0)
        buf101 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf98, (384, 3072), (1, 384), 0), view_169, out=buf101)
        del view_169
        buf102 = buf93; del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf98, (3072, 384), (384, 1), 0), permute_207, out=buf102)
        del permute_207
        buf103 = reinterpret_tensor(buf96, (8, 196, 1, 3), (588, 1, 4704, 196), 0); del buf96  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf102, primals_255, buf103, 4704, 128, grid=grid(4704), stream=stream0)
        buf104 = buf72; del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf103, buf104, 1568, 3, grid=grid(1568), stream=stream0)
        buf105 = reinterpret_tensor(buf103, (8, 196, 1, 3), (588, 3, 4704, 1), 0); del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf102, primals_255, mul_168, buf105, 4704, 128, grid=grid(4704), stream=stream0)
        buf106 = buf70; del buf70  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf105, buf106, 1568, 3, grid=grid(1568), stream=stream0)
        buf107 = reinterpret_tensor(buf90, (384, 13), (13, 1), 0); del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf102, mul_168, buf107, 4992, 121, grid=grid(4992), stream=stream0)
        buf108 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf107, buf108, 384, 13, grid=grid(384), stream=stream0)
        buf109 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf102, buf109, 384, 1568, grid=grid(384), stream=stream0)
        buf110 = buf92; del buf92  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf110, div_7, buf102, primals_255, buf104, mul_168, buf106, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del div_7
        del mul_168
        del primals_255
        buf111 = buf77; del buf77  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf110, (1568, 384), (384, 1), 0), permute_210, out=buf111)
        del permute_210
        buf112 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf110, (384, 1568), (1, 384), 0), view_167, out=buf112)
        del view_167
        buf113 = reinterpret_tensor(buf107, (1, 384, 13), (4992, 1, 384), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf110, buf113, 4992, 121, grid=grid(4992), stream=stream0)
        buf114 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf113, buf114, 384, 13, grid=grid(384), stream=stream0)
        buf115 = buf81; del buf81  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf111, getitem_167, getitem_166, buf115, 2408448, grid=grid(2408448), stream=stream0)
        del getitem_166
        del getitem_167
        buf116 = reinterpret_tensor(buf102, (1568, 384), (384, 1), 0); del buf102  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf115, (1568, 1536), (1536, 1), 0), permute_215, out=buf116)
        del permute_215
        buf117 = reinterpret_tensor(buf94, (1536, 384), (384, 1), 0); del buf94  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf115, (1536, 1568), (1, 1536), 0), view_165, out=buf117)
        del view_165
        buf118 = buf84; del buf84  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf115, buf118, 19968, 121, grid=grid(19968), stream=stream0)
        buf119 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf118, buf119, 1536, 13, grid=grid(1536), stream=stream0)
        buf126 = buf110; del buf110  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf126, buf116, primals_249, mul_164, div_8, 1568, 384, grid=grid(1568), stream=stream0)
        del div_8
        del primals_249
        buf122 = reinterpret_tensor(buf113, (384, 13), (1, 384), 0); del buf113  # reuse
        buf124 = buf88; del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf116, mul_164, buf122, buf124, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_164
        buf123 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf122, buf123, 384, 13, grid=grid(384), stream=stream0)
        buf125 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf124, buf125, 384, 13, grid=grid(384), stream=stream0)
        buf127 = reinterpret_tensor(buf116, (3072, 196), (196, 1), 0); del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf126, buf127, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf128 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf127, permute_220, out=buf128)
        del permute_220
        buf129 = empty((196, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf127, (196, 3072), (1, 196), 0), view_163, out=buf129)
        del view_163
        buf130 = reinterpret_tensor(buf105, (1, 196, 24), (4704, 1, 196), 0); del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf127, buf130, 4704, 128, grid=grid(4704), stream=stream0)
        buf131 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf130, buf131, 196, 24, grid=grid(196), stream=stream0)
        buf132 = buf98; del buf98  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_14.run(buf128, getitem_163, getitem_162, buf132, 1179648, grid=grid(1179648), stream=stream0)
        del getitem_162
        del getitem_163
        buf133 = buf99; del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf132, buf133, 9216, 128, grid=grid(9216), stream=stream0)
        buf134 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf133, buf134, 384, 24, grid=grid(384), stream=stream0)
        buf135 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf132, (384, 3072), (1, 384), 0), view_161, out=buf135)
        del view_161
        buf136 = buf127; del buf127  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf132, (3072, 384), (384, 1), 0), permute_227, out=buf136)
        del permute_227
        buf137 = reinterpret_tensor(buf130, (8, 196, 1, 3), (588, 1, 4704, 196), 0); del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf136, primals_243, buf137, 4704, 128, grid=grid(4704), stream=stream0)
        buf138 = buf106; del buf106  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf137, buf138, 1568, 3, grid=grid(1568), stream=stream0)
        buf139 = reinterpret_tensor(buf137, (8, 196, 1, 3), (588, 3, 4704, 1), 0); del buf137  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf136, primals_243, mul_160, buf139, 4704, 128, grid=grid(4704), stream=stream0)
        buf140 = buf104; del buf104  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf139, buf140, 1568, 3, grid=grid(1568), stream=stream0)
        buf141 = reinterpret_tensor(buf124, (384, 13), (13, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf136, mul_160, buf141, 4992, 121, grid=grid(4992), stream=stream0)
        buf142 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf141, buf142, 384, 13, grid=grid(384), stream=stream0)
        buf143 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf136, buf143, 384, 1568, grid=grid(384), stream=stream0)
        buf144 = buf126; del buf126  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf144, div_9, buf136, primals_243, buf138, mul_160, buf140, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del div_9
        del mul_160
        del primals_243
        buf145 = buf111; del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf144, (1568, 384), (384, 1), 0), permute_230, out=buf145)
        del permute_230
        buf146 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf144, (384, 1568), (1, 384), 0), view_159, out=buf146)
        del view_159
        buf147 = reinterpret_tensor(buf141, (1, 384, 13), (4992, 1, 384), 0); del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf144, buf147, 4992, 121, grid=grid(4992), stream=stream0)
        buf148 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf147, buf148, 384, 13, grid=grid(384), stream=stream0)
        buf149 = buf115; del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf145, getitem_159, getitem_158, buf149, 2408448, grid=grid(2408448), stream=stream0)
        del getitem_158
        del getitem_159
        buf150 = reinterpret_tensor(buf136, (1568, 384), (384, 1), 0); del buf136  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (1568, 1536), (1536, 1), 0), permute_235, out=buf150)
        del permute_235
        buf151 = reinterpret_tensor(buf128, (1536, 384), (384, 1), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf149, (1536, 1568), (1, 1536), 0), view_157, out=buf151)
        del view_157
        buf152 = buf118; del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf149, buf152, 19968, 121, grid=grid(19968), stream=stream0)
        buf153 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf152, buf153, 1536, 13, grid=grid(1536), stream=stream0)
        buf160 = buf144; del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf160, buf150, primals_237, mul_156, div_10, 1568, 384, grid=grid(1568), stream=stream0)
        del div_10
        del primals_237
        buf156 = reinterpret_tensor(buf147, (384, 13), (1, 384), 0); del buf147  # reuse
        buf158 = buf122; del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf150, mul_156, buf156, buf158, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_156
        buf157 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf156, buf157, 384, 13, grid=grid(384), stream=stream0)
        buf159 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf158, buf159, 384, 13, grid=grid(384), stream=stream0)
        buf161 = reinterpret_tensor(buf150, (3072, 196), (196, 1), 0); del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf160, buf161, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf162 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf161, permute_240, out=buf162)
        del permute_240
        buf163 = empty((196, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf161, (196, 3072), (1, 196), 0), view_155, out=buf163)
        del view_155
        buf164 = reinterpret_tensor(buf139, (1, 196, 24), (4704, 1, 196), 0); del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf161, buf164, 4704, 128, grid=grid(4704), stream=stream0)
        buf165 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf164, buf165, 196, 24, grid=grid(196), stream=stream0)
        buf166 = buf132; del buf132  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_14.run(buf162, getitem_155, getitem_154, buf166, 1179648, grid=grid(1179648), stream=stream0)
        del getitem_154
        del getitem_155
        buf167 = buf133; del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf166, buf167, 9216, 128, grid=grid(9216), stream=stream0)
        buf168 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf167, buf168, 384, 24, grid=grid(384), stream=stream0)
        buf169 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf166, (384, 3072), (1, 384), 0), view_153, out=buf169)
        del view_153
        buf170 = buf161; del buf161  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf166, (3072, 384), (384, 1), 0), permute_247, out=buf170)
        del permute_247
        buf171 = reinterpret_tensor(buf164, (8, 196, 1, 3), (588, 1, 4704, 196), 0); del buf164  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf170, primals_231, buf171, 4704, 128, grid=grid(4704), stream=stream0)
        buf172 = buf140; del buf140  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf171, buf172, 1568, 3, grid=grid(1568), stream=stream0)
        buf173 = reinterpret_tensor(buf171, (8, 196, 1, 3), (588, 3, 4704, 1), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf170, primals_231, mul_152, buf173, 4704, 128, grid=grid(4704), stream=stream0)
        buf174 = buf138; del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf173, buf174, 1568, 3, grid=grid(1568), stream=stream0)
        buf175 = reinterpret_tensor(buf158, (384, 13), (13, 1), 0); del buf158  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf170, mul_152, buf175, 4992, 121, grid=grid(4992), stream=stream0)
        buf176 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf175, buf176, 384, 13, grid=grid(384), stream=stream0)
        buf177 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf170, buf177, 384, 1568, grid=grid(384), stream=stream0)
        buf178 = buf160; del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf178, div_11, buf170, primals_231, buf172, mul_152, buf174, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del div_11
        del mul_152
        del primals_231
        buf179 = buf145; del buf145  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf178, (1568, 384), (384, 1), 0), permute_250, out=buf179)
        del permute_250
        buf180 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf178, (384, 1568), (1, 384), 0), view_151, out=buf180)
        del view_151
        buf181 = reinterpret_tensor(buf175, (1, 384, 13), (4992, 1, 384), 0); del buf175  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf178, buf181, 4992, 121, grid=grid(4992), stream=stream0)
        buf182 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf181, buf182, 384, 13, grid=grid(384), stream=stream0)
        buf183 = buf149; del buf149  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf179, getitem_151, getitem_150, buf183, 2408448, grid=grid(2408448), stream=stream0)
        del getitem_150
        del getitem_151
        buf184 = reinterpret_tensor(buf170, (1568, 384), (384, 1), 0); del buf170  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf183, (1568, 1536), (1536, 1), 0), permute_255, out=buf184)
        del permute_255
        buf185 = reinterpret_tensor(buf162, (1536, 384), (384, 1), 0); del buf162  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf183, (1536, 1568), (1, 1536), 0), view_149, out=buf185)
        del view_149
        buf186 = buf152; del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf183, buf186, 19968, 121, grid=grid(19968), stream=stream0)
        buf187 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf186, buf187, 1536, 13, grid=grid(1536), stream=stream0)
        buf194 = buf178; del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf194, buf184, primals_225, mul_148, div_12, 1568, 384, grid=grid(1568), stream=stream0)
        del div_12
        del primals_225
        buf190 = reinterpret_tensor(buf181, (384, 13), (1, 384), 0); del buf181  # reuse
        buf192 = buf156; del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf184, mul_148, buf190, buf192, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_148
        buf191 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf190, buf191, 384, 13, grid=grid(384), stream=stream0)
        buf193 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf192, buf193, 384, 13, grid=grid(384), stream=stream0)
        buf195 = reinterpret_tensor(buf184, (3072, 196), (196, 1), 0); del buf184  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf194, buf195, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf196 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf195, permute_260, out=buf196)
        del permute_260
        buf197 = empty((196, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf195, (196, 3072), (1, 196), 0), view_147, out=buf197)
        del view_147
        buf198 = reinterpret_tensor(buf173, (1, 196, 24), (4704, 1, 196), 0); del buf173  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf195, buf198, 4704, 128, grid=grid(4704), stream=stream0)
        buf199 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf198, buf199, 196, 24, grid=grid(196), stream=stream0)
        buf200 = buf166; del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_14.run(buf196, getitem_147, getitem_146, buf200, 1179648, grid=grid(1179648), stream=stream0)
        del getitem_146
        del getitem_147
        buf201 = buf167; del buf167  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf200, buf201, 9216, 128, grid=grid(9216), stream=stream0)
        buf202 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf201, buf202, 384, 24, grid=grid(384), stream=stream0)
        buf203 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf200, (384, 3072), (1, 384), 0), view_145, out=buf203)
        del view_145
        buf204 = buf195; del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf200, (3072, 384), (384, 1), 0), permute_267, out=buf204)
        del permute_267
        buf205 = reinterpret_tensor(buf198, (8, 196, 1, 3), (588, 1, 4704, 196), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf204, primals_219, buf205, 4704, 128, grid=grid(4704), stream=stream0)
        buf206 = buf174; del buf174  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf205, buf206, 1568, 3, grid=grid(1568), stream=stream0)
        buf207 = reinterpret_tensor(buf205, (8, 196, 1, 3), (588, 3, 4704, 1), 0); del buf205  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf204, primals_219, mul_144, buf207, 4704, 128, grid=grid(4704), stream=stream0)
        buf208 = buf172; del buf172  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf207, buf208, 1568, 3, grid=grid(1568), stream=stream0)
        buf209 = reinterpret_tensor(buf192, (384, 13), (13, 1), 0); del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf204, mul_144, buf209, 4992, 121, grid=grid(4992), stream=stream0)
        buf210 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf209, buf210, 384, 13, grid=grid(384), stream=stream0)
        buf211 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf204, buf211, 384, 1568, grid=grid(384), stream=stream0)
        buf212 = buf194; del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf212, div_13, buf204, primals_219, buf206, mul_144, buf208, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del div_13
        del mul_144
        del primals_219
        buf213 = buf179; del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf212, (1568, 384), (384, 1), 0), permute_270, out=buf213)
        del permute_270
        buf214 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf212, (384, 1568), (1, 384), 0), view_143, out=buf214)
        del view_143
        buf215 = reinterpret_tensor(buf209, (1, 384, 13), (4992, 1, 384), 0); del buf209  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf212, buf215, 4992, 121, grid=grid(4992), stream=stream0)
        buf216 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf215, buf216, 384, 13, grid=grid(384), stream=stream0)
        buf217 = buf183; del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf213, getitem_143, getitem_142, buf217, 2408448, grid=grid(2408448), stream=stream0)
        del getitem_142
        del getitem_143
        buf218 = reinterpret_tensor(buf204, (1568, 384), (384, 1), 0); del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf217, (1568, 1536), (1536, 1), 0), permute_275, out=buf218)
        del permute_275
        buf219 = reinterpret_tensor(buf196, (1536, 384), (384, 1), 0); del buf196  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf217, (1536, 1568), (1, 1536), 0), view_141, out=buf219)
        del view_141
        buf220 = buf186; del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf217, buf220, 19968, 121, grid=grid(19968), stream=stream0)
        buf221 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf220, buf221, 1536, 13, grid=grid(1536), stream=stream0)
        buf228 = buf212; del buf212  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf228, buf218, primals_213, mul_140, div_14, 1568, 384, grid=grid(1568), stream=stream0)
        del div_14
        del primals_213
        buf224 = reinterpret_tensor(buf215, (384, 13), (1, 384), 0); del buf215  # reuse
        buf226 = buf190; del buf190  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf218, mul_140, buf224, buf226, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_140
        buf225 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf224, buf225, 384, 13, grid=grid(384), stream=stream0)
        buf227 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf226, buf227, 384, 13, grid=grid(384), stream=stream0)
        buf229 = reinterpret_tensor(buf218, (3072, 196), (196, 1), 0); del buf218  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf228, buf229, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf230 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf229, permute_280, out=buf230)
        del permute_280
        buf231 = empty((196, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf229, (196, 3072), (1, 196), 0), view_139, out=buf231)
        del view_139
        buf232 = reinterpret_tensor(buf207, (1, 196, 24), (4704, 1, 196), 0); del buf207  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf229, buf232, 4704, 128, grid=grid(4704), stream=stream0)
        buf233 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf232, buf233, 196, 24, grid=grid(196), stream=stream0)
        buf234 = buf200; del buf200  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_14.run(buf230, getitem_139, getitem_138, buf234, 1179648, grid=grid(1179648), stream=stream0)
        del getitem_138
        del getitem_139
        buf235 = buf201; del buf201  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf234, buf235, 9216, 128, grid=grid(9216), stream=stream0)
        buf236 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf235, buf236, 384, 24, grid=grid(384), stream=stream0)
        buf237 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf234, (384, 3072), (1, 384), 0), view_137, out=buf237)
        del view_137
        buf238 = buf229; del buf229  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf234, (3072, 384), (384, 1), 0), permute_287, out=buf238)
        del permute_287
        buf239 = reinterpret_tensor(buf232, (8, 196, 1, 3), (588, 1, 4704, 196), 0); del buf232  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf238, primals_207, buf239, 4704, 128, grid=grid(4704), stream=stream0)
        buf240 = buf208; del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf239, buf240, 1568, 3, grid=grid(1568), stream=stream0)
        buf241 = reinterpret_tensor(buf239, (8, 196, 1, 3), (588, 3, 4704, 1), 0); del buf239  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf238, primals_207, mul_136, buf241, 4704, 128, grid=grid(4704), stream=stream0)
        buf242 = buf206; del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf241, buf242, 1568, 3, grid=grid(1568), stream=stream0)
        buf243 = reinterpret_tensor(buf226, (384, 13), (13, 1), 0); del buf226  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf238, mul_136, buf243, 4992, 121, grid=grid(4992), stream=stream0)
        buf244 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf243, buf244, 384, 13, grid=grid(384), stream=stream0)
        buf245 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf238, buf245, 384, 1568, grid=grid(384), stream=stream0)
        buf246 = buf228; del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf246, div_15, buf238, primals_207, buf240, mul_136, buf242, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del div_15
        del mul_136
        del primals_207
        buf247 = buf213; del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf246, (1568, 384), (384, 1), 0), permute_290, out=buf247)
        del permute_290
        buf248 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf246, (384, 1568), (1, 384), 0), view_135, out=buf248)
        del view_135
        buf249 = reinterpret_tensor(buf243, (1, 384, 13), (4992, 1, 384), 0); del buf243  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf246, buf249, 4992, 121, grid=grid(4992), stream=stream0)
        buf250 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf249, buf250, 384, 13, grid=grid(384), stream=stream0)
        buf251 = buf217; del buf217  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf247, getitem_135, getitem_134, buf251, 2408448, grid=grid(2408448), stream=stream0)
        del getitem_134
        del getitem_135
        buf252 = reinterpret_tensor(buf238, (1568, 384), (384, 1), 0); del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf251, (1568, 1536), (1536, 1), 0), permute_295, out=buf252)
        del permute_295
        buf253 = reinterpret_tensor(buf230, (1536, 384), (384, 1), 0); del buf230  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf251, (1536, 1568), (1, 1536), 0), view_133, out=buf253)
        del view_133
        buf254 = buf220; del buf220  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf251, buf254, 19968, 121, grid=grid(19968), stream=stream0)
        buf255 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf254, buf255, 1536, 13, grid=grid(1536), stream=stream0)
        buf262 = buf246; del buf246  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf262, buf252, primals_201, mul_132, div_16, 1568, 384, grid=grid(1568), stream=stream0)
        del div_16
        del primals_201
        buf258 = reinterpret_tensor(buf249, (384, 13), (1, 384), 0); del buf249  # reuse
        buf260 = buf224; del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf252, mul_132, buf258, buf260, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_132
        buf259 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf258, buf259, 384, 13, grid=grid(384), stream=stream0)
        buf261 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf260, buf261, 384, 13, grid=grid(384), stream=stream0)
        buf263 = reinterpret_tensor(buf252, (3072, 196), (196, 1), 0); del buf252  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf262, buf263, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf264 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf263, permute_300, out=buf264)
        del permute_300
        buf265 = empty((196, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf263, (196, 3072), (1, 196), 0), view_131, out=buf265)
        del view_131
        buf266 = reinterpret_tensor(buf241, (1, 196, 24), (4704, 1, 196), 0); del buf241  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf263, buf266, 4704, 128, grid=grid(4704), stream=stream0)
        buf267 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf266, buf267, 196, 24, grid=grid(196), stream=stream0)
        buf268 = buf234; del buf234  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_14.run(buf264, getitem_131, getitem_130, buf268, 1179648, grid=grid(1179648), stream=stream0)
        del getitem_130
        del getitem_131
        buf269 = buf235; del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf268, buf269, 9216, 128, grid=grid(9216), stream=stream0)
        buf270 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf269, buf270, 384, 24, grid=grid(384), stream=stream0)
        buf271 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf268, (384, 3072), (1, 384), 0), view_129, out=buf271)
        del view_129
        buf272 = buf263; del buf263  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf268, (3072, 384), (384, 1), 0), permute_307, out=buf272)
        del permute_307
        buf273 = reinterpret_tensor(buf266, (8, 196, 1, 3), (588, 1, 4704, 196), 0); del buf266  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf272, primals_195, buf273, 4704, 128, grid=grid(4704), stream=stream0)
        buf274 = buf242; del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf273, buf274, 1568, 3, grid=grid(1568), stream=stream0)
        buf275 = reinterpret_tensor(buf273, (8, 196, 1, 3), (588, 3, 4704, 1), 0); del buf273  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf272, primals_195, mul_128, buf275, 4704, 128, grid=grid(4704), stream=stream0)
        buf276 = buf240; del buf240  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf275, buf276, 1568, 3, grid=grid(1568), stream=stream0)
        buf277 = reinterpret_tensor(buf260, (384, 13), (13, 1), 0); del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf272, mul_128, buf277, 4992, 121, grid=grid(4992), stream=stream0)
        buf278 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf277, buf278, 384, 13, grid=grid(384), stream=stream0)
        buf279 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf272, buf279, 384, 1568, grid=grid(384), stream=stream0)
        buf280 = buf262; del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf280, div_17, buf272, primals_195, buf274, mul_128, buf276, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del div_17
        del mul_128
        del primals_195
        buf281 = buf247; del buf247  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf280, (1568, 384), (384, 1), 0), permute_310, out=buf281)
        del permute_310
        buf282 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf280, (384, 1568), (1, 384), 0), view_127, out=buf282)
        del view_127
        buf283 = reinterpret_tensor(buf277, (1, 384, 13), (4992, 1, 384), 0); del buf277  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf280, buf283, 4992, 121, grid=grid(4992), stream=stream0)
        buf284 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf283, buf284, 384, 13, grid=grid(384), stream=stream0)
        buf285 = buf251; del buf251  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf281, getitem_127, getitem_126, buf285, 2408448, grid=grid(2408448), stream=stream0)
        del getitem_126
        del getitem_127
        buf286 = reinterpret_tensor(buf272, (1568, 384), (384, 1), 0); del buf272  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf285, (1568, 1536), (1536, 1), 0), permute_315, out=buf286)
        del permute_315
        buf287 = reinterpret_tensor(buf264, (1536, 384), (384, 1), 0); del buf264  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf285, (1536, 1568), (1, 1536), 0), view_125, out=buf287)
        del view_125
        buf288 = buf254; del buf254  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf285, buf288, 19968, 121, grid=grid(19968), stream=stream0)
        buf289 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf288, buf289, 1536, 13, grid=grid(1536), stream=stream0)
        buf296 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf296, buf286, primals_189, mul_124, div_18, 1568, 384, grid=grid(1568), stream=stream0)
        del div_18
        del primals_189
        buf292 = reinterpret_tensor(buf283, (384, 13), (1, 384), 0); del buf283  # reuse
        buf294 = buf258; del buf258  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf286, mul_124, buf292, buf294, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_124
        buf293 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf292, buf293, 384, 13, grid=grid(384), stream=stream0)
        buf295 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf294, buf295, 384, 13, grid=grid(384), stream=stream0)
        buf297 = reinterpret_tensor(buf286, (3072, 196), (196, 1), 0); del buf286  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf296, buf297, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf298 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf297, permute_320, out=buf298)
        del permute_320
        buf299 = empty((196, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf297, (196, 3072), (1, 196), 0), view_123, out=buf299)
        del view_123
        buf300 = reinterpret_tensor(buf275, (1, 196, 24), (4704, 1, 196), 0); del buf275  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf297, buf300, 4704, 128, grid=grid(4704), stream=stream0)
        buf301 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf300, buf301, 196, 24, grid=grid(196), stream=stream0)
        buf302 = buf268; del buf268  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_14.run(buf298, getitem_123, getitem_122, buf302, 1179648, grid=grid(1179648), stream=stream0)
        del getitem_122
        del getitem_123
        buf303 = buf269; del buf269  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf302, buf303, 9216, 128, grid=grid(9216), stream=stream0)
        buf304 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf303, buf304, 384, 24, grid=grid(384), stream=stream0)
        buf305 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf302, (384, 3072), (1, 384), 0), view_121, out=buf305)
        del view_121
        buf306 = buf297; del buf297  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf302, (3072, 384), (384, 1), 0), permute_327, out=buf306)
        del permute_327
        buf307 = reinterpret_tensor(buf300, (8, 196, 1, 3), (588, 1, 4704, 196), 0); del buf300  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf306, primals_183, buf307, 4704, 128, grid=grid(4704), stream=stream0)
        buf308 = buf276; del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf307, buf308, 1568, 3, grid=grid(1568), stream=stream0)
        buf309 = reinterpret_tensor(buf307, (8, 196, 1, 3), (588, 3, 4704, 1), 0); del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf306, primals_183, mul_120, buf309, 4704, 128, grid=grid(4704), stream=stream0)
        buf310 = buf274; del buf274  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf309, buf310, 1568, 3, grid=grid(1568), stream=stream0)
        buf311 = reinterpret_tensor(buf294, (384, 13), (13, 1), 0); del buf294  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf306, mul_120, buf311, 4992, 121, grid=grid(4992), stream=stream0)
        buf312 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf311, buf312, 384, 13, grid=grid(384), stream=stream0)
        buf313 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf306, buf313, 384, 1568, grid=grid(384), stream=stream0)
        buf314 = buf296; del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf314, div_19, buf306, primals_183, buf308, mul_120, buf310, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del div_19
        del mul_120
        del primals_183
        buf315 = buf281; del buf281  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf314, (1568, 384), (384, 1), 0), permute_330, out=buf315)
        del permute_330
        buf316 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf314, (384, 1568), (1, 384), 0), view_119, out=buf316)
        del view_119
        buf317 = reinterpret_tensor(buf311, (1, 384, 13), (4992, 1, 384), 0); del buf311  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf314, buf317, 4992, 121, grid=grid(4992), stream=stream0)
        buf318 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf317, buf318, 384, 13, grid=grid(384), stream=stream0)
        buf319 = buf285; del buf285  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf315, getitem_119, getitem_118, buf319, 2408448, grid=grid(2408448), stream=stream0)
        del getitem_118
        del getitem_119
        buf320 = reinterpret_tensor(buf306, (1568, 384), (384, 1), 0); del buf306  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf319, (1568, 1536), (1536, 1), 0), permute_335, out=buf320)
        del permute_335
        buf321 = reinterpret_tensor(buf298, (1536, 384), (384, 1), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf319, (1536, 1568), (1, 1536), 0), view_117, out=buf321)
        del view_117
        buf322 = buf288; del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf319, buf322, 19968, 121, grid=grid(19968), stream=stream0)
        buf323 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf322, buf323, 1536, 13, grid=grid(1536), stream=stream0)
        buf330 = buf314; del buf314  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf330, buf320, primals_177, mul_116, div_20, 1568, 384, grid=grid(1568), stream=stream0)
        del div_20
        del primals_177
        buf326 = reinterpret_tensor(buf317, (384, 13), (1, 384), 0); del buf317  # reuse
        buf328 = buf292; del buf292  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf320, mul_116, buf326, buf328, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_116
        buf327 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf326, buf327, 384, 13, grid=grid(384), stream=stream0)
        buf329 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf328, buf329, 384, 13, grid=grid(384), stream=stream0)
        buf331 = reinterpret_tensor(buf320, (3072, 196), (196, 1), 0); del buf320  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf330, buf331, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf332 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf331, permute_340, out=buf332)
        del permute_340
        buf333 = empty((196, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf331, (196, 3072), (1, 196), 0), view_115, out=buf333)
        del view_115
        buf334 = reinterpret_tensor(buf309, (1, 196, 24), (4704, 1, 196), 0); del buf309  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf331, buf334, 4704, 128, grid=grid(4704), stream=stream0)
        buf335 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf334, buf335, 196, 24, grid=grid(196), stream=stream0)
        buf336 = buf302; del buf302  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_14.run(buf332, getitem_115, getitem_114, buf336, 1179648, grid=grid(1179648), stream=stream0)
        del getitem_114
        del getitem_115
        buf337 = buf303; del buf303  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf336, buf337, 9216, 128, grid=grid(9216), stream=stream0)
        buf338 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf337, buf338, 384, 24, grid=grid(384), stream=stream0)
        buf339 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf336, (384, 3072), (1, 384), 0), view_113, out=buf339)
        del view_113
        buf340 = buf331; del buf331  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf336, (3072, 384), (384, 1), 0), permute_347, out=buf340)
        del permute_347
        buf341 = reinterpret_tensor(buf334, (8, 196, 1, 3), (588, 1, 4704, 196), 0); del buf334  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf340, primals_171, buf341, 4704, 128, grid=grid(4704), stream=stream0)
        buf342 = buf310; del buf310  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf341, buf342, 1568, 3, grid=grid(1568), stream=stream0)
        buf343 = reinterpret_tensor(buf341, (8, 196, 1, 3), (588, 3, 4704, 1), 0); del buf341  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf340, primals_171, mul_112, buf343, 4704, 128, grid=grid(4704), stream=stream0)
        buf344 = buf308; del buf308  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf343, buf344, 1568, 3, grid=grid(1568), stream=stream0)
        buf345 = reinterpret_tensor(buf328, (384, 13), (13, 1), 0); del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf340, mul_112, buf345, 4992, 121, grid=grid(4992), stream=stream0)
        buf346 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf345, buf346, 384, 13, grid=grid(384), stream=stream0)
        buf347 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf340, buf347, 384, 1568, grid=grid(384), stream=stream0)
        buf348 = buf330; del buf330  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf348, div_21, buf340, primals_171, buf342, mul_112, buf344, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del div_21
        del mul_112
        del primals_171
        buf349 = buf315; del buf315  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf348, (1568, 384), (384, 1), 0), permute_350, out=buf349)
        del permute_350
        buf350 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf348, (384, 1568), (1, 384), 0), view_111, out=buf350)
        del view_111
        buf351 = reinterpret_tensor(buf345, (1, 384, 13), (4992, 1, 384), 0); del buf345  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf348, buf351, 4992, 121, grid=grid(4992), stream=stream0)
        buf352 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf351, buf352, 384, 13, grid=grid(384), stream=stream0)
        buf353 = buf319; del buf319  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf349, getitem_111, getitem_110, buf353, 2408448, grid=grid(2408448), stream=stream0)
        del getitem_110
        del getitem_111
        buf354 = reinterpret_tensor(buf340, (1568, 384), (384, 1), 0); del buf340  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf353, (1568, 1536), (1536, 1), 0), permute_355, out=buf354)
        del permute_355
        buf355 = reinterpret_tensor(buf332, (1536, 384), (384, 1), 0); del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf353, (1536, 1568), (1, 1536), 0), view_109, out=buf355)
        del view_109
        buf356 = buf322; del buf322  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf353, buf356, 19968, 121, grid=grid(19968), stream=stream0)
        buf357 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf356, buf357, 1536, 13, grid=grid(1536), stream=stream0)
        buf364 = buf348; del buf348  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf364, buf354, primals_165, mul_108, div_22, 1568, 384, grid=grid(1568), stream=stream0)
        del div_22
        del primals_165
        buf360 = reinterpret_tensor(buf351, (384, 13), (1, 384), 0); del buf351  # reuse
        buf362 = buf326; del buf326  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf354, mul_108, buf360, buf362, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_108
        buf361 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf360, buf361, 384, 13, grid=grid(384), stream=stream0)
        buf363 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf362, buf363, 384, 13, grid=grid(384), stream=stream0)
        buf365 = reinterpret_tensor(buf354, (3072, 196), (196, 1), 0); del buf354  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf364, buf365, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf366 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf365, permute_360, out=buf366)
        del permute_360
        buf367 = empty((196, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf365, (196, 3072), (1, 196), 0), view_107, out=buf367)
        del view_107
        buf368 = reinterpret_tensor(buf343, (1, 196, 24), (4704, 1, 196), 0); del buf343  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf365, buf368, 4704, 128, grid=grid(4704), stream=stream0)
        buf369 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf368, buf369, 196, 24, grid=grid(196), stream=stream0)
        buf370 = buf336; del buf336  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_14.run(buf366, getitem_107, getitem_106, buf370, 1179648, grid=grid(1179648), stream=stream0)
        del getitem_106
        del getitem_107
        buf371 = buf337; del buf337  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf370, buf371, 9216, 128, grid=grid(9216), stream=stream0)
        buf372 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf371, buf372, 384, 24, grid=grid(384), stream=stream0)
        buf373 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf370, (384, 3072), (1, 384), 0), view_105, out=buf373)
        del view_105
        buf374 = buf365; del buf365  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf370, (3072, 384), (384, 1), 0), permute_367, out=buf374)
        del permute_367
        buf375 = reinterpret_tensor(buf368, (8, 196, 1, 3), (588, 1, 4704, 196), 0); del buf368  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf374, primals_159, buf375, 4704, 128, grid=grid(4704), stream=stream0)
        buf376 = buf344; del buf344  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf375, buf376, 1568, 3, grid=grid(1568), stream=stream0)
        buf377 = reinterpret_tensor(buf375, (8, 196, 1, 3), (588, 3, 4704, 1), 0); del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf374, primals_159, mul_104, buf377, 4704, 128, grid=grid(4704), stream=stream0)
        buf378 = buf342; del buf342  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf377, buf378, 1568, 3, grid=grid(1568), stream=stream0)
        buf379 = reinterpret_tensor(buf362, (384, 13), (13, 1), 0); del buf362  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf374, mul_104, buf379, 4992, 121, grid=grid(4992), stream=stream0)
        buf380 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf379, buf380, 384, 13, grid=grid(384), stream=stream0)
        buf381 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf374, buf381, 384, 1568, grid=grid(384), stream=stream0)
        buf382 = buf364; del buf364  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf382, div_23, buf374, primals_159, buf376, mul_104, buf378, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del div_23
        del mul_104
        del primals_159
        buf383 = buf349; del buf349  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf382, (1568, 384), (384, 1), 0), permute_370, out=buf383)
        del permute_370
        buf384 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf382, (384, 1568), (1, 384), 0), view_103, out=buf384)
        del view_103
        buf385 = reinterpret_tensor(buf379, (1, 384, 13), (4992, 1, 384), 0); del buf379  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf382, buf385, 4992, 121, grid=grid(4992), stream=stream0)
        buf386 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf385, buf386, 384, 13, grid=grid(384), stream=stream0)
        buf387 = buf353; del buf353  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf383, getitem_103, getitem_102, buf387, 2408448, grid=grid(2408448), stream=stream0)
        del getitem_102
        del getitem_103
        buf388 = reinterpret_tensor(buf374, (1568, 384), (384, 1), 0); del buf374  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf387, (1568, 1536), (1536, 1), 0), permute_375, out=buf388)
        del permute_375
        buf389 = reinterpret_tensor(buf366, (1536, 384), (384, 1), 0); del buf366  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf387, (1536, 1568), (1, 1536), 0), view_101, out=buf389)
        del view_101
        buf390 = buf356; del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf387, buf390, 19968, 121, grid=grid(19968), stream=stream0)
        buf391 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf390, buf391, 1536, 13, grid=grid(1536), stream=stream0)
        buf398 = buf382; del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf398, buf388, primals_153, mul_100, div_24, 1568, 384, grid=grid(1568), stream=stream0)
        del div_24
        del primals_153
        buf394 = reinterpret_tensor(buf385, (384, 13), (1, 384), 0); del buf385  # reuse
        buf396 = buf360; del buf360  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf388, mul_100, buf394, buf396, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_100
        buf395 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf394, buf395, 384, 13, grid=grid(384), stream=stream0)
        buf397 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf396, buf397, 384, 13, grid=grid(384), stream=stream0)
        buf399 = reinterpret_tensor(buf388, (3072, 196), (196, 1), 0); del buf388  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf398, buf399, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf400 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf399, permute_380, out=buf400)
        del permute_380
        buf401 = empty((196, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf399, (196, 3072), (1, 196), 0), view_99, out=buf401)
        del view_99
        buf402 = reinterpret_tensor(buf377, (1, 196, 24), (4704, 1, 196), 0); del buf377  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf399, buf402, 4704, 128, grid=grid(4704), stream=stream0)
        buf403 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf402, buf403, 196, 24, grid=grid(196), stream=stream0)
        buf404 = buf370; del buf370  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_14.run(buf400, getitem_99, getitem_98, buf404, 1179648, grid=grid(1179648), stream=stream0)
        del getitem_98
        del getitem_99
        buf405 = buf371; del buf371  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf404, buf405, 9216, 128, grid=grid(9216), stream=stream0)
        buf406 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf405, buf406, 384, 24, grid=grid(384), stream=stream0)
        buf407 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf404, (384, 3072), (1, 384), 0), view_97, out=buf407)
        del view_97
        buf408 = buf399; del buf399  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf404, (3072, 384), (384, 1), 0), permute_387, out=buf408)
        del permute_387
        buf409 = reinterpret_tensor(buf402, (8, 196, 1, 3), (588, 1, 4704, 196), 0); del buf402  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf408, primals_147, buf409, 4704, 128, grid=grid(4704), stream=stream0)
        buf410 = buf378; del buf378  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf409, buf410, 1568, 3, grid=grid(1568), stream=stream0)
        buf411 = reinterpret_tensor(buf409, (8, 196, 1, 3), (588, 3, 4704, 1), 0); del buf409  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf408, primals_147, mul_96, buf411, 4704, 128, grid=grid(4704), stream=stream0)
        buf412 = buf376; del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf411, buf412, 1568, 3, grid=grid(1568), stream=stream0)
        buf413 = reinterpret_tensor(buf396, (384, 13), (13, 1), 0); del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf408, mul_96, buf413, 4992, 121, grid=grid(4992), stream=stream0)
        buf414 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf413, buf414, 384, 13, grid=grid(384), stream=stream0)
        buf415 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf408, buf415, 384, 1568, grid=grid(384), stream=stream0)
        buf416 = buf398; del buf398  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf416, div_25, buf408, primals_147, buf410, mul_96, buf412, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del div_25
        del mul_96
        del primals_147
        buf417 = buf383; del buf383  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf416, (1568, 384), (384, 1), 0), permute_390, out=buf417)
        del permute_390
        buf418 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf416, (384, 1568), (1, 384), 0), view_95, out=buf418)
        del view_95
        buf419 = reinterpret_tensor(buf413, (1, 384, 13), (4992, 1, 384), 0); del buf413  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf416, buf419, 4992, 121, grid=grid(4992), stream=stream0)
        buf420 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf419, buf420, 384, 13, grid=grid(384), stream=stream0)
        buf421 = buf387; del buf387  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf417, getitem_95, getitem_94, buf421, 2408448, grid=grid(2408448), stream=stream0)
        del getitem_94
        del getitem_95
        buf422 = reinterpret_tensor(buf408, (1568, 384), (384, 1), 0); del buf408  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf421, (1568, 1536), (1536, 1), 0), permute_395, out=buf422)
        del permute_395
        buf423 = reinterpret_tensor(buf400, (1536, 384), (384, 1), 0); del buf400  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf421, (1536, 1568), (1, 1536), 0), view_93, out=buf423)
        del view_93
        buf424 = buf390; del buf390  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf421, buf424, 19968, 121, grid=grid(19968), stream=stream0)
        buf425 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf424, buf425, 1536, 13, grid=grid(1536), stream=stream0)
        buf432 = buf416; del buf416  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf432, buf422, primals_141, mul_92, div_26, 1568, 384, grid=grid(1568), stream=stream0)
        del div_26
        del primals_141
        buf428 = reinterpret_tensor(buf419, (384, 13), (1, 384), 0); del buf419  # reuse
        buf430 = buf394; del buf394  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf422, mul_92, buf428, buf430, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_92
        buf429 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf428, buf429, 384, 13, grid=grid(384), stream=stream0)
        buf431 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf430, buf431, 384, 13, grid=grid(384), stream=stream0)
        buf433 = reinterpret_tensor(buf422, (3072, 196), (196, 1), 0); del buf422  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf432, buf433, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf434 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf433, permute_400, out=buf434)
        del permute_400
        buf435 = empty((196, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf433, (196, 3072), (1, 196), 0), view_91, out=buf435)
        del view_91
        buf436 = reinterpret_tensor(buf411, (1, 196, 24), (4704, 1, 196), 0); del buf411  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf433, buf436, 4704, 128, grid=grid(4704), stream=stream0)
        buf437 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf436, buf437, 196, 24, grid=grid(196), stream=stream0)
        buf438 = buf404; del buf404  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_14.run(buf434, getitem_91, getitem_90, buf438, 1179648, grid=grid(1179648), stream=stream0)
        del getitem_90
        del getitem_91
        buf439 = buf405; del buf405  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf438, buf439, 9216, 128, grid=grid(9216), stream=stream0)
        buf440 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf439, buf440, 384, 24, grid=grid(384), stream=stream0)
        buf441 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf438, (384, 3072), (1, 384), 0), view_89, out=buf441)
        del view_89
        buf442 = buf433; del buf433  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf438, (3072, 384), (384, 1), 0), permute_407, out=buf442)
        del permute_407
        buf443 = reinterpret_tensor(buf436, (8, 196, 1, 3), (588, 1, 4704, 196), 0); del buf436  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf442, primals_135, buf443, 4704, 128, grid=grid(4704), stream=stream0)
        buf444 = buf412; del buf412  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf443, buf444, 1568, 3, grid=grid(1568), stream=stream0)
        buf445 = reinterpret_tensor(buf443, (8, 196, 1, 3), (588, 3, 4704, 1), 0); del buf443  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf442, primals_135, mul_88, buf445, 4704, 128, grid=grid(4704), stream=stream0)
        buf446 = buf410; del buf410  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf445, buf446, 1568, 3, grid=grid(1568), stream=stream0)
        buf447 = reinterpret_tensor(buf430, (384, 13), (13, 1), 0); del buf430  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf442, mul_88, buf447, 4992, 121, grid=grid(4992), stream=stream0)
        buf448 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf447, buf448, 384, 13, grid=grid(384), stream=stream0)
        buf449 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf442, buf449, 384, 1568, grid=grid(384), stream=stream0)
        buf450 = buf432; del buf432  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf450, div_27, buf442, primals_135, buf444, mul_88, buf446, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del div_27
        del mul_88
        del primals_135
        buf451 = buf417; del buf417  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf450, (1568, 384), (384, 1), 0), permute_410, out=buf451)
        del permute_410
        buf452 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf450, (384, 1568), (1, 384), 0), view_87, out=buf452)
        del view_87
        buf453 = reinterpret_tensor(buf447, (1, 384, 13), (4992, 1, 384), 0); del buf447  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf450, buf453, 4992, 121, grid=grid(4992), stream=stream0)
        buf454 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf453, buf454, 384, 13, grid=grid(384), stream=stream0)
        buf455 = buf421; del buf421  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf451, getitem_87, getitem_86, buf455, 2408448, grid=grid(2408448), stream=stream0)
        del getitem_86
        del getitem_87
        buf456 = reinterpret_tensor(buf442, (1568, 384), (384, 1), 0); del buf442  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf455, (1568, 1536), (1536, 1), 0), permute_415, out=buf456)
        del permute_415
        buf457 = reinterpret_tensor(buf434, (1536, 384), (384, 1), 0); del buf434  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf455, (1536, 1568), (1, 1536), 0), view_85, out=buf457)
        del view_85
        buf458 = buf424; del buf424  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf455, buf458, 19968, 121, grid=grid(19968), stream=stream0)
        buf459 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf458, buf459, 1536, 13, grid=grid(1536), stream=stream0)
        buf466 = buf450; del buf450  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf466, buf456, primals_129, mul_84, div_28, 1568, 384, grid=grid(1568), stream=stream0)
        del div_28
        del primals_129
        buf462 = reinterpret_tensor(buf453, (384, 13), (1, 384), 0); del buf453  # reuse
        buf464 = buf428; del buf428  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf456, mul_84, buf462, buf464, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_84
        buf463 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf462, buf463, 384, 13, grid=grid(384), stream=stream0)
        buf465 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf464, buf465, 384, 13, grid=grid(384), stream=stream0)
        buf467 = reinterpret_tensor(buf456, (3072, 196), (196, 1), 0); del buf456  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf466, buf467, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf468 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf467, permute_420, out=buf468)
        del permute_420
        buf469 = empty((196, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf467, (196, 3072), (1, 196), 0), view_83, out=buf469)
        del view_83
        buf470 = reinterpret_tensor(buf445, (1, 196, 24), (4704, 1, 196), 0); del buf445  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf467, buf470, 4704, 128, grid=grid(4704), stream=stream0)
        buf471 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf470, buf471, 196, 24, grid=grid(196), stream=stream0)
        buf472 = buf438; del buf438  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_14.run(buf468, getitem_83, getitem_82, buf472, 1179648, grid=grid(1179648), stream=stream0)
        del getitem_82
        del getitem_83
        buf473 = buf439; del buf439  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf472, buf473, 9216, 128, grid=grid(9216), stream=stream0)
        buf474 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf473, buf474, 384, 24, grid=grid(384), stream=stream0)
        buf475 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf472, (384, 3072), (1, 384), 0), view_81, out=buf475)
        del view_81
        buf476 = buf467; del buf467  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf472, (3072, 384), (384, 1), 0), permute_427, out=buf476)
        del permute_427
        buf477 = reinterpret_tensor(buf470, (8, 196, 1, 3), (588, 1, 4704, 196), 0); del buf470  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf476, primals_123, buf477, 4704, 128, grid=grid(4704), stream=stream0)
        buf478 = buf446; del buf446  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf477, buf478, 1568, 3, grid=grid(1568), stream=stream0)
        buf479 = reinterpret_tensor(buf477, (8, 196, 1, 3), (588, 3, 4704, 1), 0); del buf477  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf476, primals_123, mul_80, buf479, 4704, 128, grid=grid(4704), stream=stream0)
        buf480 = buf444; del buf444  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf479, buf480, 1568, 3, grid=grid(1568), stream=stream0)
        buf481 = reinterpret_tensor(buf464, (384, 13), (13, 1), 0); del buf464  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf476, mul_80, buf481, 4992, 121, grid=grid(4992), stream=stream0)
        buf482 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf481, buf482, 384, 13, grid=grid(384), stream=stream0)
        buf483 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf476, buf483, 384, 1568, grid=grid(384), stream=stream0)
        buf484 = buf466; del buf466  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf484, div_29, buf476, primals_123, buf478, mul_80, buf480, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del div_29
        del mul_80
        del primals_123
        buf485 = buf451; del buf451  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf484, (1568, 384), (384, 1), 0), permute_430, out=buf485)
        del permute_430
        buf486 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf484, (384, 1568), (1, 384), 0), view_79, out=buf486)
        del view_79
        buf487 = reinterpret_tensor(buf481, (1, 384, 13), (4992, 1, 384), 0); del buf481  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf484, buf487, 4992, 121, grid=grid(4992), stream=stream0)
        buf488 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf487, buf488, 384, 13, grid=grid(384), stream=stream0)
        buf489 = buf455; del buf455  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf485, getitem_79, getitem_78, buf489, 2408448, grid=grid(2408448), stream=stream0)
        del getitem_78
        del getitem_79
        buf490 = reinterpret_tensor(buf476, (1568, 384), (384, 1), 0); del buf476  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf489, (1568, 1536), (1536, 1), 0), permute_435, out=buf490)
        del permute_435
        buf491 = reinterpret_tensor(buf468, (1536, 384), (384, 1), 0); del buf468  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf489, (1536, 1568), (1, 1536), 0), view_77, out=buf491)
        del view_77
        buf492 = buf458; del buf458  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf489, buf492, 19968, 121, grid=grid(19968), stream=stream0)
        buf493 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf492, buf493, 1536, 13, grid=grid(1536), stream=stream0)
        buf500 = buf484; del buf484  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf500, buf490, primals_117, mul_76, div_30, 1568, 384, grid=grid(1568), stream=stream0)
        del div_30
        del primals_117
        buf496 = reinterpret_tensor(buf487, (384, 13), (1, 384), 0); del buf487  # reuse
        buf498 = buf462; del buf462  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf490, mul_76, buf496, buf498, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_76
        buf497 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf496, buf497, 384, 13, grid=grid(384), stream=stream0)
        buf499 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf498, buf499, 384, 13, grid=grid(384), stream=stream0)
        buf501 = reinterpret_tensor(buf490, (3072, 196), (196, 1), 0); del buf490  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf500, buf501, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf502 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf501, permute_440, out=buf502)
        del permute_440
        buf503 = empty((196, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf501, (196, 3072), (1, 196), 0), view_75, out=buf503)
        del view_75
        buf504 = reinterpret_tensor(buf479, (1, 196, 24), (4704, 1, 196), 0); del buf479  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf501, buf504, 4704, 128, grid=grid(4704), stream=stream0)
        buf505 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf504, buf505, 196, 24, grid=grid(196), stream=stream0)
        buf506 = buf472; del buf472  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_14.run(buf502, getitem_75, getitem_74, buf506, 1179648, grid=grid(1179648), stream=stream0)
        del getitem_74
        del getitem_75
        buf507 = buf473; del buf473  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf506, buf507, 9216, 128, grid=grid(9216), stream=stream0)
        buf508 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf507, buf508, 384, 24, grid=grid(384), stream=stream0)
        buf509 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf506, (384, 3072), (1, 384), 0), view_73, out=buf509)
        del view_73
        buf510 = buf501; del buf501  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf506, (3072, 384), (384, 1), 0), permute_447, out=buf510)
        del permute_447
        buf511 = reinterpret_tensor(buf504, (8, 196, 1, 3), (588, 1, 4704, 196), 0); del buf504  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf510, primals_111, buf511, 4704, 128, grid=grid(4704), stream=stream0)
        buf512 = buf480; del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf511, buf512, 1568, 3, grid=grid(1568), stream=stream0)
        buf513 = reinterpret_tensor(buf511, (8, 196, 1, 3), (588, 3, 4704, 1), 0); del buf511  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf510, primals_111, mul_72, buf513, 4704, 128, grid=grid(4704), stream=stream0)
        buf514 = buf478; del buf478  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf513, buf514, 1568, 3, grid=grid(1568), stream=stream0)
        buf515 = reinterpret_tensor(buf498, (384, 13), (13, 1), 0); del buf498  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf510, mul_72, buf515, 4992, 121, grid=grid(4992), stream=stream0)
        buf516 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf515, buf516, 384, 13, grid=grid(384), stream=stream0)
        buf517 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf510, buf517, 384, 1568, grid=grid(384), stream=stream0)
        buf518 = buf500; del buf500  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf518, div_31, buf510, primals_111, buf512, mul_72, buf514, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del div_31
        del mul_72
        del primals_111
        buf519 = buf485; del buf485  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf518, (1568, 384), (384, 1), 0), permute_450, out=buf519)
        del permute_450
        buf520 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf518, (384, 1568), (1, 384), 0), view_71, out=buf520)
        del view_71
        buf521 = reinterpret_tensor(buf515, (1, 384, 13), (4992, 1, 384), 0); del buf515  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf518, buf521, 4992, 121, grid=grid(4992), stream=stream0)
        buf522 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf521, buf522, 384, 13, grid=grid(384), stream=stream0)
        buf523 = buf489; del buf489  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf519, getitem_71, getitem_70, buf523, 2408448, grid=grid(2408448), stream=stream0)
        del getitem_70
        del getitem_71
        buf524 = reinterpret_tensor(buf510, (1568, 384), (384, 1), 0); del buf510  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf523, (1568, 1536), (1536, 1), 0), permute_455, out=buf524)
        del permute_455
        buf525 = reinterpret_tensor(buf502, (1536, 384), (384, 1), 0); del buf502  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf523, (1536, 1568), (1, 1536), 0), view_69, out=buf525)
        del view_69
        buf526 = buf492; del buf492  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf523, buf526, 19968, 121, grid=grid(19968), stream=stream0)
        buf527 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf526, buf527, 1536, 13, grid=grid(1536), stream=stream0)
        buf534 = buf518; del buf518  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf534, buf524, primals_105, mul_68, div_32, 1568, 384, grid=grid(1568), stream=stream0)
        del div_32
        del primals_105
        buf530 = reinterpret_tensor(buf521, (384, 13), (1, 384), 0); del buf521  # reuse
        buf532 = buf496; del buf496  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf524, mul_68, buf530, buf532, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_68
        buf531 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf530, buf531, 384, 13, grid=grid(384), stream=stream0)
        buf533 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf532, buf533, 384, 13, grid=grid(384), stream=stream0)
        buf535 = reinterpret_tensor(buf524, (3072, 196), (196, 1), 0); del buf524  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf534, buf535, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf536 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf535, permute_460, out=buf536)
        del permute_460
        buf537 = empty((196, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf535, (196, 3072), (1, 196), 0), view_67, out=buf537)
        del view_67
        buf538 = reinterpret_tensor(buf513, (1, 196, 24), (4704, 1, 196), 0); del buf513  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf535, buf538, 4704, 128, grid=grid(4704), stream=stream0)
        buf539 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf538, buf539, 196, 24, grid=grid(196), stream=stream0)
        buf540 = buf506; del buf506  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_14.run(buf536, getitem_67, getitem_66, buf540, 1179648, grid=grid(1179648), stream=stream0)
        del getitem_66
        del getitem_67
        buf541 = buf507; del buf507  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf540, buf541, 9216, 128, grid=grid(9216), stream=stream0)
        buf542 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf541, buf542, 384, 24, grid=grid(384), stream=stream0)
        buf543 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf540, (384, 3072), (1, 384), 0), view_65, out=buf543)
        del view_65
        buf544 = buf535; del buf535  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf540, (3072, 384), (384, 1), 0), permute_467, out=buf544)
        del permute_467
        buf545 = reinterpret_tensor(buf538, (8, 196, 1, 3), (588, 1, 4704, 196), 0); del buf538  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf544, primals_99, buf545, 4704, 128, grid=grid(4704), stream=stream0)
        buf546 = buf514; del buf514  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf545, buf546, 1568, 3, grid=grid(1568), stream=stream0)
        buf547 = reinterpret_tensor(buf545, (8, 196, 1, 3), (588, 3, 4704, 1), 0); del buf545  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf544, primals_99, mul_64, buf547, 4704, 128, grid=grid(4704), stream=stream0)
        buf548 = buf512; del buf512  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf547, buf548, 1568, 3, grid=grid(1568), stream=stream0)
        buf549 = reinterpret_tensor(buf532, (384, 13), (13, 1), 0); del buf532  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf544, mul_64, buf549, 4992, 121, grid=grid(4992), stream=stream0)
        buf550 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf549, buf550, 384, 13, grid=grid(384), stream=stream0)
        buf551 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf544, buf551, 384, 1568, grid=grid(384), stream=stream0)
        buf552 = buf534; del buf534  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf552, div_33, buf544, primals_99, buf546, mul_64, buf548, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del div_33
        del mul_64
        del primals_99
        buf553 = buf519; del buf519  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf552, (1568, 384), (384, 1), 0), permute_470, out=buf553)
        del permute_470
        buf554 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf552, (384, 1568), (1, 384), 0), view_63, out=buf554)
        del view_63
        buf555 = reinterpret_tensor(buf549, (1, 384, 13), (4992, 1, 384), 0); del buf549  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf552, buf555, 4992, 121, grid=grid(4992), stream=stream0)
        buf556 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf555, buf556, 384, 13, grid=grid(384), stream=stream0)
        buf557 = buf523; del buf523  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf553, getitem_63, getitem_62, buf557, 2408448, grid=grid(2408448), stream=stream0)
        del getitem_62
        del getitem_63
        buf558 = reinterpret_tensor(buf544, (1568, 384), (384, 1), 0); del buf544  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf557, (1568, 1536), (1536, 1), 0), permute_475, out=buf558)
        del permute_475
        buf559 = reinterpret_tensor(buf536, (1536, 384), (384, 1), 0); del buf536  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf557, (1536, 1568), (1, 1536), 0), view_61, out=buf559)
        del view_61
        buf560 = buf526; del buf526  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf557, buf560, 19968, 121, grid=grid(19968), stream=stream0)
        buf561 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf560, buf561, 1536, 13, grid=grid(1536), stream=stream0)
        buf568 = buf552; del buf552  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf568, buf558, primals_93, mul_60, div_34, 1568, 384, grid=grid(1568), stream=stream0)
        del div_34
        del primals_93
        buf564 = reinterpret_tensor(buf555, (384, 13), (1, 384), 0); del buf555  # reuse
        buf566 = buf530; del buf530  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf558, mul_60, buf564, buf566, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_60
        buf565 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf564, buf565, 384, 13, grid=grid(384), stream=stream0)
        buf567 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf566, buf567, 384, 13, grid=grid(384), stream=stream0)
        buf569 = reinterpret_tensor(buf558, (3072, 196), (196, 1), 0); del buf558  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf568, buf569, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf570 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf569, permute_480, out=buf570)
        del permute_480
        buf571 = empty((196, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf569, (196, 3072), (1, 196), 0), view_59, out=buf571)
        del view_59
        buf572 = reinterpret_tensor(buf547, (1, 196, 24), (4704, 1, 196), 0); del buf547  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf569, buf572, 4704, 128, grid=grid(4704), stream=stream0)
        buf573 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf572, buf573, 196, 24, grid=grid(196), stream=stream0)
        buf574 = buf540; del buf540  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_14.run(buf570, getitem_59, getitem_58, buf574, 1179648, grid=grid(1179648), stream=stream0)
        del getitem_58
        del getitem_59
        buf575 = buf541; del buf541  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf574, buf575, 9216, 128, grid=grid(9216), stream=stream0)
        buf576 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf575, buf576, 384, 24, grid=grid(384), stream=stream0)
        buf577 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf574, (384, 3072), (1, 384), 0), view_57, out=buf577)
        del view_57
        buf578 = buf569; del buf569  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf574, (3072, 384), (384, 1), 0), permute_487, out=buf578)
        del permute_487
        buf579 = reinterpret_tensor(buf572, (8, 196, 1, 3), (588, 1, 4704, 196), 0); del buf572  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf578, primals_87, buf579, 4704, 128, grid=grid(4704), stream=stream0)
        buf580 = buf548; del buf548  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf579, buf580, 1568, 3, grid=grid(1568), stream=stream0)
        buf581 = reinterpret_tensor(buf579, (8, 196, 1, 3), (588, 3, 4704, 1), 0); del buf579  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf578, primals_87, mul_56, buf581, 4704, 128, grid=grid(4704), stream=stream0)
        buf582 = buf546; del buf546  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf581, buf582, 1568, 3, grid=grid(1568), stream=stream0)
        buf583 = reinterpret_tensor(buf566, (384, 13), (13, 1), 0); del buf566  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf578, mul_56, buf583, 4992, 121, grid=grid(4992), stream=stream0)
        buf584 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf583, buf584, 384, 13, grid=grid(384), stream=stream0)
        buf585 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf578, buf585, 384, 1568, grid=grid(384), stream=stream0)
        buf586 = buf568; del buf568  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf586, div_35, buf578, primals_87, buf580, mul_56, buf582, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del div_35
        del mul_56
        del primals_87
        buf587 = buf553; del buf553  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf586, (1568, 384), (384, 1), 0), permute_490, out=buf587)
        del permute_490
        buf588 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf586, (384, 1568), (1, 384), 0), view_55, out=buf588)
        del view_55
        buf589 = reinterpret_tensor(buf583, (1, 384, 13), (4992, 1, 384), 0); del buf583  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf586, buf589, 4992, 121, grid=grid(4992), stream=stream0)
        buf590 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf589, buf590, 384, 13, grid=grid(384), stream=stream0)
        buf591 = buf557; del buf557  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf587, getitem_55, getitem_54, buf591, 2408448, grid=grid(2408448), stream=stream0)
        del getitem_54
        del getitem_55
        buf592 = reinterpret_tensor(buf578, (1568, 384), (384, 1), 0); del buf578  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf591, (1568, 1536), (1536, 1), 0), permute_495, out=buf592)
        del permute_495
        buf593 = reinterpret_tensor(buf570, (1536, 384), (384, 1), 0); del buf570  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf591, (1536, 1568), (1, 1536), 0), view_53, out=buf593)
        del view_53
        buf594 = buf560; del buf560  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf591, buf594, 19968, 121, grid=grid(19968), stream=stream0)
        buf595 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf594, buf595, 1536, 13, grid=grid(1536), stream=stream0)
        buf602 = buf586; del buf586  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf602, buf592, primals_81, mul_52, div_36, 1568, 384, grid=grid(1568), stream=stream0)
        del div_36
        del primals_81
        buf598 = reinterpret_tensor(buf589, (384, 13), (1, 384), 0); del buf589  # reuse
        buf600 = buf564; del buf564  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf592, mul_52, buf598, buf600, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_52
        buf599 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf598, buf599, 384, 13, grid=grid(384), stream=stream0)
        buf601 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf600, buf601, 384, 13, grid=grid(384), stream=stream0)
        buf603 = reinterpret_tensor(buf592, (3072, 196), (196, 1), 0); del buf592  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf602, buf603, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf604 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf603, permute_500, out=buf604)
        del permute_500
        buf605 = empty((196, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf603, (196, 3072), (1, 196), 0), view_51, out=buf605)
        del view_51
        buf606 = reinterpret_tensor(buf581, (1, 196, 24), (4704, 1, 196), 0); del buf581  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf603, buf606, 4704, 128, grid=grid(4704), stream=stream0)
        buf607 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf606, buf607, 196, 24, grid=grid(196), stream=stream0)
        buf608 = buf574; del buf574  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_14.run(buf604, getitem_51, getitem_50, buf608, 1179648, grid=grid(1179648), stream=stream0)
        del getitem_50
        del getitem_51
        buf609 = buf575; del buf575  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf608, buf609, 9216, 128, grid=grid(9216), stream=stream0)
        buf610 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf609, buf610, 384, 24, grid=grid(384), stream=stream0)
        buf611 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf608, (384, 3072), (1, 384), 0), view_49, out=buf611)
        del view_49
        buf612 = buf603; del buf603  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf608, (3072, 384), (384, 1), 0), permute_507, out=buf612)
        del permute_507
        buf613 = reinterpret_tensor(buf606, (8, 196, 1, 3), (588, 1, 4704, 196), 0); del buf606  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf612, primals_75, buf613, 4704, 128, grid=grid(4704), stream=stream0)
        buf614 = buf582; del buf582  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf613, buf614, 1568, 3, grid=grid(1568), stream=stream0)
        buf615 = reinterpret_tensor(buf613, (8, 196, 1, 3), (588, 3, 4704, 1), 0); del buf613  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf612, primals_75, mul_48, buf615, 4704, 128, grid=grid(4704), stream=stream0)
        buf616 = buf580; del buf580  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf615, buf616, 1568, 3, grid=grid(1568), stream=stream0)
        buf617 = reinterpret_tensor(buf600, (384, 13), (13, 1), 0); del buf600  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf612, mul_48, buf617, 4992, 121, grid=grid(4992), stream=stream0)
        buf618 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf617, buf618, 384, 13, grid=grid(384), stream=stream0)
        buf619 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf612, buf619, 384, 1568, grid=grid(384), stream=stream0)
        buf620 = buf602; del buf602  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf620, div_37, buf612, primals_75, buf614, mul_48, buf616, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del div_37
        del mul_48
        del primals_75
        buf621 = buf587; del buf587  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf620, (1568, 384), (384, 1), 0), permute_510, out=buf621)
        del permute_510
        buf622 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf620, (384, 1568), (1, 384), 0), view_47, out=buf622)
        del view_47
        buf623 = reinterpret_tensor(buf617, (1, 384, 13), (4992, 1, 384), 0); del buf617  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf620, buf623, 4992, 121, grid=grid(4992), stream=stream0)
        buf624 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf623, buf624, 384, 13, grid=grid(384), stream=stream0)
        buf625 = buf591; del buf591  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf621, getitem_47, getitem_46, buf625, 2408448, grid=grid(2408448), stream=stream0)
        del getitem_46
        del getitem_47
        buf626 = reinterpret_tensor(buf612, (1568, 384), (384, 1), 0); del buf612  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf625, (1568, 1536), (1536, 1), 0), permute_515, out=buf626)
        del permute_515
        buf627 = reinterpret_tensor(buf604, (1536, 384), (384, 1), 0); del buf604  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf625, (1536, 1568), (1, 1536), 0), view_45, out=buf627)
        del view_45
        buf628 = buf594; del buf594  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf625, buf628, 19968, 121, grid=grid(19968), stream=stream0)
        buf629 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf628, buf629, 1536, 13, grid=grid(1536), stream=stream0)
        buf636 = buf620; del buf620  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf636, buf626, primals_69, mul_44, div_38, 1568, 384, grid=grid(1568), stream=stream0)
        del div_38
        del primals_69
        buf632 = reinterpret_tensor(buf623, (384, 13), (1, 384), 0); del buf623  # reuse
        buf634 = buf598; del buf598  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf626, mul_44, buf632, buf634, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_44
        buf633 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf632, buf633, 384, 13, grid=grid(384), stream=stream0)
        buf635 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf634, buf635, 384, 13, grid=grid(384), stream=stream0)
        buf637 = reinterpret_tensor(buf626, (3072, 196), (196, 1), 0); del buf626  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf636, buf637, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf638 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf637, permute_520, out=buf638)
        del permute_520
        buf639 = empty((196, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf637, (196, 3072), (1, 196), 0), view_43, out=buf639)
        del view_43
        buf640 = reinterpret_tensor(buf615, (1, 196, 24), (4704, 1, 196), 0); del buf615  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf637, buf640, 4704, 128, grid=grid(4704), stream=stream0)
        buf641 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf640, buf641, 196, 24, grid=grid(196), stream=stream0)
        buf642 = buf608; del buf608  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_14.run(buf638, getitem_43, getitem_42, buf642, 1179648, grid=grid(1179648), stream=stream0)
        del getitem_42
        del getitem_43
        buf643 = buf609; del buf609  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf642, buf643, 9216, 128, grid=grid(9216), stream=stream0)
        buf644 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf643, buf644, 384, 24, grid=grid(384), stream=stream0)
        buf645 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf642, (384, 3072), (1, 384), 0), view_41, out=buf645)
        del view_41
        buf646 = buf637; del buf637  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf642, (3072, 384), (384, 1), 0), permute_527, out=buf646)
        del permute_527
        buf647 = reinterpret_tensor(buf640, (8, 196, 1, 3), (588, 1, 4704, 196), 0); del buf640  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf646, primals_63, buf647, 4704, 128, grid=grid(4704), stream=stream0)
        buf648 = buf616; del buf616  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf647, buf648, 1568, 3, grid=grid(1568), stream=stream0)
        buf649 = reinterpret_tensor(buf647, (8, 196, 1, 3), (588, 3, 4704, 1), 0); del buf647  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf646, primals_63, mul_40, buf649, 4704, 128, grid=grid(4704), stream=stream0)
        buf650 = buf614; del buf614  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf649, buf650, 1568, 3, grid=grid(1568), stream=stream0)
        buf651 = reinterpret_tensor(buf634, (384, 13), (13, 1), 0); del buf634  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf646, mul_40, buf651, 4992, 121, grid=grid(4992), stream=stream0)
        buf652 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf651, buf652, 384, 13, grid=grid(384), stream=stream0)
        buf653 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf646, buf653, 384, 1568, grid=grid(384), stream=stream0)
        buf654 = buf636; del buf636  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf654, div_39, buf646, primals_63, buf648, mul_40, buf650, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del div_39
        del mul_40
        del primals_63
        buf655 = buf621; del buf621  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf654, (1568, 384), (384, 1), 0), permute_530, out=buf655)
        del permute_530
        buf656 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf654, (384, 1568), (1, 384), 0), view_39, out=buf656)
        del view_39
        buf657 = reinterpret_tensor(buf651, (1, 384, 13), (4992, 1, 384), 0); del buf651  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf654, buf657, 4992, 121, grid=grid(4992), stream=stream0)
        buf658 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf657, buf658, 384, 13, grid=grid(384), stream=stream0)
        buf659 = buf625; del buf625  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf655, getitem_39, getitem_38, buf659, 2408448, grid=grid(2408448), stream=stream0)
        del getitem_38
        del getitem_39
        buf660 = reinterpret_tensor(buf646, (1568, 384), (384, 1), 0); del buf646  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf659, (1568, 1536), (1536, 1), 0), permute_535, out=buf660)
        del permute_535
        buf661 = reinterpret_tensor(buf638, (1536, 384), (384, 1), 0); del buf638  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf659, (1536, 1568), (1, 1536), 0), view_37, out=buf661)
        del view_37
        buf662 = buf628; del buf628  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf659, buf662, 19968, 121, grid=grid(19968), stream=stream0)
        buf663 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf662, buf663, 1536, 13, grid=grid(1536), stream=stream0)
        buf670 = buf654; del buf654  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf670, buf660, primals_57, mul_36, div_40, 1568, 384, grid=grid(1568), stream=stream0)
        del div_40
        del primals_57
        buf666 = reinterpret_tensor(buf657, (384, 13), (1, 384), 0); del buf657  # reuse
        buf668 = buf632; del buf632  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf660, mul_36, buf666, buf668, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_36
        buf667 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf666, buf667, 384, 13, grid=grid(384), stream=stream0)
        buf669 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf668, buf669, 384, 13, grid=grid(384), stream=stream0)
        buf671 = reinterpret_tensor(buf660, (3072, 196), (196, 1), 0); del buf660  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf670, buf671, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf672 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf671, permute_540, out=buf672)
        del permute_540
        buf673 = empty((196, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf671, (196, 3072), (1, 196), 0), view_35, out=buf673)
        del view_35
        buf674 = reinterpret_tensor(buf649, (1, 196, 24), (4704, 1, 196), 0); del buf649  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf671, buf674, 4704, 128, grid=grid(4704), stream=stream0)
        buf675 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf674, buf675, 196, 24, grid=grid(196), stream=stream0)
        buf676 = buf642; del buf642  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_14.run(buf672, getitem_35, getitem_34, buf676, 1179648, grid=grid(1179648), stream=stream0)
        del getitem_34
        del getitem_35
        buf677 = buf643; del buf643  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf676, buf677, 9216, 128, grid=grid(9216), stream=stream0)
        buf678 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf677, buf678, 384, 24, grid=grid(384), stream=stream0)
        buf679 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf676, (384, 3072), (1, 384), 0), view_33, out=buf679)
        del view_33
        buf680 = buf671; del buf671  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf676, (3072, 384), (384, 1), 0), permute_547, out=buf680)
        del permute_547
        buf681 = reinterpret_tensor(buf674, (8, 196, 1, 3), (588, 1, 4704, 196), 0); del buf674  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf680, primals_51, buf681, 4704, 128, grid=grid(4704), stream=stream0)
        buf682 = buf650; del buf650  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf681, buf682, 1568, 3, grid=grid(1568), stream=stream0)
        buf683 = reinterpret_tensor(buf681, (8, 196, 1, 3), (588, 3, 4704, 1), 0); del buf681  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf680, primals_51, mul_32, buf683, 4704, 128, grid=grid(4704), stream=stream0)
        buf684 = buf648; del buf648  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf683, buf684, 1568, 3, grid=grid(1568), stream=stream0)
        buf685 = reinterpret_tensor(buf668, (384, 13), (13, 1), 0); del buf668  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf680, mul_32, buf685, 4992, 121, grid=grid(4992), stream=stream0)
        buf686 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf685, buf686, 384, 13, grid=grid(384), stream=stream0)
        buf687 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf680, buf687, 384, 1568, grid=grid(384), stream=stream0)
        buf688 = buf670; del buf670  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf688, div_41, buf680, primals_51, buf682, mul_32, buf684, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del div_41
        del mul_32
        del primals_51
        buf689 = buf655; del buf655  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf688, (1568, 384), (384, 1), 0), permute_550, out=buf689)
        del permute_550
        buf690 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf688, (384, 1568), (1, 384), 0), view_31, out=buf690)
        del view_31
        buf691 = reinterpret_tensor(buf685, (1, 384, 13), (4992, 1, 384), 0); del buf685  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf688, buf691, 4992, 121, grid=grid(4992), stream=stream0)
        buf692 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf691, buf692, 384, 13, grid=grid(384), stream=stream0)
        buf693 = buf659; del buf659  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf689, getitem_31, getitem_30, buf693, 2408448, grid=grid(2408448), stream=stream0)
        del getitem_30
        del getitem_31
        buf694 = reinterpret_tensor(buf680, (1568, 384), (384, 1), 0); del buf680  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf693, (1568, 1536), (1536, 1), 0), permute_555, out=buf694)
        del permute_555
        buf695 = reinterpret_tensor(buf672, (1536, 384), (384, 1), 0); del buf672  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf693, (1536, 1568), (1, 1536), 0), view_29, out=buf695)
        del view_29
        buf696 = buf662; del buf662  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf693, buf696, 19968, 121, grid=grid(19968), stream=stream0)
        buf697 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf696, buf697, 1536, 13, grid=grid(1536), stream=stream0)
        buf704 = buf688; del buf688  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf704, buf694, primals_45, mul_28, div_42, 1568, 384, grid=grid(1568), stream=stream0)
        del div_42
        del primals_45
        buf700 = reinterpret_tensor(buf691, (384, 13), (1, 384), 0); del buf691  # reuse
        buf702 = buf666; del buf666  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf694, mul_28, buf700, buf702, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_28
        buf701 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf700, buf701, 384, 13, grid=grid(384), stream=stream0)
        buf703 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf702, buf703, 384, 13, grid=grid(384), stream=stream0)
        buf705 = reinterpret_tensor(buf694, (3072, 196), (196, 1), 0); del buf694  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf704, buf705, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf706 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf705, permute_560, out=buf706)
        del permute_560
        buf707 = empty((196, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf705, (196, 3072), (1, 196), 0), view_27, out=buf707)
        del view_27
        buf708 = reinterpret_tensor(buf683, (1, 196, 24), (4704, 1, 196), 0); del buf683  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf705, buf708, 4704, 128, grid=grid(4704), stream=stream0)
        buf709 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf708, buf709, 196, 24, grid=grid(196), stream=stream0)
        buf710 = buf676; del buf676  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_14.run(buf706, getitem_27, getitem_26, buf710, 1179648, grid=grid(1179648), stream=stream0)
        del getitem_26
        del getitem_27
        buf711 = buf677; del buf677  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf710, buf711, 9216, 128, grid=grid(9216), stream=stream0)
        buf712 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf711, buf712, 384, 24, grid=grid(384), stream=stream0)
        buf713 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf710, (384, 3072), (1, 384), 0), view_25, out=buf713)
        del view_25
        buf714 = buf705; del buf705  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf710, (3072, 384), (384, 1), 0), permute_567, out=buf714)
        del permute_567
        buf715 = reinterpret_tensor(buf708, (8, 196, 1, 3), (588, 1, 4704, 196), 0); del buf708  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf714, primals_39, buf715, 4704, 128, grid=grid(4704), stream=stream0)
        buf716 = buf684; del buf684  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf715, buf716, 1568, 3, grid=grid(1568), stream=stream0)
        buf717 = reinterpret_tensor(buf715, (8, 196, 1, 3), (588, 3, 4704, 1), 0); del buf715  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf714, primals_39, mul_24, buf717, 4704, 128, grid=grid(4704), stream=stream0)
        buf718 = buf682; del buf682  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf717, buf718, 1568, 3, grid=grid(1568), stream=stream0)
        buf719 = reinterpret_tensor(buf702, (384, 13), (13, 1), 0); del buf702  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf714, mul_24, buf719, 4992, 121, grid=grid(4992), stream=stream0)
        buf720 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf719, buf720, 384, 13, grid=grid(384), stream=stream0)
        buf721 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf714, buf721, 384, 1568, grid=grid(384), stream=stream0)
        buf722 = buf704; del buf704  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf722, div_43, buf714, primals_39, buf716, mul_24, buf718, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del div_43
        del mul_24
        del primals_39
        buf723 = buf689; del buf689  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf722, (1568, 384), (384, 1), 0), permute_570, out=buf723)
        del permute_570
        buf724 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf722, (384, 1568), (1, 384), 0), view_23, out=buf724)
        del view_23
        buf725 = reinterpret_tensor(buf719, (1, 384, 13), (4992, 1, 384), 0); del buf719  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf722, buf725, 4992, 121, grid=grid(4992), stream=stream0)
        buf726 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf725, buf726, 384, 13, grid=grid(384), stream=stream0)
        buf727 = buf693; del buf693  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf723, getitem_23, getitem_22, buf727, 2408448, grid=grid(2408448), stream=stream0)
        del getitem_22
        del getitem_23
        buf728 = reinterpret_tensor(buf714, (1568, 384), (384, 1), 0); del buf714  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf727, (1568, 1536), (1536, 1), 0), permute_575, out=buf728)
        del permute_575
        buf729 = reinterpret_tensor(buf706, (1536, 384), (384, 1), 0); del buf706  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf727, (1536, 1568), (1, 1536), 0), view_21, out=buf729)
        del view_21
        buf730 = buf696; del buf696  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf727, buf730, 19968, 121, grid=grid(19968), stream=stream0)
        buf731 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf730, buf731, 1536, 13, grid=grid(1536), stream=stream0)
        buf738 = buf722; del buf722  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf738, buf728, primals_33, mul_20, div_44, 1568, 384, grid=grid(1568), stream=stream0)
        del div_44
        del primals_33
        buf734 = reinterpret_tensor(buf725, (384, 13), (1, 384), 0); del buf725  # reuse
        buf736 = buf700; del buf700  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf728, mul_20, buf734, buf736, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_20
        buf735 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf734, buf735, 384, 13, grid=grid(384), stream=stream0)
        buf737 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf736, buf737, 384, 13, grid=grid(384), stream=stream0)
        buf739 = reinterpret_tensor(buf728, (3072, 196), (196, 1), 0); del buf728  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf738, buf739, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf740 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf739, permute_580, out=buf740)
        del permute_580
        buf741 = empty((196, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf739, (196, 3072), (1, 196), 0), view_19, out=buf741)
        del view_19
        buf742 = reinterpret_tensor(buf717, (1, 196, 24), (4704, 1, 196), 0); del buf717  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf739, buf742, 4704, 128, grid=grid(4704), stream=stream0)
        buf743 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf742, buf743, 196, 24, grid=grid(196), stream=stream0)
        buf744 = buf710; del buf710  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_14.run(buf740, getitem_19, getitem_18, buf744, 1179648, grid=grid(1179648), stream=stream0)
        del getitem_18
        del getitem_19
        buf745 = buf711; del buf711  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf744, buf745, 9216, 128, grid=grid(9216), stream=stream0)
        buf746 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf745, buf746, 384, 24, grid=grid(384), stream=stream0)
        buf747 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf744, (384, 3072), (1, 384), 0), view_17, out=buf747)
        del view_17
        buf748 = buf739; del buf739  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf744, (3072, 384), (384, 1), 0), permute_587, out=buf748)
        del permute_587
        buf749 = reinterpret_tensor(buf742, (8, 196, 1, 3), (588, 1, 4704, 196), 0); del buf742  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf748, primals_27, buf749, 4704, 128, grid=grid(4704), stream=stream0)
        buf750 = buf718; del buf718  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf749, buf750, 1568, 3, grid=grid(1568), stream=stream0)
        buf751 = reinterpret_tensor(buf749, (8, 196, 1, 3), (588, 3, 4704, 1), 0); del buf749  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf748, primals_27, mul_16, buf751, 4704, 128, grid=grid(4704), stream=stream0)
        buf752 = buf716; del buf716  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf751, buf752, 1568, 3, grid=grid(1568), stream=stream0)
        buf753 = reinterpret_tensor(buf736, (384, 13), (13, 1), 0); del buf736  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf748, mul_16, buf753, 4992, 121, grid=grid(4992), stream=stream0)
        buf754 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf753, buf754, 384, 13, grid=grid(384), stream=stream0)
        buf755 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf748, buf755, 384, 1568, grid=grid(384), stream=stream0)
        buf756 = buf738; del buf738  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf756, div_45, buf748, primals_27, buf750, mul_16, buf752, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del div_45
        del mul_16
        del primals_27
        buf757 = buf723; del buf723  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf756, (1568, 384), (384, 1), 0), permute_590, out=buf757)
        del permute_590
        buf758 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf756, (384, 1568), (1, 384), 0), view_15, out=buf758)
        del view_15
        buf759 = reinterpret_tensor(buf753, (1, 384, 13), (4992, 1, 384), 0); del buf753  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf756, buf759, 4992, 121, grid=grid(4992), stream=stream0)
        buf760 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf759, buf760, 384, 13, grid=grid(384), stream=stream0)
        buf761 = buf727; del buf727  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf757, getitem_15, getitem_14, buf761, 2408448, grid=grid(2408448), stream=stream0)
        del getitem_14
        del getitem_15
        buf762 = reinterpret_tensor(buf748, (1568, 384), (384, 1), 0); del buf748  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf761, (1568, 1536), (1536, 1), 0), permute_595, out=buf762)
        del permute_595
        buf763 = reinterpret_tensor(buf740, (1536, 384), (384, 1), 0); del buf740  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf761, (1536, 1568), (1, 1536), 0), view_13, out=buf763)
        del view_13
        buf764 = buf730; del buf730  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf761, buf764, 19968, 121, grid=grid(19968), stream=stream0)
        buf765 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf764, buf765, 1536, 13, grid=grid(1536), stream=stream0)
        buf772 = buf756; del buf756  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf772, buf762, primals_21, mul_12, div_46, 1568, 384, grid=grid(1568), stream=stream0)
        del div_46
        del primals_21
        buf768 = reinterpret_tensor(buf759, (384, 13), (1, 384), 0); del buf759  # reuse
        buf770 = buf734; del buf734  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf762, mul_12, buf768, buf770, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_12
        buf769 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf768, buf769, 384, 13, grid=grid(384), stream=stream0)
        buf771 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf770, buf771, 384, 13, grid=grid(384), stream=stream0)
        buf773 = reinterpret_tensor(buf762, (3072, 196), (196, 1), 0); del buf762  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf772, buf773, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf774 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf773, permute_600, out=buf774)
        del permute_600
        buf775 = empty((196, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf773, (196, 3072), (1, 196), 0), view_11, out=buf775)
        del view_11
        buf776 = reinterpret_tensor(buf751, (1, 196, 24), (4704, 1, 196), 0); del buf751  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf773, buf776, 4704, 128, grid=grid(4704), stream=stream0)
        buf777 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf776, buf777, 196, 24, grid=grid(196), stream=stream0)
        buf778 = buf744; del buf744  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_14.run(buf774, getitem_11, getitem_10, buf778, 1179648, grid=grid(1179648), stream=stream0)
        del getitem_10
        del getitem_11
        buf779 = buf745; del buf745  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf778, buf779, 9216, 128, grid=grid(9216), stream=stream0)
        buf780 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf779, buf780, 384, 24, grid=grid(384), stream=stream0)
        buf781 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf778, (384, 3072), (1, 384), 0), view_9, out=buf781)
        del view_9
        buf782 = buf773; del buf773  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf778, (3072, 384), (384, 1), 0), permute_607, out=buf782)
        del permute_607
        buf783 = reinterpret_tensor(buf776, (8, 196, 1, 3), (588, 1, 4704, 196), 0); del buf776  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf782, primals_15, buf783, 4704, 128, grid=grid(4704), stream=stream0)
        buf784 = buf752; del buf752  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf783, buf784, 1568, 3, grid=grid(1568), stream=stream0)
        buf785 = reinterpret_tensor(buf783, (8, 196, 1, 3), (588, 3, 4704, 1), 0); del buf783  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf782, primals_15, mul_8, buf785, 4704, 128, grid=grid(4704), stream=stream0)
        buf786 = buf750; del buf750  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf785, buf786, 1568, 3, grid=grid(1568), stream=stream0)
        buf787 = reinterpret_tensor(buf770, (384, 13), (13, 1), 0); del buf770  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf782, mul_8, buf787, 4992, 121, grid=grid(4992), stream=stream0)
        buf788 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf787, buf788, 384, 13, grid=grid(384), stream=stream0)
        buf789 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf782, buf789, 384, 1568, grid=grid(384), stream=stream0)
        buf790 = buf772; del buf772  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf790, div_47, buf782, primals_15, buf784, mul_8, buf786, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del div_47
        del mul_8
        del primals_15
        buf791 = buf757; del buf757  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf790, (1568, 384), (384, 1), 0), permute_610, out=buf791)
        del permute_610
        buf792 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf790, (384, 1568), (1, 384), 0), view_7, out=buf792)
        del view_7
        buf793 = reinterpret_tensor(buf787, (1, 384, 13), (4992, 1, 384), 0); del buf787  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf790, buf793, 4992, 121, grid=grid(4992), stream=stream0)
        buf794 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf793, buf794, 384, 13, grid=grid(384), stream=stream0)
        buf795 = buf761; del buf761  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf791, getitem_7, getitem_6, buf795, 2408448, grid=grid(2408448), stream=stream0)
        del buf791
        del getitem_6
        del getitem_7
        buf796 = reinterpret_tensor(buf782, (1568, 384), (384, 1), 0); del buf782  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf795, (1568, 1536), (1536, 1), 0), permute_615, out=buf796)
        del permute_615
        buf797 = reinterpret_tensor(buf774, (1536, 384), (384, 1), 0); del buf774  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf795, (1536, 1568), (1, 1536), 0), view_5, out=buf797)
        del view_5
        buf798 = buf764; del buf764  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf795, buf798, 19968, 121, grid=grid(19968), stream=stream0)
        del buf795
        buf799 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf798, buf799, 1536, 13, grid=grid(1536), stream=stream0)
        del buf798
        buf806 = buf790; del buf790  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf806, buf796, primals_9, mul_4, div_48, 1568, 384, grid=grid(1568), stream=stream0)
        del div_48
        del primals_9
        buf802 = reinterpret_tensor(buf793, (384, 13), (1, 384), 0); del buf793  # reuse
        buf804 = buf768; del buf768  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf796, mul_4, buf802, buf804, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_4
        buf803 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf802, buf803, 384, 13, grid=grid(384), stream=stream0)
        del buf802
        buf805 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf804, buf805, 384, 13, grid=grid(384), stream=stream0)
        buf807 = reinterpret_tensor(buf796, (3072, 196), (196, 1), 0); del buf796  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_11.run(buf806, buf807, 3072, 196, grid=grid(3072, 196), stream=stream0)
        buf808 = empty((3072, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf807, permute_620, out=buf808)
        del permute_620
        buf809 = empty((196, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf807, (196, 3072), (1, 196), 0), view_3, out=buf809)
        del view_3
        buf810 = reinterpret_tensor(buf785, (1, 196, 24), (4704, 1, 196), 0); del buf785  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf807, buf810, 4704, 128, grid=grid(4704), stream=stream0)
        buf811 = empty((1, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf810, buf811, 196, 24, grid=grid(196), stream=stream0)
        buf812 = buf778; del buf778  # reuse
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_14.run(buf808, getitem_3, getitem_2, buf812, 1179648, grid=grid(1179648), stream=stream0)
        del buf808
        del getitem_2
        del getitem_3
        buf813 = buf779; del buf779  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_15.run(buf812, buf813, 9216, 128, grid=grid(9216), stream=stream0)
        buf814 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf813, buf814, 384, 24, grid=grid(384), stream=stream0)
        del buf813
        buf815 = empty((384, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf812, (384, 3072), (1, 384), 0), view_1, out=buf815)
        del view_1
        buf816 = buf807; del buf807  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf812, (3072, 384), (384, 1), 0), permute_627, out=buf816)
        del buf812
        del permute_627
        buf817 = reinterpret_tensor(buf810, (8, 196, 1, 3), (588, 1, 4704, 196), 0); del buf810  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_17.run(buf816, primals_3, buf817, 4704, 128, grid=grid(4704), stream=stream0)
        buf818 = buf786; del buf786  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_18.run(buf817, buf818, 1568, 3, grid=grid(1568), stream=stream0)
        buf819 = reinterpret_tensor(buf817, (8, 196, 1, 3), (588, 3, 4704, 1), 0); del buf817  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_19.run(buf816, primals_3, mul, buf819, 4704, 128, grid=grid(4704), stream=stream0)
        buf820 = buf784; del buf784  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_20.run(buf819, buf820, 1568, 3, grid=grid(1568), stream=stream0)
        del buf819
        buf821 = reinterpret_tensor(buf804, (384, 13), (13, 1), 0); del buf804  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_21.run(buf816, mul, buf821, 4992, 121, grid=grid(4992), stream=stream0)
        buf822 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_22.run(buf821, buf822, 384, 13, grid=grid(384), stream=stream0)
        buf823 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_23.run(buf816, buf823, 384, 1568, grid=grid(384), stream=stream0)
        buf824 = buf806; del buf806  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_poi_fused_add_native_layer_norm_backward_24.run(buf824, div_49, buf816, primals_3, buf818, mul, buf820, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del buf816
        del buf818
        del buf820
        del div_49
        del mul
        del primals_3
        buf825 = reinterpret_tensor(buf821, (384, 13), (1, 384), 0); del buf821  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_25.run(buf824, buf825, 4992, 121, grid=grid(4992), stream=stream0)
        buf826 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_div_native_layer_norm_backward_3.run(buf825, buf826, 384, 13, grid=grid(384), stream=stream0)
        del buf825
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf827 = aten.convolution_backward(reinterpret_tensor(buf824, (8, 384, 14, 14), (75264, 1, 5376, 384), 0), primals_295, primals_1, [384], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf824
        del primals_1
        del primals_295
        buf828 = buf827[1]
        return (buf828, buf826, buf822, buf823, reinterpret_tensor(buf815, (384, 196), (196, 1), 0), reinterpret_tensor(buf814, (384, ), (1, ), 0), reinterpret_tensor(buf809, (196, 192), (192, 1), 0), reinterpret_tensor(buf811, (196, ), (1, ), 0), buf803, buf805, reinterpret_tensor(buf797, (1536, 384), (384, 1), 0), reinterpret_tensor(buf799, (1536, ), (1, ), 0), reinterpret_tensor(buf792, (384, 768), (768, 1), 0), reinterpret_tensor(buf794, (384, ), (1, ), 0), buf788, buf789, reinterpret_tensor(buf781, (384, 196), (196, 1), 0), reinterpret_tensor(buf780, (384, ), (1, ), 0), reinterpret_tensor(buf775, (196, 192), (192, 1), 0), reinterpret_tensor(buf777, (196, ), (1, ), 0), buf769, buf771, reinterpret_tensor(buf763, (1536, 384), (384, 1), 0), reinterpret_tensor(buf765, (1536, ), (1, ), 0), reinterpret_tensor(buf758, (384, 768), (768, 1), 0), reinterpret_tensor(buf760, (384, ), (1, ), 0), buf754, buf755, reinterpret_tensor(buf747, (384, 196), (196, 1), 0), reinterpret_tensor(buf746, (384, ), (1, ), 0), reinterpret_tensor(buf741, (196, 192), (192, 1), 0), reinterpret_tensor(buf743, (196, ), (1, ), 0), buf735, buf737, reinterpret_tensor(buf729, (1536, 384), (384, 1), 0), reinterpret_tensor(buf731, (1536, ), (1, ), 0), reinterpret_tensor(buf724, (384, 768), (768, 1), 0), reinterpret_tensor(buf726, (384, ), (1, ), 0), buf720, buf721, reinterpret_tensor(buf713, (384, 196), (196, 1), 0), reinterpret_tensor(buf712, (384, ), (1, ), 0), reinterpret_tensor(buf707, (196, 192), (192, 1), 0), reinterpret_tensor(buf709, (196, ), (1, ), 0), buf701, buf703, reinterpret_tensor(buf695, (1536, 384), (384, 1), 0), reinterpret_tensor(buf697, (1536, ), (1, ), 0), reinterpret_tensor(buf690, (384, 768), (768, 1), 0), reinterpret_tensor(buf692, (384, ), (1, ), 0), buf686, buf687, reinterpret_tensor(buf679, (384, 196), (196, 1), 0), reinterpret_tensor(buf678, (384, ), (1, ), 0), reinterpret_tensor(buf673, (196, 192), (192, 1), 0), reinterpret_tensor(buf675, (196, ), (1, ), 0), buf667, buf669, reinterpret_tensor(buf661, (1536, 384), (384, 1), 0), reinterpret_tensor(buf663, (1536, ), (1, ), 0), reinterpret_tensor(buf656, (384, 768), (768, 1), 0), reinterpret_tensor(buf658, (384, ), (1, ), 0), buf652, buf653, reinterpret_tensor(buf645, (384, 196), (196, 1), 0), reinterpret_tensor(buf644, (384, ), (1, ), 0), reinterpret_tensor(buf639, (196, 192), (192, 1), 0), reinterpret_tensor(buf641, (196, ), (1, ), 0), buf633, buf635, reinterpret_tensor(buf627, (1536, 384), (384, 1), 0), reinterpret_tensor(buf629, (1536, ), (1, ), 0), reinterpret_tensor(buf622, (384, 768), (768, 1), 0), reinterpret_tensor(buf624, (384, ), (1, ), 0), buf618, buf619, reinterpret_tensor(buf611, (384, 196), (196, 1), 0), reinterpret_tensor(buf610, (384, ), (1, ), 0), reinterpret_tensor(buf605, (196, 192), (192, 1), 0), reinterpret_tensor(buf607, (196, ), (1, ), 0), buf599, buf601, reinterpret_tensor(buf593, (1536, 384), (384, 1), 0), reinterpret_tensor(buf595, (1536, ), (1, ), 0), reinterpret_tensor(buf588, (384, 768), (768, 1), 0), reinterpret_tensor(buf590, (384, ), (1, ), 0), buf584, buf585, reinterpret_tensor(buf577, (384, 196), (196, 1), 0), reinterpret_tensor(buf576, (384, ), (1, ), 0), reinterpret_tensor(buf571, (196, 192), (192, 1), 0), reinterpret_tensor(buf573, (196, ), (1, ), 0), buf565, buf567, reinterpret_tensor(buf559, (1536, 384), (384, 1), 0), reinterpret_tensor(buf561, (1536, ), (1, ), 0), reinterpret_tensor(buf554, (384, 768), (768, 1), 0), reinterpret_tensor(buf556, (384, ), (1, ), 0), buf550, buf551, reinterpret_tensor(buf543, (384, 196), (196, 1), 0), reinterpret_tensor(buf542, (384, ), (1, ), 0), reinterpret_tensor(buf537, (196, 192), (192, 1), 0), reinterpret_tensor(buf539, (196, ), (1, ), 0), buf531, buf533, reinterpret_tensor(buf525, (1536, 384), (384, 1), 0), reinterpret_tensor(buf527, (1536, ), (1, ), 0), reinterpret_tensor(buf520, (384, 768), (768, 1), 0), reinterpret_tensor(buf522, (384, ), (1, ), 0), buf516, buf517, reinterpret_tensor(buf509, (384, 196), (196, 1), 0), reinterpret_tensor(buf508, (384, ), (1, ), 0), reinterpret_tensor(buf503, (196, 192), (192, 1), 0), reinterpret_tensor(buf505, (196, ), (1, ), 0), buf497, buf499, reinterpret_tensor(buf491, (1536, 384), (384, 1), 0), reinterpret_tensor(buf493, (1536, ), (1, ), 0), reinterpret_tensor(buf486, (384, 768), (768, 1), 0), reinterpret_tensor(buf488, (384, ), (1, ), 0), buf482, buf483, reinterpret_tensor(buf475, (384, 196), (196, 1), 0), reinterpret_tensor(buf474, (384, ), (1, ), 0), reinterpret_tensor(buf469, (196, 192), (192, 1), 0), reinterpret_tensor(buf471, (196, ), (1, ), 0), buf463, buf465, reinterpret_tensor(buf457, (1536, 384), (384, 1), 0), reinterpret_tensor(buf459, (1536, ), (1, ), 0), reinterpret_tensor(buf452, (384, 768), (768, 1), 0), reinterpret_tensor(buf454, (384, ), (1, ), 0), buf448, buf449, reinterpret_tensor(buf441, (384, 196), (196, 1), 0), reinterpret_tensor(buf440, (384, ), (1, ), 0), reinterpret_tensor(buf435, (196, 192), (192, 1), 0), reinterpret_tensor(buf437, (196, ), (1, ), 0), buf429, buf431, reinterpret_tensor(buf423, (1536, 384), (384, 1), 0), reinterpret_tensor(buf425, (1536, ), (1, ), 0), reinterpret_tensor(buf418, (384, 768), (768, 1), 0), reinterpret_tensor(buf420, (384, ), (1, ), 0), buf414, buf415, reinterpret_tensor(buf407, (384, 196), (196, 1), 0), reinterpret_tensor(buf406, (384, ), (1, ), 0), reinterpret_tensor(buf401, (196, 192), (192, 1), 0), reinterpret_tensor(buf403, (196, ), (1, ), 0), buf395, buf397, reinterpret_tensor(buf389, (1536, 384), (384, 1), 0), reinterpret_tensor(buf391, (1536, ), (1, ), 0), reinterpret_tensor(buf384, (384, 768), (768, 1), 0), reinterpret_tensor(buf386, (384, ), (1, ), 0), buf380, buf381, reinterpret_tensor(buf373, (384, 196), (196, 1), 0), reinterpret_tensor(buf372, (384, ), (1, ), 0), reinterpret_tensor(buf367, (196, 192), (192, 1), 0), reinterpret_tensor(buf369, (196, ), (1, ), 0), buf361, buf363, reinterpret_tensor(buf355, (1536, 384), (384, 1), 0), reinterpret_tensor(buf357, (1536, ), (1, ), 0), reinterpret_tensor(buf350, (384, 768), (768, 1), 0), reinterpret_tensor(buf352, (384, ), (1, ), 0), buf346, buf347, reinterpret_tensor(buf339, (384, 196), (196, 1), 0), reinterpret_tensor(buf338, (384, ), (1, ), 0), reinterpret_tensor(buf333, (196, 192), (192, 1), 0), reinterpret_tensor(buf335, (196, ), (1, ), 0), buf327, buf329, reinterpret_tensor(buf321, (1536, 384), (384, 1), 0), reinterpret_tensor(buf323, (1536, ), (1, ), 0), reinterpret_tensor(buf316, (384, 768), (768, 1), 0), reinterpret_tensor(buf318, (384, ), (1, ), 0), buf312, buf313, reinterpret_tensor(buf305, (384, 196), (196, 1), 0), reinterpret_tensor(buf304, (384, ), (1, ), 0), reinterpret_tensor(buf299, (196, 192), (192, 1), 0), reinterpret_tensor(buf301, (196, ), (1, ), 0), buf293, buf295, reinterpret_tensor(buf287, (1536, 384), (384, 1), 0), reinterpret_tensor(buf289, (1536, ), (1, ), 0), reinterpret_tensor(buf282, (384, 768), (768, 1), 0), reinterpret_tensor(buf284, (384, ), (1, ), 0), buf278, buf279, reinterpret_tensor(buf271, (384, 196), (196, 1), 0), reinterpret_tensor(buf270, (384, ), (1, ), 0), reinterpret_tensor(buf265, (196, 192), (192, 1), 0), reinterpret_tensor(buf267, (196, ), (1, ), 0), buf259, buf261, reinterpret_tensor(buf253, (1536, 384), (384, 1), 0), reinterpret_tensor(buf255, (1536, ), (1, ), 0), reinterpret_tensor(buf248, (384, 768), (768, 1), 0), reinterpret_tensor(buf250, (384, ), (1, ), 0), buf244, buf245, reinterpret_tensor(buf237, (384, 196), (196, 1), 0), reinterpret_tensor(buf236, (384, ), (1, ), 0), reinterpret_tensor(buf231, (196, 192), (192, 1), 0), reinterpret_tensor(buf233, (196, ), (1, ), 0), buf225, buf227, reinterpret_tensor(buf219, (1536, 384), (384, 1), 0), reinterpret_tensor(buf221, (1536, ), (1, ), 0), reinterpret_tensor(buf214, (384, 768), (768, 1), 0), reinterpret_tensor(buf216, (384, ), (1, ), 0), buf210, buf211, reinterpret_tensor(buf203, (384, 196), (196, 1), 0), reinterpret_tensor(buf202, (384, ), (1, ), 0), reinterpret_tensor(buf197, (196, 192), (192, 1), 0), reinterpret_tensor(buf199, (196, ), (1, ), 0), buf191, buf193, reinterpret_tensor(buf185, (1536, 384), (384, 1), 0), reinterpret_tensor(buf187, (1536, ), (1, ), 0), reinterpret_tensor(buf180, (384, 768), (768, 1), 0), reinterpret_tensor(buf182, (384, ), (1, ), 0), buf176, buf177, reinterpret_tensor(buf169, (384, 196), (196, 1), 0), reinterpret_tensor(buf168, (384, ), (1, ), 0), reinterpret_tensor(buf163, (196, 192), (192, 1), 0), reinterpret_tensor(buf165, (196, ), (1, ), 0), buf157, buf159, reinterpret_tensor(buf151, (1536, 384), (384, 1), 0), reinterpret_tensor(buf153, (1536, ), (1, ), 0), reinterpret_tensor(buf146, (384, 768), (768, 1), 0), reinterpret_tensor(buf148, (384, ), (1, ), 0), buf142, buf143, reinterpret_tensor(buf135, (384, 196), (196, 1), 0), reinterpret_tensor(buf134, (384, ), (1, ), 0), reinterpret_tensor(buf129, (196, 192), (192, 1), 0), reinterpret_tensor(buf131, (196, ), (1, ), 0), buf123, buf125, reinterpret_tensor(buf117, (1536, 384), (384, 1), 0), reinterpret_tensor(buf119, (1536, ), (1, ), 0), reinterpret_tensor(buf112, (384, 768), (768, 1), 0), reinterpret_tensor(buf114, (384, ), (1, ), 0), buf108, buf109, reinterpret_tensor(buf101, (384, 196), (196, 1), 0), reinterpret_tensor(buf100, (384, ), (1, ), 0), reinterpret_tensor(buf95, (196, 192), (192, 1), 0), reinterpret_tensor(buf97, (196, ), (1, ), 0), buf89, buf91, reinterpret_tensor(buf83, (1536, 384), (384, 1), 0), reinterpret_tensor(buf85, (1536, ), (1, ), 0), reinterpret_tensor(buf78, (384, 768), (768, 1), 0), reinterpret_tensor(buf80, (384, ), (1, ), 0), buf74, buf75, reinterpret_tensor(buf67, (384, 196), (196, 1), 0), reinterpret_tensor(buf66, (384, ), (1, ), 0), reinterpret_tensor(buf61, (196, 192), (192, 1), 0), reinterpret_tensor(buf63, (196, ), (1, ), 0), buf55, buf57, reinterpret_tensor(buf49, (1536, 384), (384, 1), 0), reinterpret_tensor(buf51, (1536, ), (1, ), 0), reinterpret_tensor(buf44, (384, 768), (768, 1), 0), reinterpret_tensor(buf46, (384, ), (1, ), 0), buf40, buf41, reinterpret_tensor(buf33, (384, 196), (196, 1), 0), reinterpret_tensor(buf32, (384, ), (1, ), 0), reinterpret_tensor(buf27, (196, 192), (192, 1), 0), reinterpret_tensor(buf29, (196, ), (1, ), 0), buf21, buf23, reinterpret_tensor(buf15, (1536, 384), (384, 1), 0), reinterpret_tensor(buf17, (1536, ), (1, ), 0), reinterpret_tensor(buf10, (384, 768), (768, 1), 0), reinterpret_tensor(buf12, (384, ), (1, ), 0), buf7, buf8, reinterpret_tensor(buf1, (1000, 384), (384, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((384, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    mul = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_1 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    getitem_2 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    view_3 = rand_strided((3072, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_4 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_5 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_6 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_7 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_8 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_9 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    getitem_10 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_11 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    view_11 = rand_strided((3072, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_12 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_13 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_14 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_15 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_15 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_16 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_17 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    getitem_18 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_19 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    view_19 = rand_strided((3072, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_20 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_21 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_22 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_23 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_23 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_24 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_25 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    getitem_26 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_27 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    view_27 = rand_strided((3072, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_28 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_29 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_30 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_31 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_31 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_32 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_33 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    getitem_34 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_35 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    view_35 = rand_strided((3072, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_36 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_37 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_38 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_39 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_39 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_40 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_41 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    getitem_42 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_43 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    view_43 = rand_strided((3072, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_44 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_45 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_46 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_47 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_47 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_48 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_49 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    getitem_50 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_51 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    view_51 = rand_strided((3072, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_52 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_53 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_54 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_55 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_55 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_56 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_57 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    getitem_58 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_59 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    view_59 = rand_strided((3072, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_60 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_61 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_62 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_63 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_63 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_64 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_65 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    getitem_66 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_67 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    view_67 = rand_strided((3072, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_68 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_69 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_70 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_71 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_71 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_72 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_73 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    getitem_74 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_75 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    view_75 = rand_strided((3072, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_76 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_77 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_78 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_79 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_79 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_80 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_81 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    getitem_82 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_83 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    view_83 = rand_strided((3072, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_84 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_85 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_86 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_87 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_87 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_88 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_89 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    getitem_90 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_91 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    view_91 = rand_strided((3072, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_92 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_93 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_94 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_95 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_95 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_96 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_97 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    getitem_98 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_99 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    view_99 = rand_strided((3072, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_100 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_101 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_102 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_103 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_103 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_104 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_105 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    getitem_106 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_107 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    view_107 = rand_strided((3072, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_108 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_109 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_110 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_111 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_111 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_112 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_113 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    getitem_114 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_115 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    view_115 = rand_strided((3072, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_116 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_117 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_118 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_119 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_119 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_120 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_121 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    getitem_122 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_123 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    view_123 = rand_strided((3072, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_124 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_125 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_126 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_127 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_127 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_128 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_129 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    getitem_130 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_131 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    view_131 = rand_strided((3072, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_132 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_133 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_134 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_135 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_135 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_136 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_137 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    getitem_138 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_139 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    view_139 = rand_strided((3072, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_140 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_141 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_142 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_143 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_143 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_144 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_145 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    getitem_146 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_147 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    view_147 = rand_strided((3072, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_148 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_149 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_150 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_151 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_151 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_152 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_153 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    getitem_154 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_155 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    view_155 = rand_strided((3072, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_156 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_157 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_158 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_159 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_159 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_160 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_161 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    getitem_162 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_163 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    view_163 = rand_strided((3072, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_164 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_165 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_166 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_167 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_167 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_168 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_169 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    getitem_170 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_171 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    view_171 = rand_strided((3072, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_172 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_173 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_174 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_175 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_175 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_176 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_177 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    getitem_178 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_179 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    view_179 = rand_strided((3072, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_180 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_181 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_182 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_183 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_183 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_184 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_185 = rand_strided((3072, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    getitem_186 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_187 = rand_strided((8, 384, 192), (147456, 384, 1), device='cuda:0', dtype=torch.float32)
    view_187 = rand_strided((3072, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_188 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    view_189 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    getitem_190 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_191 = rand_strided((8, 196, 768), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    view_191 = rand_strided((1568, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_192 = rand_strided((8, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    clone_169 = rand_strided((8, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_146 = rand_strided((1000, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_1 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_150 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_155 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_2 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_160 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_167 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_170 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_175 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_4 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_180 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_187 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_5 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_190 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_195 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_6 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_200 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_207 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_7 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_210 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_215 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_8 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_220 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_227 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_9 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_230 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_235 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_10 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_240 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_247 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_11 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_250 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_255 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_12 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_260 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_267 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_13 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_270 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_275 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_14 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_280 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_287 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_15 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_290 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_295 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_16 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_300 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_307 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_17 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_310 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_315 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_18 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_320 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_327 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_19 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_330 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_335 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_20 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_340 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_347 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_21 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_350 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_355 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_22 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_360 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_367 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_23 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_370 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_375 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_24 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_380 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_387 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_25 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_390 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_395 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_26 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_400 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_407 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_27 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_410 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_415 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_28 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_420 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_427 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_29 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_430 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_435 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_30 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_440 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_447 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_31 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_450 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_455 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_32 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_460 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_467 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_33 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_470 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_475 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_34 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_480 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_487 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_35 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_490 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_495 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_36 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_500 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_507 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_37 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_510 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_515 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_38 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_520 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_527 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_39 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_530 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_535 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_40 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_540 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_547 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_41 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_550 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_555 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_42 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_560 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_567 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_43 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_570 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_575 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_44 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_580 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_587 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_45 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_590 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_595 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_46 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_600 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_607 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_47 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_610 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_615 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_48 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_620 = rand_strided((196, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_627 = rand_strided((384, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    div_49 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_9, primals_15, primals_21, primals_27, primals_33, primals_39, primals_45, primals_51, primals_57, primals_63, primals_69, primals_75, primals_81, primals_87, primals_93, primals_99, primals_105, primals_111, primals_117, primals_123, primals_129, primals_135, primals_141, primals_147, primals_153, primals_159, primals_165, primals_171, primals_177, primals_183, primals_189, primals_195, primals_201, primals_207, primals_213, primals_219, primals_225, primals_231, primals_237, primals_243, primals_249, primals_255, primals_261, primals_267, primals_273, primals_279, primals_285, primals_291, primals_295, mul, view_1, getitem_2, getitem_3, view_3, mul_4, view_5, getitem_6, getitem_7, view_7, mul_8, view_9, getitem_10, getitem_11, view_11, mul_12, view_13, getitem_14, getitem_15, view_15, mul_16, view_17, getitem_18, getitem_19, view_19, mul_20, view_21, getitem_22, getitem_23, view_23, mul_24, view_25, getitem_26, getitem_27, view_27, mul_28, view_29, getitem_30, getitem_31, view_31, mul_32, view_33, getitem_34, getitem_35, view_35, mul_36, view_37, getitem_38, getitem_39, view_39, mul_40, view_41, getitem_42, getitem_43, view_43, mul_44, view_45, getitem_46, getitem_47, view_47, mul_48, view_49, getitem_50, getitem_51, view_51, mul_52, view_53, getitem_54, getitem_55, view_55, mul_56, view_57, getitem_58, getitem_59, view_59, mul_60, view_61, getitem_62, getitem_63, view_63, mul_64, view_65, getitem_66, getitem_67, view_67, mul_68, view_69, getitem_70, getitem_71, view_71, mul_72, view_73, getitem_74, getitem_75, view_75, mul_76, view_77, getitem_78, getitem_79, view_79, mul_80, view_81, getitem_82, getitem_83, view_83, mul_84, view_85, getitem_86, getitem_87, view_87, mul_88, view_89, getitem_90, getitem_91, view_91, mul_92, view_93, getitem_94, getitem_95, view_95, mul_96, view_97, getitem_98, getitem_99, view_99, mul_100, view_101, getitem_102, getitem_103, view_103, mul_104, view_105, getitem_106, getitem_107, view_107, mul_108, view_109, getitem_110, getitem_111, view_111, mul_112, view_113, getitem_114, getitem_115, view_115, mul_116, view_117, getitem_118, getitem_119, view_119, mul_120, view_121, getitem_122, getitem_123, view_123, mul_124, view_125, getitem_126, getitem_127, view_127, mul_128, view_129, getitem_130, getitem_131, view_131, mul_132, view_133, getitem_134, getitem_135, view_135, mul_136, view_137, getitem_138, getitem_139, view_139, mul_140, view_141, getitem_142, getitem_143, view_143, mul_144, view_145, getitem_146, getitem_147, view_147, mul_148, view_149, getitem_150, getitem_151, view_151, mul_152, view_153, getitem_154, getitem_155, view_155, mul_156, view_157, getitem_158, getitem_159, view_159, mul_160, view_161, getitem_162, getitem_163, view_163, mul_164, view_165, getitem_166, getitem_167, view_167, mul_168, view_169, getitem_170, getitem_171, view_171, mul_172, view_173, getitem_174, getitem_175, view_175, mul_176, view_177, getitem_178, getitem_179, view_179, mul_180, view_181, getitem_182, getitem_183, view_183, mul_184, view_185, getitem_186, getitem_187, view_187, mul_188, view_189, getitem_190, getitem_191, view_191, mul_192, clone_169, permute_146, div_1, permute_150, permute_155, div_2, permute_160, permute_167, div_3, permute_170, permute_175, div_4, permute_180, permute_187, div_5, permute_190, permute_195, div_6, permute_200, permute_207, div_7, permute_210, permute_215, div_8, permute_220, permute_227, div_9, permute_230, permute_235, div_10, permute_240, permute_247, div_11, permute_250, permute_255, div_12, permute_260, permute_267, div_13, permute_270, permute_275, div_14, permute_280, permute_287, div_15, permute_290, permute_295, div_16, permute_300, permute_307, div_17, permute_310, permute_315, div_18, permute_320, permute_327, div_19, permute_330, permute_335, div_20, permute_340, permute_347, div_21, permute_350, permute_355, div_22, permute_360, permute_367, div_23, permute_370, permute_375, div_24, permute_380, permute_387, div_25, permute_390, permute_395, div_26, permute_400, permute_407, div_27, permute_410, permute_415, div_28, permute_420, permute_427, div_29, permute_430, permute_435, div_30, permute_440, permute_447, div_31, permute_450, permute_455, div_32, permute_460, permute_467, div_33, permute_470, permute_475, div_34, permute_480, permute_487, div_35, permute_490, permute_495, div_36, permute_500, permute_507, div_37, permute_510, permute_515, div_38, permute_520, permute_527, div_39, permute_530, permute_535, div_40, permute_540, permute_547, div_41, permute_550, permute_555, div_42, permute_560, permute_567, div_43, permute_570, permute_575, div_44, permute_580, permute_587, div_45, permute_590, permute_595, div_46, permute_600, permute_607, div_47, permute_610, permute_615, div_48, permute_620, permute_627, div_49, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('gmixer_24_224', benchmark_compiled_module)
