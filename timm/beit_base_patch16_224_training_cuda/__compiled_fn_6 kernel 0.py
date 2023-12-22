
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


# kernel path: /tmp/torchinductor_youkaichao/sk/cskjxz7a3k2b666q5yreto43w254s4nrvprxcjds5fv7orwpe2mz.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
    xnumel = 8
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
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 768.0
    tmp15 = tmp2 * tmp14
    tmp16 = tmp15 - tmp6
    tmp17 = tmp7 * tmp12
    tmp18 = tmp16 - tmp17
    tmp19 = tmp13 * tmp18
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp19, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xr/cxr3gqmtxvs6da2raazzghpngroyja6dtfh3jnopivtr6csx2rqh.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cg/ccg2rpxlsxgy6rues2ierdokasq36wp75uzjnptuiwryx2yc5vx7.py
# Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.slice_backward, aten.sum]

triton_red_fused_div_mul_slice_backward_sum_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_mul_slice_backward_sum_3', 'mutated_arg_names': []}
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
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 1576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = (r2 + (122*x1)) % 197
        tmp4 = tl.full([1, 1], 1, tl.int64)
        tmp5 = tmp3 >= tmp4
        tmp6 = tmp5 & tmp2
        tmp7 = tl.load(in_ptr0 + (x0 + (768*(((r2 + (122*x1)) // 197) % 8))), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = 196.0
        tmp9 = tmp7 / tmp8
        tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
        tmp11 = tl.where(tmp6, tmp9, tmp10)
        tmp12 = 0.0
        tmp13 = tl.where(tmp5, tmp11, tmp12)
        tmp14 = tl.load(in_ptr1 + (x0 + (768*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tmp13 * tmp14
        tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
        tmp17 = tl.where(tmp2, tmp15, tmp16)
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k7/ck7faycdeh2evc5ga5nn4dkmafggoypivpjbd4gog75xfg4uaay6.py
# Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.slice_backward, aten.sum]

triton_per_fused_div_mul_slice_backward_sum_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_mul_slice_backward_sum_4', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/3u/c3uuidkq7ufq6ae3kotcndgoh5swvuuuexnlda2zgglm7bu3yxac.py
# Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.slice_backward]

triton_poi_fused_div_mul_slice_backward_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_mul_slice_backward_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1210368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 768) % 197
    x0 = xindex % 768
    x2 = (xindex // 151296)
    x3 = xindex
    tmp10 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + (768*x2)), tmp2, eviction_policy='evict_last', other=0.0)
    tmp4 = 196.0
    tmp5 = tmp3 / tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp11 = tmp9 * tmp10
    tl.store(out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bu/cbukzoh7ii3qi6wncgyrbyazeg7q3mmbgr7ufqjuonefklqxikm4.py
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


# kernel path: /tmp/torchinductor_youkaichao/6t/c6tq5ooskqvcmiibnuhmqqoyirzrzkt5kh5zeoq6wqicgsarplby.py
# Source Nodes: [x_179], Original ATen: [aten.gelu, aten.gelu_backward]
# x_179 => add_82, erf_11, mul_105
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


# kernel path: /tmp/torchinductor_youkaichao/wo/cwoomxwvy7sobxlua2lj72amba5mgg2fyl4u3s2gxb6t7ssjju2f.py
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


# kernel path: /tmp/torchinductor_youkaichao/32/c32pj3ey7mw6s5xpg23tisarscfgfxfyaqyfyfcxjk3pcsdt2u34.py
# Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward, aten.slice_backward]

triton_per_fused_add_div_mul_native_layer_norm_backward_slice_backward_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_slice_backward_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
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
    x2 = xindex % 197
    x3 = (xindex // 197)
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = x2
    tmp14 = tl.full([1], 1, tl.int64)
    tmp15 = tmp13 >= tmp14
    tmp16 = tl.load(in_ptr3 + (r1 + (768*x3)), rmask & tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = 196.0
    tmp18 = tmp16 / tmp17
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = 0.0
    tmp22 = tl.where(tmp15, tmp20, tmp21)
    tmp24 = 768.0
    tmp25 = tmp2 * tmp24
    tmp26 = tmp25 - tmp6
    tmp27 = tmp7 * tmp12
    tmp28 = tmp26 - tmp27
    tmp29 = tmp23 * tmp28
    tmp30 = tmp22 + tmp29
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp30, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp32, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r5/cr5ssgryjesdrsnx343nx2hcpad55hcftfvzt2qxhhfzy5pwj7gx.py
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


# kernel path: /tmp/torchinductor_youkaichao/or/corrta7wk56dypjegklfl7efqmnosgbit2jblwqtw5t27wqego5r.py
# Source Nodes: [], Original ATen: [aten.mul, aten.sum]

triton_red_fused_mul_sum_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_12', 'mutated_arg_names': []}
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
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 1576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (768*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (768*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hs/chs3ei7qw47xlv6oqigysjpy7otdemyjnmf7azfif7snsdctbgpz.py
# Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.slice_backward, aten.sum]

triton_per_fused_constant_pad_nd_slice_backward_sum_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[524288, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_constant_pad_nd_slice_backward_sum_13', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 465708
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 197
    x1 = (xindex // 197)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (208*x1) + (491712*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = x0
    tmp6 = tl.full([1, 1], 200, tl.int64)
    tmp7 = tmp5 < tmp6
    tmp8 = tl.full([1, 1], 197, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp7
    tmp11 = tl.full(tmp4.shape, 0.0, tmp4.dtype)
    tmp12 = tl.where(tmp10, tmp4, tmp11)
    tmp13 = 0.0
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp7, tmp14, tmp15)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n2/cn2q7o6r6ynq7zr5jotspdw4fpqktbo2wjd7q65o4gcz5bg3nlzj.py
# Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]

triton_poi_fused_index_put_new_zeros_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_new_zeros_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tq/ctqtaoqsufvrpbz35pdqufz35dx3ecuf5wokaoge2ojkrhhdsi6r.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_15', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ut/cutudygwca46qlyce7ip2zxbctmnbm5xi7gwlv4puleiqdqnnsuv.py
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


# kernel path: /tmp/torchinductor_youkaichao/6t/c6t7dxxfmof74igd7qgcodsmkagkdzb64g2toizhfewejio3ovvb.py
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
    size_hints=[4096, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_17', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/7e/c7echwnarrjczellvtvwimmmz7qpi4lmsojsfhaka5azehw5n3kx.py
# Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]

triton_per_fused_add_mul_native_layer_norm_backward_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_backward_18', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
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
    tmp22 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp23 = tmp21 * tmp22
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp23, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ff/cffu2pfpwraftcvy5hz6kjd3ofdhwrq67z3nhitqq3stsipcumm3.py
# Source Nodes: [x_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# x_6 => mul, sub
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_19', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/co/ccoyaz6dlgf6xz5bzl3trs4qaeaif6wy2uyhvm2h6kggnehekogv.py
# Source Nodes: [x_6], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# x_6 => mul, sub
triton_red_fused_native_layer_norm_native_layer_norm_backward_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_20', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/jx/cjx2mvw2cx4sihhrfe4eevj5ic7vctku7lu3dux4yiuwvd52fzog.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_21', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ec/cecco7n7qgs7fxcfl2vwvo43c7cc2qjunkpri727qgz7yctasv3n.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_22', 'mutated_arg_names': []}
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
    primals_2, primals_3, primals_9, primals_10, primals_12, primals_13, primals_19, primals_20, primals_22, primals_23, primals_29, primals_30, primals_32, primals_33, primals_39, primals_40, primals_42, primals_43, primals_49, primals_50, primals_52, primals_53, primals_59, primals_60, primals_62, primals_63, primals_69, primals_70, primals_72, primals_73, primals_79, primals_80, primals_82, primals_83, primals_89, primals_90, primals_92, primals_93, primals_99, primals_100, primals_102, primals_103, primals_109, primals_110, primals_112, primals_113, primals_119, primals_120, primals_122, primals_124, primals_224, cat, getitem_1, rsqrt, view_1, getitem_2, getitem_3, getitem_4, view_4, expand_1, getitem_6, getitem_7, getitem_8, view_7, addmm_1, mul_3, view_9, addmm_2, view_11, addmm_3, mul_9, view_13, getitem_13, getitem_14, getitem_15, view_16, expand_2, getitem_17, getitem_18, getitem_19, view_19, addmm_5, mul_12, view_21, addmm_6, view_23, addmm_7, mul_18, view_25, getitem_24, getitem_25, getitem_26, view_28, expand_3, getitem_28, getitem_29, getitem_30, view_31, addmm_9, mul_21, view_33, addmm_10, view_35, addmm_11, mul_27, view_37, getitem_35, getitem_36, getitem_37, view_40, expand_4, getitem_39, getitem_40, getitem_41, view_43, addmm_13, mul_30, view_45, addmm_14, view_47, addmm_15, mul_36, view_49, getitem_46, getitem_47, getitem_48, view_52, expand_5, getitem_50, getitem_51, getitem_52, view_55, addmm_17, mul_39, view_57, addmm_18, view_59, addmm_19, mul_45, view_61, getitem_57, getitem_58, getitem_59, view_64, expand_6, getitem_61, getitem_62, getitem_63, view_67, addmm_21, mul_48, view_69, addmm_22, view_71, addmm_23, mul_54, view_73, getitem_68, getitem_69, getitem_70, view_76, expand_7, getitem_72, getitem_73, getitem_74, view_79, addmm_25, mul_57, view_81, addmm_26, view_83, addmm_27, mul_63, view_85, getitem_79, getitem_80, getitem_81, view_88, expand_8, getitem_83, getitem_84, getitem_85, view_91, addmm_29, mul_66, view_93, addmm_30, view_95, addmm_31, mul_72, view_97, getitem_90, getitem_91, getitem_92, view_100, expand_9, getitem_94, getitem_95, getitem_96, view_103, addmm_33, mul_75, view_105, addmm_34, view_107, addmm_35, mul_81, view_109, getitem_101, getitem_102, getitem_103, view_112, expand_10, getitem_105, getitem_106, getitem_107, view_115, addmm_37, mul_84, view_117, addmm_38, view_119, addmm_39, mul_90, view_121, getitem_112, getitem_113, getitem_114, view_124, expand_11, getitem_116, getitem_117, getitem_118, view_127, addmm_41, mul_93, view_129, addmm_42, view_131, addmm_43, mul_99, view_133, getitem_123, getitem_124, getitem_125, view_136, expand_12, getitem_127, getitem_128, getitem_129, view_139, addmm_45, mul_102, view_141, addmm_46, view_143, addmm_47, mul_108, clone_49, permute_86, div, permute_90, permute_94, div_2, permute_98, alias_12, permute_105, div_3, permute_109, permute_113, div_4, permute_117, alias_13, permute_124, div_5, permute_128, permute_132, div_6, permute_136, alias_14, permute_143, div_7, permute_147, permute_151, div_8, permute_155, alias_15, permute_162, div_9, permute_166, permute_170, div_10, permute_174, alias_16, permute_181, div_11, permute_185, permute_189, div_12, permute_193, alias_17, permute_200, div_13, permute_204, permute_208, div_14, permute_212, alias_18, permute_219, div_15, permute_223, permute_227, div_16, permute_231, alias_19, permute_238, div_17, permute_242, permute_246, div_18, permute_250, alias_20, permute_257, div_19, permute_261, permute_265, div_20, permute_269, alias_21, permute_276, div_21, permute_280, permute_284, div_22, permute_288, alias_22, permute_295, div_23, permute_299, permute_303, div_24, permute_307, alias_23, permute_314, tangents_1 = args
    args.clear()
    assert_size_stride(primals_2, (768, ), (1, ))
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_10, (768, ), (1, ))
    assert_size_stride(primals_12, (768, ), (1, ))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_20, (768, ), (1, ))
    assert_size_stride(primals_22, (768, ), (1, ))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_30, (768, ), (1, ))
    assert_size_stride(primals_32, (768, ), (1, ))
    assert_size_stride(primals_33, (768, ), (1, ))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_40, (768, ), (1, ))
    assert_size_stride(primals_42, (768, ), (1, ))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_49, (768, ), (1, ))
    assert_size_stride(primals_50, (768, ), (1, ))
    assert_size_stride(primals_52, (768, ), (1, ))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_60, (768, ), (1, ))
    assert_size_stride(primals_62, (768, ), (1, ))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_69, (768, ), (1, ))
    assert_size_stride(primals_70, (768, ), (1, ))
    assert_size_stride(primals_72, (768, ), (1, ))
    assert_size_stride(primals_73, (768, ), (1, ))
    assert_size_stride(primals_79, (768, ), (1, ))
    assert_size_stride(primals_80, (768, ), (1, ))
    assert_size_stride(primals_82, (768, ), (1, ))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_89, (768, ), (1, ))
    assert_size_stride(primals_90, (768, ), (1, ))
    assert_size_stride(primals_92, (768, ), (1, ))
    assert_size_stride(primals_93, (768, ), (1, ))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_100, (768, ), (1, ))
    assert_size_stride(primals_102, (768, ), (1, ))
    assert_size_stride(primals_103, (768, ), (1, ))
    assert_size_stride(primals_109, (768, ), (1, ))
    assert_size_stride(primals_110, (768, ), (1, ))
    assert_size_stride(primals_112, (768, ), (1, ))
    assert_size_stride(primals_113, (768, ), (1, ))
    assert_size_stride(primals_119, (768, ), (1, ))
    assert_size_stride(primals_120, (768, ), (1, ))
    assert_size_stride(primals_122, (768, ), (1, ))
    assert_size_stride(primals_124, (768, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(primals_224, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(cat, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(getitem_1, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_1, (1576, 768), (768, 1))
    assert_size_stride(getitem_2, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_3, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_4, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(view_4, (38809, ), (1, ))
    assert_size_stride(expand_1, (8, 12, 197, 197), (0, 39400, 200, 1))
    assert_size_stride(getitem_6, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_7, (), ())
    assert_size_stride(getitem_8, (), ())
    assert_size_stride(view_7, (1576, 768), (768, 1))
    assert_size_stride(addmm_1, (1576, 768), (768, 1))
    assert_size_stride(mul_3, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_9, (1576, 768), (768, 1))
    assert_size_stride(addmm_2, (1576, 3072), (3072, 1))
    assert_size_stride(view_11, (1576, 3072), (3072, 1))
    assert_size_stride(addmm_3, (1576, 768), (768, 1))
    assert_size_stride(mul_9, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_13, (1576, 768), (768, 1))
    assert_size_stride(getitem_13, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_14, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_15, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(view_16, (38809, ), (1, ))
    assert_size_stride(expand_2, (8, 12, 197, 197), (0, 39400, 200, 1))
    assert_size_stride(getitem_17, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_18, (), ())
    assert_size_stride(getitem_19, (), ())
    assert_size_stride(view_19, (1576, 768), (768, 1))
    assert_size_stride(addmm_5, (1576, 768), (768, 1))
    assert_size_stride(mul_12, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_21, (1576, 768), (768, 1))
    assert_size_stride(addmm_6, (1576, 3072), (3072, 1))
    assert_size_stride(view_23, (1576, 3072), (3072, 1))
    assert_size_stride(addmm_7, (1576, 768), (768, 1))
    assert_size_stride(mul_18, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_25, (1576, 768), (768, 1))
    assert_size_stride(getitem_24, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_25, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_26, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(view_28, (38809, ), (1, ))
    assert_size_stride(expand_3, (8, 12, 197, 197), (0, 39400, 200, 1))
    assert_size_stride(getitem_28, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_29, (), ())
    assert_size_stride(getitem_30, (), ())
    assert_size_stride(view_31, (1576, 768), (768, 1))
    assert_size_stride(addmm_9, (1576, 768), (768, 1))
    assert_size_stride(mul_21, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_33, (1576, 768), (768, 1))
    assert_size_stride(addmm_10, (1576, 3072), (3072, 1))
    assert_size_stride(view_35, (1576, 3072), (3072, 1))
    assert_size_stride(addmm_11, (1576, 768), (768, 1))
    assert_size_stride(mul_27, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_37, (1576, 768), (768, 1))
    assert_size_stride(getitem_35, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_36, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_37, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(view_40, (38809, ), (1, ))
    assert_size_stride(expand_4, (8, 12, 197, 197), (0, 39400, 200, 1))
    assert_size_stride(getitem_39, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_40, (), ())
    assert_size_stride(getitem_41, (), ())
    assert_size_stride(view_43, (1576, 768), (768, 1))
    assert_size_stride(addmm_13, (1576, 768), (768, 1))
    assert_size_stride(mul_30, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_45, (1576, 768), (768, 1))
    assert_size_stride(addmm_14, (1576, 3072), (3072, 1))
    assert_size_stride(view_47, (1576, 3072), (3072, 1))
    assert_size_stride(addmm_15, (1576, 768), (768, 1))
    assert_size_stride(mul_36, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_49, (1576, 768), (768, 1))
    assert_size_stride(getitem_46, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_47, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_48, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(view_52, (38809, ), (1, ))
    assert_size_stride(expand_5, (8, 12, 197, 197), (0, 39400, 200, 1))
    assert_size_stride(getitem_50, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_51, (), ())
    assert_size_stride(getitem_52, (), ())
    assert_size_stride(view_55, (1576, 768), (768, 1))
    assert_size_stride(addmm_17, (1576, 768), (768, 1))
    assert_size_stride(mul_39, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_57, (1576, 768), (768, 1))
    assert_size_stride(addmm_18, (1576, 3072), (3072, 1))
    assert_size_stride(view_59, (1576, 3072), (3072, 1))
    assert_size_stride(addmm_19, (1576, 768), (768, 1))
    assert_size_stride(mul_45, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_61, (1576, 768), (768, 1))
    assert_size_stride(getitem_57, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_58, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_59, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(view_64, (38809, ), (1, ))
    assert_size_stride(expand_6, (8, 12, 197, 197), (0, 39400, 200, 1))
    assert_size_stride(getitem_61, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_62, (), ())
    assert_size_stride(getitem_63, (), ())
    assert_size_stride(view_67, (1576, 768), (768, 1))
    assert_size_stride(addmm_21, (1576, 768), (768, 1))
    assert_size_stride(mul_48, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_69, (1576, 768), (768, 1))
    assert_size_stride(addmm_22, (1576, 3072), (3072, 1))
    assert_size_stride(view_71, (1576, 3072), (3072, 1))
    assert_size_stride(addmm_23, (1576, 768), (768, 1))
    assert_size_stride(mul_54, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_73, (1576, 768), (768, 1))
    assert_size_stride(getitem_68, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_69, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_70, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(view_76, (38809, ), (1, ))
    assert_size_stride(expand_7, (8, 12, 197, 197), (0, 39400, 200, 1))
    assert_size_stride(getitem_72, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_73, (), ())
    assert_size_stride(getitem_74, (), ())
    assert_size_stride(view_79, (1576, 768), (768, 1))
    assert_size_stride(addmm_25, (1576, 768), (768, 1))
    assert_size_stride(mul_57, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_81, (1576, 768), (768, 1))
    assert_size_stride(addmm_26, (1576, 3072), (3072, 1))
    assert_size_stride(view_83, (1576, 3072), (3072, 1))
    assert_size_stride(addmm_27, (1576, 768), (768, 1))
    assert_size_stride(mul_63, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_85, (1576, 768), (768, 1))
    assert_size_stride(getitem_79, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_80, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_81, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(view_88, (38809, ), (1, ))
    assert_size_stride(expand_8, (8, 12, 197, 197), (0, 39400, 200, 1))
    assert_size_stride(getitem_83, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_84, (), ())
    assert_size_stride(getitem_85, (), ())
    assert_size_stride(view_91, (1576, 768), (768, 1))
    assert_size_stride(addmm_29, (1576, 768), (768, 1))
    assert_size_stride(mul_66, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_93, (1576, 768), (768, 1))
    assert_size_stride(addmm_30, (1576, 3072), (3072, 1))
    assert_size_stride(view_95, (1576, 3072), (3072, 1))
    assert_size_stride(addmm_31, (1576, 768), (768, 1))
    assert_size_stride(mul_72, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_97, (1576, 768), (768, 1))
    assert_size_stride(getitem_90, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_91, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_92, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(view_100, (38809, ), (1, ))
    assert_size_stride(expand_9, (8, 12, 197, 197), (0, 39400, 200, 1))
    assert_size_stride(getitem_94, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_95, (), ())
    assert_size_stride(getitem_96, (), ())
    assert_size_stride(view_103, (1576, 768), (768, 1))
    assert_size_stride(addmm_33, (1576, 768), (768, 1))
    assert_size_stride(mul_75, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_105, (1576, 768), (768, 1))
    assert_size_stride(addmm_34, (1576, 3072), (3072, 1))
    assert_size_stride(view_107, (1576, 3072), (3072, 1))
    assert_size_stride(addmm_35, (1576, 768), (768, 1))
    assert_size_stride(mul_81, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_109, (1576, 768), (768, 1))
    assert_size_stride(getitem_101, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_102, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_103, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(view_112, (38809, ), (1, ))
    assert_size_stride(expand_10, (8, 12, 197, 197), (0, 39400, 200, 1))
    assert_size_stride(getitem_105, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_106, (), ())
    assert_size_stride(getitem_107, (), ())
    assert_size_stride(view_115, (1576, 768), (768, 1))
    assert_size_stride(addmm_37, (1576, 768), (768, 1))
    assert_size_stride(mul_84, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_117, (1576, 768), (768, 1))
    assert_size_stride(addmm_38, (1576, 3072), (3072, 1))
    assert_size_stride(view_119, (1576, 3072), (3072, 1))
    assert_size_stride(addmm_39, (1576, 768), (768, 1))
    assert_size_stride(mul_90, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_121, (1576, 768), (768, 1))
    assert_size_stride(getitem_112, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_113, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_114, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(view_124, (38809, ), (1, ))
    assert_size_stride(expand_11, (8, 12, 197, 197), (0, 39400, 200, 1))
    assert_size_stride(getitem_116, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_117, (), ())
    assert_size_stride(getitem_118, (), ())
    assert_size_stride(view_127, (1576, 768), (768, 1))
    assert_size_stride(addmm_41, (1576, 768), (768, 1))
    assert_size_stride(mul_93, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_129, (1576, 768), (768, 1))
    assert_size_stride(addmm_42, (1576, 3072), (3072, 1))
    assert_size_stride(view_131, (1576, 3072), (3072, 1))
    assert_size_stride(addmm_43, (1576, 768), (768, 1))
    assert_size_stride(mul_99, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_133, (1576, 768), (768, 1))
    assert_size_stride(getitem_123, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_124, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(getitem_125, (8, 12, 197, 64), (453888, 64, 2304, 1))
    assert_size_stride(view_136, (38809, ), (1, ))
    assert_size_stride(expand_12, (8, 12, 197, 197), (0, 39400, 200, 1))
    assert_size_stride(getitem_127, (8, 12, 224), (2688, 224, 1))
    assert_size_stride(getitem_128, (), ())
    assert_size_stride(getitem_129, (), ())
    assert_size_stride(view_139, (1576, 768), (768, 1))
    assert_size_stride(addmm_45, (1576, 768), (768, 1))
    assert_size_stride(mul_102, (8, 197, 768), (151296, 768, 1))
    assert_size_stride(view_141, (1576, 768), (768, 1))
    assert_size_stride(addmm_46, (1576, 3072), (3072, 1))
    assert_size_stride(view_143, (1576, 3072), (3072, 1))
    assert_size_stride(addmm_47, (1576, 768), (768, 1))
    assert_size_stride(mul_108, (8, 768), (768, 1))
    assert_size_stride(clone_49, (8, 768), (768, 1))
    assert_size_stride(permute_86, (1000, 768), (768, 1))
    assert_size_stride(div, (8, 1), (1, 1))
    assert_size_stride(permute_90, (768, 3072), (3072, 1))
    assert_size_stride(permute_94, (3072, 768), (768, 1))
    assert_size_stride(div_2, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_98, (768, 768), (768, 1))
    assert_size_stride(alias_12, (8, 12, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(permute_105, (2304, 768), (768, 1))
    assert_size_stride(div_3, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_109, (768, 3072), (3072, 1))
    assert_size_stride(permute_113, (3072, 768), (768, 1))
    assert_size_stride(div_4, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_117, (768, 768), (768, 1))
    assert_size_stride(alias_13, (8, 12, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(permute_124, (2304, 768), (768, 1))
    assert_size_stride(div_5, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_128, (768, 3072), (3072, 1))
    assert_size_stride(permute_132, (3072, 768), (768, 1))
    assert_size_stride(div_6, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_136, (768, 768), (768, 1))
    assert_size_stride(alias_14, (8, 12, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(permute_143, (2304, 768), (768, 1))
    assert_size_stride(div_7, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_147, (768, 3072), (3072, 1))
    assert_size_stride(permute_151, (3072, 768), (768, 1))
    assert_size_stride(div_8, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_155, (768, 768), (768, 1))
    assert_size_stride(alias_15, (8, 12, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(permute_162, (2304, 768), (768, 1))
    assert_size_stride(div_9, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_166, (768, 3072), (3072, 1))
    assert_size_stride(permute_170, (3072, 768), (768, 1))
    assert_size_stride(div_10, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_174, (768, 768), (768, 1))
    assert_size_stride(alias_16, (8, 12, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(permute_181, (2304, 768), (768, 1))
    assert_size_stride(div_11, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_185, (768, 3072), (3072, 1))
    assert_size_stride(permute_189, (3072, 768), (768, 1))
    assert_size_stride(div_12, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_193, (768, 768), (768, 1))
    assert_size_stride(alias_17, (8, 12, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(permute_200, (2304, 768), (768, 1))
    assert_size_stride(div_13, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_204, (768, 3072), (3072, 1))
    assert_size_stride(permute_208, (3072, 768), (768, 1))
    assert_size_stride(div_14, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_212, (768, 768), (768, 1))
    assert_size_stride(alias_18, (8, 12, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(permute_219, (2304, 768), (768, 1))
    assert_size_stride(div_15, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_223, (768, 3072), (3072, 1))
    assert_size_stride(permute_227, (3072, 768), (768, 1))
    assert_size_stride(div_16, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_231, (768, 768), (768, 1))
    assert_size_stride(alias_19, (8, 12, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(permute_238, (2304, 768), (768, 1))
    assert_size_stride(div_17, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_242, (768, 3072), (3072, 1))
    assert_size_stride(permute_246, (3072, 768), (768, 1))
    assert_size_stride(div_18, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_250, (768, 768), (768, 1))
    assert_size_stride(alias_20, (8, 12, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(permute_257, (2304, 768), (768, 1))
    assert_size_stride(div_19, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_261, (768, 3072), (3072, 1))
    assert_size_stride(permute_265, (3072, 768), (768, 1))
    assert_size_stride(div_20, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_269, (768, 768), (768, 1))
    assert_size_stride(alias_21, (8, 12, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(permute_276, (2304, 768), (768, 1))
    assert_size_stride(div_21, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_280, (768, 3072), (3072, 1))
    assert_size_stride(permute_284, (3072, 768), (768, 1))
    assert_size_stride(div_22, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_288, (768, 768), (768, 1))
    assert_size_stride(alias_22, (8, 12, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(permute_295, (2304, 768), (768, 1))
    assert_size_stride(div_23, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_299, (768, 3072), (3072, 1))
    assert_size_stride(permute_303, (3072, 768), (768, 1))
    assert_size_stride(div_24, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_307, (768, 768), (768, 1))
    assert_size_stride(alias_23, (8, 12, 197, 64), (151296, 64, 768, 1))
    assert_size_stride(permute_314, (2304, 768), (768, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_86, out=buf0)
        del permute_86
        buf1 = empty((1000, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_49, out=buf1)
        del clone_49
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf7 = empty((8, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_1.run(buf0, primals_122, mul_108, div, buf7, 8, 768, grid=grid(8), stream=stream0)
        del div
        del primals_122
        buf5 = empty((768, ), device='cuda', dtype=torch.float32)
        buf6 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf0, mul_108, buf5, buf6, 768, 8, grid=grid(768), stream=stream0)
        del buf0
        del mul_108
        buf8 = empty_strided((1, 1, 768, 13), (9984, 9984, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.slice_backward, aten.sum]
        triton_red_fused_div_mul_slice_backward_sum_3.run(buf7, addmm_47, buf8, 9984, 122, grid=grid(9984), stream=stream0)
        del addmm_47
        buf9 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.slice_backward, aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf8, buf9, 768, 13, grid=grid(768), stream=stream0)
        buf10 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div, aten.mul, aten.slice_backward]
        triton_poi_fused_div_mul_slice_backward_5.run(buf7, primals_119, buf10, 1210368, grid=grid(1210368), stream=stream0)
        del primals_119
        buf11 = empty((1576, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (1576, 768), (768, 1), 0), permute_90, out=buf11)
        del permute_90
        buf12 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (768, 1576), (1, 768), 0), view_143, out=buf12)
        del view_143
        buf13 = reinterpret_tensor(buf8, (1, 768, 13), (9984, 1, 768), 0); del buf8  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf10, buf13, 9984, 122, grid=grid(9984), stream=stream0)
        buf14 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf13, buf14, 768, 13, grid=grid(768), stream=stream0)
        buf15 = reinterpret_tensor(buf11, (8, 197, 3072), (605184, 3072, 1), 0); del buf11  # reuse
        # Source Nodes: [x_179], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_7.run(buf15, addmm_46, 4841472, grid=grid(4841472), stream=stream0)
        del addmm_46
        buf16 = reinterpret_tensor(buf10, (1576, 768), (768, 1), 0); del buf10  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (1576, 3072), (3072, 1), 0), permute_94, out=buf16)
        del permute_94
        buf17 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (3072, 1576), (1, 3072), 0), view_141, out=buf17)
        del view_141
        buf18 = empty_strided((1, 3072, 13), (39936, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf15, buf18, 39936, 122, grid=grid(39936), stream=stream0)
        buf19 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf18, buf19, 3072, 13, grid=grid(3072), stream=stream0)
        buf26 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        buf29 = empty((8, 197, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward, aten.slice_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_slice_backward_10.run(buf16, primals_120, mul_102, buf7, div_2, primals_112, buf26, buf29, 1576, 768, grid=grid(1576), stream=stream0)
        del buf7
        del div_2
        del primals_112
        del primals_120
        buf22 = reinterpret_tensor(buf13, (768, 13), (1, 768), 0); del buf13  # reuse
        buf24 = empty_strided((768, 13), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf16, mul_102, buf22, buf24, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_102
        buf23 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf22, buf23, 768, 13, grid=grid(768), stream=stream0)
        buf25 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf24, buf25, 768, 13, grid=grid(768), stream=stream0)
        buf27 = reinterpret_tensor(buf24, (1, 1, 768, 13), (9984, 9984, 1, 768), 0); del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_12.run(buf26, addmm_45, buf27, 9984, 122, grid=grid(9984), stream=stream0)
        del addmm_45
        buf28 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf27, buf28, 768, 13, grid=grid(768), stream=stream0)
        buf30 = buf16; del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf29, (1576, 768), (768, 1), 0), permute_98, out=buf30)
        del permute_98
        buf31 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf29, (768, 1576), (1, 768), 0), view_139, out=buf31)
        del view_139
        buf32 = reinterpret_tensor(buf27, (1, 768, 13), (9984, 1, 768), 0); del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf29, buf32, 9984, 122, grid=grid(9984), stream=stream0)
        del buf29
        buf33 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf32, buf33, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf34 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf30, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_123, getitem_124, getitem_125, expand_12, alias_12, getitem_127, getitem_128, getitem_129, 0.0, [True, True, True, True])
        del alias_12
        del buf30
        del expand_12
        del getitem_123
        del getitem_124
        del getitem_125
        del getitem_127
        del getitem_128
        del getitem_129
        buf35 = buf34[0]
        buf36 = buf34[1]
        buf37 = buf34[2]
        buf38 = buf34[3]
        del buf34
        buf39 = empty((1, 12, 197, 197), device='cuda', dtype=torch.float32)
        buf41 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.slice_backward, aten.sum]
        triton_per_fused_constant_pad_nd_slice_backward_sum_13.run(buf41, buf38, 465708, 8, grid=grid(465708), stream=stream0)
        del buf38
        buf40 = empty((732, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_14.run(buf40, 8784, grid=grid(8784), stream=stream0)
        aten.index_put_(buf40, [view_136], reinterpret_tensor(buf41, (38809, 12), (1, 38809), 0), True)
        del view_136
        buf44 = empty((8, 197, 3, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf35, buf36, buf37, buf44, 3631104, grid=grid(3631104), stream=stream0)
        del buf35
        buf45 = reinterpret_tensor(buf37, (1576, 768), (768, 1), 0); del buf37  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf44, (1576, 2304), (2304, 1), 0), permute_105, out=buf45)
        del permute_105
        buf46 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf44, (2304, 1576), (1, 2304), 0), view_133, out=buf46)
        del view_133
        buf47 = empty_strided((1, 2304, 13), (29952, 1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf44, buf47, 29952, 122, grid=grid(29952), stream=stream0)
        buf48 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf47, buf48, 2304, 13, grid=grid(2304), stream=stream0)
        buf55 = buf26; del buf26  # reuse
        buf58 = reinterpret_tensor(buf36, (8, 197, 768), (151296, 768, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_18.run(buf55, buf45, primals_113, mul_99, div_3, primals_109, buf58, 1576, 768, grid=grid(1576), stream=stream0)
        del div_3
        del primals_109
        del primals_113
        buf51 = reinterpret_tensor(buf32, (768, 13), (1, 768), 0); del buf32  # reuse
        buf53 = buf22; del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf45, mul_99, buf51, buf53, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_99
        buf52 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf51, buf52, 768, 13, grid=grid(768), stream=stream0)
        buf54 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf53, buf54, 768, 13, grid=grid(768), stream=stream0)
        buf56 = reinterpret_tensor(buf53, (1, 1, 768, 13), (9984, 9984, 1, 768), 0); del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_12.run(buf55, addmm_43, buf56, 9984, 122, grid=grid(9984), stream=stream0)
        del addmm_43
        buf57 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf56, buf57, 768, 13, grid=grid(768), stream=stream0)
        buf59 = reinterpret_tensor(buf15, (1576, 3072), (3072, 1), 0); del buf15  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf58, (1576, 768), (768, 1), 0), permute_109, out=buf59)
        del permute_109
        buf60 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf58, (768, 1576), (1, 768), 0), view_131, out=buf60)
        del view_131
        buf61 = reinterpret_tensor(buf56, (1, 768, 13), (9984, 1, 768), 0); del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf58, buf61, 9984, 122, grid=grid(9984), stream=stream0)
        buf62 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf61, buf62, 768, 13, grid=grid(768), stream=stream0)
        buf63 = reinterpret_tensor(buf59, (8, 197, 3072), (605184, 3072, 1), 0); del buf59  # reuse
        # Source Nodes: [x_164], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_7.run(buf63, addmm_42, 4841472, grid=grid(4841472), stream=stream0)
        del addmm_42
        buf64 = reinterpret_tensor(buf58, (1576, 768), (768, 1), 0); del buf58  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf63, (1576, 3072), (3072, 1), 0), permute_113, out=buf64)
        del permute_113
        buf65 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf63, (3072, 1576), (1, 3072), 0), view_129, out=buf65)
        del view_129
        buf66 = buf18; del buf18  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf63, buf66, 39936, 122, grid=grid(39936), stream=stream0)
        buf67 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf66, buf67, 3072, 13, grid=grid(3072), stream=stream0)
        buf74 = buf55; del buf55  # reuse
        buf77 = reinterpret_tensor(buf45, (8, 197, 768), (151296, 768, 1), 0); del buf45  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_18.run(buf74, buf64, primals_110, mul_93, div_4, primals_102, buf77, 1576, 768, grid=grid(1576), stream=stream0)
        del div_4
        del primals_102
        del primals_110
        buf70 = reinterpret_tensor(buf61, (768, 13), (1, 768), 0); del buf61  # reuse
        buf72 = buf51; del buf51  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf64, mul_93, buf70, buf72, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_93
        buf71 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf70, buf71, 768, 13, grid=grid(768), stream=stream0)
        buf73 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf72, buf73, 768, 13, grid=grid(768), stream=stream0)
        buf75 = reinterpret_tensor(buf72, (1, 1, 768, 13), (9984, 9984, 1, 768), 0); del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_12.run(buf74, addmm_41, buf75, 9984, 122, grid=grid(9984), stream=stream0)
        del addmm_41
        buf76 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf75, buf76, 768, 13, grid=grid(768), stream=stream0)
        buf78 = buf64; del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf77, (1576, 768), (768, 1), 0), permute_117, out=buf78)
        del permute_117
        buf79 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf77, (768, 1576), (1, 768), 0), view_127, out=buf79)
        del view_127
        buf80 = reinterpret_tensor(buf75, (1, 768, 13), (9984, 1, 768), 0); del buf75  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf77, buf80, 9984, 122, grid=grid(9984), stream=stream0)
        del buf77
        buf81 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf80, buf81, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf82 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf78, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_112, getitem_113, getitem_114, expand_11, alias_13, getitem_116, getitem_117, getitem_118, 0.0, [True, True, True, True])
        del alias_13
        del buf78
        del expand_11
        del getitem_112
        del getitem_113
        del getitem_114
        del getitem_116
        del getitem_117
        del getitem_118
        buf83 = buf82[0]
        buf84 = buf82[1]
        buf85 = buf82[2]
        buf86 = buf82[3]
        del buf82
        buf87 = buf41; del buf41  # reuse
        buf89 = buf87; del buf87  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.slice_backward, aten.sum]
        triton_per_fused_constant_pad_nd_slice_backward_sum_13.run(buf89, buf86, 465708, 8, grid=grid(465708), stream=stream0)
        del buf86
        buf88 = empty((732, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_14.run(buf88, 8784, grid=grid(8784), stream=stream0)
        aten.index_put_(buf88, [view_124], reinterpret_tensor(buf89, (38809, 12), (1, 38809), 0), True)
        del view_124
        buf92 = buf44; del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf83, buf84, buf85, buf92, 3631104, grid=grid(3631104), stream=stream0)
        del buf83
        buf93 = reinterpret_tensor(buf85, (1576, 768), (768, 1), 0); del buf85  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf92, (1576, 2304), (2304, 1), 0), permute_124, out=buf93)
        del permute_124
        buf94 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf92, (2304, 1576), (1, 2304), 0), view_121, out=buf94)
        del view_121
        buf95 = buf47; del buf47  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf92, buf95, 29952, 122, grid=grid(29952), stream=stream0)
        buf96 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf95, buf96, 2304, 13, grid=grid(2304), stream=stream0)
        buf103 = buf74; del buf74  # reuse
        buf106 = reinterpret_tensor(buf84, (8, 197, 768), (151296, 768, 1), 0); del buf84  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_18.run(buf103, buf93, primals_103, mul_90, div_5, primals_99, buf106, 1576, 768, grid=grid(1576), stream=stream0)
        del div_5
        del primals_103
        del primals_99
        buf99 = reinterpret_tensor(buf80, (768, 13), (1, 768), 0); del buf80  # reuse
        buf101 = buf70; del buf70  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf93, mul_90, buf99, buf101, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_90
        buf100 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf99, buf100, 768, 13, grid=grid(768), stream=stream0)
        buf102 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf101, buf102, 768, 13, grid=grid(768), stream=stream0)
        buf104 = reinterpret_tensor(buf101, (1, 1, 768, 13), (9984, 9984, 1, 768), 0); del buf101  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_12.run(buf103, addmm_39, buf104, 9984, 122, grid=grid(9984), stream=stream0)
        del addmm_39
        buf105 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf104, buf105, 768, 13, grid=grid(768), stream=stream0)
        buf107 = reinterpret_tensor(buf63, (1576, 3072), (3072, 1), 0); del buf63  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf106, (1576, 768), (768, 1), 0), permute_128, out=buf107)
        del permute_128
        buf108 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf106, (768, 1576), (1, 768), 0), view_119, out=buf108)
        del view_119
        buf109 = reinterpret_tensor(buf104, (1, 768, 13), (9984, 1, 768), 0); del buf104  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf106, buf109, 9984, 122, grid=grid(9984), stream=stream0)
        buf110 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf109, buf110, 768, 13, grid=grid(768), stream=stream0)
        buf111 = reinterpret_tensor(buf107, (8, 197, 3072), (605184, 3072, 1), 0); del buf107  # reuse
        # Source Nodes: [x_149], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_7.run(buf111, addmm_38, 4841472, grid=grid(4841472), stream=stream0)
        del addmm_38
        buf112 = reinterpret_tensor(buf106, (1576, 768), (768, 1), 0); del buf106  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf111, (1576, 3072), (3072, 1), 0), permute_132, out=buf112)
        del permute_132
        buf113 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf111, (3072, 1576), (1, 3072), 0), view_117, out=buf113)
        del view_117
        buf114 = buf66; del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf111, buf114, 39936, 122, grid=grid(39936), stream=stream0)
        buf115 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf114, buf115, 3072, 13, grid=grid(3072), stream=stream0)
        buf122 = buf103; del buf103  # reuse
        buf125 = reinterpret_tensor(buf93, (8, 197, 768), (151296, 768, 1), 0); del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_18.run(buf122, buf112, primals_100, mul_84, div_6, primals_92, buf125, 1576, 768, grid=grid(1576), stream=stream0)
        del div_6
        del primals_100
        del primals_92
        buf118 = reinterpret_tensor(buf109, (768, 13), (1, 768), 0); del buf109  # reuse
        buf120 = buf99; del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf112, mul_84, buf118, buf120, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_84
        buf119 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf118, buf119, 768, 13, grid=grid(768), stream=stream0)
        buf121 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf120, buf121, 768, 13, grid=grid(768), stream=stream0)
        buf123 = reinterpret_tensor(buf120, (1, 1, 768, 13), (9984, 9984, 1, 768), 0); del buf120  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_12.run(buf122, addmm_37, buf123, 9984, 122, grid=grid(9984), stream=stream0)
        del addmm_37
        buf124 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf123, buf124, 768, 13, grid=grid(768), stream=stream0)
        buf126 = buf112; del buf112  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (1576, 768), (768, 1), 0), permute_136, out=buf126)
        del permute_136
        buf127 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (768, 1576), (1, 768), 0), view_115, out=buf127)
        del view_115
        buf128 = reinterpret_tensor(buf123, (1, 768, 13), (9984, 1, 768), 0); del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf125, buf128, 9984, 122, grid=grid(9984), stream=stream0)
        del buf125
        buf129 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf128, buf129, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf130 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf126, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_101, getitem_102, getitem_103, expand_10, alias_14, getitem_105, getitem_106, getitem_107, 0.0, [True, True, True, True])
        del alias_14
        del buf126
        del expand_10
        del getitem_101
        del getitem_102
        del getitem_103
        del getitem_105
        del getitem_106
        del getitem_107
        buf131 = buf130[0]
        buf132 = buf130[1]
        buf133 = buf130[2]
        buf134 = buf130[3]
        del buf130
        buf135 = buf89; del buf89  # reuse
        buf137 = buf135; del buf135  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.slice_backward, aten.sum]
        triton_per_fused_constant_pad_nd_slice_backward_sum_13.run(buf137, buf134, 465708, 8, grid=grid(465708), stream=stream0)
        del buf134
        buf136 = empty((732, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_14.run(buf136, 8784, grid=grid(8784), stream=stream0)
        aten.index_put_(buf136, [view_112], reinterpret_tensor(buf137, (38809, 12), (1, 38809), 0), True)
        del view_112
        buf140 = buf92; del buf92  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf131, buf132, buf133, buf140, 3631104, grid=grid(3631104), stream=stream0)
        del buf131
        buf141 = reinterpret_tensor(buf133, (1576, 768), (768, 1), 0); del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf140, (1576, 2304), (2304, 1), 0), permute_143, out=buf141)
        del permute_143
        buf142 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf140, (2304, 1576), (1, 2304), 0), view_109, out=buf142)
        del view_109
        buf143 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf140, buf143, 29952, 122, grid=grid(29952), stream=stream0)
        buf144 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf143, buf144, 2304, 13, grid=grid(2304), stream=stream0)
        buf151 = buf122; del buf122  # reuse
        buf154 = reinterpret_tensor(buf132, (8, 197, 768), (151296, 768, 1), 0); del buf132  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_18.run(buf151, buf141, primals_93, mul_81, div_7, primals_89, buf154, 1576, 768, grid=grid(1576), stream=stream0)
        del div_7
        del primals_89
        del primals_93
        buf147 = reinterpret_tensor(buf128, (768, 13), (1, 768), 0); del buf128  # reuse
        buf149 = buf118; del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf141, mul_81, buf147, buf149, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_81
        buf148 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf147, buf148, 768, 13, grid=grid(768), stream=stream0)
        buf150 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf149, buf150, 768, 13, grid=grid(768), stream=stream0)
        buf152 = reinterpret_tensor(buf149, (1, 1, 768, 13), (9984, 9984, 1, 768), 0); del buf149  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_12.run(buf151, addmm_35, buf152, 9984, 122, grid=grid(9984), stream=stream0)
        del addmm_35
        buf153 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf152, buf153, 768, 13, grid=grid(768), stream=stream0)
        buf155 = reinterpret_tensor(buf111, (1576, 3072), (3072, 1), 0); del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (1576, 768), (768, 1), 0), permute_147, out=buf155)
        del permute_147
        buf156 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (768, 1576), (1, 768), 0), view_107, out=buf156)
        del view_107
        buf157 = reinterpret_tensor(buf152, (1, 768, 13), (9984, 1, 768), 0); del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf154, buf157, 9984, 122, grid=grid(9984), stream=stream0)
        buf158 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf157, buf158, 768, 13, grid=grid(768), stream=stream0)
        buf159 = reinterpret_tensor(buf155, (8, 197, 3072), (605184, 3072, 1), 0); del buf155  # reuse
        # Source Nodes: [x_134], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_7.run(buf159, addmm_34, 4841472, grid=grid(4841472), stream=stream0)
        del addmm_34
        buf160 = reinterpret_tensor(buf154, (1576, 768), (768, 1), 0); del buf154  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf159, (1576, 3072), (3072, 1), 0), permute_151, out=buf160)
        del permute_151
        buf161 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf159, (3072, 1576), (1, 3072), 0), view_105, out=buf161)
        del view_105
        buf162 = buf114; del buf114  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf159, buf162, 39936, 122, grid=grid(39936), stream=stream0)
        buf163 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf162, buf163, 3072, 13, grid=grid(3072), stream=stream0)
        buf170 = buf151; del buf151  # reuse
        buf173 = reinterpret_tensor(buf141, (8, 197, 768), (151296, 768, 1), 0); del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_18.run(buf170, buf160, primals_90, mul_75, div_8, primals_82, buf173, 1576, 768, grid=grid(1576), stream=stream0)
        del div_8
        del primals_82
        del primals_90
        buf166 = reinterpret_tensor(buf157, (768, 13), (1, 768), 0); del buf157  # reuse
        buf168 = buf147; del buf147  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf160, mul_75, buf166, buf168, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_75
        buf167 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf166, buf167, 768, 13, grid=grid(768), stream=stream0)
        buf169 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf168, buf169, 768, 13, grid=grid(768), stream=stream0)
        buf171 = reinterpret_tensor(buf168, (1, 1, 768, 13), (9984, 9984, 1, 768), 0); del buf168  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_12.run(buf170, addmm_33, buf171, 9984, 122, grid=grid(9984), stream=stream0)
        del addmm_33
        buf172 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf171, buf172, 768, 13, grid=grid(768), stream=stream0)
        buf174 = buf160; del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf173, (1576, 768), (768, 1), 0), permute_155, out=buf174)
        del permute_155
        buf175 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf173, (768, 1576), (1, 768), 0), view_103, out=buf175)
        del view_103
        buf176 = reinterpret_tensor(buf171, (1, 768, 13), (9984, 1, 768), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf173, buf176, 9984, 122, grid=grid(9984), stream=stream0)
        del buf173
        buf177 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf176, buf177, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf178 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf174, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_90, getitem_91, getitem_92, expand_9, alias_15, getitem_94, getitem_95, getitem_96, 0.0, [True, True, True, True])
        del alias_15
        del buf174
        del expand_9
        del getitem_90
        del getitem_91
        del getitem_92
        del getitem_94
        del getitem_95
        del getitem_96
        buf179 = buf178[0]
        buf180 = buf178[1]
        buf181 = buf178[2]
        buf182 = buf178[3]
        del buf178
        buf183 = buf137; del buf137  # reuse
        buf185 = buf183; del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.slice_backward, aten.sum]
        triton_per_fused_constant_pad_nd_slice_backward_sum_13.run(buf185, buf182, 465708, 8, grid=grid(465708), stream=stream0)
        del buf182
        buf184 = empty((732, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_14.run(buf184, 8784, grid=grid(8784), stream=stream0)
        aten.index_put_(buf184, [view_100], reinterpret_tensor(buf185, (38809, 12), (1, 38809), 0), True)
        del view_100
        buf188 = buf140; del buf140  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf179, buf180, buf181, buf188, 3631104, grid=grid(3631104), stream=stream0)
        del buf179
        buf189 = reinterpret_tensor(buf181, (1576, 768), (768, 1), 0); del buf181  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf188, (1576, 2304), (2304, 1), 0), permute_162, out=buf189)
        del permute_162
        buf190 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf188, (2304, 1576), (1, 2304), 0), view_97, out=buf190)
        del view_97
        buf191 = buf143; del buf143  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf188, buf191, 29952, 122, grid=grid(29952), stream=stream0)
        buf192 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf191, buf192, 2304, 13, grid=grid(2304), stream=stream0)
        buf199 = buf170; del buf170  # reuse
        buf202 = reinterpret_tensor(buf180, (8, 197, 768), (151296, 768, 1), 0); del buf180  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_18.run(buf199, buf189, primals_83, mul_72, div_9, primals_79, buf202, 1576, 768, grid=grid(1576), stream=stream0)
        del div_9
        del primals_79
        del primals_83
        buf195 = reinterpret_tensor(buf176, (768, 13), (1, 768), 0); del buf176  # reuse
        buf197 = buf166; del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf189, mul_72, buf195, buf197, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_72
        buf196 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf195, buf196, 768, 13, grid=grid(768), stream=stream0)
        buf198 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf197, buf198, 768, 13, grid=grid(768), stream=stream0)
        buf200 = reinterpret_tensor(buf197, (1, 1, 768, 13), (9984, 9984, 1, 768), 0); del buf197  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_12.run(buf199, addmm_31, buf200, 9984, 122, grid=grid(9984), stream=stream0)
        del addmm_31
        buf201 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf200, buf201, 768, 13, grid=grid(768), stream=stream0)
        buf203 = reinterpret_tensor(buf159, (1576, 3072), (3072, 1), 0); del buf159  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf202, (1576, 768), (768, 1), 0), permute_166, out=buf203)
        del permute_166
        buf204 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf202, (768, 1576), (1, 768), 0), view_95, out=buf204)
        del view_95
        buf205 = reinterpret_tensor(buf200, (1, 768, 13), (9984, 1, 768), 0); del buf200  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf202, buf205, 9984, 122, grid=grid(9984), stream=stream0)
        buf206 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf205, buf206, 768, 13, grid=grid(768), stream=stream0)
        buf207 = reinterpret_tensor(buf203, (8, 197, 3072), (605184, 3072, 1), 0); del buf203  # reuse
        # Source Nodes: [x_119], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_7.run(buf207, addmm_30, 4841472, grid=grid(4841472), stream=stream0)
        del addmm_30
        buf208 = reinterpret_tensor(buf202, (1576, 768), (768, 1), 0); del buf202  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf207, (1576, 3072), (3072, 1), 0), permute_170, out=buf208)
        del permute_170
        buf209 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf207, (3072, 1576), (1, 3072), 0), view_93, out=buf209)
        del view_93
        buf210 = buf162; del buf162  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf207, buf210, 39936, 122, grid=grid(39936), stream=stream0)
        buf211 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf210, buf211, 3072, 13, grid=grid(3072), stream=stream0)
        buf218 = buf199; del buf199  # reuse
        buf221 = reinterpret_tensor(buf189, (8, 197, 768), (151296, 768, 1), 0); del buf189  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_18.run(buf218, buf208, primals_80, mul_66, div_10, primals_72, buf221, 1576, 768, grid=grid(1576), stream=stream0)
        del div_10
        del primals_72
        del primals_80
        buf214 = reinterpret_tensor(buf205, (768, 13), (1, 768), 0); del buf205  # reuse
        buf216 = buf195; del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf208, mul_66, buf214, buf216, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_66
        buf215 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf214, buf215, 768, 13, grid=grid(768), stream=stream0)
        buf217 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf216, buf217, 768, 13, grid=grid(768), stream=stream0)
        buf219 = reinterpret_tensor(buf216, (1, 1, 768, 13), (9984, 9984, 1, 768), 0); del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_12.run(buf218, addmm_29, buf219, 9984, 122, grid=grid(9984), stream=stream0)
        del addmm_29
        buf220 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf219, buf220, 768, 13, grid=grid(768), stream=stream0)
        buf222 = buf208; del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf221, (1576, 768), (768, 1), 0), permute_174, out=buf222)
        del permute_174
        buf223 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf221, (768, 1576), (1, 768), 0), view_91, out=buf223)
        del view_91
        buf224 = reinterpret_tensor(buf219, (1, 768, 13), (9984, 1, 768), 0); del buf219  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf221, buf224, 9984, 122, grid=grid(9984), stream=stream0)
        del buf221
        buf225 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf224, buf225, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf226 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf222, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_79, getitem_80, getitem_81, expand_8, alias_16, getitem_83, getitem_84, getitem_85, 0.0, [True, True, True, True])
        del alias_16
        del buf222
        del expand_8
        del getitem_79
        del getitem_80
        del getitem_81
        del getitem_83
        del getitem_84
        del getitem_85
        buf227 = buf226[0]
        buf228 = buf226[1]
        buf229 = buf226[2]
        buf230 = buf226[3]
        del buf226
        buf231 = buf185; del buf185  # reuse
        buf233 = buf231; del buf231  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.slice_backward, aten.sum]
        triton_per_fused_constant_pad_nd_slice_backward_sum_13.run(buf233, buf230, 465708, 8, grid=grid(465708), stream=stream0)
        del buf230
        buf232 = empty((732, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_14.run(buf232, 8784, grid=grid(8784), stream=stream0)
        aten.index_put_(buf232, [view_88], reinterpret_tensor(buf233, (38809, 12), (1, 38809), 0), True)
        del view_88
        buf236 = buf188; del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf227, buf228, buf229, buf236, 3631104, grid=grid(3631104), stream=stream0)
        del buf227
        buf237 = reinterpret_tensor(buf229, (1576, 768), (768, 1), 0); del buf229  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf236, (1576, 2304), (2304, 1), 0), permute_181, out=buf237)
        del permute_181
        buf238 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf236, (2304, 1576), (1, 2304), 0), view_85, out=buf238)
        del view_85
        buf239 = buf191; del buf191  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf236, buf239, 29952, 122, grid=grid(29952), stream=stream0)
        buf240 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf239, buf240, 2304, 13, grid=grid(2304), stream=stream0)
        buf247 = buf218; del buf218  # reuse
        buf250 = reinterpret_tensor(buf228, (8, 197, 768), (151296, 768, 1), 0); del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_18.run(buf247, buf237, primals_73, mul_63, div_11, primals_69, buf250, 1576, 768, grid=grid(1576), stream=stream0)
        del div_11
        del primals_69
        del primals_73
        buf243 = reinterpret_tensor(buf224, (768, 13), (1, 768), 0); del buf224  # reuse
        buf245 = buf214; del buf214  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf237, mul_63, buf243, buf245, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_63
        buf244 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf243, buf244, 768, 13, grid=grid(768), stream=stream0)
        buf246 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf245, buf246, 768, 13, grid=grid(768), stream=stream0)
        buf248 = reinterpret_tensor(buf245, (1, 1, 768, 13), (9984, 9984, 1, 768), 0); del buf245  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_12.run(buf247, addmm_27, buf248, 9984, 122, grid=grid(9984), stream=stream0)
        del addmm_27
        buf249 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf248, buf249, 768, 13, grid=grid(768), stream=stream0)
        buf251 = reinterpret_tensor(buf207, (1576, 3072), (3072, 1), 0); del buf207  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf250, (1576, 768), (768, 1), 0), permute_185, out=buf251)
        del permute_185
        buf252 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf250, (768, 1576), (1, 768), 0), view_83, out=buf252)
        del view_83
        buf253 = reinterpret_tensor(buf248, (1, 768, 13), (9984, 1, 768), 0); del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf250, buf253, 9984, 122, grid=grid(9984), stream=stream0)
        buf254 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf253, buf254, 768, 13, grid=grid(768), stream=stream0)
        buf255 = reinterpret_tensor(buf251, (8, 197, 3072), (605184, 3072, 1), 0); del buf251  # reuse
        # Source Nodes: [x_104], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_7.run(buf255, addmm_26, 4841472, grid=grid(4841472), stream=stream0)
        del addmm_26
        buf256 = reinterpret_tensor(buf250, (1576, 768), (768, 1), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf255, (1576, 3072), (3072, 1), 0), permute_189, out=buf256)
        del permute_189
        buf257 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf255, (3072, 1576), (1, 3072), 0), view_81, out=buf257)
        del view_81
        buf258 = buf210; del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf255, buf258, 39936, 122, grid=grid(39936), stream=stream0)
        buf259 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf258, buf259, 3072, 13, grid=grid(3072), stream=stream0)
        buf266 = buf247; del buf247  # reuse
        buf269 = reinterpret_tensor(buf237, (8, 197, 768), (151296, 768, 1), 0); del buf237  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_18.run(buf266, buf256, primals_70, mul_57, div_12, primals_62, buf269, 1576, 768, grid=grid(1576), stream=stream0)
        del div_12
        del primals_62
        del primals_70
        buf262 = reinterpret_tensor(buf253, (768, 13), (1, 768), 0); del buf253  # reuse
        buf264 = buf243; del buf243  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf256, mul_57, buf262, buf264, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_57
        buf263 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf262, buf263, 768, 13, grid=grid(768), stream=stream0)
        buf265 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf264, buf265, 768, 13, grid=grid(768), stream=stream0)
        buf267 = reinterpret_tensor(buf264, (1, 1, 768, 13), (9984, 9984, 1, 768), 0); del buf264  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_12.run(buf266, addmm_25, buf267, 9984, 122, grid=grid(9984), stream=stream0)
        del addmm_25
        buf268 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf267, buf268, 768, 13, grid=grid(768), stream=stream0)
        buf270 = buf256; del buf256  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf269, (1576, 768), (768, 1), 0), permute_193, out=buf270)
        del permute_193
        buf271 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf269, (768, 1576), (1, 768), 0), view_79, out=buf271)
        del view_79
        buf272 = reinterpret_tensor(buf267, (1, 768, 13), (9984, 1, 768), 0); del buf267  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf269, buf272, 9984, 122, grid=grid(9984), stream=stream0)
        del buf269
        buf273 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf272, buf273, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf274 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf270, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_68, getitem_69, getitem_70, expand_7, alias_17, getitem_72, getitem_73, getitem_74, 0.0, [True, True, True, True])
        del alias_17
        del buf270
        del expand_7
        del getitem_68
        del getitem_69
        del getitem_70
        del getitem_72
        del getitem_73
        del getitem_74
        buf275 = buf274[0]
        buf276 = buf274[1]
        buf277 = buf274[2]
        buf278 = buf274[3]
        del buf274
        buf279 = buf233; del buf233  # reuse
        buf281 = buf279; del buf279  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.slice_backward, aten.sum]
        triton_per_fused_constant_pad_nd_slice_backward_sum_13.run(buf281, buf278, 465708, 8, grid=grid(465708), stream=stream0)
        del buf278
        buf280 = empty((732, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_14.run(buf280, 8784, grid=grid(8784), stream=stream0)
        aten.index_put_(buf280, [view_76], reinterpret_tensor(buf281, (38809, 12), (1, 38809), 0), True)
        del view_76
        buf284 = buf236; del buf236  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf275, buf276, buf277, buf284, 3631104, grid=grid(3631104), stream=stream0)
        del buf275
        buf285 = reinterpret_tensor(buf277, (1576, 768), (768, 1), 0); del buf277  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf284, (1576, 2304), (2304, 1), 0), permute_200, out=buf285)
        del permute_200
        buf286 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf284, (2304, 1576), (1, 2304), 0), view_73, out=buf286)
        del view_73
        buf287 = buf239; del buf239  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf284, buf287, 29952, 122, grid=grid(29952), stream=stream0)
        buf288 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf287, buf288, 2304, 13, grid=grid(2304), stream=stream0)
        buf295 = buf266; del buf266  # reuse
        buf298 = reinterpret_tensor(buf276, (8, 197, 768), (151296, 768, 1), 0); del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_18.run(buf295, buf285, primals_63, mul_54, div_13, primals_59, buf298, 1576, 768, grid=grid(1576), stream=stream0)
        del div_13
        del primals_59
        del primals_63
        buf291 = reinterpret_tensor(buf272, (768, 13), (1, 768), 0); del buf272  # reuse
        buf293 = buf262; del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf285, mul_54, buf291, buf293, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_54
        buf292 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf291, buf292, 768, 13, grid=grid(768), stream=stream0)
        buf294 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf293, buf294, 768, 13, grid=grid(768), stream=stream0)
        buf296 = reinterpret_tensor(buf293, (1, 1, 768, 13), (9984, 9984, 1, 768), 0); del buf293  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_12.run(buf295, addmm_23, buf296, 9984, 122, grid=grid(9984), stream=stream0)
        del addmm_23
        buf297 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf296, buf297, 768, 13, grid=grid(768), stream=stream0)
        buf299 = reinterpret_tensor(buf255, (1576, 3072), (3072, 1), 0); del buf255  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf298, (1576, 768), (768, 1), 0), permute_204, out=buf299)
        del permute_204
        buf300 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf298, (768, 1576), (1, 768), 0), view_71, out=buf300)
        del view_71
        buf301 = reinterpret_tensor(buf296, (1, 768, 13), (9984, 1, 768), 0); del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf298, buf301, 9984, 122, grid=grid(9984), stream=stream0)
        buf302 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf301, buf302, 768, 13, grid=grid(768), stream=stream0)
        buf303 = reinterpret_tensor(buf299, (8, 197, 3072), (605184, 3072, 1), 0); del buf299  # reuse
        # Source Nodes: [x_89], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_7.run(buf303, addmm_22, 4841472, grid=grid(4841472), stream=stream0)
        del addmm_22
        buf304 = reinterpret_tensor(buf298, (1576, 768), (768, 1), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf303, (1576, 3072), (3072, 1), 0), permute_208, out=buf304)
        del permute_208
        buf305 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf303, (3072, 1576), (1, 3072), 0), view_69, out=buf305)
        del view_69
        buf306 = buf258; del buf258  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf303, buf306, 39936, 122, grid=grid(39936), stream=stream0)
        buf307 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf306, buf307, 3072, 13, grid=grid(3072), stream=stream0)
        buf314 = buf295; del buf295  # reuse
        buf317 = reinterpret_tensor(buf285, (8, 197, 768), (151296, 768, 1), 0); del buf285  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_18.run(buf314, buf304, primals_60, mul_48, div_14, primals_52, buf317, 1576, 768, grid=grid(1576), stream=stream0)
        del div_14
        del primals_52
        del primals_60
        buf310 = reinterpret_tensor(buf301, (768, 13), (1, 768), 0); del buf301  # reuse
        buf312 = buf291; del buf291  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf304, mul_48, buf310, buf312, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_48
        buf311 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf310, buf311, 768, 13, grid=grid(768), stream=stream0)
        buf313 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf312, buf313, 768, 13, grid=grid(768), stream=stream0)
        buf315 = reinterpret_tensor(buf312, (1, 1, 768, 13), (9984, 9984, 1, 768), 0); del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_12.run(buf314, addmm_21, buf315, 9984, 122, grid=grid(9984), stream=stream0)
        del addmm_21
        buf316 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf315, buf316, 768, 13, grid=grid(768), stream=stream0)
        buf318 = buf304; del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf317, (1576, 768), (768, 1), 0), permute_212, out=buf318)
        del permute_212
        buf319 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf317, (768, 1576), (1, 768), 0), view_67, out=buf319)
        del view_67
        buf320 = reinterpret_tensor(buf315, (1, 768, 13), (9984, 1, 768), 0); del buf315  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf317, buf320, 9984, 122, grid=grid(9984), stream=stream0)
        del buf317
        buf321 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf320, buf321, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf322 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf318, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_57, getitem_58, getitem_59, expand_6, alias_18, getitem_61, getitem_62, getitem_63, 0.0, [True, True, True, True])
        del alias_18
        del buf318
        del expand_6
        del getitem_57
        del getitem_58
        del getitem_59
        del getitem_61
        del getitem_62
        del getitem_63
        buf323 = buf322[0]
        buf324 = buf322[1]
        buf325 = buf322[2]
        buf326 = buf322[3]
        del buf322
        buf327 = buf281; del buf281  # reuse
        buf329 = buf327; del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.slice_backward, aten.sum]
        triton_per_fused_constant_pad_nd_slice_backward_sum_13.run(buf329, buf326, 465708, 8, grid=grid(465708), stream=stream0)
        del buf326
        buf328 = empty((732, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_14.run(buf328, 8784, grid=grid(8784), stream=stream0)
        aten.index_put_(buf328, [view_64], reinterpret_tensor(buf329, (38809, 12), (1, 38809), 0), True)
        del view_64
        buf332 = buf284; del buf284  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf323, buf324, buf325, buf332, 3631104, grid=grid(3631104), stream=stream0)
        del buf323
        buf333 = reinterpret_tensor(buf325, (1576, 768), (768, 1), 0); del buf325  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf332, (1576, 2304), (2304, 1), 0), permute_219, out=buf333)
        del permute_219
        buf334 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf332, (2304, 1576), (1, 2304), 0), view_61, out=buf334)
        del view_61
        buf335 = buf287; del buf287  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf332, buf335, 29952, 122, grid=grid(29952), stream=stream0)
        buf336 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf335, buf336, 2304, 13, grid=grid(2304), stream=stream0)
        buf343 = buf314; del buf314  # reuse
        buf346 = reinterpret_tensor(buf324, (8, 197, 768), (151296, 768, 1), 0); del buf324  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_18.run(buf343, buf333, primals_53, mul_45, div_15, primals_49, buf346, 1576, 768, grid=grid(1576), stream=stream0)
        del div_15
        del primals_49
        del primals_53
        buf339 = reinterpret_tensor(buf320, (768, 13), (1, 768), 0); del buf320  # reuse
        buf341 = buf310; del buf310  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf333, mul_45, buf339, buf341, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_45
        buf340 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf339, buf340, 768, 13, grid=grid(768), stream=stream0)
        buf342 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf341, buf342, 768, 13, grid=grid(768), stream=stream0)
        buf344 = reinterpret_tensor(buf341, (1, 1, 768, 13), (9984, 9984, 1, 768), 0); del buf341  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_12.run(buf343, addmm_19, buf344, 9984, 122, grid=grid(9984), stream=stream0)
        del addmm_19
        buf345 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf344, buf345, 768, 13, grid=grid(768), stream=stream0)
        buf347 = reinterpret_tensor(buf303, (1576, 3072), (3072, 1), 0); del buf303  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf346, (1576, 768), (768, 1), 0), permute_223, out=buf347)
        del permute_223
        buf348 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf346, (768, 1576), (1, 768), 0), view_59, out=buf348)
        del view_59
        buf349 = reinterpret_tensor(buf344, (1, 768, 13), (9984, 1, 768), 0); del buf344  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf346, buf349, 9984, 122, grid=grid(9984), stream=stream0)
        buf350 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf349, buf350, 768, 13, grid=grid(768), stream=stream0)
        buf351 = reinterpret_tensor(buf347, (8, 197, 3072), (605184, 3072, 1), 0); del buf347  # reuse
        # Source Nodes: [x_74], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_7.run(buf351, addmm_18, 4841472, grid=grid(4841472), stream=stream0)
        del addmm_18
        buf352 = reinterpret_tensor(buf346, (1576, 768), (768, 1), 0); del buf346  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf351, (1576, 3072), (3072, 1), 0), permute_227, out=buf352)
        del permute_227
        buf353 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf351, (3072, 1576), (1, 3072), 0), view_57, out=buf353)
        del view_57
        buf354 = buf306; del buf306  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf351, buf354, 39936, 122, grid=grid(39936), stream=stream0)
        buf355 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf354, buf355, 3072, 13, grid=grid(3072), stream=stream0)
        buf362 = buf343; del buf343  # reuse
        buf365 = reinterpret_tensor(buf333, (8, 197, 768), (151296, 768, 1), 0); del buf333  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_18.run(buf362, buf352, primals_50, mul_39, div_16, primals_42, buf365, 1576, 768, grid=grid(1576), stream=stream0)
        del div_16
        del primals_42
        del primals_50
        buf358 = reinterpret_tensor(buf349, (768, 13), (1, 768), 0); del buf349  # reuse
        buf360 = buf339; del buf339  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf352, mul_39, buf358, buf360, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_39
        buf359 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf358, buf359, 768, 13, grid=grid(768), stream=stream0)
        buf361 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf360, buf361, 768, 13, grid=grid(768), stream=stream0)
        buf363 = reinterpret_tensor(buf360, (1, 1, 768, 13), (9984, 9984, 1, 768), 0); del buf360  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_12.run(buf362, addmm_17, buf363, 9984, 122, grid=grid(9984), stream=stream0)
        del addmm_17
        buf364 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf363, buf364, 768, 13, grid=grid(768), stream=stream0)
        buf366 = buf352; del buf352  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf365, (1576, 768), (768, 1), 0), permute_231, out=buf366)
        del permute_231
        buf367 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf365, (768, 1576), (1, 768), 0), view_55, out=buf367)
        del view_55
        buf368 = reinterpret_tensor(buf363, (1, 768, 13), (9984, 1, 768), 0); del buf363  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf365, buf368, 9984, 122, grid=grid(9984), stream=stream0)
        del buf365
        buf369 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf368, buf369, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf370 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf366, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_46, getitem_47, getitem_48, expand_5, alias_19, getitem_50, getitem_51, getitem_52, 0.0, [True, True, True, True])
        del alias_19
        del buf366
        del expand_5
        del getitem_46
        del getitem_47
        del getitem_48
        del getitem_50
        del getitem_51
        del getitem_52
        buf371 = buf370[0]
        buf372 = buf370[1]
        buf373 = buf370[2]
        buf374 = buf370[3]
        del buf370
        buf375 = buf329; del buf329  # reuse
        buf377 = buf375; del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.slice_backward, aten.sum]
        triton_per_fused_constant_pad_nd_slice_backward_sum_13.run(buf377, buf374, 465708, 8, grid=grid(465708), stream=stream0)
        del buf374
        buf376 = empty((732, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_14.run(buf376, 8784, grid=grid(8784), stream=stream0)
        aten.index_put_(buf376, [view_52], reinterpret_tensor(buf377, (38809, 12), (1, 38809), 0), True)
        del view_52
        buf380 = buf332; del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf371, buf372, buf373, buf380, 3631104, grid=grid(3631104), stream=stream0)
        del buf371
        buf381 = reinterpret_tensor(buf373, (1576, 768), (768, 1), 0); del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf380, (1576, 2304), (2304, 1), 0), permute_238, out=buf381)
        del permute_238
        buf382 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf380, (2304, 1576), (1, 2304), 0), view_49, out=buf382)
        del view_49
        buf383 = buf335; del buf335  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf380, buf383, 29952, 122, grid=grid(29952), stream=stream0)
        buf384 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf383, buf384, 2304, 13, grid=grid(2304), stream=stream0)
        buf391 = buf362; del buf362  # reuse
        buf394 = reinterpret_tensor(buf372, (8, 197, 768), (151296, 768, 1), 0); del buf372  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_18.run(buf391, buf381, primals_43, mul_36, div_17, primals_39, buf394, 1576, 768, grid=grid(1576), stream=stream0)
        del div_17
        del primals_39
        del primals_43
        buf387 = reinterpret_tensor(buf368, (768, 13), (1, 768), 0); del buf368  # reuse
        buf389 = buf358; del buf358  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf381, mul_36, buf387, buf389, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_36
        buf388 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf387, buf388, 768, 13, grid=grid(768), stream=stream0)
        buf390 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf389, buf390, 768, 13, grid=grid(768), stream=stream0)
        buf392 = reinterpret_tensor(buf389, (1, 1, 768, 13), (9984, 9984, 1, 768), 0); del buf389  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_12.run(buf391, addmm_15, buf392, 9984, 122, grid=grid(9984), stream=stream0)
        del addmm_15
        buf393 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf392, buf393, 768, 13, grid=grid(768), stream=stream0)
        buf395 = reinterpret_tensor(buf351, (1576, 3072), (3072, 1), 0); del buf351  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf394, (1576, 768), (768, 1), 0), permute_242, out=buf395)
        del permute_242
        buf396 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf394, (768, 1576), (1, 768), 0), view_47, out=buf396)
        del view_47
        buf397 = reinterpret_tensor(buf392, (1, 768, 13), (9984, 1, 768), 0); del buf392  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf394, buf397, 9984, 122, grid=grid(9984), stream=stream0)
        buf398 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf397, buf398, 768, 13, grid=grid(768), stream=stream0)
        buf399 = reinterpret_tensor(buf395, (8, 197, 3072), (605184, 3072, 1), 0); del buf395  # reuse
        # Source Nodes: [x_59], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_7.run(buf399, addmm_14, 4841472, grid=grid(4841472), stream=stream0)
        del addmm_14
        buf400 = reinterpret_tensor(buf394, (1576, 768), (768, 1), 0); del buf394  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf399, (1576, 3072), (3072, 1), 0), permute_246, out=buf400)
        del permute_246
        buf401 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf399, (3072, 1576), (1, 3072), 0), view_45, out=buf401)
        del view_45
        buf402 = buf354; del buf354  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf399, buf402, 39936, 122, grid=grid(39936), stream=stream0)
        buf403 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf402, buf403, 3072, 13, grid=grid(3072), stream=stream0)
        buf410 = buf391; del buf391  # reuse
        buf413 = reinterpret_tensor(buf381, (8, 197, 768), (151296, 768, 1), 0); del buf381  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_18.run(buf410, buf400, primals_40, mul_30, div_18, primals_32, buf413, 1576, 768, grid=grid(1576), stream=stream0)
        del div_18
        del primals_32
        del primals_40
        buf406 = reinterpret_tensor(buf397, (768, 13), (1, 768), 0); del buf397  # reuse
        buf408 = buf387; del buf387  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf400, mul_30, buf406, buf408, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_30
        buf407 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf406, buf407, 768, 13, grid=grid(768), stream=stream0)
        buf409 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf408, buf409, 768, 13, grid=grid(768), stream=stream0)
        buf411 = reinterpret_tensor(buf408, (1, 1, 768, 13), (9984, 9984, 1, 768), 0); del buf408  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_12.run(buf410, addmm_13, buf411, 9984, 122, grid=grid(9984), stream=stream0)
        del addmm_13
        buf412 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf411, buf412, 768, 13, grid=grid(768), stream=stream0)
        buf414 = buf400; del buf400  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf413, (1576, 768), (768, 1), 0), permute_250, out=buf414)
        del permute_250
        buf415 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf413, (768, 1576), (1, 768), 0), view_43, out=buf415)
        del view_43
        buf416 = reinterpret_tensor(buf411, (1, 768, 13), (9984, 1, 768), 0); del buf411  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf413, buf416, 9984, 122, grid=grid(9984), stream=stream0)
        del buf413
        buf417 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf416, buf417, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf418 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf414, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_35, getitem_36, getitem_37, expand_4, alias_20, getitem_39, getitem_40, getitem_41, 0.0, [True, True, True, True])
        del alias_20
        del buf414
        del expand_4
        del getitem_35
        del getitem_36
        del getitem_37
        del getitem_39
        del getitem_40
        del getitem_41
        buf419 = buf418[0]
        buf420 = buf418[1]
        buf421 = buf418[2]
        buf422 = buf418[3]
        del buf418
        buf423 = buf377; del buf377  # reuse
        buf425 = buf423; del buf423  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.slice_backward, aten.sum]
        triton_per_fused_constant_pad_nd_slice_backward_sum_13.run(buf425, buf422, 465708, 8, grid=grid(465708), stream=stream0)
        del buf422
        buf424 = empty((732, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_14.run(buf424, 8784, grid=grid(8784), stream=stream0)
        aten.index_put_(buf424, [view_40], reinterpret_tensor(buf425, (38809, 12), (1, 38809), 0), True)
        del view_40
        buf428 = buf380; del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf419, buf420, buf421, buf428, 3631104, grid=grid(3631104), stream=stream0)
        del buf419
        buf429 = reinterpret_tensor(buf421, (1576, 768), (768, 1), 0); del buf421  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf428, (1576, 2304), (2304, 1), 0), permute_257, out=buf429)
        del permute_257
        buf430 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf428, (2304, 1576), (1, 2304), 0), view_37, out=buf430)
        del view_37
        buf431 = buf383; del buf383  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf428, buf431, 29952, 122, grid=grid(29952), stream=stream0)
        buf432 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf431, buf432, 2304, 13, grid=grid(2304), stream=stream0)
        buf439 = buf410; del buf410  # reuse
        buf442 = reinterpret_tensor(buf420, (8, 197, 768), (151296, 768, 1), 0); del buf420  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_18.run(buf439, buf429, primals_33, mul_27, div_19, primals_29, buf442, 1576, 768, grid=grid(1576), stream=stream0)
        del div_19
        del primals_29
        del primals_33
        buf435 = reinterpret_tensor(buf416, (768, 13), (1, 768), 0); del buf416  # reuse
        buf437 = buf406; del buf406  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf429, mul_27, buf435, buf437, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_27
        buf436 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf435, buf436, 768, 13, grid=grid(768), stream=stream0)
        buf438 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf437, buf438, 768, 13, grid=grid(768), stream=stream0)
        buf440 = reinterpret_tensor(buf437, (1, 1, 768, 13), (9984, 9984, 1, 768), 0); del buf437  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_12.run(buf439, addmm_11, buf440, 9984, 122, grid=grid(9984), stream=stream0)
        del addmm_11
        buf441 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf440, buf441, 768, 13, grid=grid(768), stream=stream0)
        buf443 = reinterpret_tensor(buf399, (1576, 3072), (3072, 1), 0); del buf399  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf442, (1576, 768), (768, 1), 0), permute_261, out=buf443)
        del permute_261
        buf444 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf442, (768, 1576), (1, 768), 0), view_35, out=buf444)
        del view_35
        buf445 = reinterpret_tensor(buf440, (1, 768, 13), (9984, 1, 768), 0); del buf440  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf442, buf445, 9984, 122, grid=grid(9984), stream=stream0)
        buf446 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf445, buf446, 768, 13, grid=grid(768), stream=stream0)
        buf447 = reinterpret_tensor(buf443, (8, 197, 3072), (605184, 3072, 1), 0); del buf443  # reuse
        # Source Nodes: [x_44], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_7.run(buf447, addmm_10, 4841472, grid=grid(4841472), stream=stream0)
        del addmm_10
        buf448 = reinterpret_tensor(buf442, (1576, 768), (768, 1), 0); del buf442  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf447, (1576, 3072), (3072, 1), 0), permute_265, out=buf448)
        del permute_265
        buf449 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf447, (3072, 1576), (1, 3072), 0), view_33, out=buf449)
        del view_33
        buf450 = buf402; del buf402  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf447, buf450, 39936, 122, grid=grid(39936), stream=stream0)
        buf451 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf450, buf451, 3072, 13, grid=grid(3072), stream=stream0)
        buf458 = buf439; del buf439  # reuse
        buf461 = reinterpret_tensor(buf429, (8, 197, 768), (151296, 768, 1), 0); del buf429  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_18.run(buf458, buf448, primals_30, mul_21, div_20, primals_22, buf461, 1576, 768, grid=grid(1576), stream=stream0)
        del div_20
        del primals_22
        del primals_30
        buf454 = reinterpret_tensor(buf445, (768, 13), (1, 768), 0); del buf445  # reuse
        buf456 = buf435; del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf448, mul_21, buf454, buf456, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_21
        buf455 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf454, buf455, 768, 13, grid=grid(768), stream=stream0)
        buf457 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf456, buf457, 768, 13, grid=grid(768), stream=stream0)
        buf459 = reinterpret_tensor(buf456, (1, 1, 768, 13), (9984, 9984, 1, 768), 0); del buf456  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_12.run(buf458, addmm_9, buf459, 9984, 122, grid=grid(9984), stream=stream0)
        del addmm_9
        buf460 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf459, buf460, 768, 13, grid=grid(768), stream=stream0)
        buf462 = buf448; del buf448  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf461, (1576, 768), (768, 1), 0), permute_269, out=buf462)
        del permute_269
        buf463 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf461, (768, 1576), (1, 768), 0), view_31, out=buf463)
        del view_31
        buf464 = reinterpret_tensor(buf459, (1, 768, 13), (9984, 1, 768), 0); del buf459  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf461, buf464, 9984, 122, grid=grid(9984), stream=stream0)
        del buf461
        buf465 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf464, buf465, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf466 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf462, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_24, getitem_25, getitem_26, expand_3, alias_21, getitem_28, getitem_29, getitem_30, 0.0, [True, True, True, True])
        del alias_21
        del buf462
        del expand_3
        del getitem_24
        del getitem_25
        del getitem_26
        del getitem_28
        del getitem_29
        del getitem_30
        buf467 = buf466[0]
        buf468 = buf466[1]
        buf469 = buf466[2]
        buf470 = buf466[3]
        del buf466
        buf471 = buf425; del buf425  # reuse
        buf473 = buf471; del buf471  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.slice_backward, aten.sum]
        triton_per_fused_constant_pad_nd_slice_backward_sum_13.run(buf473, buf470, 465708, 8, grid=grid(465708), stream=stream0)
        del buf470
        buf472 = empty((732, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_14.run(buf472, 8784, grid=grid(8784), stream=stream0)
        aten.index_put_(buf472, [view_28], reinterpret_tensor(buf473, (38809, 12), (1, 38809), 0), True)
        del view_28
        buf476 = buf428; del buf428  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf467, buf468, buf469, buf476, 3631104, grid=grid(3631104), stream=stream0)
        del buf467
        buf477 = reinterpret_tensor(buf469, (1576, 768), (768, 1), 0); del buf469  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf476, (1576, 2304), (2304, 1), 0), permute_276, out=buf477)
        del permute_276
        buf478 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf476, (2304, 1576), (1, 2304), 0), view_25, out=buf478)
        del view_25
        buf479 = buf431; del buf431  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf476, buf479, 29952, 122, grid=grid(29952), stream=stream0)
        buf480 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf479, buf480, 2304, 13, grid=grid(2304), stream=stream0)
        buf487 = buf458; del buf458  # reuse
        buf490 = reinterpret_tensor(buf468, (8, 197, 768), (151296, 768, 1), 0); del buf468  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_18.run(buf487, buf477, primals_23, mul_18, div_21, primals_19, buf490, 1576, 768, grid=grid(1576), stream=stream0)
        del div_21
        del primals_19
        del primals_23
        buf483 = reinterpret_tensor(buf464, (768, 13), (1, 768), 0); del buf464  # reuse
        buf485 = buf454; del buf454  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf477, mul_18, buf483, buf485, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_18
        buf484 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf483, buf484, 768, 13, grid=grid(768), stream=stream0)
        buf486 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf485, buf486, 768, 13, grid=grid(768), stream=stream0)
        buf488 = reinterpret_tensor(buf485, (1, 1, 768, 13), (9984, 9984, 1, 768), 0); del buf485  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_12.run(buf487, addmm_7, buf488, 9984, 122, grid=grid(9984), stream=stream0)
        del addmm_7
        buf489 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf488, buf489, 768, 13, grid=grid(768), stream=stream0)
        buf491 = reinterpret_tensor(buf447, (1576, 3072), (3072, 1), 0); del buf447  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf490, (1576, 768), (768, 1), 0), permute_280, out=buf491)
        del permute_280
        buf492 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf490, (768, 1576), (1, 768), 0), view_23, out=buf492)
        del view_23
        buf493 = reinterpret_tensor(buf488, (1, 768, 13), (9984, 1, 768), 0); del buf488  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf490, buf493, 9984, 122, grid=grid(9984), stream=stream0)
        buf494 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf493, buf494, 768, 13, grid=grid(768), stream=stream0)
        buf495 = reinterpret_tensor(buf491, (8, 197, 3072), (605184, 3072, 1), 0); del buf491  # reuse
        # Source Nodes: [x_29], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_7.run(buf495, addmm_6, 4841472, grid=grid(4841472), stream=stream0)
        del addmm_6
        buf496 = reinterpret_tensor(buf490, (1576, 768), (768, 1), 0); del buf490  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf495, (1576, 3072), (3072, 1), 0), permute_284, out=buf496)
        del permute_284
        buf497 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf495, (3072, 1576), (1, 3072), 0), view_21, out=buf497)
        del view_21
        buf498 = buf450; del buf450  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf495, buf498, 39936, 122, grid=grid(39936), stream=stream0)
        buf499 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf498, buf499, 3072, 13, grid=grid(3072), stream=stream0)
        buf506 = buf487; del buf487  # reuse
        buf509 = reinterpret_tensor(buf477, (8, 197, 768), (151296, 768, 1), 0); del buf477  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_18.run(buf506, buf496, primals_20, mul_12, div_22, primals_12, buf509, 1576, 768, grid=grid(1576), stream=stream0)
        del div_22
        del primals_12
        del primals_20
        buf502 = reinterpret_tensor(buf493, (768, 13), (1, 768), 0); del buf493  # reuse
        buf504 = buf483; del buf483  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf496, mul_12, buf502, buf504, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_12
        buf503 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf502, buf503, 768, 13, grid=grid(768), stream=stream0)
        buf505 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf504, buf505, 768, 13, grid=grid(768), stream=stream0)
        buf507 = reinterpret_tensor(buf504, (1, 1, 768, 13), (9984, 9984, 1, 768), 0); del buf504  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_12.run(buf506, addmm_5, buf507, 9984, 122, grid=grid(9984), stream=stream0)
        del addmm_5
        buf508 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf507, buf508, 768, 13, grid=grid(768), stream=stream0)
        buf510 = buf496; del buf496  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf509, (1576, 768), (768, 1), 0), permute_288, out=buf510)
        del permute_288
        buf511 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf509, (768, 1576), (1, 768), 0), view_19, out=buf511)
        del view_19
        buf512 = reinterpret_tensor(buf507, (1, 768, 13), (9984, 1, 768), 0); del buf507  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf509, buf512, 9984, 122, grid=grid(9984), stream=stream0)
        del buf509
        buf513 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf512, buf513, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf514 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf510, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_13, getitem_14, getitem_15, expand_2, alias_22, getitem_17, getitem_18, getitem_19, 0.0, [True, True, True, True])
        del alias_22
        del buf510
        del expand_2
        del getitem_13
        del getitem_14
        del getitem_15
        del getitem_17
        del getitem_18
        del getitem_19
        buf515 = buf514[0]
        buf516 = buf514[1]
        buf517 = buf514[2]
        buf518 = buf514[3]
        del buf514
        buf519 = buf473; del buf473  # reuse
        buf521 = buf519; del buf519  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.slice_backward, aten.sum]
        triton_per_fused_constant_pad_nd_slice_backward_sum_13.run(buf521, buf518, 465708, 8, grid=grid(465708), stream=stream0)
        del buf518
        buf520 = empty((732, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_14.run(buf520, 8784, grid=grid(8784), stream=stream0)
        aten.index_put_(buf520, [view_16], reinterpret_tensor(buf521, (38809, 12), (1, 38809), 0), True)
        del view_16
        buf524 = buf476; del buf476  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf515, buf516, buf517, buf524, 3631104, grid=grid(3631104), stream=stream0)
        del buf515
        buf525 = reinterpret_tensor(buf517, (1576, 768), (768, 1), 0); del buf517  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf524, (1576, 2304), (2304, 1), 0), permute_295, out=buf525)
        del permute_295
        buf526 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf524, (2304, 1576), (1, 2304), 0), view_13, out=buf526)
        del view_13
        buf527 = buf479; del buf479  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf524, buf527, 29952, 122, grid=grid(29952), stream=stream0)
        buf528 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf527, buf528, 2304, 13, grid=grid(2304), stream=stream0)
        buf535 = buf506; del buf506  # reuse
        buf538 = reinterpret_tensor(buf516, (8, 197, 768), (151296, 768, 1), 0); del buf516  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_18.run(buf535, buf525, primals_13, mul_9, div_23, primals_9, buf538, 1576, 768, grid=grid(1576), stream=stream0)
        del div_23
        del primals_13
        del primals_9
        buf531 = reinterpret_tensor(buf512, (768, 13), (1, 768), 0); del buf512  # reuse
        buf533 = buf502; del buf502  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf525, mul_9, buf531, buf533, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_9
        buf532 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf531, buf532, 768, 13, grid=grid(768), stream=stream0)
        buf534 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf533, buf534, 768, 13, grid=grid(768), stream=stream0)
        buf536 = reinterpret_tensor(buf533, (1, 1, 768, 13), (9984, 9984, 1, 768), 0); del buf533  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_12.run(buf535, addmm_3, buf536, 9984, 122, grid=grid(9984), stream=stream0)
        del addmm_3
        buf537 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf536, buf537, 768, 13, grid=grid(768), stream=stream0)
        buf539 = reinterpret_tensor(buf495, (1576, 3072), (3072, 1), 0); del buf495  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf538, (1576, 768), (768, 1), 0), permute_299, out=buf539)
        del permute_299
        buf540 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf538, (768, 1576), (1, 768), 0), view_11, out=buf540)
        del view_11
        buf541 = reinterpret_tensor(buf536, (1, 768, 13), (9984, 1, 768), 0); del buf536  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf538, buf541, 9984, 122, grid=grid(9984), stream=stream0)
        buf542 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf541, buf542, 768, 13, grid=grid(768), stream=stream0)
        buf543 = reinterpret_tensor(buf539, (8, 197, 3072), (605184, 3072, 1), 0); del buf539  # reuse
        # Source Nodes: [x_14], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_7.run(buf543, addmm_2, 4841472, grid=grid(4841472), stream=stream0)
        del addmm_2
        buf544 = reinterpret_tensor(buf538, (1576, 768), (768, 1), 0); del buf538  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf543, (1576, 3072), (3072, 1), 0), permute_303, out=buf544)
        del permute_303
        buf545 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf543, (3072, 1576), (1, 3072), 0), view_9, out=buf545)
        del view_9
        buf546 = buf498; del buf498  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_8.run(buf543, buf546, 39936, 122, grid=grid(39936), stream=stream0)
        del buf543
        buf547 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_9.run(buf546, buf547, 3072, 13, grid=grid(3072), stream=stream0)
        del buf546
        buf554 = buf535; del buf535  # reuse
        buf557 = reinterpret_tensor(buf525, (8, 197, 768), (151296, 768, 1), 0); del buf525  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_mul_native_layer_norm_backward_18.run(buf554, buf544, primals_10, mul_3, div_24, primals_2, buf557, 1576, 768, grid=grid(1576), stream=stream0)
        del div_24
        del primals_10
        del primals_2
        buf550 = reinterpret_tensor(buf541, (768, 13), (1, 768), 0); del buf541  # reuse
        buf552 = buf531; del buf531  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_11.run(buf544, mul_3, buf550, buf552, 9984, 122, grid=grid(9984), stream=stream0)
        del mul_3
        buf551 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf550, buf551, 768, 13, grid=grid(768), stream=stream0)
        buf553 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf552, buf553, 768, 13, grid=grid(768), stream=stream0)
        buf555 = reinterpret_tensor(buf552, (1, 1, 768, 13), (9984, 9984, 1, 768), 0); del buf552  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_12.run(buf554, addmm_1, buf555, 9984, 122, grid=grid(9984), stream=stream0)
        del addmm_1
        buf556 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf555, buf556, 768, 13, grid=grid(768), stream=stream0)
        buf558 = buf544; del buf544  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf557, (1576, 768), (768, 1), 0), permute_307, out=buf558)
        del permute_307
        buf559 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf557, (768, 1576), (1, 768), 0), view_7, out=buf559)
        del view_7
        buf560 = reinterpret_tensor(buf555, (1, 768, 13), (9984, 1, 768), 0); del buf555  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf557, buf560, 9984, 122, grid=grid(9984), stream=stream0)
        del buf557
        buf561 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf560, buf561, 768, 13, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf562 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf558, (8, 12, 197, 64), (151296, 64, 768, 1), 0), getitem_2, getitem_3, getitem_4, expand_1, alias_23, getitem_6, getitem_7, getitem_8, 0.0, [True, True, True, True])
        del alias_23
        del buf558
        del expand_1
        del getitem_2
        del getitem_3
        del getitem_4
        del getitem_6
        del getitem_7
        del getitem_8
        buf563 = buf562[0]
        buf564 = buf562[1]
        buf565 = buf562[2]
        buf566 = buf562[3]
        del buf562
        buf567 = buf521; del buf521  # reuse
        buf569 = buf567; del buf567  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.slice_backward, aten.sum]
        triton_per_fused_constant_pad_nd_slice_backward_sum_13.run(buf569, buf566, 465708, 8, grid=grid(465708), stream=stream0)
        del buf566
        buf568 = empty((732, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_14.run(buf568, 8784, grid=grid(8784), stream=stream0)
        aten.index_put_(buf568, [view_4], reinterpret_tensor(buf569, (38809, 12), (1, 38809), 0), True)
        del buf569
        del view_4
        buf572 = buf524; del buf524  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf563, buf564, buf565, buf572, 3631104, grid=grid(3631104), stream=stream0)
        del buf563
        del buf564
        buf573 = reinterpret_tensor(buf565, (1576, 768), (768, 1), 0); del buf565  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf572, (1576, 2304), (2304, 1), 0), permute_314, out=buf573)
        del permute_314
        buf574 = empty((2304, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf572, (2304, 1576), (1, 2304), 0), view_1, out=buf574)
        del view_1
        buf575 = buf527; del buf527  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf572, buf575, 29952, 122, grid=grid(29952), stream=stream0)
        del buf572
        buf576 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf575, buf576, 2304, 13, grid=grid(2304), stream=stream0)
        del buf575
        buf583 = buf554; del buf554  # reuse
        # Source Nodes: [x_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_19.run(buf583, buf573, primals_3, cat, getitem_1, rsqrt, 1576, 768, grid=grid(1576), stream=stream0)
        del primals_3
        buf579 = reinterpret_tensor(buf560, (768, 13), (1, 768), 0); del buf560  # reuse
        buf581 = buf550; del buf550  # reuse
        # Source Nodes: [x_6], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_20.run(buf573, cat, getitem_1, rsqrt, buf579, buf581, 9984, 122, grid=grid(9984), stream=stream0)
        del buf573
        del cat
        del getitem_1
        del rsqrt
        buf580 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf579, buf580, 768, 13, grid=grid(768), stream=stream0)
        del buf579
        buf582 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf581, buf582, 768, 13, grid=grid(768), stream=stream0)
        buf584 = empty((1, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_21.run(buf583, buf584, 768, 8, grid=grid(768), stream=stream0)
        buf585 = buf581; del buf581  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_22.run(buf583, buf585, 9984, 121, grid=grid(9984), stream=stream0)
        buf586 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_div_mul_slice_backward_sum_4.run(buf585, buf586, 768, 13, grid=grid(768), stream=stream0)
        del buf585
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf587 = aten.convolution_backward(reinterpret_tensor(buf583, (8, 768, 14, 14), (151296, 1, 10752, 768), 768), primals_224, primals_124, [768], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf583
        del primals_124
        del primals_224
        buf588 = buf587[1]
        return (buf584, reinterpret_tensor(buf556, (768, ), (1, ), 0), buf580, buf582, reinterpret_tensor(buf576, (768, ), (1, ), 0), reinterpret_tensor(buf576, (768, ), (1, ), 1536), reinterpret_tensor(buf574, (2304, 768), (768, 1), 0), buf568, reinterpret_tensor(buf537, (768, ), (1, ), 0), buf551, buf553, reinterpret_tensor(buf508, (768, ), (1, ), 0), buf532, buf534, reinterpret_tensor(buf528, (768, ), (1, ), 0), reinterpret_tensor(buf528, (768, ), (1, ), 1536), reinterpret_tensor(buf526, (2304, 768), (768, 1), 0), buf520, reinterpret_tensor(buf489, (768, ), (1, ), 0), buf503, buf505, reinterpret_tensor(buf460, (768, ), (1, ), 0), buf484, buf486, reinterpret_tensor(buf480, (768, ), (1, ), 0), reinterpret_tensor(buf480, (768, ), (1, ), 1536), reinterpret_tensor(buf478, (2304, 768), (768, 1), 0), buf472, reinterpret_tensor(buf441, (768, ), (1, ), 0), buf455, buf457, reinterpret_tensor(buf412, (768, ), (1, ), 0), buf436, buf438, reinterpret_tensor(buf432, (768, ), (1, ), 0), reinterpret_tensor(buf432, (768, ), (1, ), 1536), reinterpret_tensor(buf430, (2304, 768), (768, 1), 0), buf424, reinterpret_tensor(buf393, (768, ), (1, ), 0), buf407, buf409, reinterpret_tensor(buf364, (768, ), (1, ), 0), buf388, buf390, reinterpret_tensor(buf384, (768, ), (1, ), 0), reinterpret_tensor(buf384, (768, ), (1, ), 1536), reinterpret_tensor(buf382, (2304, 768), (768, 1), 0), buf376, reinterpret_tensor(buf345, (768, ), (1, ), 0), buf359, buf361, reinterpret_tensor(buf316, (768, ), (1, ), 0), buf340, buf342, reinterpret_tensor(buf336, (768, ), (1, ), 0), reinterpret_tensor(buf336, (768, ), (1, ), 1536), reinterpret_tensor(buf334, (2304, 768), (768, 1), 0), buf328, reinterpret_tensor(buf297, (768, ), (1, ), 0), buf311, buf313, reinterpret_tensor(buf268, (768, ), (1, ), 0), buf292, buf294, reinterpret_tensor(buf288, (768, ), (1, ), 0), reinterpret_tensor(buf288, (768, ), (1, ), 1536), reinterpret_tensor(buf286, (2304, 768), (768, 1), 0), buf280, reinterpret_tensor(buf249, (768, ), (1, ), 0), buf263, buf265, reinterpret_tensor(buf220, (768, ), (1, ), 0), buf244, buf246, reinterpret_tensor(buf240, (768, ), (1, ), 0), reinterpret_tensor(buf240, (768, ), (1, ), 1536), reinterpret_tensor(buf238, (2304, 768), (768, 1), 0), buf232, reinterpret_tensor(buf201, (768, ), (1, ), 0), buf215, buf217, reinterpret_tensor(buf172, (768, ), (1, ), 0), buf196, buf198, reinterpret_tensor(buf192, (768, ), (1, ), 0), reinterpret_tensor(buf192, (768, ), (1, ), 1536), reinterpret_tensor(buf190, (2304, 768), (768, 1), 0), buf184, reinterpret_tensor(buf153, (768, ), (1, ), 0), buf167, buf169, reinterpret_tensor(buf124, (768, ), (1, ), 0), buf148, buf150, reinterpret_tensor(buf144, (768, ), (1, ), 0), reinterpret_tensor(buf144, (768, ), (1, ), 1536), reinterpret_tensor(buf142, (2304, 768), (768, 1), 0), buf136, reinterpret_tensor(buf105, (768, ), (1, ), 0), buf119, buf121, reinterpret_tensor(buf76, (768, ), (1, ), 0), buf100, buf102, reinterpret_tensor(buf96, (768, ), (1, ), 0), reinterpret_tensor(buf96, (768, ), (1, ), 1536), reinterpret_tensor(buf94, (2304, 768), (768, 1), 0), buf88, reinterpret_tensor(buf57, (768, ), (1, ), 0), buf71, buf73, reinterpret_tensor(buf28, (768, ), (1, ), 0), buf52, buf54, reinterpret_tensor(buf48, (768, ), (1, ), 0), reinterpret_tensor(buf48, (768, ), (1, ), 1536), reinterpret_tensor(buf46, (2304, 768), (768, 1), 0), buf40, reinterpret_tensor(buf9, (768, ), (1, ), 0), buf23, buf25, buf5, buf6, buf588, buf586, reinterpret_tensor(buf559, (768, 768), (768, 1), 0), reinterpret_tensor(buf561, (768, ), (1, ), 0), reinterpret_tensor(buf545, (3072, 768), (768, 1), 0), reinterpret_tensor(buf547, (3072, ), (1, ), 0), reinterpret_tensor(buf540, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf542, (768, ), (1, ), 0), reinterpret_tensor(buf511, (768, 768), (768, 1), 0), reinterpret_tensor(buf513, (768, ), (1, ), 0), reinterpret_tensor(buf497, (3072, 768), (768, 1), 0), reinterpret_tensor(buf499, (3072, ), (1, ), 0), reinterpret_tensor(buf492, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf494, (768, ), (1, ), 0), reinterpret_tensor(buf463, (768, 768), (768, 1), 0), reinterpret_tensor(buf465, (768, ), (1, ), 0), reinterpret_tensor(buf449, (3072, 768), (768, 1), 0), reinterpret_tensor(buf451, (3072, ), (1, ), 0), reinterpret_tensor(buf444, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf446, (768, ), (1, ), 0), reinterpret_tensor(buf415, (768, 768), (768, 1), 0), reinterpret_tensor(buf417, (768, ), (1, ), 0), reinterpret_tensor(buf401, (3072, 768), (768, 1), 0), reinterpret_tensor(buf403, (3072, ), (1, ), 0), reinterpret_tensor(buf396, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf398, (768, ), (1, ), 0), reinterpret_tensor(buf367, (768, 768), (768, 1), 0), reinterpret_tensor(buf369, (768, ), (1, ), 0), reinterpret_tensor(buf353, (3072, 768), (768, 1), 0), reinterpret_tensor(buf355, (3072, ), (1, ), 0), reinterpret_tensor(buf348, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf350, (768, ), (1, ), 0), reinterpret_tensor(buf319, (768, 768), (768, 1), 0), reinterpret_tensor(buf321, (768, ), (1, ), 0), reinterpret_tensor(buf305, (3072, 768), (768, 1), 0), reinterpret_tensor(buf307, (3072, ), (1, ), 0), reinterpret_tensor(buf300, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf302, (768, ), (1, ), 0), reinterpret_tensor(buf271, (768, 768), (768, 1), 0), reinterpret_tensor(buf273, (768, ), (1, ), 0), reinterpret_tensor(buf257, (3072, 768), (768, 1), 0), reinterpret_tensor(buf259, (3072, ), (1, ), 0), reinterpret_tensor(buf252, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf254, (768, ), (1, ), 0), reinterpret_tensor(buf223, (768, 768), (768, 1), 0), reinterpret_tensor(buf225, (768, ), (1, ), 0), reinterpret_tensor(buf209, (3072, 768), (768, 1), 0), reinterpret_tensor(buf211, (3072, ), (1, ), 0), reinterpret_tensor(buf204, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf206, (768, ), (1, ), 0), reinterpret_tensor(buf175, (768, 768), (768, 1), 0), reinterpret_tensor(buf177, (768, ), (1, ), 0), reinterpret_tensor(buf161, (3072, 768), (768, 1), 0), reinterpret_tensor(buf163, (3072, ), (1, ), 0), reinterpret_tensor(buf156, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf158, (768, ), (1, ), 0), reinterpret_tensor(buf127, (768, 768), (768, 1), 0), reinterpret_tensor(buf129, (768, ), (1, ), 0), reinterpret_tensor(buf113, (3072, 768), (768, 1), 0), reinterpret_tensor(buf115, (3072, ), (1, ), 0), reinterpret_tensor(buf108, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf110, (768, ), (1, ), 0), reinterpret_tensor(buf79, (768, 768), (768, 1), 0), reinterpret_tensor(buf81, (768, ), (1, ), 0), reinterpret_tensor(buf65, (3072, 768), (768, 1), 0), reinterpret_tensor(buf67, (3072, ), (1, ), 0), reinterpret_tensor(buf60, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf62, (768, ), (1, ), 0), reinterpret_tensor(buf31, (768, 768), (768, 1), 0), reinterpret_tensor(buf33, (768, ), (1, ), 0), reinterpret_tensor(buf17, (3072, 768), (768, 1), 0), reinterpret_tensor(buf19, (3072, ), (1, ), 0), reinterpret_tensor(buf12, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf14, (768, ), (1, ), 0), reinterpret_tensor(buf1, (1000, 768), (768, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_2 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((768, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    cat = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_1 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_2 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_4 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    view_4 = rand_strided((38809, ), (1, ), device='cuda:0', dtype=torch.int64)
    expand_1 = rand_strided((8, 12, 197, 197), (0, 39400, 200, 1), device='cuda:0', dtype=torch.float32)
    getitem_6 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_8 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_7 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_1 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_3 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_9 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_2 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_11 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_3 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_9 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_13 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_13 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_14 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_15 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    view_16 = rand_strided((38809, ), (1, ), device='cuda:0', dtype=torch.int64)
    expand_2 = rand_strided((8, 12, 197, 197), (0, 39400, 200, 1), device='cuda:0', dtype=torch.float32)
    getitem_17 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_18 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_19 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_19 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_5 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_12 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_21 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_6 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_23 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_7 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_18 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_25 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_24 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_25 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_26 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    view_28 = rand_strided((38809, ), (1, ), device='cuda:0', dtype=torch.int64)
    expand_3 = rand_strided((8, 12, 197, 197), (0, 39400, 200, 1), device='cuda:0', dtype=torch.float32)
    getitem_28 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_29 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_30 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_31 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_9 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_21 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_33 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_35 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_11 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_27 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_37 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_35 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_36 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_37 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    view_40 = rand_strided((38809, ), (1, ), device='cuda:0', dtype=torch.int64)
    expand_4 = rand_strided((8, 12, 197, 197), (0, 39400, 200, 1), device='cuda:0', dtype=torch.float32)
    getitem_39 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_40 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_41 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_43 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_13 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_30 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_45 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_14 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_47 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_15 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_36 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_49 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_46 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_47 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_48 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    view_52 = rand_strided((38809, ), (1, ), device='cuda:0', dtype=torch.int64)
    expand_5 = rand_strided((8, 12, 197, 197), (0, 39400, 200, 1), device='cuda:0', dtype=torch.float32)
    getitem_50 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_51 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_52 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_55 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_17 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_39 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_57 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_18 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_59 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_19 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_45 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_61 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_57 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_58 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_59 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    view_64 = rand_strided((38809, ), (1, ), device='cuda:0', dtype=torch.int64)
    expand_6 = rand_strided((8, 12, 197, 197), (0, 39400, 200, 1), device='cuda:0', dtype=torch.float32)
    getitem_61 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_62 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_63 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_67 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_21 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_48 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_69 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_71 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_23 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_54 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_73 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_68 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_69 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_70 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    view_76 = rand_strided((38809, ), (1, ), device='cuda:0', dtype=torch.int64)
    expand_7 = rand_strided((8, 12, 197, 197), (0, 39400, 200, 1), device='cuda:0', dtype=torch.float32)
    getitem_72 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_73 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_74 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_79 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_25 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_57 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_81 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_26 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_83 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_27 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_63 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_85 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_79 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_80 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_81 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    view_88 = rand_strided((38809, ), (1, ), device='cuda:0', dtype=torch.int64)
    expand_8 = rand_strided((8, 12, 197, 197), (0, 39400, 200, 1), device='cuda:0', dtype=torch.float32)
    getitem_83 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_84 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_85 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_91 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_29 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_66 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_93 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_30 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_95 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_31 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_72 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_97 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_90 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_91 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_92 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    view_100 = rand_strided((38809, ), (1, ), device='cuda:0', dtype=torch.int64)
    expand_9 = rand_strided((8, 12, 197, 197), (0, 39400, 200, 1), device='cuda:0', dtype=torch.float32)
    getitem_94 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_95 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_96 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_103 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_33 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_75 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_105 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_107 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_35 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_81 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_109 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_101 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_102 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_103 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    view_112 = rand_strided((38809, ), (1, ), device='cuda:0', dtype=torch.int64)
    expand_10 = rand_strided((8, 12, 197, 197), (0, 39400, 200, 1), device='cuda:0', dtype=torch.float32)
    getitem_105 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_106 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_107 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_115 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_37 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_84 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_117 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_38 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_119 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_39 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_90 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_121 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_112 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_113 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_114 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    view_124 = rand_strided((38809, ), (1, ), device='cuda:0', dtype=torch.int64)
    expand_11 = rand_strided((8, 12, 197, 197), (0, 39400, 200, 1), device='cuda:0', dtype=torch.float32)
    getitem_116 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_117 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_118 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_127 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_41 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_93 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_129 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_42 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_131 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_43 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_99 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_133 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_123 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_124 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    getitem_125 = rand_strided((8, 12, 197, 64), (453888, 64, 2304, 1), device='cuda:0', dtype=torch.float32)
    view_136 = rand_strided((38809, ), (1, ), device='cuda:0', dtype=torch.int64)
    expand_12 = rand_strided((8, 12, 197, 197), (0, 39400, 200, 1), device='cuda:0', dtype=torch.float32)
    getitem_127 = rand_strided((8, 12, 224), (2688, 224, 1), device='cuda:0', dtype=torch.float32)
    getitem_128 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_129 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_139 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_45 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_102 = rand_strided((8, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    view_141 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_46 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_143 = rand_strided((1576, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    addmm_47 = rand_strided((1576, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mul_108 = rand_strided((8, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    clone_49 = rand_strided((8, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_86 = rand_strided((1000, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    permute_90 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_94 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_2 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_98 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_12 = rand_strided((8, 12, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_105 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_109 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_113 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_4 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_117 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_13 = rand_strided((8, 12, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_124 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_5 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_128 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_132 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_6 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_136 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_14 = rand_strided((8, 12, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_143 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_7 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_147 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_151 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_8 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_155 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_15 = rand_strided((8, 12, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_162 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_9 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_166 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_170 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_10 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_174 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_16 = rand_strided((8, 12, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_181 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_11 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_185 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_189 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_12 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_193 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_17 = rand_strided((8, 12, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_200 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_13 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_204 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_208 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_14 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_212 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_18 = rand_strided((8, 12, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_219 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_15 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_223 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_227 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_16 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_231 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_19 = rand_strided((8, 12, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_238 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_17 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_242 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_246 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_18 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_250 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_20 = rand_strided((8, 12, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_257 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_19 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_261 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_265 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_20 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_269 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_21 = rand_strided((8, 12, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_276 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_21 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_280 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_284 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_22 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_288 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_22 = rand_strided((8, 12, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_295 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_23 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_299 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_303 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_24 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_307 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    alias_23 = rand_strided((8, 12, 197, 64), (151296, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    permute_314 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_2, primals_3, primals_9, primals_10, primals_12, primals_13, primals_19, primals_20, primals_22, primals_23, primals_29, primals_30, primals_32, primals_33, primals_39, primals_40, primals_42, primals_43, primals_49, primals_50, primals_52, primals_53, primals_59, primals_60, primals_62, primals_63, primals_69, primals_70, primals_72, primals_73, primals_79, primals_80, primals_82, primals_83, primals_89, primals_90, primals_92, primals_93, primals_99, primals_100, primals_102, primals_103, primals_109, primals_110, primals_112, primals_113, primals_119, primals_120, primals_122, primals_124, primals_224, cat, getitem_1, rsqrt, view_1, getitem_2, getitem_3, getitem_4, view_4, expand_1, getitem_6, getitem_7, getitem_8, view_7, addmm_1, mul_3, view_9, addmm_2, view_11, addmm_3, mul_9, view_13, getitem_13, getitem_14, getitem_15, view_16, expand_2, getitem_17, getitem_18, getitem_19, view_19, addmm_5, mul_12, view_21, addmm_6, view_23, addmm_7, mul_18, view_25, getitem_24, getitem_25, getitem_26, view_28, expand_3, getitem_28, getitem_29, getitem_30, view_31, addmm_9, mul_21, view_33, addmm_10, view_35, addmm_11, mul_27, view_37, getitem_35, getitem_36, getitem_37, view_40, expand_4, getitem_39, getitem_40, getitem_41, view_43, addmm_13, mul_30, view_45, addmm_14, view_47, addmm_15, mul_36, view_49, getitem_46, getitem_47, getitem_48, view_52, expand_5, getitem_50, getitem_51, getitem_52, view_55, addmm_17, mul_39, view_57, addmm_18, view_59, addmm_19, mul_45, view_61, getitem_57, getitem_58, getitem_59, view_64, expand_6, getitem_61, getitem_62, getitem_63, view_67, addmm_21, mul_48, view_69, addmm_22, view_71, addmm_23, mul_54, view_73, getitem_68, getitem_69, getitem_70, view_76, expand_7, getitem_72, getitem_73, getitem_74, view_79, addmm_25, mul_57, view_81, addmm_26, view_83, addmm_27, mul_63, view_85, getitem_79, getitem_80, getitem_81, view_88, expand_8, getitem_83, getitem_84, getitem_85, view_91, addmm_29, mul_66, view_93, addmm_30, view_95, addmm_31, mul_72, view_97, getitem_90, getitem_91, getitem_92, view_100, expand_9, getitem_94, getitem_95, getitem_96, view_103, addmm_33, mul_75, view_105, addmm_34, view_107, addmm_35, mul_81, view_109, getitem_101, getitem_102, getitem_103, view_112, expand_10, getitem_105, getitem_106, getitem_107, view_115, addmm_37, mul_84, view_117, addmm_38, view_119, addmm_39, mul_90, view_121, getitem_112, getitem_113, getitem_114, view_124, expand_11, getitem_116, getitem_117, getitem_118, view_127, addmm_41, mul_93, view_129, addmm_42, view_131, addmm_43, mul_99, view_133, getitem_123, getitem_124, getitem_125, view_136, expand_12, getitem_127, getitem_128, getitem_129, view_139, addmm_45, mul_102, view_141, addmm_46, view_143, addmm_47, mul_108, clone_49, permute_86, div, permute_90, permute_94, div_2, permute_98, alias_12, permute_105, div_3, permute_109, permute_113, div_4, permute_117, alias_13, permute_124, div_5, permute_128, permute_132, div_6, permute_136, alias_14, permute_143, div_7, permute_147, permute_151, div_8, permute_155, alias_15, permute_162, div_9, permute_166, permute_170, div_10, permute_174, alias_16, permute_181, div_11, permute_185, permute_189, div_12, permute_193, alias_17, permute_200, div_13, permute_204, permute_208, div_14, permute_212, alias_18, permute_219, div_15, permute_223, permute_227, div_16, permute_231, alias_19, permute_238, div_17, permute_242, permute_246, div_18, permute_250, alias_20, permute_257, div_19, permute_261, permute_265, div_20, permute_269, alias_21, permute_276, div_21, permute_280, permute_284, div_22, permute_288, alias_22, permute_295, div_23, permute_299, permute_303, div_24, permute_307, alias_23, permute_314, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('beit_base_patch16_224', benchmark_compiled_module)
