
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


# kernel path: /tmp/torchinductor_youkaichao/r2/cr2mfq7afvrxx3ov3rrz4zqlapxzai6gv5nkbfzrafveysdcwuil.py
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
    size_hints=[512, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_select_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 400
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 50
    x1 = (xindex // 50)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp3 = tl.load(in_ptr0 + (r2 + (512*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr2 + (r2 + (512*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
        tmp20 = tl.load(in_ptr0 + (r2 + (512*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.load(in_ptr2 + (r2 + (512*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = x0
        tmp18 = tl.full([1, 1], 0, tl.int32)
        tmp19 = tmp17 == tmp18
        tmp21 = 0.0
        tmp22 = tl.where(tmp19, tmp20, tmp21)
        tmp24 = tmp22 * tmp23
        tmp25 = 512.0
        tmp26 = tmp24 * tmp25
        tmp27 = tmp26 - tmp9
        tmp29 = tmp28 * tmp14
        tmp30 = tmp27 - tmp29
        tmp31 = tmp16 * tmp30
        tl.store(out_ptr2 + (r2 + (512*x3)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kk/ckk4tavz4wcjqpi7qhmwcif45pcda24xqdm6it6zthqskmf6fcpv.py
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
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_select_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 100
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp3 = tl.load(in_ptr0 + (x0 + (512*(r2 // 50)) + (1024*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr1 + (x0 + (512*r2) + (51200*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp0 = r2 % 50
        tmp1 = tl.full([1, 1], 0, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp4 = 0.0
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp7 = tmp5 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3z/c3zctxdgljqws2l77fh7wmgxijb45cewukcnwfpgnll4uzxstqtp.py
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
    size_hints=[512, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_select_backward_3', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/35/c35f6q77c7fox74va2yspdqwiwzcvwnk2tfy5s7aarrsvqssonvu.py
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
    size_hints=[512, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_select_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 50
        r2 = (rindex // 50)
        tmp3 = tl.load(in_ptr0 + (x0 + (512*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/46/c46vo5bjvzcy4gpois5zeyr2cow3y2kwdodyefrzwa763a6hrnnt.py
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
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 100
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (51200*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sq/csqxvmsinmuqlkmr7paukwcqtodv6e33mi7xwmtapjl5hazeiluc.py
# Source Nodes: [x_154], Original ATen: [aten.gelu, aten.gelu_backward]
# x_154 => add_78, erf_7, mul_78
triton_poi_fused_gelu_gelu_backward_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 819200
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


# kernel path: /tmp/torchinductor_youkaichao/5r/c5rxotbgrbsat7adjyoasfivylqydkelpmtiyxzf4db6kmj4a2uc.py
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
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 100
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (2048*r2) + (204800*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7q/c7qogtm527csp2gwqgzz35722rxgyyvumvkhxugdvfih2t6zgtk7.py
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
    size_hints=[2048, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2048*r1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qp/cqpjtnjk7mj67uycueudzb6xtm7wxvthampdzoa7xy6rsvxehg32.py
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
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_9', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 400
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 512.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v2/cv2uvdr37qzgf5jqjdm42bkba4jjlqkquogzerazbxcz2fskamzu.py
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
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 100
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (51200*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (512*r2) + (51200*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
        tmp6 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lq/clqdj5sizdlluifh2jeazd2ufyblzszbyn42wq4dfyciaxhqgj6x.py
# Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul]

triton_poi_fused_constant_pad_nd_mul_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_mul_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 512) % 49
    x3 = (xindex // 25088)
    x5 = xindex % 25088
    x4 = xindex % 512
    x0 = xindex % 64
    x1 = (xindex // 64) % 8
    tmp6 = tl.load(in_ptr1 + (x4 + (1536*x2) + (76800*x3)), None)
    tmp0 = 1 + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (512 + x5 + (25600*x3)), tmp2, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp7 = tmp5 * tmp6
    tl.store(out_ptr0 + (x0 + (64*x2) + (3136*x1) + (25088*x3)), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4d/c4dkvfswjraglkfeir5l7uld5rq5kmy3x4t52flqg5nimgvu7f5n.py
# Source Nodes: [], Original ATen: [aten.slice]

triton_poi_fused_slice_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + ((64*x2) + (3136*(y0 // 64)) + (25088*y1) + (y0 % 64)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (6272*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hv/chvejkldt5dkjndcjal4burqmtollxhxrekjb3jsjc7pjjueufvh.py
# Source Nodes: [], Original ATen: [aten.slice]

triton_poi_fused_slice_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + (6272 + (64*x2) + (3136*(y0 // 64)) + (25088*y1) + (y0 % 64)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (192*x2) + (9408*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/47/c47vl5jz27lm4ghabn4epmeoxyjz3mdwcuk62uz37sbvpd7defrz.py
# Source Nodes: [], Original ATen: [aten.slice]

triton_poi_fused_slice_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + (15680 + (64*x2) + (3136*(y0 // 64)) + (25088*y1) + (y0 % 64)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (192*x2) + (9408*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o4/co4enaanrp4ycaopzoyliz4xwbpivtsfbdd6douviimvn5r43iog.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 192
    x1 = (xindex // 192)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r2) + (18816*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pm/cpm7xcbkdmp7mmmj73e4hbarxhamhhyc34wfpss4jxa7pxwzxk3p.py
# Source Nodes: [], Original ATen: [aten.clone, aten.mul]

triton_poi_fused_clone_mul_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 204800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 50
    x2 = (xindex // 3200) % 8
    x3 = (xindex // 25600)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (512*x1) + (25600*x3)), None)
    tmp1 = 0.125
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mc/cmciazgkqe3h6mwkt3vlukazp2ixmuepi3n2pduo2vuzc22c2tda.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_per_fused__softmax_backward_data_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 50
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64) % 8
    x2 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (r3 + (50*x4)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x1 + (8*x0) + (512*r3) + (25600*x2)), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/i5/ci5gn64kw44g7ovngjgzn3b6bg6fgzid6o3m2nf3ps6il5lohvq3.py
# Source Nodes: [], Original ATen: [aten.stack]

triton_poi_fused_stack_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9600
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y2 = (yindex // 400)
    y0 = yindex % 50
    x3 = xindex
    y1 = (yindex // 50) % 8
    y4 = yindex
    y5 = (yindex // 50)
    tmp0 = y2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.broadcast_to(y0, [XBLOCK, YBLOCK])
    tmp6 = tl.full([1, 1], 1, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tmp7 & tmp4
    tmp9 = tmp5 >= tmp1
    tmp10 = tmp9 & tmp8
    tmp11 = tl.load(in_ptr0 + (x3 + (64*y1) + (512*y0) + (25600*y2)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = tl.load(in_ptr1 + (x3 + (64*y1) + (512*(((-1) + y0) % 49)) + (25088*y2)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 * tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = 0.0
    tmp19 = tl.where(tmp7, tmp17, tmp18)
    tmp20 = tl.load(in_ptr2 + (x3 + (64*y4)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp4, tmp21, tmp22)
    tmp24 = tmp0 >= tmp3
    tmp25 = tl.full([1, 1], 16, tl.int64)
    tmp26 = tmp0 < tmp25
    tmp27 = tmp24 & tmp26
    tmp28 = tl.load(in_ptr3 + ((-204800) + y0 + (50*x3) + (3200*y5)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr4 + ((-204800) + y1 + (8*x3) + (512*y0) + (25600*y2)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 * tmp29
    tmp31 = tl.load(in_ptr5 + ((-4096) + x3 + (64*y5)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp29 * tmp31
    tmp33 = tmp30 - tmp32
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp27, tmp33, tmp34)
    tmp36 = tmp0 >= tmp25
    tmp37 = tl.full([1, 1], 24, tl.int64)
    tmp38 = tmp0 < tmp37
    tmp39 = tmp7 & tmp36
    tmp40 = x3 + (64*y1)
    tmp41 = tmp40 >= tmp1
    tmp42 = tl.full([1, 1], 128, tl.int64)
    tmp43 = tmp40 < tmp42
    tmp44 = tmp43 & tmp39
    tmp45 = tl.load(in_ptr6 + ((-100352) + (49*x3) + (3136*y1) + (6272*y2) + (((-1) + y0) % 49)), tmp44 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp44, tmp45, tmp46)
    tmp48 = tmp40 >= tmp42
    tmp49 = tl.full([1, 1], 320, tl.int64)
    tmp50 = tmp40 < tmp49
    tmp51 = tmp48 & tmp50
    tmp52 = tmp51 & tmp39
    tmp53 = tl.load(in_ptr7 + ((-156800) + (49*x3) + (3136*y1) + (9408*y2) + (((-1) + y0) % 49)), tmp52 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp52, tmp53, tmp54)
    tmp56 = tmp40 >= tmp49
    tmp57 = tl.full([1, 1], 512, tl.int64)
    tmp58 = tmp40 < tmp57
    tmp59 = tmp56 & tmp39
    tmp60 = tl.load(in_ptr8 + ((-166208) + (49*x3) + (3136*y1) + (9408*y2) + (((-1) + y0) % 49)), tmp59 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp61 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
    tmp62 = tl.where(tmp59, tmp60, tmp61)
    tmp63 = tl.where(tmp51, tmp55, tmp62)
    tmp64 = tl.where(tmp43, tmp47, tmp63)
    tmp65 = tl.full(tmp64.shape, 0.0, tmp64.dtype)
    tmp66 = tl.where(tmp39, tmp64, tmp65)
    tmp67 = tl.where(tmp7, tmp66, tmp18)
    tmp68 = tl.load(in_ptr9 + ((-409600) + x3 + (64*y4)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp69 = tmp67 + tmp68
    tmp70 = tl.full(tmp69.shape, 0.0, tmp69.dtype)
    tmp71 = tl.where(tmp36, tmp69, tmp70)
    tmp72 = tl.where(tmp27, tmp35, tmp71)
    tmp73 = tl.where(tmp4, tmp23, tmp72)
    tl.store(out_ptr0 + (x3 + (64*y4)), tmp73, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4i/c4iolxvyrkplktzqommlvy6nniwr5wwvdjplyzirchr3m5ogivjv.py
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
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 614400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1536
    x1 = (xindex // 1536)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*(x1 % 50)) + (3200*((x0 // 64) % 8)) + (25600*(x1 // 50)) + (204800*(x0 // 512)) + (x0 % 64)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6c/c6chdsljrpfgwdy7u5jt3ogjtzuziyzkm4falvcno6lktudq4qzk.py
# Source Nodes: [cur_28], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# cur_28 => mul_71, sub_25
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_20', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel):
    xnumel = 400
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
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
    tmp18 = 512.0
    tmp19 = tmp10 / tmp18
    tmp20 = tmp2 * tmp18
    tmp21 = tmp20 - tmp6
    tmp22 = tmp11 * tmp16
    tmp23 = tmp21 - tmp22
    tmp24 = tmp19 * tmp23
    tmp25 = tmp17 + tmp24
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5u/c5ujzjtmuiwnaumxjodbqrb7emzpwyyg66znumv7yuqenxhs3dow.py
# Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward]

triton_poi_fused_add_select_backward_slice_backward_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_select_backward_slice_backward_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 204800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 512) % 50
    x0 = xindex % 512
    x2 = (xindex // 25600)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (512 + x0 + (512*(((-1) + x1) % 49)) + (25600*x2)), tmp2, other=0.0)
    tmp4 = tl.load(in_ptr1 + ((49*x0) + (25088*x2) + (((-1) + x1) % 49)), tmp2, eviction_policy='evict_last', other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tmp0 < tmp1
    tmp11 = tl.load(in_ptr0 + (x0 + (25600*x2)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = tl.where(tmp10, tmp13, tmp8)
    tmp15 = tmp9 + tmp14
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ai/cai7elwfzeae6cnlq5bz74aqazkav7lokjw5jh7eaone6u4hq7cl.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 400
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 512.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ao/caoiyfusrpnnkzpoqgazp66hvw3bdjtelietrapyn24beomek42p.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward]

triton_per_fused_add_convolution_backward_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_backward_23', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (192*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0 + (192*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tmp4 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c2/cc24xjca66o4etobi37wysfjy5dg4ek2oipm3dkczdoczg5puhva.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 192
    x1 = (xindex // 192)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r2) + (18816*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ri/crijjgtwdvh4qa6oiof3n3r6w7w4zc2xg3ihcttwsrc3pg5s4mat.py
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
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (12544*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/44/c44r6hum7l7ab3qffpaxemmrwcr2yhki7fv4guavlkye66ekg7wl.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 98
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
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (12544*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sn/csngmmh2mwix3blk6kdpjru6sop2cqsrrdezyvtvdh2ei5ul7r45.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward]

triton_per_fused_add_convolution_backward_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_backward_27', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tmp4 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/de/cded7zh6w2scfhlvkdva3yumtkawqpjq4zyat2uhdlfz2jhmzpkj.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 100
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1536
    x1 = (xindex // 1536)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1536*r2) + (153600*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bo/cbod56zqzy4xpgxiykabocsjms3d6o6j4zuu2uu5s7oz2fc4q52h.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_29', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/un/cuntyb24fwalafjfzf2kykrp62v5danc3p2n2mmcoslpet72wrbs.py
# Source Nodes: [cur_28], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# cur_28 => mul_71, sub_25
triton_red_fused_native_layer_norm_native_layer_norm_backward_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 100
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (51200*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (512*r2) + (51200*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r2 + (100*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr3 + (r2 + (100*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp5 = tmp3 * tmp4
        tmp6 = tmp0 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
        tmp10 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask, tmp12, _tmp11)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, None)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4c/c4ccfs5bt26dq7rbsbytaayaaebatnvjnd6ft7vxgzo27jol75g6.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (512 + x0 + (512*(r2 % 49)) + (25600*(r2 // 49)) + (51200*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ys/cysrdouxryde566lpdeqmj7eukoxhzaieh2p4tk4vxk4hyja2ylp.py
# Source Nodes: [cur_24], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# cur_24 => mul_62, sub_22
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 400
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (r1 + (512*x0)), rmask & xmask, other=0.0)
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
    tmp18 = 512.0
    tmp19 = tmp10 / tmp18
    tmp20 = tmp2 * tmp18
    tmp21 = tmp20 - tmp6
    tmp22 = tmp11 * tmp16
    tmp23 = tmp21 - tmp22
    tmp24 = tmp19 * tmp23
    tmp25 = tmp17 + tmp24
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rl/crlfw6xhdl64au45e26tjafn5vxvqwwojm5su5xruqszgseazsrd.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_layer_norm_backward]

triton_red_fused_convolution_backward_native_layer_norm_backward_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_native_layer_norm_backward_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (512 + x0 + (512*(r2 % 49)) + (25600*(r2 // 49)) + (51200*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr2 + (x0 + (512*r2) + (50176*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
        tmp4 = 1 + (r2 % 49)
        tmp5 = tl.full([1, 1], 1, tl.int64)
        tmp6 = tmp4 >= tmp5
        tmp7 = tl.load(in_ptr0 + (512 + x0 + (512*(r2 % 49)) + (25600*(r2 // 49)) + (51200*x1)), rmask & tmp6, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr1 + ((49*x0) + (25088*(r2 // 49)) + (50176*x1) + (r2 % 49)), rmask & tmp6, eviction_policy='evict_first', other=0.0)
        tmp9 = tmp7 + tmp8
        tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
        tmp11 = tl.where(tmp6, tmp9, tmp10)
        tmp12 = 0.0
        tmp13 = tl.where(tmp6, tmp11, tmp12)
        tmp14 = tmp4 < tmp5
        tmp15 = tl.load(in_ptr0 + (x0 + (25600*(r2 // 49)) + (51200*x1)), rmask & tmp14, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
        tmp17 = tl.where(tmp14, tmp15, tmp16)
        tmp18 = tl.where(tmp14, tmp17, tmp12)
        tmp19 = tmp13 + tmp18
        tmp21 = tmp19 * tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask, tmp24, _tmp23)
        tmp25 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask, tmp27, _tmp26)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp23, None)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp26, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qt/cqtkfookimw4uyighsdromlwgc5ivkhai64sq53yxgwj553n3bwu.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward]

triton_per_fused_add_convolution_backward_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_backward_34', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp5 = tl.load(in_ptr1 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tmp4 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4c/c4csomt4a4hzptgr66jjlxlsvf6y6fgndjldsl42ik2wloqe2dru.py
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
    size_hints=[2048, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 100
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (51200*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xx/cxxm3jvg3sbctyrr7gbwyfqa2w2ntyi456xf222pccsnwylk6ksn.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_36', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xk/cxkk5kwhgznffrqrtpyncmgtjrnb5vc7wusrw56pckzwyfv4er4u.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_37', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rp/crppnpapemw3ilnmfeipvso2f2x6d553mxvognydmun4c2bolxex.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_38', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wc/cwcz4jxwdz4ted6cxd2hfveqip5cb6lowqrm2buf3a7po5bfqbcs.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_39', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2e/c2exf56sappetny3zjqumlzn35m3yhaikaysk6ltw2bn5okkll6t.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.full([1, 1], 0, tl.int64)
    tmp1 = tl.full([1, 1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (25088 + x0 + (25600*r1)), rmask & tmp2 & xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (48 + (49*x0) + (25088*r1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tmp0 < tmp1
    tmp11 = tl.load(in_ptr0 + (x0 + (25600*r1)), rmask & tmp10 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = tl.where(tmp10, tmp13, tmp8)
    tmp15 = tmp9 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hf/chffxiibe4vbfnckwvpmvoboruigyqnmvew2dans3o4ewowg6f5i.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 4) % 49
    x2 = (xindex // 196)
    x4 = xindex % 196
    x0 = xindex % 4
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x6 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp16 = tl.load(in_ptr2 + (r3 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = 1 + x1
        tmp1 = tl.full([1, 1], 1, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.load(in_ptr0 + (512 + r3 + (128*x4) + (25600*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x1 + (49*r3) + (6272*x0) + (25088*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = 0.0
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tmp0 < tmp1
        tmp11 = tl.load(in_ptr0 + (r3 + (128*x0) + (25600*x2)), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
        tmp13 = tl.where(tmp10, tmp11, tmp12)
        tmp14 = tl.where(tmp10, tmp13, tmp8)
        tmp15 = tmp9 + tmp14
        tmp17 = tmp15 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr0 + (x6), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4p/c4ppgqcmjo7jitqniosawxic27wlp7ot3sx7z4utoioxvh7itsht.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 392
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (4*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/os/cosxod4ppmedhdqnwelbbdnonwmteximkxhioyqd3xcxrmm3kjcg.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel):
    xnumel = 392
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 49
    r2 = rindex
    x1 = (xindex // 49)
    x3 = xindex
    tmp16 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr3 + (r2 + (512*x3)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = 1 + x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (512 + r2 + (512*x0) + (25600*x1)), rmask & tmp2 & xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (x0 + (49*r2) + (25088*x1)), rmask & tmp2 & xmask, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tmp0 < tmp1
    tmp11 = tl.load(in_ptr0 + (r2 + (25600*x1)), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = tl.where(tmp10, tmp13, tmp8)
    tmp15 = tmp9 + tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp25 = 512.0
    tmp26 = tmp17 * tmp25
    tmp28 = tmp26 - tmp27
    tmp29 = tmp18 * tmp23
    tmp30 = tmp28 - tmp29
    tmp31 = tmp24 * tmp30
    tl.store(out_ptr1 + (r2 + (512*x3)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ck/cckbbwzz5fsxzfr2rzsg4hqjpvzhhcpfvzexgsmc4wkauez6icho.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (50176*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ws/cws2jvnavwajwsmdlqjiurtglibdgc5m2ofoaf2zrd2xksftlvxc.py
# Source Nodes: [], Original ATen: [aten.slice_backward, aten.view]

triton_poi_fused_slice_backward_view_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_backward_view_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 504320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 320)
    x0 = xindex % 320
    x2 = xindex
    tmp0 = x1 % 197
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((196*x0) + (62720*(x1 // 197)) + (((-1) + (x1 % 197)) % 196)), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tl.store(out_ptr0 + (x2), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5y/c5yqtihljmqgxonboku6bv6oehtpfc57onxs63ummqsfj3bfuwbx.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4160
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 320)
    x0 = xindex % 320
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 1576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (320*r2) + (39040*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o4/co4cz4il4sg5evsu5x23lvaxnas4qsnurn7f3ytqaimdwpy3u35r.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 320
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (320*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hu/chugo5szjwhbucqg7wqgzb2jsp3bpi3eqji2mhpeocok3ovj3bv6.py
# Source Nodes: [x_114], Original ATen: [aten.gelu, aten.gelu_backward]
# x_114 => add_58, erf_5, mul_58
triton_poi_fused_gelu_gelu_backward_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_48', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2017280
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


# kernel path: /tmp/torchinductor_youkaichao/vl/cvl5hmbndohgevfodzze4ofzaut4b34hl5mpefluyrnkmx66xp77.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16640
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 1280)
    x0 = xindex % 1280
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 1576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (1280*r2) + (156160*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hm/chmir4hd5ryvfx5c27zhupvctlwyz3kxpzxqjrb5sg7zlk7vgsui.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    rnumel = 13
    RBLOCK: tl.constexpr = 16
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


# kernel path: /tmp/torchinductor_youkaichao/se/cse5epbhbbn75n7pzikxlqq5pg7dy4cmls7vrrfnfah76cbioslm.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.slice_backward]

triton_per_fused_add_native_layer_norm_backward_slice_backward_51 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_slice_backward_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 197
    x3 = (xindex // 197)
    tmp0 = tl.load(in_ptr0 + (r1 + (320*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (320*x0)), rmask & xmask, other=0.0)
    tmp21 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = tl.load(in_ptr3 + ((196*r1) + (62720*x3) + (((-1) + x2) % 196)), rmask & tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp15, tmp16, tmp17)
    tmp19 = 0.0
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp22 = 320.0
    tmp23 = tmp2 * tmp22
    tmp24 = tmp23 - tmp6
    tmp25 = tmp7 * tmp12
    tmp26 = tmp24 - tmp25
    tmp27 = tmp21 * tmp26
    tmp28 = tmp20 + tmp27
    tl.store(out_ptr2 + (r1 + (320*x0)), tmp28, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qv/cqvirzoowxmimpkizyanfnbukzaimtfma3c2bze2ckaa24xgds5a.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_52', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4160
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 320)
    x0 = xindex % 320
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
        tmp3 = tl.load(in_ptr0 + (x0 + (320*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (320*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/sq/csqqw2fpwshq2vgyeqcwxllizj4hevezu4ammadq3ndsrmmwhj65.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4160
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 320)
    x0 = xindex % 320
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 1576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (320*r2) + (39040*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rl/crlbz257w7lgocfp4syfn6waw2ox2nqp74mdrnyrtflr6ltrivxw.py
# Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul]

triton_poi_fused_constant_pad_nd_mul_54 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_mul_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 320) % 196
    x3 = (xindex // 62720)
    x5 = xindex % 62720
    x4 = xindex % 320
    x0 = xindex % 40
    x1 = (xindex // 40) % 8
    tmp6 = tl.load(in_ptr1 + (x4 + (960*x2) + (189120*x3)), None)
    tmp0 = 1 + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (320 + x5 + (63040*x3)), tmp2, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp7 = tmp5 * tmp6
    tl.store(out_ptr0 + (x0 + (40*x2) + (7840*x1) + (62720*x3)), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fk/cfk6agf52u5olf6idslp3iy35hrl7uxivf27q24cxnxc2zx5bk6g.py
# Source Nodes: [], Original ATen: [aten.slice]

triton_poi_fused_slice_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 640
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 80
    y1 = (yindex // 80)
    tmp0 = tl.load(in_ptr0 + ((40*x2) + (7840*(y0 // 40)) + (62720*y1) + (y0 % 40)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (80*x2) + (15680*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ul/culoyqiuu6wn7fzsesyf3rhyfnslqbvjp6rias36a7msfzf4xkca.py
# Source Nodes: [], Original ATen: [aten.slice]

triton_poi_fused_slice_56 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_56', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 120
    y1 = (yindex // 120)
    tmp0 = tl.load(in_ptr0 + (15680 + (40*x2) + (7840*(y0 // 40)) + (62720*y1) + (y0 % 40)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (120*x2) + (23520*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ah/cahmfheahw6cknals53uukkyp7q7l5is24sqsgob2m7l63ncpep7.py
# Source Nodes: [], Original ATen: [aten.slice]

triton_poi_fused_slice_57 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_57', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 120
    y1 = (yindex // 120)
    tmp0 = tl.load(in_ptr0 + (39200 + (40*x2) + (7840*(y0 // 40)) + (62720*y1) + (y0 % 40)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (120*x2) + (23520*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mb/cmbsgekkftazcgaxuw6w4iaty446zqdvsj7wsyhdxjsb62yjml3s.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_58 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_58', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1560
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 120)
    x0 = xindex % 120
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (120*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lv/clvdeedol3igblhq6efzgucngwlboyqtrqxzzyiyk54rilqtnhub.py
# Source Nodes: [], Original ATen: [aten.clone, aten.mul]

triton_poi_fused_clone_mul_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_59', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 504320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 40
    x1 = (xindex // 40) % 197
    x2 = (xindex // 7880) % 8
    x3 = (xindex // 63040)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (40*x2) + (320*x1) + (63040*x3)), xmask)
    tmp1 = 0.15811388300841897
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x4), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hy/chyg3u7us26t7r7dyvxm2bv2vyfdqpuybpzjpi4oqexuu554e6yr.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_red_fused__softmax_backward_data_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_backward_data_60', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5120
    rnumel = 99
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x5 = (xindex // 2)
    x1 = (xindex // 2) % 40
    x2 = (xindex // 80) % 8
    x3 = (xindex // 640)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x6 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r4 = rindex
        tmp0 = r4 + (99*x0)
        tmp1 = tl.full([1, 1], 197, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (r4 + (99*x0) + (197*x5)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x2 + (8*x1) + (320*r4) + (31680*x0) + (63040*x3)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x6), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u2/cu2xaamcye54wq3axxargs745h7mckpqzlrvjhltrf7j3e3d4457.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_per_fused__softmax_backward_data_61 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_61', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2560
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wi/cwizqi27drkp3ufz65lcwdh6iupkkmqqaqgp5qnm5o6py6iclqwo.py
# Source Nodes: [], Original ATen: [aten.stack]

triton_poi_fused_stack_62 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_62', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 37824
    xnumel = 40
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y2 = (yindex // 1576)
    y0 = yindex % 197
    x3 = xindex
    y1 = (yindex // 197) % 8
    y4 = yindex
    y5 = (yindex // 197)
    tmp0 = y2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.broadcast_to(y0, [XBLOCK, YBLOCK])
    tmp6 = tl.full([1, 1], 1, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tmp7 & tmp4
    tmp9 = tmp5 >= tmp1
    tmp10 = tmp9 & tmp8
    tmp11 = tl.load(in_ptr0 + (x3 + (40*y1) + (320*y0) + (63040*y2)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = tl.load(in_ptr1 + (x3 + (40*y1) + (320*(((-1) + y0) % 196)) + (62720*y2)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 * tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = 0.0
    tmp19 = tl.where(tmp7, tmp17, tmp18)
    tmp20 = tl.load(in_ptr2 + (x3 + (40*y4)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp4, tmp21, tmp22)
    tmp24 = tmp0 >= tmp3
    tmp25 = tl.full([1, 1], 16, tl.int64)
    tmp26 = tmp0 < tmp25
    tmp27 = tmp24 & tmp26
    tmp28 = tl.load(in_ptr3 + ((-504320) + y0 + (197*x3) + (7880*y5)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr4 + ((-504320) + y1 + (8*x3) + (320*y0) + (63040*y2)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 * tmp29
    tmp31 = tl.load(in_ptr5 + ((-2560) + x3 + (40*y5)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp29 * tmp31
    tmp33 = tmp30 - tmp32
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp27, tmp33, tmp34)
    tmp36 = tmp0 >= tmp25
    tmp37 = tl.full([1, 1], 24, tl.int64)
    tmp38 = tmp0 < tmp37
    tmp39 = tmp7 & tmp36
    tmp40 = x3 + (40*y1)
    tmp41 = tmp40 >= tmp1
    tmp42 = tl.full([1, 1], 80, tl.int64)
    tmp43 = tmp40 < tmp42
    tmp44 = tmp43 & tmp39
    tmp45 = tl.load(in_ptr6 + ((-250880) + (196*x3) + (7840*y1) + (15680*y2) + (((-1) + y0) % 196)), tmp44 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp44, tmp45, tmp46)
    tmp48 = tmp40 >= tmp42
    tmp49 = tl.full([1, 1], 200, tl.int64)
    tmp50 = tmp40 < tmp49
    tmp51 = tmp48 & tmp50
    tmp52 = tmp51 & tmp39
    tmp53 = tl.load(in_ptr7 + ((-392000) + (196*x3) + (7840*y1) + (23520*y2) + (((-1) + y0) % 196)), tmp52 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp52, tmp53, tmp54)
    tmp56 = tmp40 >= tmp49
    tmp57 = tl.full([1, 1], 320, tl.int64)
    tmp58 = tmp40 < tmp57
    tmp59 = tmp56 & tmp39
    tmp60 = tl.load(in_ptr8 + ((-415520) + (196*x3) + (7840*y1) + (23520*y2) + (((-1) + y0) % 196)), tmp59 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp61 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
    tmp62 = tl.where(tmp59, tmp60, tmp61)
    tmp63 = tl.where(tmp51, tmp55, tmp62)
    tmp64 = tl.where(tmp43, tmp47, tmp63)
    tmp65 = tl.full(tmp64.shape, 0.0, tmp64.dtype)
    tmp66 = tl.where(tmp39, tmp64, tmp65)
    tmp67 = tl.where(tmp7, tmp66, tmp18)
    tmp68 = tl.load(in_ptr9 + ((-1008640) + x3 + (40*y4)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp69 = tmp67 + tmp68
    tmp70 = tl.full(tmp69.shape, 0.0, tmp69.dtype)
    tmp71 = tl.where(tmp36, tmp69, tmp70)
    tmp72 = tl.where(tmp27, tmp35, tmp71)
    tmp73 = tl.where(tmp4, tmp23, tmp72)
    tl.store(out_ptr0 + (x3 + (40*y4)), tmp73, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e7/ce7ti3umarrztjmv4s3m4m5k2vxowha5ylbzb4l7lhd2hzqynoj6.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_63', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1512960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 960
    x1 = (xindex // 960)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((40*(x1 % 197)) + (7880*((x0 // 40) % 8)) + (63040*(x1 // 197)) + (504320*(x0 // 320)) + (x0 % 40)), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r4/cr462zmnrst3n7gwobtlw6ji5ec37naoz52ojuh4m7mer34dg7fu.py
# Source Nodes: [cur_20], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# cur_20 => mul_51, sub_18
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_64 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_64', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (320*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (320*x0)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_out_ptr0 + (r1 + (320*x0)), rmask & xmask, other=0.0)
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
    tmp18 = 320.0
    tmp19 = tmp10 / tmp18
    tmp20 = tmp2 * tmp18
    tmp21 = tmp20 - tmp6
    tmp22 = tmp11 * tmp16
    tmp23 = tmp21 - tmp22
    tmp24 = tmp19 * tmp23
    tmp25 = tmp17 + tmp24
    tl.store(in_out_ptr0 + (r1 + (320*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6v/c6vgeqtrybjncfao24czxerg2iuwm5nx52tw6665lkz6y4oijiwa.py
# Source Nodes: [], Original ATen: [aten.add, aten.slice_backward]

triton_poi_fused_add_slice_backward_65 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_slice_backward_65', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 504320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 320) % 197
    x0 = xindex % 320
    x2 = (xindex // 63040)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (320 + x0 + (320*(((-1) + x1) % 196)) + (63040*x2)), tmp2 & xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + ((196*x0) + (62720*x2) + (((-1) + x1) % 196)), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tmp0 < tmp1
    tmp11 = tl.load(in_ptr0 + (x0 + (63040*x2)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = tl.where(tmp10, tmp13, tmp8)
    tmp15 = tmp9 + tmp14
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zs/czs5qgf74ytef4xd6gert7ipjrgypzj62cgprodufxlp3lzhdg4c.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_66 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_66', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (320*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (320*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (r1 + (320*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 320.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(out_ptr2 + (r1 + (320*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/un/cun2jm7frglp7tjyqguzntjr2vkqrbjaeiiezmj3wimattks7hzg.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward]

triton_per_fused_add_convolution_backward_67 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_backward_67', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 120
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (120*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0 + (120*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tmp4 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xb/cxb7pitj4yk3epiw3emdwq23akxlvvcx56z7tyg3lmmz2xwxpgne.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_68 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_68', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1560
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 120)
    x0 = xindex % 120
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (120*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jj/cjjda25h6pvhgufnqxsu2kso7wsnnvvwlpvod5g5w6mxeb5znhwx.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_69 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_69', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1040
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 80)
    x0 = xindex % 80
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (80*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gu/cguvetb7flnwe3gbp4pqpj73pawryhygyiw26ptmmepgjmhnyz3q.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_70 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_70', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1040
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 80)
    x0 = xindex % 80
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (80*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fn/cfnrc3oohsvr66jq6btip3lbz6k26stsybn4ylombmoj4zzzbv4d.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward]

triton_per_fused_add_convolution_backward_71 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_backward_71', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 80
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (80*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0 + (80*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tmp4 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p5/cp5rjdpo4llmomofgfuekzcfdazkkzpylhpeiace3vi25tnartce.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_72 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_72', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12480
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 960)
    x0 = xindex % 960
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 1576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (960*r2) + (117120*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bj/cbjorhja5jpglfjmiua5yglj4mcmiaeyfi7srkeernktjmidlmv4.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_73 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_73', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (960*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bu/cbul3w7tmijwm76ukeodw4lyzjglthyxis4cp57x2ri4wtohuauo.py
# Source Nodes: [cur_20], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# cur_20 => mul_51, sub_18
triton_red_fused_native_layer_norm_native_layer_norm_backward_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_74', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4160
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 320)
    x0 = xindex % 320
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
        tmp3 = tl.load(in_ptr0 + (x0 + (320*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (320*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/z7/cz73gwxdmhrjqrwceln4wvx5xarwupzck2ch3dxwfoefs6ary724.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_75 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_75', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4160
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 320)
    x0 = xindex % 320
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (320 + x0 + (320*((r2 + (121*x1)) % 196)) + (63040*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ig/cigzqln54hv36ypo6aapa7dpbglpswhdceudmzklmhqtbqgrqxpr.py
# Source Nodes: [], Original ATen: [aten.stack]

triton_poi_fused_stack_76 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 8192], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_76', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 7880
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y1 = (yindex // 8)
    x3 = (xindex // 40)
    x2 = xindex % 40
    y0 = yindex % 8
    x5 = xindex
    y4 = yindex
    tmp0 = y1
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.broadcast_to(x3, [XBLOCK, YBLOCK])
    tmp6 = tl.full([1, 1], 1, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tmp7 & tmp4
    tmp9 = tmp5 >= tmp1
    tmp10 = tmp9 & tmp8
    tmp11 = tl.load(in_ptr0 + (x2 + (40*y0) + (320*x3) + (63040*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = tl.load(in_ptr1 + (x2 + (40*y0) + (320*(((-1) + x3) % 196)) + (62720*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 * tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = 0.0
    tmp19 = tl.where(tmp7, tmp17, tmp18)
    tmp20 = tl.load(in_ptr2 + (x5 + (7880*y4)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp4, tmp21, tmp22)
    tmp24 = tmp0 >= tmp3
    tmp25 = tl.full([1, 1], 16, tl.int64)
    tmp26 = tmp0 < tmp25
    tmp27 = tmp24 & tmp26
    tmp28 = tl.load(in_ptr3 + ((-504320) + x3 + (197*x2) + (7880*y4)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr4 + ((-504320) + y0 + (8*x5) + (63040*y1)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 * tmp29
    tmp31 = tl.load(in_ptr5 + ((-2560) + x2 + (40*y4)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp29 * tmp31
    tmp33 = tmp30 - tmp32
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp27, tmp33, tmp34)
    tmp36 = tmp0 >= tmp25
    tmp37 = tl.full([1, 1], 24, tl.int64)
    tmp38 = tmp0 < tmp37
    tmp39 = tmp7 & tmp36
    tmp40 = x2 + (40*y0)
    tmp41 = tmp40 >= tmp1
    tmp42 = tl.full([1, 1], 80, tl.int64)
    tmp43 = tmp40 < tmp42
    tmp44 = tmp43 & tmp39
    tmp45 = tl.load(in_ptr6 + ((-250880) + (196*x2) + (7840*y0) + (15680*y1) + (((-1) + x3) % 196)), tmp44 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp44, tmp45, tmp46)
    tmp48 = tmp40 >= tmp42
    tmp49 = tl.full([1, 1], 200, tl.int64)
    tmp50 = tmp40 < tmp49
    tmp51 = tmp48 & tmp50
    tmp52 = tmp51 & tmp39
    tmp53 = tl.load(in_ptr7 + ((-392000) + (196*x2) + (7840*y0) + (23520*y1) + (((-1) + x3) % 196)), tmp52 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp52, tmp53, tmp54)
    tmp56 = tmp40 >= tmp49
    tmp57 = tl.full([1, 1], 320, tl.int64)
    tmp58 = tmp40 < tmp57
    tmp59 = tmp56 & tmp39
    tmp60 = tl.load(in_ptr8 + ((-415520) + (196*x2) + (7840*y0) + (23520*y1) + (((-1) + x3) % 196)), tmp59 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp61 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
    tmp62 = tl.where(tmp59, tmp60, tmp61)
    tmp63 = tl.where(tmp51, tmp55, tmp62)
    tmp64 = tl.where(tmp43, tmp47, tmp63)
    tmp65 = tl.full(tmp64.shape, 0.0, tmp64.dtype)
    tmp66 = tl.where(tmp39, tmp64, tmp65)
    tmp67 = tl.where(tmp7, tmp66, tmp18)
    tmp68 = tl.load(in_ptr9 + ((-1008640) + x5 + (7880*y4)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp69 = tmp67 + tmp68
    tmp70 = tl.full(tmp69.shape, 0.0, tmp69.dtype)
    tmp71 = tl.where(tmp36, tmp69, tmp70)
    tmp72 = tl.where(tmp27, tmp35, tmp71)
    tmp73 = tl.where(tmp4, tmp23, tmp72)
    tl.store(out_ptr0 + (x5 + (7880*y4)), tmp73, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2d/c2djpz65yl3dfjsakn4dpvf6vpwdmxbqjgpxo3vteugjvh4xbr4n.py
# Source Nodes: [cur_16], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# cur_16 => mul_42, sub_15
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_77 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_77', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (320*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (320*x0)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (r1 + (320*x0)), rmask & xmask, other=0.0)
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
    tmp18 = 320.0
    tmp19 = tmp10 / tmp18
    tmp20 = tmp2 * tmp18
    tmp21 = tmp20 - tmp6
    tmp22 = tmp11 * tmp16
    tmp23 = tmp21 - tmp22
    tmp24 = tmp19 * tmp23
    tmp25 = tmp17 + tmp24
    tl.store(out_ptr2 + (r1 + (320*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bo/cbobl6vwatyhipbawzd2xhfdx7djmcv6gwe6yhcubhjyussoblxr.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_layer_norm_backward]

triton_red_fused_convolution_backward_native_layer_norm_backward_78 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_native_layer_norm_backward_78', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4160
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 320)
    x0 = xindex % 320
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp32 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (320 + x0 + (320*((r2 + (121*x1)) % 196)) + (63040*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
        tmp9 = 1 + ((r2 + (121*x1)) % 196)
        tmp10 = tl.full([1, 1], 1, tl.int64)
        tmp11 = tmp9 >= tmp10
        tmp12 = tmp11 & tmp2
        tmp13 = tl.load(in_ptr0 + (320 + x0 + (320*((r2 + (121*x1)) % 196)) + (63040*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp12 & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr1 + ((196*x0) + (62720*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp12 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 + tmp14
        tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
        tmp17 = tl.where(tmp12, tmp15, tmp16)
        tmp18 = 0.0
        tmp19 = tl.where(tmp11, tmp17, tmp18)
        tmp20 = tmp9 < tmp10
        tmp21 = tmp20 & tmp2
        tmp22 = tl.load(in_ptr0 + (x0 + (63040*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp21 & xmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
        tmp24 = tl.where(tmp21, tmp22, tmp23)
        tmp25 = tl.where(tmp20, tmp24, tmp18)
        tmp26 = tmp19 + tmp25
        tmp27 = tl.load(in_ptr2 + (x0 + (320*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tmp26 * tmp27
        tmp29 = tl.full(tmp28.shape, 0, tmp28.dtype)
        tmp30 = tl.where(tmp2, tmp28, tmp29)
        tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
        tmp33 = _tmp32 + tmp31
        _tmp32 = tl.where(rmask & xmask, tmp33, _tmp32)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
    tmp32 = tl.sum(_tmp32, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/36/c36yz7gkerimxvcc337rrgelspzczlbmdojvoqwrq5reucqnzwpq.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward]

triton_per_fused_add_convolution_backward_79 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_backward_79', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 320
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (320*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0 + (320*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tmp4 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ot/cotux62go3pnmayjzbhped76tdssqn34ilklwn5dqjtuw6x74dxr.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_80 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_80', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2i/c2ip6svh3hugjreghosbqmzk44mizyqlne3o4bpkmmj4x6elcjhm.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_81 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_81', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e6/ce6dwmjmttuinejzcxa22c2qcwu4yxrrz25usim7guejhtto7qdr.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_82 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_82', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wm/cwmmuhorwzl2h5k6uvauasfqn4ozm5kl6i2rhduigslnfcw46zen.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_83 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_83', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xj/cxjfdmytznjszafyjtclniqwsq2ugecgosljrixbmxhubyjkuulc.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_84 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_84', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 320
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.full([1, 1], 0, tl.int64)
    tmp1 = tl.full([1, 1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (62720 + x0 + (63040*r1)), rmask & tmp2 & xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (195 + (196*x0) + (62720*r1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tmp0 < tmp1
    tmp11 = tl.load(in_ptr0 + (x0 + (63040*r1)), rmask & tmp10 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = tl.where(tmp10, tmp13, tmp8)
    tmp15 = tmp9 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cz/cczxj66tcdaulp3wek4km7xjxnghk223f5zwqe2iizshlnyif4yy.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_85 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_85', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 107
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3
    x1 = (xindex // 3) % 196
    x2 = (xindex // 588)
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (107*x0)
        tmp1 = tl.full([1, 1], 320, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.broadcast_to(1 + x1, [XBLOCK, RBLOCK])
        tmp4 = tl.full([1, 1], 1, tl.int64)
        tmp5 = tmp3 >= tmp4
        tmp6 = tmp5 & tmp2
        tmp7 = tl.load(in_ptr0 + (320 + r3 + (107*x0) + (320*x1) + (63040*x2)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr1 + (x1 + (196*r3) + (20972*x0) + (62720*x2)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp7 + tmp8
        tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
        tmp11 = tl.where(tmp6, tmp9, tmp10)
        tmp12 = 0.0
        tmp13 = tl.where(tmp5, tmp11, tmp12)
        tmp14 = tmp3 < tmp4
        tmp15 = tmp14 & tmp2
        tmp16 = tl.load(in_ptr0 + (r3 + (107*x0) + (63040*x2)), rmask & tmp15 & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
        tmp18 = tl.where(tmp15, tmp16, tmp17)
        tmp19 = tl.where(tmp14, tmp18, tmp12)
        tmp20 = tmp13 + tmp19
        tmp21 = tl.load(in_ptr2 + (r3 + (107*x0)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tmp20 * tmp21
        tmp23 = tl.full(tmp22.shape, 0, tmp22.dtype)
        tmp24 = tl.where(tmp2, tmp22, tmp23)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/km/ckmnup6nng4b4tug6ab55i7dzteurkdx2m6prkdq362zbg3j7qi6.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_86 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_86', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/su/csujhyamu3rwld7a4mr2n5uqvjnckj6ubpueo3toebp3rrzyg2yi.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_87 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_87', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 320
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 196
    r2 = rindex
    x1 = (xindex // 196)
    x3 = xindex
    tmp16 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr3 + (r2 + (320*x3)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = 1 + x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (320 + r2 + (320*x0) + (63040*x1)), rmask & tmp2 & xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (x0 + (196*r2) + (62720*x1)), rmask & tmp2 & xmask, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tmp0 < tmp1
    tmp11 = tl.load(in_ptr0 + (r2 + (63040*x1)), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = tl.where(tmp10, tmp13, tmp8)
    tmp15 = tmp9 + tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp25 = 320.0
    tmp26 = tmp17 * tmp25
    tmp28 = tmp26 - tmp27
    tmp29 = tmp18 * tmp23
    tmp30 = tmp28 - tmp29
    tmp31 = tmp24 * tmp30
    tl.store(out_ptr1 + (r2 + (320*x3)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gq/cgqpzqlkggz547iuexyke6inigr3ft3uxw42bkp4emgryfytw6ls.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_88 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_88', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4160
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = 1 + ((r2 + (121*x0)) % 196)
        tmp4 = tl.full([1, 1], 1, tl.int64)
        tmp5 = tmp3 >= tmp4
        tmp6 = tmp5 & tmp2
        tmp7 = tl.load(in_ptr0 + (320 + x1 + (320*((r2 + (121*x0)) % 196)) + (63040*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr1 + ((196*x1) + (62720*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp7 + tmp8
        tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
        tmp11 = tl.where(tmp6, tmp9, tmp10)
        tmp12 = 0.0
        tmp13 = tl.where(tmp5, tmp11, tmp12)
        tmp14 = tmp3 < tmp4
        tmp15 = tmp14 & tmp2
        tmp16 = tl.load(in_ptr0 + (x1 + (63040*(((r2 + (121*x0)) // 196) % 8))), rmask & tmp15 & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
        tmp18 = tl.where(tmp15, tmp16, tmp17)
        tmp19 = tl.where(tmp14, tmp18, tmp12)
        tmp20 = tmp13 + tmp19
        tmp21 = tl.full(tmp20.shape, 0, tmp20.dtype)
        tmp22 = tl.where(tmp2, tmp20, tmp21)
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7l/c7lc7tuzzicv6f6nvjvmlakku4w6xgbchhiycjprxdnly2z2b7hw.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_89 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_89', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 320
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


# kernel path: /tmp/torchinductor_youkaichao/sx/csx4b6snt2osmr3xma3v2gbujdouqhcllwobs3jtmn7gg3viiktb.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_90 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_90', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4160
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 320)
    x0 = xindex % 320
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (320*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ro/crogxubuhpqhdsosxaf7y4el3mljvw6bjpl6amfgwcpc7bnxwkfs.py
# Source Nodes: [], Original ATen: [aten.slice_backward, aten.view]

triton_poi_fused_slice_backward_view_91 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_backward_view_91', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 803840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 128)
    x0 = xindex % 128
    x2 = xindex
    tmp0 = x1 % 785
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((784*x0) + (100352*(x1 // 785)) + (((-1) + (x1 % 785)) % 784)), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tl.store(out_ptr0 + (x2), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ba/cbalxavirwvfoy2m2wwuq6ounpwbc5ndc3ivzxd24z52mkshyqxj.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_92 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_92', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6400
    rnumel = 126
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
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 6280, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (128*r2) + (16128*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rs/crszbfv7kqqzcr7rx6jfc2uprdt7mngl7qxrefy5hit3tjtruybz.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_93 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_93', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 50
    RBLOCK: tl.constexpr = 64
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


# kernel path: /tmp/torchinductor_youkaichao/mv/cmvtuqiu5kvnmvgpthy7xe64zzdgo7kmz46hfqdecjphb32qc273.py
# Source Nodes: [x_74], Original ATen: [aten.gelu, aten.gelu_backward]
# x_74 => add_38, erf_3, mul_38
triton_poi_fused_gelu_gelu_backward_94 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_94', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6430720
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


# kernel path: /tmp/torchinductor_youkaichao/is/cismu4pzep4ecgg4aoq54tecqrlm5gr743j2dmyhfom7byjke2t5.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_95 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_95', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 51200
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 1024)
    x0 = xindex % 1024
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 6280, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (1024*r2) + (129024*x1)), rmask & tmp2, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/j5/cj53kocevlokss6j6mjue3kliimmurowhyq5vm7tkny2xzzvexbq.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_96 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_96', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 50
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ky/cky6pafoiaeqzh27lh3mrjcbrdtqxkhltkliid5qh45tmhnajgre.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.slice_backward]

triton_per_fused_add_native_layer_norm_backward_slice_backward_97 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_slice_backward_97', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6280
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 785
    x3 = (xindex // 785)
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp21 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = x2
    tmp14 = tl.full([1, 1], 1, tl.int64)
    tmp15 = tmp13 >= tmp14
    tmp16 = tl.load(in_ptr3 + ((784*r1) + (100352*x3) + (((-1) + x2) % 784)), rmask & tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp15, tmp16, tmp17)
    tmp19 = 0.0
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp22 = 128.0
    tmp23 = tmp2 * tmp22
    tmp24 = tmp23 - tmp6
    tmp25 = tmp7 * tmp12
    tmp26 = tmp24 - tmp25
    tmp27 = tmp21 * tmp26
    tmp28 = tmp20 + tmp27
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp28, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/da/cda4zroaywnjnulhof2gp6ipo65jgn4y72dergbzbsbhe6mrvvdb.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_98 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_98', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6400
    rnumel = 126
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
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 6280, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (128*((r2 + (126*x1)) % 6280))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (128*((r2 + (126*x1)) % 6280))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/7l/c7lmdnvbxhxmd5arkhrocx6hinnwsieskea3pexbivnelgxklvzj.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_99 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_99', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6400
    rnumel = 126
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
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 6280, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (128*r2) + (16128*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rr/crr5f7iszsmksqrxv6omxwlz6npymdnhhaowvrwf5p6prdh2kylg.py
# Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul]

triton_poi_fused_constant_pad_nd_mul_100 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_mul_100', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 128) % 784
    x3 = (xindex // 100352)
    x5 = xindex % 100352
    x4 = xindex % 128
    x0 = xindex % 16
    x1 = (xindex // 16) % 8
    tmp6 = tl.load(in_ptr1 + (x4 + (384*x2) + (301440*x3)), None)
    tmp0 = 1 + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (128 + x5 + (100480*x3)), tmp2, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp7 = tmp5 * tmp6
    tl.store(out_ptr0 + (x0 + (16*x2) + (12544*x1) + (100352*x3)), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/f6/cf6etracheai4kkji5oawi7d3jlvrpi5tgf7hnocucukxkoamidl.py
# Source Nodes: [], Original ATen: [aten.slice]

triton_poi_fused_slice_101 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_101', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = (yindex // 32)
    tmp0 = tl.load(in_ptr0 + ((16*x2) + (12544*(y0 // 16)) + (100352*y1) + (y0 % 16)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (32*x2) + (25088*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gp/cgpiboubdk6gglwl5jihwpt4a2wcfsvkr24ystxg5tzpr7nius6z.py
# Source Nodes: [], Original ATen: [aten.slice]

triton_poi_fused_slice_102 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_102', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 48
    y1 = (yindex // 48)
    tmp0 = tl.load(in_ptr0 + (25088 + (16*x2) + (12544*(y0 // 16)) + (100352*y1) + (y0 % 16)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (48*x2) + (37632*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/54/c54yowtrbu67zb7r3s7mkm6yryjytngobfkrxlge2w47pvqilegr.py
# Source Nodes: [], Original ATen: [aten.slice]

triton_poi_fused_slice_103 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_103', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 48
    y1 = (yindex // 48)
    tmp0 = tl.load(in_ptr0 + (62720 + (16*x2) + (12544*(y0 // 16)) + (100352*y1) + (y0 % 16)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (48*x2) + (37632*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h5/ch5t2wnnivkjgju5q2x7uct2x32tv6wtq4qffzzkzrhgpohdukwx.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_104 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_104', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2352
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 48
    x1 = (xindex // 48)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (48*r2) + (6144*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cr/ccrpkz4eytkl5pvbfz4xo7dtpaedcxxfntchn6jy5wvskbyh4tvf.py
# Source Nodes: [], Original ATen: [aten.clone, aten.mul]

triton_poi_fused_clone_mul_105 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_105', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 803840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 785
    x2 = (xindex // 12560) % 8
    x3 = (xindex // 100480)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16*x2) + (128*x1) + (100480*x3)), xmask)
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x4), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/px/cpxvbishhmbyy4qlujp6axsnfjixsmc5xl3q6wx2uhtwvn5nfphx.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_red_fused__softmax_backward_data_106 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_backward_data_106', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 7168
    rnumel = 113
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 7
    x5 = (xindex // 7)
    x1 = (xindex // 7) % 16
    x2 = (xindex // 112) % 8
    x3 = (xindex // 896)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x6 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r4 = rindex
        tmp0 = r4 + (113*x0)
        tmp1 = tl.full([1, 1], 785, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (r4 + (113*x0) + (785*x5)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x2 + (8*x1) + (128*r4) + (14464*x0) + (100480*x3)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x6), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ww/cwwkqz2ftobpzaeh45f2nc7hocxht5zabbgtqdg2izua5pwv3ztq.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_per_fused__softmax_backward_data_107 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_107', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tk/ctk2bovzv3q3ihemiultgdynxq3mkz3crsoqclcx2k2ubesmfpif.py
# Source Nodes: [], Original ATen: [aten.stack]

triton_poi_fused_stack_108 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_108', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 150720
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y2 = (yindex // 6280)
    y0 = yindex % 785
    x3 = xindex
    y1 = (yindex // 785) % 8
    y4 = yindex
    y5 = (yindex // 785)
    tmp0 = y2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.broadcast_to(y0, [XBLOCK, YBLOCK])
    tmp6 = tl.full([1, 1], 1, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tmp7 & tmp4
    tmp9 = tmp5 >= tmp1
    tmp10 = tmp9 & tmp8
    tmp11 = tl.load(in_ptr0 + (x3 + (16*y1) + (128*y0) + (100480*y2)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = tl.load(in_ptr1 + (x3 + (16*y1) + (128*(((-1) + y0) % 784)) + (100352*y2)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 * tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = 0.0
    tmp19 = tl.where(tmp7, tmp17, tmp18)
    tmp20 = tl.load(in_ptr2 + (x3 + (16*y4)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp4, tmp21, tmp22)
    tmp24 = tmp0 >= tmp3
    tmp25 = tl.full([1, 1], 16, tl.int64)
    tmp26 = tmp0 < tmp25
    tmp27 = tmp24 & tmp26
    tmp28 = tl.load(in_ptr3 + ((-803840) + y0 + (785*x3) + (12560*y5)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr4 + ((-803840) + y1 + (8*x3) + (128*y0) + (100480*y2)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 * tmp29
    tmp31 = tl.load(in_ptr5 + ((-1024) + x3 + (16*y5)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp29 * tmp31
    tmp33 = tmp30 - tmp32
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp27, tmp33, tmp34)
    tmp36 = tmp0 >= tmp25
    tmp37 = tl.full([1, 1], 24, tl.int64)
    tmp38 = tmp0 < tmp37
    tmp39 = tmp7 & tmp36
    tmp40 = x3 + (16*y1)
    tmp41 = tmp40 >= tmp1
    tmp42 = tl.full([1, 1], 32, tl.int64)
    tmp43 = tmp40 < tmp42
    tmp44 = tmp43 & tmp39
    tmp45 = tl.load(in_ptr6 + ((-401408) + (784*x3) + (12544*y1) + (25088*y2) + (((-1) + y0) % 784)), tmp44 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp44, tmp45, tmp46)
    tmp48 = tmp40 >= tmp42
    tmp49 = tl.full([1, 1], 80, tl.int64)
    tmp50 = tmp40 < tmp49
    tmp51 = tmp48 & tmp50
    tmp52 = tmp51 & tmp39
    tmp53 = tl.load(in_ptr7 + ((-627200) + (784*x3) + (12544*y1) + (37632*y2) + (((-1) + y0) % 784)), tmp52 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp52, tmp53, tmp54)
    tmp56 = tmp40 >= tmp49
    tmp57 = tl.full([1, 1], 128, tl.int64)
    tmp58 = tmp40 < tmp57
    tmp59 = tmp56 & tmp39
    tmp60 = tl.load(in_ptr8 + ((-664832) + (784*x3) + (12544*y1) + (37632*y2) + (((-1) + y0) % 784)), tmp59 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp61 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
    tmp62 = tl.where(tmp59, tmp60, tmp61)
    tmp63 = tl.where(tmp51, tmp55, tmp62)
    tmp64 = tl.where(tmp43, tmp47, tmp63)
    tmp65 = tl.full(tmp64.shape, 0.0, tmp64.dtype)
    tmp66 = tl.where(tmp39, tmp64, tmp65)
    tmp67 = tl.where(tmp7, tmp66, tmp18)
    tmp68 = tl.load(in_ptr9 + ((-1607680) + x3 + (16*y4)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp69 = tmp67 + tmp68
    tmp70 = tl.full(tmp69.shape, 0.0, tmp69.dtype)
    tmp71 = tl.where(tmp36, tmp69, tmp70)
    tmp72 = tl.where(tmp27, tmp35, tmp71)
    tmp73 = tl.where(tmp4, tmp23, tmp72)
    tl.store(out_ptr0 + (x3 + (16*y4)), tmp73, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ae/caeul33qt44a6ivn7e6b6m2ina7dwciyzux3fsprg7wjnd7gowkx.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_109 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_109', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2411520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 384
    x1 = (xindex // 384)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((16*(x1 % 785)) + (12560*((x0 // 16) % 8)) + (100480*(x1 // 785)) + (803840*(x0 // 128)) + (x0 % 16)), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nw/cnwgwcr3wwohqqz5lepmqxfy4u4bpwi4fqvawuneuw3bzdeasrgk.py
# Source Nodes: [cur_12], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# cur_12 => mul_31, sub_11
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_110 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_110', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6280
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


# kernel path: /tmp/torchinductor_youkaichao/b5/cb57ew262ptfswu24ubsokjha3nyxhxk2il7buwkmfb6x2qd3biq.py
# Source Nodes: [], Original ATen: [aten.add, aten.slice_backward]

triton_poi_fused_add_slice_backward_111 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_slice_backward_111', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 803840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 128) % 785
    x0 = xindex % 128
    x2 = (xindex // 100480)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (128 + x0 + (128*(((-1) + x1) % 784)) + (100480*x2)), tmp2 & xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + ((784*x0) + (100352*x2) + (((-1) + x1) % 784)), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tmp0 < tmp1
    tmp11 = tl.load(in_ptr0 + (x0 + (100480*x2)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = tl.where(tmp10, tmp13, tmp8)
    tmp15 = tmp9 + tmp14
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i6/ci6fd3gwmhf52pieslg5viz3altpfhbs3t44zf2lq7ennoydwlet.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_112 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_112', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6280
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
    tmp13 = tl.load(in_ptr3 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ww/cwwig5bzib7fsiubnm22meezi7l5wne3ixw6f6b2xcbqloc2zuar.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward]

triton_per_fused_add_convolution_backward_113 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_backward_113', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 48
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (48*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0 + (48*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tmp4 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tx/ctxsctiavaafdqkh3pxghow2et7rku3y3kdshwy34qiqsrlfh2us.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_114 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_114', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2352
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 48
    x1 = (xindex // 48)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (48*r2) + (6144*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pu/cpu7ag25sxxqupjf7l5yimypbcv5yh6o4uya254pqznwmtgxmhnx.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_115 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_115', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r2) + (4096*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fi/cfirvupdmf75rv22hrgczyon5u7kwbydy2p6u4d7i6mwb2epcmp5.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_116 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_116', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (32*r2) + (4096*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ic/cich3ssyhgheglsn5qkyisly75hcllxksppz3ufrkulras4nbbym.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward]

triton_per_fused_add_convolution_backward_117 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_backward_117', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tmp4 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qv/cqvqaq5gdvj3flftydxts2gxa6whrwxptm3d2k72d5z5vquqxmxc.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_118 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_118', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 19200
    rnumel = 126
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
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 6280, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*r2) + (48384*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r2/cr22duf4b3qvn7mvlccrx7urhaobihx2g674tqaiz3jjlsm2san2.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_119 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_119', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 50
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


# kernel path: /tmp/torchinductor_youkaichao/oh/cohgr5lbn466x7av4kkzajysjwgmskr3hyv66yichjmwczs3gye2.py
# Source Nodes: [cur_12], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# cur_12 => mul_31, sub_11
triton_red_fused_native_layer_norm_native_layer_norm_backward_120 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_120', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6400
    rnumel = 126
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
        tmp0 = r2 + (126*x1)
        tmp1 = tl.full([1, 1], 6280, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (128*((r2 + (126*x1)) % 6280))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (128*((r2 + (126*x1)) % 6280))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + ((r2 + (126*x1)) % 6280), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 - tmp5
        tmp7 = tl.load(in_ptr3 + ((r2 + (126*x1)) % 6280), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/el/celf5mz5oths5uubbh24wmskshmd6nys5ky56dh3nqdhujowbqud.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_121 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_121', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
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
        tmp0 = tl.load(in_ptr0 + (128 + x0 + (128*((r2 + (128*x1)) % 784)) + (100480*((r2 + (128*x1)) // 784))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qi/cqiloyeo73sgh7kmjrjioeg3qtuq63zpacpmlpsp2ci7h36aztgu.py
# Source Nodes: [cur_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# cur_8 => mul_22, sub_8
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_122 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_122', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6280
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
    tmp17 = tl.load(in_ptr5 + (r1 + (128*x0)), rmask & xmask, other=0.0)
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
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ps/cpsv3s6fspdodaf6ryktiwwx6eoq6dfyfrnymbzn5jbvs2iu6ux6.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_layer_norm_backward]

triton_red_fused_convolution_backward_native_layer_norm_backward_123 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_native_layer_norm_backward_123', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (128 + x0 + (128*((r2 + (128*x1)) % 784)) + (100480*((r2 + (128*x1)) // 784))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr2 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp4 = 1 + ((r2 + (128*x1)) % 784)
        tmp5 = tl.full([1, 1], 1, tl.int64)
        tmp6 = tmp4 >= tmp5
        tmp7 = tl.load(in_ptr0 + (128 + x0 + (128*((r2 + (128*x1)) % 784)) + (100480*((r2 + (128*x1)) // 784))), rmask & tmp6 & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr1 + ((784*x0) + (100352*((r2 + (128*x1)) // 784)) + ((r2 + (128*x1)) % 784)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp7 + tmp8
        tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
        tmp11 = tl.where(tmp6, tmp9, tmp10)
        tmp12 = 0.0
        tmp13 = tl.where(tmp6, tmp11, tmp12)
        tmp14 = tmp4 < tmp5
        tmp15 = tl.load(in_ptr0 + (x0 + (100480*((r2 + (128*x1)) // 784))), rmask & tmp14 & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
        tmp17 = tl.where(tmp14, tmp15, tmp16)
        tmp18 = tl.where(tmp14, tmp17, tmp12)
        tmp19 = tmp13 + tmp18
        tmp21 = tmp19 * tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4f/c4fxdqlktw2ltmmfxod5b4s7fm23pjp7hz5iuaszcmhelsbbbjg5.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward]

triton_per_fused_add_convolution_backward_124 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_backward_124', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tmp4 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zq/czqzs5izdmk3rd4krj3kbgsdc4i42dzmobasjqa6sdgepwlceu4t.py
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
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_125', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oa/coar3seizlazuzrtboja7baduwwgiujir2phflwio3zez37ji5fx.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_126 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_126', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u3/cu3q7wpypjtpud3azo7k2me5pnvqjjpb7dmucdw6gdrjfmuexp7k.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_127 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_127', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7b/c7b4brctji5e76v5st5t6jfatleb5wvpgtjomxmoh7tp3ghrvrhc.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_128 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_128', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ms/cmsbukshokdybe4hh66szfmhd2s34fmu6e45u34lqk3g3gkpzm6w.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_129 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_129', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.full([1, 1], 0, tl.int64)
    tmp1 = tl.full([1, 1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (100352 + x0 + (100480*r1)), rmask & tmp2 & xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (783 + (784*x0) + (100352*r1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tmp0 < tmp1
    tmp11 = tl.load(in_ptr0 + (x0 + (100480*r1)), rmask & tmp10 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = tl.where(tmp10, tmp13, tmp8)
    tmp15 = tmp9 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y5/cy5dunzazdslnsioovzohimkgxfa53se3epzqjkec4modbcjxfbp.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_130 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_130', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp16 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr3 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = 1 + x0
        tmp1 = tl.full([1, 1], 1, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.load(in_ptr0 + (128 + r2 + (128*x0) + (100480*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (784*r2) + (100352*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = 0.0
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tmp0 < tmp1
        tmp11 = tl.load(in_ptr0 + (r2 + (100480*x1)), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
        tmp13 = tl.where(tmp10, tmp11, tmp12)
        tmp14 = tl.where(tmp10, tmp13, tmp8)
        tmp15 = tmp9 + tmp14
        tmp17 = tmp15 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
        tmp22 = tmp17 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tmp26 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp43 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp48 = tl.load(in_ptr3 + (r2 + (128*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp27 = 1 + x0
        tmp28 = tl.full([1, 1], 1, tl.int64)
        tmp29 = tmp27 >= tmp28
        tmp30 = tl.load(in_ptr0 + (128 + r2 + (128*x0) + (100480*x1)), rmask & tmp29 & xmask, eviction_policy='evict_first', other=0.0)
        tmp31 = tl.load(in_ptr1 + (x0 + (784*r2) + (100352*x1)), rmask & tmp29 & xmask, eviction_policy='evict_first', other=0.0)
        tmp32 = tmp30 + tmp31
        tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
        tmp34 = tl.where(tmp29, tmp32, tmp33)
        tmp35 = 0.0
        tmp36 = tl.where(tmp29, tmp34, tmp35)
        tmp37 = tmp27 < tmp28
        tmp38 = tl.load(in_ptr0 + (r2 + (100480*x1)), rmask & tmp37 & xmask, eviction_policy='evict_last', other=0.0)
        tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
        tmp40 = tl.where(tmp37, tmp38, tmp39)
        tmp41 = tl.where(tmp37, tmp40, tmp35)
        tmp42 = tmp36 + tmp41
        tmp44 = tmp42 * tmp43
        tmp45 = 128.0
        tmp46 = tmp44 * tmp45
        tmp47 = tmp46 - tmp19
        tmp49 = tmp48 * tmp24
        tmp50 = tmp47 - tmp49
        tmp51 = tmp26 * tmp50
        tl.store(out_ptr2 + (r2 + (128*x3)), tmp51, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s4/cs4smvrwmatoxvuymijvgpbo6n6fv4id7e7fnfiqtl4qfhnt5qdz.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_131 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_131', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 49
    RBLOCK: tl.constexpr = 64
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


# kernel path: /tmp/torchinductor_youkaichao/6p/c6pc5ot7gol2f47igd32q4n2cwz7fi2invc7aifac4n5nokjub3y.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_132 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_132', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
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
        tmp0 = 1 + ((r2 + (128*x0)) % 784)
        tmp1 = tl.full([1, 1], 1, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.load(in_ptr0 + (128 + x1 + (128*((r2 + (128*x0)) % 784)) + (100480*((r2 + (128*x0)) // 784))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + ((784*x1) + (100352*((r2 + (128*x0)) // 784)) + ((r2 + (128*x0)) % 784)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = 0.0
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tmp0 < tmp1
        tmp11 = tl.load(in_ptr0 + (x1 + (100480*((r2 + (128*x0)) // 784))), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
        tmp13 = tl.where(tmp10, tmp11, tmp12)
        tmp14 = tl.where(tmp10, tmp13, tmp8)
        tmp15 = tmp9 + tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/al/calfkyy7tchfyalk2tyrq5fnfjpbjkhscl2iptwdhsbhwgkmyrpe.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_133 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_133', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
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


# kernel path: /tmp/torchinductor_youkaichao/od/codtlzitpl5oo4ab7uisbfnqfeptv4tkykxgy6ogih33e6shb2dz.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_134 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_134', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
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
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y6/cy6qyb2fvg26rsp3365li3lglaygmc3rotkvbvtvgb3xxjkxfbxh.py
# Source Nodes: [], Original ATen: [aten.slice_backward, aten.view]

triton_poi_fused_slice_backward_view_135 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_backward_view_135', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1606144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 64)
    x0 = xindex % 64
    x2 = xindex
    tmp0 = x1 % 3137
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((3136*x0) + (200704*(x1 // 3137)) + (((-1) + (x1 % 3137)) % 3136)), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tl.store(out_ptr0 + (x2), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ip/cipnriii2ves3j7gg7gaewm7kftlg47syxkh2vthljtpm7gah7kt.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_136 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_136', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12608
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 25096, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kj/ckjrglsm6xig6cf7r45arss7jyivkzclvqgwwexrg3g5dng37nix.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_137 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_137', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 197
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
''')


# kernel path: /tmp/torchinductor_youkaichao/zc/czcwgwqf7dzjlsld3oce2uqih623uxsuyxe3p55tujvtvsqpn2qi.py
# Source Nodes: [x_34], Original ATen: [aten.gelu, aten.gelu_backward]
# x_34 => add_18, erf_1, mul_18
triton_poi_fused_gelu_gelu_backward_138 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_138', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12849152
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


# kernel path: /tmp/torchinductor_youkaichao/xz/cxzvros2i5erigqur6s76gfjz3vyl2if5xef2la5yaruvebdkdhn.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_139 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_139', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 100864
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 512)
    x0 = xindex % 512
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 25096, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/db/cdbojflcxhp4tpbxd2yy4t5sw2hpps62pvibstdjy7egaa2la44n.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_140 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 256],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_140', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 197
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
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yk/cykmqes2qbibjqn4msprvdj57anomb4ipc7lt6guyzi27bknsp43.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.slice_backward]

triton_per_fused_add_native_layer_norm_backward_slice_backward_141 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_slice_backward_141', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25096
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 3137
    x3 = (xindex // 3137)
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp21 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp13 = x2
    tmp14 = tl.full([1, 1], 1, tl.int64)
    tmp15 = tmp13 >= tmp14
    tmp16 = tl.load(in_ptr3 + ((3136*r1) + (200704*x3) + (((-1) + x2) % 3136)), rmask & tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp15, tmp16, tmp17)
    tmp19 = 0.0
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp22 = 64.0
    tmp23 = tmp2 * tmp22
    tmp24 = tmp23 - tmp6
    tmp25 = tmp7 * tmp12
    tmp26 = tmp24 - tmp25
    tmp27 = tmp21 * tmp26
    tmp28 = tmp20 + tmp27
    tl.store(out_ptr2 + (r1 + (64*x0)), tmp28, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2i/c2iptwbwgwejb3szptny5i4fnth6pwgkqyrb2sueqhyqxmvpo7i7.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_142 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_142', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12608
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 25096, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (64*((r2 + (128*x1)) % 25096))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (64*((r2 + (128*x1)) % 25096))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/i6/ci6ywytcdcuy2kx2ractfhrlbeoqsngzjl4ot3afakpimouaigh2.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_143 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_143', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12608
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 25096, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ag/cagzg7rvpbnsqahgpsxzzjbpbqmu5vempnbx4ydkt3k2ny3ogzxm.py
# Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul]

triton_poi_fused_constant_pad_nd_mul_144 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_mul_144', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 64) % 3136
    x3 = (xindex // 200704)
    x5 = xindex % 200704
    x4 = xindex % 64
    x0 = xindex % 8
    x1 = (xindex // 8) % 8
    tmp6 = tl.load(in_ptr1 + (x4 + (192*x2) + (602304*x3)), None)
    tmp0 = 1 + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (64 + x5 + (200768*x3)), tmp2, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp7 = tmp5 * tmp6
    tl.store(out_ptr0 + (x0 + (8*x2) + (25088*x1) + (200704*x3)), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jz/cjz73rj2pnfeba3ofiv7nokh5lakdusgwfa6qrpfebxrvi6wgdjy.py
# Source Nodes: [], Original ATen: [aten.slice]

triton_poi_fused_slice_145 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_145', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16)
    tmp0 = tl.load(in_ptr0 + ((8*x2) + (25088*(y0 // 8)) + (200704*y1) + (y0 % 8)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (16*x2) + (50176*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6b/c6byjmnu2aegcpmqo7j2q3vma5mnuj5ueg36foumf5m7eiufschm.py
# Source Nodes: [], Original ATen: [aten.slice]

triton_poi_fused_slice_146 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_146', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 24
    y1 = (yindex // 24)
    tmp0 = tl.load(in_ptr0 + (50176 + (8*x2) + (25088*(y0 // 8)) + (200704*y1) + (y0 % 8)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (24*x2) + (75264*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s5/cs5ha35hlbbwalaliajigolahwlgru7csfkhgci377mu7t7tpqbz.py
# Source Nodes: [], Original ATen: [aten.slice]

triton_poi_fused_slice_147 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_147', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 24
    y1 = (yindex // 24)
    tmp0 = tl.load(in_ptr0 + (125440 + (8*x2) + (25088*(y0 // 8)) + (200704*y1) + (y0 % 8)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (24*x2) + (75264*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uu/cuudguk5sxq2dtlq6etf76vo3l72p4w3j6l45odt3aaqump3yuys.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_148 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_148', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 24
    x1 = (xindex // 24)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (24*r2) + (3072*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/os/cosym7uuc6frf4ekgtzwys5sqloaqsu5a2xlqsvat6wg6ena4uvx.py
# Source Nodes: [], Original ATen: [aten.clone, aten.mul]

triton_poi_fused_clone_mul_149 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_149', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1606144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 8
    x1 = (xindex // 8) % 3137
    x2 = (xindex // 25096) % 8
    x3 = (xindex // 200768)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (8*x2) + (64*x1) + (200768*x3)), xmask)
    tmp1 = 0.3535533905932738
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x4), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jc/cjccc7h7osu3zqgo2uowuw7tguoyxt6moxnsf3vmlmdxgowr3bbt.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_red_fused__softmax_backward_data_150 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_backward_data_150', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12800
    rnumel = 126
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 25
    x5 = (xindex // 25)
    x1 = (xindex // 25) % 8
    x2 = (xindex // 200) % 8
    x3 = (xindex // 1600)
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x6 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r4 = rindex
        tmp0 = r4 + (126*x0)
        tmp1 = tl.full([1, 1], 3137, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (r4 + (126*x0) + (3137*x5)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x2 + (8*x1) + (64*r4) + (8064*x0) + (200768*x3)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x6), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ef/cefrs4fkijnhlj3sqzs3hdqyjj4gngd33abgl7v2fxa5hobagqdw.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_per_fused__softmax_backward_data_151 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_151', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (25*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4l/c4luuwe62hwrgrrsugjsuo7qkuzwkgmyphhq3igpvmil6pp5i5a3.py
# Source Nodes: [], Original ATen: [aten.stack]

triton_poi_fused_stack_152 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576, 8], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_152', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 602304
    xnumel = 8
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y2 = (yindex // 25096)
    y0 = yindex % 3137
    x3 = xindex
    y1 = (yindex // 3137) % 8
    y4 = yindex
    y5 = (yindex // 3137)
    tmp0 = y2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.broadcast_to(y0, [XBLOCK, YBLOCK])
    tmp6 = tl.full([1, 1], 1, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tmp7 & tmp4
    tmp9 = tmp5 >= tmp1
    tmp10 = tmp9 & tmp8
    tmp11 = tl.load(in_ptr0 + (x3 + (8*y1) + (64*y0) + (200768*y2)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = tl.load(in_ptr1 + (x3 + (8*y1) + (64*(((-1) + y0) % 3136)) + (200704*y2)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 * tmp14
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp8, tmp15, tmp16)
    tmp18 = 0.0
    tmp19 = tl.where(tmp7, tmp17, tmp18)
    tmp20 = tl.load(in_ptr2 + (x3 + (8*y4)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp4, tmp21, tmp22)
    tmp24 = tmp0 >= tmp3
    tmp25 = tl.full([1, 1], 16, tl.int64)
    tmp26 = tmp0 < tmp25
    tmp27 = tmp24 & tmp26
    tmp28 = tl.load(in_ptr3 + ((-1606144) + y0 + (3137*x3) + (25096*y5)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr4 + ((-1606144) + y1 + (8*x3) + (64*y0) + (200768*y2)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 * tmp29
    tmp31 = tl.load(in_ptr5 + ((-512) + x3 + (8*y5)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp29 * tmp31
    tmp33 = tmp30 - tmp32
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp27, tmp33, tmp34)
    tmp36 = tmp0 >= tmp25
    tmp37 = tl.full([1, 1], 24, tl.int64)
    tmp38 = tmp0 < tmp37
    tmp39 = tmp7 & tmp36
    tmp40 = x3 + (8*y1)
    tmp41 = tmp40 >= tmp1
    tmp42 = tmp40 < tmp25
    tmp43 = tmp42 & tmp39
    tmp44 = tl.load(in_ptr6 + ((-802816) + (3136*x3) + (25088*y1) + (50176*y2) + (((-1) + y0) % 3136)), tmp43 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp43, tmp44, tmp45)
    tmp47 = tmp40 >= tmp25
    tmp48 = tl.full([1, 1], 40, tl.int64)
    tmp49 = tmp40 < tmp48
    tmp50 = tmp47 & tmp49
    tmp51 = tmp50 & tmp39
    tmp52 = tl.load(in_ptr7 + ((-1254400) + (3136*x3) + (25088*y1) + (75264*y2) + (((-1) + y0) % 3136)), tmp51 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp53 = tl.full(tmp52.shape, 0.0, tmp52.dtype)
    tmp54 = tl.where(tmp51, tmp52, tmp53)
    tmp55 = tmp40 >= tmp48
    tmp56 = tl.full([1, 1], 64, tl.int64)
    tmp57 = tmp40 < tmp56
    tmp58 = tmp55 & tmp39
    tmp59 = tl.load(in_ptr8 + ((-1329664) + (3136*x3) + (25088*y1) + (75264*y2) + (((-1) + y0) % 3136)), tmp58 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp60 = tl.full(tmp59.shape, 0.0, tmp59.dtype)
    tmp61 = tl.where(tmp58, tmp59, tmp60)
    tmp62 = tl.where(tmp50, tmp54, tmp61)
    tmp63 = tl.where(tmp42, tmp46, tmp62)
    tmp64 = tl.full(tmp63.shape, 0.0, tmp63.dtype)
    tmp65 = tl.where(tmp39, tmp63, tmp64)
    tmp66 = tl.where(tmp7, tmp65, tmp18)
    tmp67 = tl.load(in_ptr9 + ((-3212288) + x3 + (8*y4)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp68 = tmp66 + tmp67
    tmp69 = tl.full(tmp68.shape, 0.0, tmp68.dtype)
    tmp70 = tl.where(tmp36, tmp68, tmp69)
    tmp71 = tl.where(tmp27, tmp35, tmp70)
    tmp72 = tl.where(tmp4, tmp23, tmp71)
    tl.store(out_ptr0 + (x3 + (8*y4)), tmp72, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gf/cgf52ax572b7etliko65liegrgbqdipjlhpg2dy3bysqzrooy3rj.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_153 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_153', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4818432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 192
    x1 = (xindex // 192)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((8*(x1 % 3137)) + (25096*((x0 // 8) % 8)) + (200768*(x1 // 3137)) + (1606144*(x0 // 64)) + (x0 % 8)), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lz/clzrsylasbfqr4hg3guyr3mgijtqw5dbr24zsekvtqtrcxwb3iwp.py
# Source Nodes: [cur_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# cur_4 => mul_11, sub_4
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_154 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_154', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25096
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_out_ptr0 + (r1 + (64*x0)), rmask & xmask, other=0.0)
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
    tmp18 = 64.0
    tmp19 = tmp10 / tmp18
    tmp20 = tmp2 * tmp18
    tmp21 = tmp20 - tmp6
    tmp22 = tmp11 * tmp16
    tmp23 = tmp21 - tmp22
    tmp24 = tmp19 * tmp23
    tmp25 = tmp17 + tmp24
    tl.store(in_out_ptr0 + (r1 + (64*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xm/cxmccb3eriyc5dsmlk6hdajzoxue3vjpbywpebstmewyjswhict4.py
# Source Nodes: [], Original ATen: [aten.add, aten.slice_backward]

triton_poi_fused_add_slice_backward_155 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_slice_backward_155', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1606144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 64) % 3137
    x0 = xindex % 64
    x2 = (xindex // 200768)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (64 + x0 + (64*(((-1) + x1) % 3136)) + (200768*x2)), tmp2 & xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + ((3136*x0) + (200704*x2) + (((-1) + x1) % 3136)), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tmp0 < tmp1
    tmp11 = tl.load(in_ptr0 + (x0 + (200768*x2)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = tl.where(tmp10, tmp13, tmp8)
    tmp15 = tmp9 + tmp14
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d4/cd4h5bid7ii6tk5h72jvdntv4wfiizzwm4k47ucieivpeh64fwtn.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_156 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_156', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25096
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp15 = 64.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(out_ptr2 + (r1 + (64*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m2/cm26ja7zpe6ndsq3lcd6lvnlkbec2bvckivuno26xkrl7o52idtz.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward]

triton_red_fused_add_convolution_backward_157 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32, 256],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_backward_157', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (24*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (24*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp8 = tmp2 + tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sc/csc6klatcj7vmsywujthbew5sr6jp5le6eg7sweb7anidsvw476r.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_158 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_158', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 24
    x1 = (xindex // 24)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (24*r2) + (3072*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xv/cxvvdarcjyeuksu7vbh6icpcspqmmbif4bncnkwx2zgn5r3ydiay.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_159 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_159', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3136
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (16*r2) + (2048*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2m/c2m2h7lu4hqkcrafwovhuijlqbxrzzgmewii3ipsenuatiq4rg3k.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_160 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_160', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3136
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (16*r2) + (2048*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fo/cfoynayuam2mvx4n3upy5d5bll5yaw6uvpjgzg5j3d2bjnmrnvry.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward]

triton_red_fused_add_convolution_backward_161 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16, 256],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_backward_161', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (16*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (16*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp8 = tmp2 + tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kc/ckcjtjc5mcvuotl4nhylv56tdgpid4t6mqolrmm2aq7cdqwo7h2c.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_162 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_162', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 37824
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 192)
    x0 = xindex % 192
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 25096, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (192*r2) + (24576*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hg/chgvcm2cfswqgbey7oui3wavxkcjfz6izyhzaori2tvywchn3imw.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_163 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_163', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 197
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
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pd/cpddhnzi4we3cqp26dwg64lj57ppgb76s2xmwrblkz3z2fjehscs.py
# Source Nodes: [cur_4], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# cur_4 => mul_11, sub_4
triton_red_fused_native_layer_norm_native_layer_norm_backward_164 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_164', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12608
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 64)
    x0 = xindex % 64
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (128*x1)
        tmp1 = tl.full([1, 1], 25096, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (64*((r2 + (128*x1)) % 25096))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (64*((r2 + (128*x1)) % 25096))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + ((r2 + (128*x1)) % 25096), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 - tmp5
        tmp7 = tl.load(in_ptr3 + ((r2 + (128*x1)) % 25096), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/pe/cpef637gezxgouciiath2k33jq3fhbjlsubuiewd5oq6fe76zyxh.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_165 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_165', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (64 + x0 + (64*((r2 + (128*x1)) % 3136)) + (200768*((r2 + (128*x1)) // 3136))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hg/chgalegfht2ciyauzh2fiv3tmbqdq7ftlpumen64ftzne2d55poq.py
# Source Nodes: [cur], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# cur => mul_2, sub_1
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_166 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_166', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25096
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (r1 + (64*x0)), rmask & xmask, other=0.0)
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
    tmp18 = 64.0
    tmp19 = tmp10 / tmp18
    tmp20 = tmp2 * tmp18
    tmp21 = tmp20 - tmp6
    tmp22 = tmp11 * tmp16
    tmp23 = tmp21 - tmp22
    tmp24 = tmp19 * tmp23
    tmp25 = tmp17 + tmp24
    tl.store(out_ptr2 + (r1 + (64*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p4/cp4qkum4xvja76wby7fzfjgubbrzjz7irkgub7j32fuwhsfltvq6.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_layer_norm_backward]

triton_red_fused_convolution_backward_native_layer_norm_backward_167 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_native_layer_norm_backward_167', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (64 + x0 + (64*((r2 + (128*x1)) % 3136)) + (200768*((r2 + (128*x1)) // 3136))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr2 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp4 = 1 + ((r2 + (128*x1)) % 3136)
        tmp5 = tl.full([1, 1], 1, tl.int64)
        tmp6 = tmp4 >= tmp5
        tmp7 = tl.load(in_ptr0 + (64 + x0 + (64*((r2 + (128*x1)) % 3136)) + (200768*((r2 + (128*x1)) // 3136))), rmask & tmp6 & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr1 + ((3136*x0) + (200704*((r2 + (128*x1)) // 3136)) + ((r2 + (128*x1)) % 3136)), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp7 + tmp8
        tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
        tmp11 = tl.where(tmp6, tmp9, tmp10)
        tmp12 = 0.0
        tmp13 = tl.where(tmp6, tmp11, tmp12)
        tmp14 = tmp4 < tmp5
        tmp15 = tl.load(in_ptr0 + (x0 + (200768*((r2 + (128*x1)) // 3136))), rmask & tmp14 & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
        tmp17 = tl.where(tmp14, tmp15, tmp16)
        tmp18 = tl.where(tmp14, tmp17, tmp12)
        tmp19 = tmp13 + tmp18
        tmp21 = tmp19 * tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pz/cpzynbvi5jb5lcspg7ryd5oerk27l53sgv7lvb5ibqoddsbpdbzj.py
# Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward]

triton_red_fused_add_convolution_backward_168 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_backward_168', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp8 = tmp2 + tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nl/cnleofpuldg3b3wapcdkepeq2lj5rahgduie7ufk5r2mz4dlvtr3.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_169 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_169', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/so/csoljylkgo2kw4aek7fjk5u6egxzoj5gkrtpzy3ofbiicacff5cp.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_170 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_170', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rp/crpbo6lbhru6gysinfahbhbz7flxqwahdoo7tqau3aikh5bzpgjk.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_171 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_171', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/26/c26zq36wn7spya2wt4qlypdxxppebwl3zr56mc7qhrvj6kkrb4qq.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_172 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_172', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gg/cggvbmegp6bobilb2lo2usdcp7w3zklds3ns5tk72atbqsmefsy3.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_173 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_173', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.full([1, 1], 0, tl.int64)
    tmp1 = tl.full([1, 1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (200704 + x0 + (200768*r1)), rmask & tmp2 & xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (3135 + (3136*x0) + (200704*r1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tmp0 < tmp1
    tmp11 = tl.load(in_ptr0 + (x0 + (200768*r1)), rmask & tmp10 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = tl.where(tmp10, tmp13, tmp8)
    tmp15 = tmp9 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yx/cyx3zrgpmgz5jugsga2n3y6qn6gzrhd3lj5ycondptgfs66rbf2d.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_174 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_174', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x0 = xindex % 3136
    r2 = rindex
    x1 = (xindex // 3136)
    x3 = xindex
    tmp16 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.load(in_ptr3 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp28 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = 1 + x0
    tmp1 = tl.full([1, 1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + (64 + r2 + (64*x0) + (200768*x1)), rmask & tmp2 & xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (x0 + (3136*r2) + (200704*x1)), rmask & tmp2 & xmask, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tmp0 < tmp1
    tmp11 = tl.load(in_ptr0 + (r2 + (200768*x1)), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = tl.where(tmp10, tmp13, tmp8)
    tmp15 = tmp9 + tmp14
    tmp17 = tmp15 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp20 = tl.where(rmask & xmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tmp23 = tmp17 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
    tmp26 = tl.where(rmask & xmask, tmp24, 0)
    tmp27 = tl.sum(tmp26, 1)[:, None]
    tmp29 = 64.0
    tmp30 = tmp17 * tmp29
    tmp31 = tmp30 - tmp21
    tmp32 = tmp22 * tmp27
    tmp33 = tmp31 - tmp32
    tmp34 = tmp28 * tmp33
    tl.store(out_ptr2 + (r2 + (64*x3)), tmp34, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x7/cx7mr6r36avs5mszs27dcaexjwa2f4fqiy7fxijwe7ebs4w6e3nb.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_175 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_175', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
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
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kq/ckqzc2anamvdwmtvhdrzt3ybuaq76wf7pp2muvsdz53sw4loubeh.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_176 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_176', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = 1 + ((r2 + (128*x0)) % 3136)
        tmp1 = tl.full([1, 1], 1, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.load(in_ptr0 + (64 + x1 + (64*((r2 + (128*x0)) % 3136)) + (200768*((r2 + (128*x0)) // 3136))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + ((3136*x1) + (200704*((r2 + (128*x0)) // 3136)) + ((r2 + (128*x0)) % 3136)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = 0.0
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tmp0 < tmp1
        tmp11 = tl.load(in_ptr0 + (x1 + (200768*((r2 + (128*x0)) // 3136))), rmask & tmp10 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
        tmp13 = tl.where(tmp10, tmp11, tmp12)
        tmp14 = tl.where(tmp10, tmp13, tmp8)
        tmp15 = tmp9 + tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4x/c4xqo55rq4jxmupxdb2rzpnfsf73aq2dfme4g7lrr6wxa3z2zlo3.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_177 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_177', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
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


# kernel path: /tmp/torchinductor_youkaichao/ev/cevay2zb322e5hmkuvneflycigmbfv7frn5i7a5vn5own6tjj4qz.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_178 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_178', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (64*r2) + (8192*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_2, primals_4, primals_6, primals_8, primals_11, primals_13, primals_15, primals_17, primals_20, primals_22, primals_24, primals_26, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_47, primals_49, primals_51, primals_67, primals_69, primals_71, primals_75, primals_77, primals_79, primals_95, primals_97, primals_99, primals_103, primals_105, primals_107, primals_123, primals_125, primals_127, primals_131, primals_133, primals_135, primals_153, mul, view_1, cat_1, getitem_3, rsqrt_1, view_3, slice_8, getitem_7, getitem_8, getitem_9, cat_2, view_15, mul_6, view_17, addmm_2, view_19, view_21, cat_3, getitem_13, rsqrt_3, view_23, slice_20, getitem_17, getitem_18, getitem_19, cat_4, view_35, mul_15, view_37, addmm_6, view_39, clone_15, mul_20, view_43, cat_6, getitem_25, rsqrt_6, view_45, slice_35, getitem_29, getitem_30, getitem_31, cat_7, view_57, mul_26, view_59, addmm_10, view_61, view_63, cat_8, getitem_35, rsqrt_8, view_65, slice_47, getitem_39, getitem_40, getitem_41, cat_9, view_77, mul_35, view_79, addmm_14, view_81, clone_31, mul_40, view_85, cat_11, getitem_47, rsqrt_11, view_87, slice_62, getitem_51, getitem_52, getitem_53, cat_12, view_99, mul_46, view_101, addmm_18, view_103, view_105, cat_13, getitem_57, rsqrt_13, view_107, slice_74, getitem_61, getitem_62, getitem_63, cat_14, view_119, mul_55, view_121, addmm_22, view_123, clone_47, mul_60, view_127, cat_16, getitem_69, rsqrt_16, view_129, slice_89, getitem_73, getitem_74, getitem_75, cat_17, view_141, mul_66, view_143, addmm_26, view_145, view_147, cat_18, getitem_79, rsqrt_18, view_149, slice_101, getitem_83, getitem_84, getitem_85, cat_19, view_161, mul_75, view_163, addmm_30, view_165, mul_80, clone_64, permute_97, div_8, permute_101, permute_105, div_9, permute_109, permute_116, permute_117, permute_118, permute_119, alias_8, permute_122, permute_128, permute_132, div_11, permute_136, permute_143, permute_144, permute_145, permute_146, alias_9, permute_149, div_13, permute_157, permute_161, div_14, permute_165, permute_172, permute_173, permute_174, permute_175, alias_10, permute_178, permute_184, permute_188, div_16, permute_192, permute_199, permute_200, permute_201, permute_202, alias_11, permute_205, div_18, permute_213, permute_217, div_19, permute_221, permute_228, permute_229, permute_230, permute_231, alias_12, permute_234, permute_240, permute_244, div_21, permute_248, permute_255, permute_256, permute_257, permute_258, alias_13, permute_261, div_23, permute_269, permute_273, div_24, permute_277, permute_284, permute_285, permute_286, permute_287, alias_14, permute_290, permute_296, permute_300, div_26, permute_304, permute_311, permute_312, permute_313, permute_314, alias_15, permute_317, div_28, tangents_1 = args
    args.clear()
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_13, (128, ), (1, ))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_17, (128, ), (1, ))
    assert_size_stride(primals_20, (320, ), (1, ))
    assert_size_stride(primals_22, (320, ), (1, ))
    assert_size_stride(primals_24, (320, ), (1, ))
    assert_size_stride(primals_26, (320, ), (1, ))
    assert_size_stride(primals_29, (512, ), (1, ))
    assert_size_stride(primals_31, (512, ), (1, ))
    assert_size_stride(primals_33, (512, ), (1, ))
    assert_size_stride(primals_35, (512, ), (1, ))
    assert_size_stride(primals_37, (512, ), (1, ))
    assert_size_stride(primals_39, (64, 3, 4, 4), (48, 1, 12, 3))
    assert_size_stride(primals_41, (64, ), (1, ))
    assert_size_stride(primals_43, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_47, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_49, (24, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_51, (24, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_67, (128, 64, 2, 2), (256, 1, 128, 64))
    assert_size_stride(primals_69, (128, ), (1, ))
    assert_size_stride(primals_71, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_75, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_77, (48, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_79, (48, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_95, (320, 128, 2, 2), (512, 1, 256, 128))
    assert_size_stride(primals_97, (320, ), (1, ))
    assert_size_stride(primals_99, (320, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_103, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_105, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_107, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_123, (512, 320, 2, 2), (1280, 1, 640, 320))
    assert_size_stride(primals_125, (512, ), (1, ))
    assert_size_stride(primals_127, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_131, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_133, (192, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_135, (192, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_153, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(mul, (8, 3136, 64), (200704, 64, 1))
    assert_size_stride(view_1, (8, 64, 56, 56), (200768, 1, 3584, 64))
    assert_size_stride(cat_1, (8, 3137, 64), (200768, 64, 1))
    assert_size_stride(getitem_3, (8, 3137, 1), (3137, 1, 1))
    assert_size_stride(rsqrt_1, (8, 3137, 1), (3137, 1, 1))
    assert_size_stride(view_3, (25096, 64), (64, 1))
    assert_size_stride(slice_8, (8, 8, 3136, 8), (602304, 8, 192, 1))
    assert_size_stride(getitem_7, (8, 16, 56, 56), (602304, 1, 10752, 192))
    assert_size_stride(getitem_8, (8, 24, 56, 56), (602304, 1, 10752, 192))
    assert_size_stride(getitem_9, (8, 24, 56, 56), (602304, 1, 10752, 192))
    assert_size_stride(cat_2, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(view_15, (25096, 64), (64, 1))
    assert_size_stride(mul_6, (8, 3137, 64), (200768, 64, 1))
    assert_size_stride(view_17, (25096, 64), (64, 1))
    assert_size_stride(addmm_2, (25096, 512), (512, 1))
    assert_size_stride(view_19, (25096, 512), (512, 1))
    assert_size_stride(view_21, (8, 64, 56, 56), (200768, 1, 3584, 64))
    assert_size_stride(cat_3, (8, 3137, 64), (200768, 64, 1))
    assert_size_stride(getitem_13, (8, 3137, 1), (3137, 1, 1))
    assert_size_stride(rsqrt_3, (8, 3137, 1), (3137, 1, 1))
    assert_size_stride(view_23, (25096, 64), (64, 1))
    assert_size_stride(slice_20, (8, 8, 3136, 8), (602304, 8, 192, 1))
    assert_size_stride(getitem_17, (8, 16, 56, 56), (602304, 1, 10752, 192))
    assert_size_stride(getitem_18, (8, 24, 56, 56), (602304, 1, 10752, 192))
    assert_size_stride(getitem_19, (8, 24, 56, 56), (602304, 1, 10752, 192))
    assert_size_stride(cat_4, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(view_35, (25096, 64), (64, 1))
    assert_size_stride(mul_15, (8, 3137, 64), (200768, 64, 1))
    assert_size_stride(view_37, (25096, 64), (64, 1))
    assert_size_stride(addmm_6, (25096, 512), (512, 1))
    assert_size_stride(view_39, (25096, 512), (512, 1))
    assert_size_stride(clone_15, (8, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(mul_20, (8, 784, 128), (100352, 128, 1))
    assert_size_stride(view_43, (8, 128, 28, 28), (100480, 1, 3584, 128))
    assert_size_stride(cat_6, (8, 785, 128), (100480, 128, 1))
    assert_size_stride(getitem_25, (8, 785, 1), (785, 1, 1))
    assert_size_stride(rsqrt_6, (8, 785, 1), (785, 1, 1))
    assert_size_stride(view_45, (6280, 128), (128, 1))
    assert_size_stride(slice_35, (8, 8, 784, 16), (301440, 16, 384, 1))
    assert_size_stride(getitem_29, (8, 32, 28, 28), (301440, 1, 10752, 384))
    assert_size_stride(getitem_30, (8, 48, 28, 28), (301440, 1, 10752, 384))
    assert_size_stride(getitem_31, (8, 48, 28, 28), (301440, 1, 10752, 384))
    assert_size_stride(cat_7, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(view_57, (6280, 128), (128, 1))
    assert_size_stride(mul_26, (8, 785, 128), (100480, 128, 1))
    assert_size_stride(view_59, (6280, 128), (128, 1))
    assert_size_stride(addmm_10, (6280, 1024), (1024, 1))
    assert_size_stride(view_61, (6280, 1024), (1024, 1))
    assert_size_stride(view_63, (8, 128, 28, 28), (100480, 1, 3584, 128))
    assert_size_stride(cat_8, (8, 785, 128), (100480, 128, 1))
    assert_size_stride(getitem_35, (8, 785, 1), (785, 1, 1))
    assert_size_stride(rsqrt_8, (8, 785, 1), (785, 1, 1))
    assert_size_stride(view_65, (6280, 128), (128, 1))
    assert_size_stride(slice_47, (8, 8, 784, 16), (301440, 16, 384, 1))
    assert_size_stride(getitem_39, (8, 32, 28, 28), (301440, 1, 10752, 384))
    assert_size_stride(getitem_40, (8, 48, 28, 28), (301440, 1, 10752, 384))
    assert_size_stride(getitem_41, (8, 48, 28, 28), (301440, 1, 10752, 384))
    assert_size_stride(cat_9, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(view_77, (6280, 128), (128, 1))
    assert_size_stride(mul_35, (8, 785, 128), (100480, 128, 1))
    assert_size_stride(view_79, (6280, 128), (128, 1))
    assert_size_stride(addmm_14, (6280, 1024), (1024, 1))
    assert_size_stride(view_81, (6280, 1024), (1024, 1))
    assert_size_stride(clone_31, (8, 128, 28, 28), (100352, 1, 3584, 128))
    assert_size_stride(mul_40, (8, 196, 320), (62720, 320, 1))
    assert_size_stride(view_85, (8, 320, 14, 14), (63040, 1, 4480, 320))
    assert_size_stride(cat_11, (8, 197, 320), (63040, 320, 1))
    assert_size_stride(getitem_47, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_11, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_87, (1576, 320), (320, 1))
    assert_size_stride(slice_62, (8, 8, 196, 40), (189120, 40, 960, 1))
    assert_size_stride(getitem_51, (8, 80, 14, 14), (189120, 1, 13440, 960))
    assert_size_stride(getitem_52, (8, 120, 14, 14), (189120, 1, 13440, 960))
    assert_size_stride(getitem_53, (8, 120, 14, 14), (189120, 1, 13440, 960))
    assert_size_stride(cat_12, (8, 320, 14, 14), (62720, 1, 4480, 320))
    assert_size_stride(view_99, (1576, 320), (320, 1))
    assert_size_stride(mul_46, (8, 197, 320), (63040, 320, 1))
    assert_size_stride(view_101, (1576, 320), (320, 1))
    assert_size_stride(addmm_18, (1576, 1280), (1280, 1))
    assert_size_stride(view_103, (1576, 1280), (1280, 1))
    assert_size_stride(view_105, (8, 320, 14, 14), (63040, 1, 4480, 320))
    assert_size_stride(cat_13, (8, 197, 320), (63040, 320, 1))
    assert_size_stride(getitem_57, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_13, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_107, (1576, 320), (320, 1))
    assert_size_stride(slice_74, (8, 8, 196, 40), (189120, 40, 960, 1))
    assert_size_stride(getitem_61, (8, 80, 14, 14), (189120, 1, 13440, 960))
    assert_size_stride(getitem_62, (8, 120, 14, 14), (189120, 1, 13440, 960))
    assert_size_stride(getitem_63, (8, 120, 14, 14), (189120, 1, 13440, 960))
    assert_size_stride(cat_14, (8, 320, 14, 14), (62720, 1, 4480, 320))
    assert_size_stride(view_119, (1576, 320), (320, 1))
    assert_size_stride(mul_55, (8, 197, 320), (63040, 320, 1))
    assert_size_stride(view_121, (1576, 320), (320, 1))
    assert_size_stride(addmm_22, (1576, 1280), (1280, 1))
    assert_size_stride(view_123, (1576, 1280), (1280, 1))
    assert_size_stride(clone_47, (8, 320, 14, 14), (62720, 1, 4480, 320))
    assert_size_stride(mul_60, (8, 49, 512), (25088, 512, 1))
    assert_size_stride(view_127, (8, 512, 7, 7), (25600, 1, 3584, 512))
    assert_size_stride(cat_16, (8, 50, 512), (25600, 512, 1))
    assert_size_stride(getitem_69, (8, 50, 1), (50, 1, 1))
    assert_size_stride(rsqrt_16, (8, 50, 1), (50, 1, 1))
    assert_size_stride(view_129, (400, 512), (512, 1))
    assert_size_stride(slice_89, (8, 8, 49, 64), (76800, 64, 1536, 1))
    assert_size_stride(getitem_73, (8, 128, 7, 7), (76800, 1, 10752, 1536))
    assert_size_stride(getitem_74, (8, 192, 7, 7), (76800, 1, 10752, 1536))
    assert_size_stride(getitem_75, (8, 192, 7, 7), (76800, 1, 10752, 1536))
    assert_size_stride(cat_17, (8, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(view_141, (400, 512), (512, 1))
    assert_size_stride(mul_66, (8, 50, 512), (25600, 512, 1))
    assert_size_stride(view_143, (400, 512), (512, 1))
    assert_size_stride(addmm_26, (400, 2048), (2048, 1))
    assert_size_stride(view_145, (400, 2048), (2048, 1))
    assert_size_stride(view_147, (8, 512, 7, 7), (25600, 1, 3584, 512))
    assert_size_stride(cat_18, (8, 50, 512), (25600, 512, 1))
    assert_size_stride(getitem_79, (8, 50, 1), (50, 1, 1))
    assert_size_stride(rsqrt_18, (8, 50, 1), (50, 1, 1))
    assert_size_stride(view_149, (400, 512), (512, 1))
    assert_size_stride(slice_101, (8, 8, 49, 64), (76800, 64, 1536, 1))
    assert_size_stride(getitem_83, (8, 128, 7, 7), (76800, 1, 10752, 1536))
    assert_size_stride(getitem_84, (8, 192, 7, 7), (76800, 1, 10752, 1536))
    assert_size_stride(getitem_85, (8, 192, 7, 7), (76800, 1, 10752, 1536))
    assert_size_stride(cat_19, (8, 512, 7, 7), (25088, 1, 3584, 512))
    assert_size_stride(view_161, (400, 512), (512, 1))
    assert_size_stride(mul_75, (8, 50, 512), (25600, 512, 1))
    assert_size_stride(view_163, (400, 512), (512, 1))
    assert_size_stride(addmm_30, (400, 2048), (2048, 1))
    assert_size_stride(view_165, (400, 2048), (2048, 1))
    assert_size_stride(mul_80, (8, 50, 512), (25600, 512, 1))
    assert_size_stride(clone_64, (8, 512), (512, 1))
    assert_size_stride(permute_97, (1000, 512), (512, 1))
    assert_size_stride(div_8, (8, 50, 1), (50, 1, 1))
    assert_size_stride(permute_101, (512, 2048), (2048, 1))
    assert_size_stride(permute_105, (2048, 512), (512, 1))
    assert_size_stride(div_9, (8, 50, 1), (50, 1, 1))
    assert_size_stride(permute_109, (512, 512), (512, 1))
    assert_size_stride(permute_116, (64, 64, 50), (3200, 1, 64))
    assert_size_stride(permute_117, (64, 64, 64), (4096, 1, 64))
    assert_size_stride(permute_118, (64, 50, 64), (3200, 64, 1))
    assert_size_stride(permute_119, (64, 64, 50), (3200, 1, 64))
    assert_size_stride(alias_8, (8, 8, 50, 64), (25600, 1, 512, 8))
    assert_size_stride(permute_122, (1536, 512), (512, 1))
    assert_size_stride(permute_128, (512, 2048), (2048, 1))
    assert_size_stride(permute_132, (2048, 512), (512, 1))
    assert_size_stride(div_11, (8, 50, 1), (50, 1, 1))
    assert_size_stride(permute_136, (512, 512), (512, 1))
    assert_size_stride(permute_143, (64, 64, 50), (3200, 1, 64))
    assert_size_stride(permute_144, (64, 64, 64), (4096, 1, 64))
    assert_size_stride(permute_145, (64, 50, 64), (3200, 64, 1))
    assert_size_stride(permute_146, (64, 64, 50), (3200, 1, 64))
    assert_size_stride(alias_9, (8, 8, 50, 64), (25600, 1, 512, 8))
    assert_size_stride(permute_149, (1536, 512), (512, 1))
    assert_size_stride(div_13, (8, 49, 1), (49, 1, 1))
    assert_size_stride(permute_157, (320, 1280), (1280, 1))
    assert_size_stride(permute_161, (1280, 320), (320, 1))
    assert_size_stride(div_14, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_165, (320, 320), (320, 1))
    assert_size_stride(permute_172, (64, 40, 197), (7880, 1, 40))
    assert_size_stride(permute_173, (64, 40, 40), (1600, 1, 40))
    assert_size_stride(permute_174, (64, 197, 40), (7880, 40, 1))
    assert_size_stride(permute_175, (64, 40, 197), (7880, 1, 40))
    assert_size_stride(alias_10, (8, 8, 197, 40), (63040, 1, 320, 8))
    assert_size_stride(permute_178, (960, 320), (320, 1))
    assert_size_stride(permute_184, (320, 1280), (1280, 1))
    assert_size_stride(permute_188, (1280, 320), (320, 1))
    assert_size_stride(div_16, (8, 197, 1), (197, 1, 1))
    assert_size_stride(permute_192, (320, 320), (320, 1))
    assert_size_stride(permute_199, (64, 40, 197), (7880, 1, 40))
    assert_size_stride(permute_200, (64, 40, 40), (1600, 1, 40))
    assert_size_stride(permute_201, (64, 197, 40), (7880, 40, 1))
    assert_size_stride(permute_202, (64, 40, 197), (7880, 1, 40))
    assert_size_stride(alias_11, (8, 8, 197, 40), (63040, 1, 320, 8))
    assert_size_stride(permute_205, (960, 320), (320, 1))
    assert_size_stride(div_18, (8, 196, 1), (196, 1, 1))
    assert_size_stride(permute_213, (128, 1024), (1024, 1))
    assert_size_stride(permute_217, (1024, 128), (128, 1))
    assert_size_stride(div_19, (8, 785, 1), (785, 1, 1))
    assert_size_stride(permute_221, (128, 128), (128, 1))
    assert_size_stride(permute_228, (64, 16, 785), (12560, 1, 16))
    assert_size_stride(permute_229, (64, 16, 16), (256, 1, 16))
    assert_size_stride(permute_230, (64, 785, 16), (12560, 16, 1))
    assert_size_stride(permute_231, (64, 16, 785), (12560, 1, 16))
    assert_size_stride(alias_12, (8, 8, 785, 16), (100480, 1, 128, 8))
    assert_size_stride(permute_234, (384, 128), (128, 1))
    assert_size_stride(permute_240, (128, 1024), (1024, 1))
    assert_size_stride(permute_244, (1024, 128), (128, 1))
    assert_size_stride(div_21, (8, 785, 1), (785, 1, 1))
    assert_size_stride(permute_248, (128, 128), (128, 1))
    assert_size_stride(permute_255, (64, 16, 785), (12560, 1, 16))
    assert_size_stride(permute_256, (64, 16, 16), (256, 1, 16))
    assert_size_stride(permute_257, (64, 785, 16), (12560, 16, 1))
    assert_size_stride(permute_258, (64, 16, 785), (12560, 1, 16))
    assert_size_stride(alias_13, (8, 8, 785, 16), (100480, 1, 128, 8))
    assert_size_stride(permute_261, (384, 128), (128, 1))
    assert_size_stride(div_23, (8, 784, 1), (784, 1, 1))
    assert_size_stride(permute_269, (64, 512), (512, 1))
    assert_size_stride(permute_273, (512, 64), (64, 1))
    assert_size_stride(div_24, (8, 3137, 1), (3137, 1, 1))
    assert_size_stride(permute_277, (64, 64), (64, 1))
    assert_size_stride(permute_284, (64, 8, 3137), (25096, 1, 8))
    assert_size_stride(permute_285, (64, 8, 8), (64, 1, 8))
    assert_size_stride(permute_286, (64, 3137, 8), (25096, 8, 1))
    assert_size_stride(permute_287, (64, 8, 3137), (25096, 1, 8))
    assert_size_stride(alias_14, (8, 8, 3137, 8), (200768, 1, 64, 8))
    assert_size_stride(permute_290, (192, 64), (64, 1))
    assert_size_stride(permute_296, (64, 512), (512, 1))
    assert_size_stride(permute_300, (512, 64), (64, 1))
    assert_size_stride(div_26, (8, 3137, 1), (3137, 1, 1))
    assert_size_stride(permute_304, (64, 64), (64, 1))
    assert_size_stride(permute_311, (64, 8, 3137), (25096, 1, 8))
    assert_size_stride(permute_312, (64, 8, 8), (64, 1, 8))
    assert_size_stride(permute_313, (64, 3137, 8), (25096, 8, 1))
    assert_size_stride(permute_314, (64, 8, 3137), (25096, 1, 8))
    assert_size_stride(alias_15, (8, 8, 3137, 8), (200768, 1, 64, 8))
    assert_size_stride(permute_317, (192, 64), (64, 1))
    assert_size_stride(div_28, (8, 3136, 1), (3136, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_97, out=buf0)
        del permute_97
        buf1 = empty((1000, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_64, out=buf1)
        del clone_64
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf5 = empty((8, 50, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]
        triton_red_fused_native_layer_norm_backward_select_backward_1.run(buf0, primals_37, mul_80, div_8, buf5, 400, 512, grid=grid(400), stream=stream0)
        del div_8
        del primals_37
        buf6 = empty_strided((512, 4), (1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]
        triton_red_fused_native_layer_norm_backward_select_backward_2.run(buf0, mul_80, buf6, 2048, 100, grid=grid(2048), stream=stream0)
        del mul_80
        buf7 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf6, buf7, 512, 4, grid=grid(512), stream=stream0)
        buf8 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]
        triton_red_fused_native_layer_norm_backward_select_backward_4.run(buf0, buf8, 512, 400, grid=grid(512), stream=stream0)
        buf9 = empty((400, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (400, 512), (512, 1), 0), permute_101, out=buf9)
        del permute_101
        buf10 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (512, 400), (1, 512), 0), view_165, out=buf10)
        del view_165
        buf11 = reinterpret_tensor(buf6, (1, 512, 4), (2048, 1, 512), 0); del buf6  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf5, buf11, 2048, 100, grid=grid(2048), stream=stream0)
        buf12 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf11, buf12, 512, 4, grid=grid(512), stream=stream0)
        buf13 = reinterpret_tensor(buf9, (8, 50, 2048), (102400, 2048, 1), 0); del buf9  # reuse
        # Source Nodes: [x_154], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf13, addmm_30, 819200, grid=grid(819200), stream=stream0)
        del addmm_30
        buf14 = empty((400, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (400, 2048), (2048, 1), 0), permute_105, out=buf14)
        del permute_105
        buf15 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (2048, 400), (1, 2048), 0), view_163, out=buf15)
        del view_163
        buf16 = empty_strided((1, 2048, 4), (8192, 1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf13, buf16, 8192, 100, grid=grid(8192), stream=stream0)
        buf17 = reinterpret_tensor(buf11, (1, 2048), (2048, 1), 0); del buf11  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf16, buf17, 2048, 4, grid=grid(2048), stream=stream0)
        buf24 = buf5; del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf24, buf14, primals_35, mul_75, div_9, 400, 512, grid=grid(400), stream=stream0)
        del div_9
        del primals_35
        buf20 = empty_strided((512, 4), (1, 512), device='cuda', dtype=torch.float32)
        buf22 = empty_strided((512, 4), (1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf14, mul_75, buf20, buf22, 2048, 100, grid=grid(2048), stream=stream0)
        del mul_75
        buf21 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf20, buf21, 512, 4, grid=grid(512), stream=stream0)
        buf23 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf22, buf23, 512, 4, grid=grid(512), stream=stream0)
        buf25 = buf14; del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf24, (400, 512), (512, 1), 0), permute_109, out=buf25)
        del permute_109
        buf26 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf24, (512, 400), (1, 512), 0), view_161, out=buf26)
        del view_161
        buf27 = reinterpret_tensor(buf22, (1, 512, 4), (2048, 1, 512), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf24, buf27, 2048, 100, grid=grid(2048), stream=stream0)
        buf28 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf27, buf28, 512, 4, grid=grid(512), stream=stream0)
        buf29 = empty((8, 8, 49, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul]
        triton_poi_fused_constant_pad_nd_mul_11.run(buf25, slice_101, buf29, 200704, grid=grid(200704), stream=stream0)
        del slice_101
        buf30 = empty_strided((8, 128, 7, 7), (6272, 1, 896, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.slice]
        triton_poi_fused_slice_12.run(buf29, buf30, 1024, 49, grid=grid(1024, 49), stream=stream0)
        buf31 = empty_strided((8, 192, 7, 7), (9408, 1, 1344, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.slice]
        triton_poi_fused_slice_13.run(buf29, buf31, 1536, 49, grid=grid(1536, 49), stream=stream0)
        buf32 = empty_strided((8, 192, 7, 7), (9408, 1, 1344, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.slice]
        triton_poi_fused_slice_14.run(buf29, buf32, 1536, 49, grid=grid(1536, 49), stream=stream0)
        del buf29
        buf33 = empty_strided((192, 4), (1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_15.run(buf32, buf33, 768, 98, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf35 = aten.convolution_backward(buf32, getitem_85, primals_135, [192], [1, 1], [3, 3], [1, 1], False, [0, 0], 192, [True, True, False])
        del buf32
        del getitem_85
        buf36 = buf35[0]
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf40 = aten.convolution_backward(buf31, getitem_84, primals_133, [192], [1, 1], [2, 2], [1, 1], False, [0, 0], 192, [True, True, False])
        del getitem_84
        buf41 = buf40[0]
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf45 = aten.convolution_backward(buf30, getitem_83, primals_131, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, False])
        del getitem_83
        buf46 = buf45[0]
        buf48 = empty((8, 8, 50, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_16.run(buf25, buf48, 204800, grid=grid(204800), stream=stream0)
        buf50 = empty((64, 50, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf48, (64, 50, 64), (3200, 64, 1), 0), permute_117, out=buf50)
        del permute_117
        buf49 = empty((64, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_116, reinterpret_tensor(buf48, (64, 50, 64), (3200, 64, 1), 0), out=buf49)
        del permute_116
        buf51 = reinterpret_tensor(buf48, (64, 50, 64), (3200, 64, 1), 0); del buf48  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_118, reinterpret_tensor(buf49, (64, 64, 64), (4096, 64, 1), 0), out=buf51)
        del permute_118
        buf52 = empty((64, 64, 50), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf49, (64, 64, 64), (4096, 64, 1), 0), permute_119, out=buf52)
        del permute_119
        buf53 = reinterpret_tensor(buf0, (8, 8, 1, 64), (512, 64, 4096, 1), 0); del buf0  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_17.run(buf52, alias_8, buf53, 4096, 50, grid=grid(4096), stream=stream0)
        buf54 = empty((24, 8, 50, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_18.run(buf25, cat_19, buf50, buf52, alias_8, buf53, buf46, buf41, buf36, buf51, buf54, 9600, 64, grid=grid(9600, 64), stream=stream0)
        del alias_8
        del buf36
        del buf46
        del cat_19
        buf55 = empty((400, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf54, buf55, 614400, grid=grid(614400), stream=stream0)
        buf56 = reinterpret_tensor(buf52, (400, 512), (512, 1), 0); del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf55, permute_122, out=buf56)
        del permute_122
        buf66 = buf24; del buf24  # reuse
        # Source Nodes: [cur_28], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_20.run(buf66, buf56, primals_33, cat_18, getitem_79, rsqrt_18, 400, 512, grid=grid(400), stream=stream0)
        del primals_33
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf69 = aten.convolution_backward(reinterpret_tensor(buf66, (8, 512, 7, 7), (25600, 1, 3584, 512), 512), view_147, primals_127, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 512, [True, True, False])
        del view_147
        buf70 = buf69[0]
        buf72 = reinterpret_tensor(buf51, (8, 50, 512), (25600, 512, 1), 0); del buf51  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward]
        triton_poi_fused_add_select_backward_slice_backward_21.run(buf66, buf70, buf72, 204800, grid=grid(204800), stream=stream0)
        buf73 = reinterpret_tensor(buf13, (400, 2048), (2048, 1), 0); del buf13  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf72, (400, 512), (512, 1), 0), permute_128, out=buf73)
        del permute_128
        buf77 = reinterpret_tensor(buf73, (8, 50, 2048), (102400, 2048, 1), 0); del buf73  # reuse
        # Source Nodes: [x_136], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf77, addmm_26, 819200, grid=grid(819200), stream=stream0)
        del addmm_26
        buf78 = reinterpret_tensor(buf50, (400, 512), (512, 1), 0); del buf50  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf77, (400, 2048), (2048, 1), 0), permute_132, out=buf78)
        del permute_132
        buf88 = reinterpret_tensor(buf25, (8, 50, 512), (25600, 512, 1), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_22.run(buf78, primals_31, mul_66, buf72, div_11, buf88, 400, 512, grid=grid(400), stream=stream0)
        del div_11
        del primals_31
        buf89 = empty((400, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf88, (400, 512), (512, 1), 0), permute_136, out=buf89)
        del permute_136
        buf93 = reinterpret_tensor(buf70, (8, 8, 49, 64), (25088, 3136, 64, 1), 0); del buf70  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul]
        triton_poi_fused_constant_pad_nd_mul_11.run(buf89, slice_89, buf93, 200704, grid=grid(200704), stream=stream0)
        del slice_89
        buf96 = reinterpret_tensor(buf41, (8, 192, 7, 7), (9408, 1, 1344, 192), 0); del buf41  # reuse
        # Source Nodes: [], Original ATen: [aten.slice]
        triton_poi_fused_slice_14.run(buf93, buf96, 1536, 49, grid=grid(1536, 49), stream=stream0)
        buf97 = empty_strided((192, 4), (1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_15.run(buf96, buf97, 768, 98, grid=grid(768), stream=stream0)
        buf34 = empty((192, ), device='cuda', dtype=torch.float32)
        buf103 = buf34; del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward]
        triton_per_fused_add_convolution_backward_23.run(buf103, buf33, buf97, 192, 4, grid=grid(192), stream=stream0)
        buf37 = buf35[1]
        del buf35
        buf38 = buf97; del buf97  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_24.run(buf31, buf38, 768, 98, grid=grid(768), stream=stream0)
        buf95 = buf31; del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten.slice]
        triton_poi_fused_slice_13.run(buf93, buf95, 1536, 49, grid=grid(1536, 49), stream=stream0)
        buf104 = buf33; del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_15.run(buf95, buf104, 768, 98, grid=grid(768), stream=stream0)
        buf105 = empty((192, ), device='cuda', dtype=torch.float32)
        buf110 = buf105; del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward]
        triton_per_fused_add_convolution_backward_23.run(buf110, buf38, buf104, 192, 4, grid=grid(192), stream=stream0)
        del buf104
        del buf38
        buf42 = buf40[1]
        del buf40
        buf43 = empty_strided((128, 4), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_25.run(buf30, buf43, 512, 98, grid=grid(512), stream=stream0)
        buf94 = buf30; del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.slice]
        triton_poi_fused_slice_12.run(buf93, buf94, 1024, 49, grid=grid(1024, 49), stream=stream0)
        buf111 = empty_strided((128, 4), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_26.run(buf94, buf111, 512, 98, grid=grid(512), stream=stream0)
        buf112 = empty((128, ), device='cuda', dtype=torch.float32)
        buf117 = buf112; del buf112  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward]
        triton_per_fused_add_convolution_backward_27.run(buf117, buf43, buf111, 128, 4, grid=grid(128), stream=stream0)
        buf47 = buf45[1]
        del buf45
        buf57 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf55, (1536, 400), (1, 1536), 0), view_149, out=buf57)
        del view_149
        buf58 = empty_strided((1, 1536, 4), (6144, 1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf55, buf58, 6144, 100, grid=grid(6144), stream=stream0)
        buf59 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_29.run(buf58, buf59, 1536, 4, grid=grid(1536), stream=stream0)
        buf62 = reinterpret_tensor(buf27, (512, 4), (1, 512), 0); del buf27  # reuse
        buf64 = buf20; del buf20  # reuse
        # Source Nodes: [cur_28], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_30.run(buf56, cat_18, getitem_79, rsqrt_18, buf62, buf64, 2048, 100, grid=grid(2048), stream=stream0)
        del cat_18
        del getitem_79
        del rsqrt_18
        buf63 = reinterpret_tensor(buf43, (512, ), (1, ), 0); del buf43  # reuse
        # Source Nodes: [cur_28], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf62, buf63, 512, 4, grid=grid(512), stream=stream0)
        buf65 = reinterpret_tensor(buf111, (512, ), (1, ), 0); del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf64, buf65, 512, 4, grid=grid(512), stream=stream0)
        buf67 = buf64; del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_31.run(buf66, buf67, 2048, 98, grid=grid(2048), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf99 = aten.convolution_backward(buf96, getitem_75, primals_135, [192], [1, 1], [3, 3], [1, 1], False, [0, 0], 192, [True, True, False])
        del buf96
        del getitem_75
        del primals_135
        buf100 = buf99[0]
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf106 = aten.convolution_backward(buf95, getitem_74, primals_133, [192], [1, 1], [2, 2], [1, 1], False, [0, 0], 192, [True, True, False])
        del buf95
        del getitem_74
        del primals_133
        buf107 = buf106[0]
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf113 = aten.convolution_backward(buf94, getitem_73, primals_131, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, False])
        del buf94
        del getitem_73
        del primals_131
        buf114 = buf113[0]
        buf118 = reinterpret_tensor(buf66, (8, 8, 50, 64), (25600, 3200, 64, 1), 0); del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_16.run(buf89, buf118, 204800, grid=grid(204800), stream=stream0)
        buf120 = reinterpret_tensor(buf56, (64, 50, 64), (3200, 64, 1), 0); del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf118, (64, 50, 64), (3200, 64, 1), 0), permute_144, out=buf120)
        del permute_144
        buf119 = buf49; del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_143, reinterpret_tensor(buf118, (64, 50, 64), (3200, 64, 1), 0), out=buf119)
        del permute_143
        buf121 = reinterpret_tensor(buf118, (64, 50, 64), (3200, 64, 1), 0); del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_145, reinterpret_tensor(buf119, (64, 64, 64), (4096, 64, 1), 0), out=buf121)
        del permute_145
        buf122 = empty((64, 64, 50), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf119, (64, 64, 64), (4096, 64, 1), 0), permute_146, out=buf122)
        del permute_146
        buf123 = buf53; del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_17.run(buf122, alias_9, buf123, 4096, 50, grid=grid(4096), stream=stream0)
        buf124 = reinterpret_tensor(buf55, (24, 8, 50, 64), (25600, 3200, 64, 1), 0); del buf55  # reuse
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_18.run(buf89, cat_17, buf120, buf122, alias_9, buf123, buf114, buf107, buf100, buf121, buf124, 9600, 64, grid=grid(9600, 64), stream=stream0)
        del alias_9
        del buf100
        del buf107
        del buf114
        del buf120
        del buf121
        del cat_17
        buf125 = reinterpret_tensor(buf54, (400, 1536), (1536, 1), 0); del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_19.run(buf124, buf125, 614400, grid=grid(614400), stream=stream0)
        del buf124
        buf126 = buf89; del buf89  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf125, permute_149, out=buf126)
        del permute_149
        buf136 = reinterpret_tensor(buf122, (8, 50, 512), (25600, 512, 1), 0); del buf122  # reuse
        # Source Nodes: [cur_24], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_32.run(buf126, primals_29, cat_16, getitem_69, rsqrt_16, buf88, buf136, 400, 512, grid=grid(400), stream=stream0)
        del primals_29
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf139 = aten.convolution_backward(reinterpret_tensor(buf136, (8, 512, 7, 7), (25600, 1, 3584, 512), 512), view_127, primals_127, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 512, [True, True, False])
        del primals_127
        del view_127
        buf140 = buf139[0]
        buf137 = buf62; del buf62  # reuse
        buf148 = empty_strided((512, 4), (1, 512), device='cuda', dtype=torch.float32)
        buf150 = empty_strided((512, 4), (1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_layer_norm_backward]
        triton_red_fused_convolution_backward_native_layer_norm_backward_33.run(buf136, buf140, mul_60, buf137, buf148, buf150, 2048, 98, grid=grid(2048), stream=stream0)
        buf138 = empty((512, ), device='cuda', dtype=torch.float32)
        buf143 = buf138; del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward]
        triton_per_fused_add_convolution_backward_34.run(buf143, buf67, buf137, 512, 4, grid=grid(512), stream=stream0)
        buf71 = buf69[1]
        del buf69
        buf74 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf72, (512, 400), (1, 512), 0), view_145, out=buf74)
        del view_145
        buf75 = reinterpret_tensor(buf67, (1, 512, 4), (2048, 1, 512), 0); del buf67  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_35.run(buf72, buf75, 2048, 100, grid=grid(2048), stream=stream0)
        del buf72
        buf76 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf75, buf76, 512, 4, grid=grid(512), stream=stream0)
        buf79 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf77, (2048, 400), (1, 2048), 0), view_143, out=buf79)
        del view_143
        buf80 = buf16; del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf77, buf80, 8192, 100, grid=grid(8192), stream=stream0)
        del buf77
        buf81 = reinterpret_tensor(buf75, (1, 2048), (2048, 1), 0); del buf75  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf80, buf81, 2048, 4, grid=grid(2048), stream=stream0)
        del buf80
        buf84 = buf137; del buf137  # reuse
        buf86 = empty_strided((512, 4), (1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf78, mul_66, buf84, buf86, 2048, 100, grid=grid(2048), stream=stream0)
        del buf78
        del mul_66
        buf85 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf84, buf85, 512, 4, grid=grid(512), stream=stream0)
        buf87 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf86, buf87, 512, 4, grid=grid(512), stream=stream0)
        buf90 = reinterpret_tensor(buf119, (512, 512), (512, 1), 0); del buf119  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf88, (512, 400), (1, 512), 0), view_141, out=buf90)
        del view_141
        buf91 = reinterpret_tensor(buf86, (1, 512, 4), (2048, 1, 512), 0); del buf86  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_35.run(buf88, buf91, 2048, 100, grid=grid(2048), stream=stream0)
        del buf88
        buf92 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf91, buf92, 512, 4, grid=grid(512), stream=stream0)
        buf101 = buf99[1]
        del buf99
        buf102 = buf101; del buf101  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_36.run(buf102, buf37, 9408, grid=grid(9408), stream=stream0)
        del buf37
        buf108 = buf106[1]
        del buf106
        buf109 = buf108; del buf108  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_37.run(buf109, buf42, 4800, grid=grid(4800), stream=stream0)
        del buf42
        buf115 = buf113[1]
        del buf113
        buf116 = buf115; del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_38.run(buf116, buf47, 1152, grid=grid(1152), stream=stream0)
        del buf47
        buf127 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (1536, 400), (1, 1536), 0), view_129, out=buf127)
        del view_129
        buf128 = buf58; del buf58  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf125, buf128, 6144, 100, grid=grid(6144), stream=stream0)
        del buf125
        buf129 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_29.run(buf128, buf129, 1536, 4, grid=grid(1536), stream=stream0)
        del buf128
        buf132 = reinterpret_tensor(buf91, (512, 4), (1, 512), 0); del buf91  # reuse
        buf134 = buf84; del buf84  # reuse
        # Source Nodes: [cur_24], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_30.run(buf126, cat_16, getitem_69, rsqrt_16, buf132, buf134, 2048, 100, grid=grid(2048), stream=stream0)
        del buf126
        del cat_16
        del getitem_69
        del rsqrt_16
        buf133 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [cur_24], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf132, buf133, 512, 4, grid=grid(512), stream=stream0)
        del buf132
        buf135 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf134, buf135, 512, 4, grid=grid(512), stream=stream0)
        del buf134
        buf141 = buf139[1]
        del buf139
        buf142 = buf141; del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_39.run(buf142, buf71, 4608, grid=grid(4608), stream=stream0)
        del buf71
        buf144 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf136, buf140, buf144, 512, 8, grid=grid(512), stream=stream0)
        buf145 = empty_strided((8, 49, 1, 4), (196, 4, 1568, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_41.run(buf136, buf140, primals_125, buf145, 1568, 128, grid=grid(1568), stream=stream0)
        buf146 = empty_strided((8, 49, 1), (49, 1, 392), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_42.run(buf145, buf146, 392, 4, grid=grid(392), stream=stream0)
        buf152 = reinterpret_tensor(buf93, (8, 49, 512), (25088, 512, 1), 0); del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_43.run(buf136, buf140, primals_125, mul_60, div_13, buf146, buf152, 392, 512, grid=grid(392), stream=stream0)
        del buf136
        del buf140
        del buf146
        del div_13
        del mul_60
        del primals_125
        buf149 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf148, buf149, 512, 4, grid=grid(512), stream=stream0)
        del buf148
        buf151 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf150, buf151, 512, 4, grid=grid(512), stream=stream0)
        buf153 = buf150; del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_44.run(buf152, buf153, 2048, 98, grid=grid(2048), stream=stream0)
        buf154 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_3.run(buf153, buf154, 512, 4, grid=grid(512), stream=stream0)
        del buf153
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf155 = aten.convolution_backward(reinterpret_tensor(buf152, (8, 512, 7, 7), (25088, 1, 3584, 512), 0), clone_47, primals_123, [512], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del clone_47
        del primals_123
        buf156 = buf155[0]
        buf157 = buf155[1]
        del buf155
        buf158 = empty((1576, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.slice_backward, aten.view]
        triton_poi_fused_slice_backward_view_45.run(buf156, buf158, 504320, grid=grid(504320), stream=stream0)
        buf159 = empty((1576, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf158, permute_157, out=buf159)
        del permute_157
        buf160 = empty((320, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf158, (320, 1576), (1, 320), 0), view_123, out=buf160)
        del view_123
        buf161 = empty_strided((1, 320, 13), (4160, 1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_46.run(buf158, buf161, 4160, 122, grid=grid(4160), stream=stream0)
        buf162 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_47.run(buf161, buf162, 320, 13, grid=grid(320), stream=stream0)
        buf163 = reinterpret_tensor(buf159, (8, 197, 1280), (252160, 1280, 1), 0); del buf159  # reuse
        # Source Nodes: [x_114], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_48.run(buf163, addmm_22, 2017280, grid=grid(2017280), stream=stream0)
        del addmm_22
        buf164 = buf158; del buf158  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (1576, 1280), (1280, 1), 0), permute_161, out=buf164)
        del permute_161
        buf165 = empty((1280, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (1280, 1576), (1, 1280), 0), view_121, out=buf165)
        del view_121
        buf166 = empty_strided((1, 1280, 13), (16640, 1, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_49.run(buf163, buf166, 16640, 122, grid=grid(16640), stream=stream0)
        buf167 = empty((1, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_50.run(buf166, buf167, 1280, 13, grid=grid(1280), stream=stream0)
        buf174 = empty((8, 197, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.slice_backward]
        triton_per_fused_add_native_layer_norm_backward_slice_backward_51.run(buf164, primals_26, mul_55, buf156, div_14, buf174, 1576, 320, grid=grid(1576), stream=stream0)
        del div_14
        del primals_26
        buf170 = reinterpret_tensor(buf161, (320, 13), (1, 320), 0); del buf161  # reuse
        buf172 = empty_strided((320, 13), (1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_52.run(buf164, mul_55, buf170, buf172, 4160, 122, grid=grid(4160), stream=stream0)
        del mul_55
        buf171 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_47.run(buf170, buf171, 320, 13, grid=grid(320), stream=stream0)
        buf173 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_47.run(buf172, buf173, 320, 13, grid=grid(320), stream=stream0)
        buf175 = buf164; del buf164  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf174, (1576, 320), (320, 1), 0), permute_165, out=buf175)
        del permute_165
        buf176 = empty((320, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf174, (320, 1576), (1, 320), 0), view_119, out=buf176)
        del view_119
        buf177 = reinterpret_tensor(buf172, (1, 320, 13), (4160, 1, 320), 0); del buf172  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_53.run(buf174, buf177, 4160, 122, grid=grid(4160), stream=stream0)
        buf178 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_47.run(buf177, buf178, 320, 13, grid=grid(320), stream=stream0)
        buf179 = reinterpret_tensor(buf156, (8, 8, 196, 40), (62720, 7840, 40, 1), 0); del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul]
        triton_poi_fused_constant_pad_nd_mul_54.run(buf175, slice_74, buf179, 501760, grid=grid(501760), stream=stream0)
        del slice_74
        buf180 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.slice]
        triton_poi_fused_slice_55.run(buf179, buf180, 640, 196, grid=grid(640, 196), stream=stream0)
        buf181 = empty_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.slice]
        triton_poi_fused_slice_56.run(buf179, buf181, 960, 196, grid=grid(960, 196), stream=stream0)
        buf182 = empty_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.slice]
        triton_poi_fused_slice_57.run(buf179, buf182, 960, 196, grid=grid(960, 196), stream=stream0)
        del buf179
        buf183 = empty_strided((120, 13), (1, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_58.run(buf182, buf183, 1560, 121, grid=grid(1560), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf185 = aten.convolution_backward(buf182, getitem_63, primals_107, [120], [1, 1], [3, 3], [1, 1], False, [0, 0], 120, [True, True, False])
        del buf182
        del getitem_63
        buf186 = buf185[0]
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf190 = aten.convolution_backward(buf181, getitem_62, primals_105, [120], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
        del getitem_62
        buf191 = buf190[0]
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf195 = aten.convolution_backward(buf180, getitem_61, primals_103, [80], [1, 1], [1, 1], [1, 1], False, [0, 0], 80, [True, True, False])
        del getitem_61
        buf196 = buf195[0]
        buf198 = empty((8, 8, 197, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_59.run(buf175, buf198, 504320, grid=grid(504320), stream=stream0)
        buf200 = empty((64, 197, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf198, (64, 197, 40), (7880, 40, 1), 0), permute_173, out=buf200)
        del permute_173
        buf199 = empty((64, 40, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_172, reinterpret_tensor(buf198, (64, 197, 40), (7880, 40, 1), 0), out=buf199)
        del permute_172
        buf201 = reinterpret_tensor(buf198, (64, 197, 40), (7880, 40, 1), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_174, reinterpret_tensor(buf199, (64, 40, 40), (1600, 40, 1), 0), out=buf201)
        del permute_174
        buf202 = empty((64, 40, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf199, (64, 40, 40), (1600, 40, 1), 0), permute_175, out=buf202)
        del permute_175
        buf203 = empty_strided((8, 8, 1, 40, 2), (640, 80, 5120, 2, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_red_fused__softmax_backward_data_60.run(buf202, alias_10, buf203, 5120, 99, grid=grid(5120), stream=stream0)
        buf204 = empty_strided((8, 8, 1, 40), (320, 40, 2560, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_61.run(buf203, buf204, 2560, 2, grid=grid(2560), stream=stream0)
        buf205 = empty((24, 8, 197, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_62.run(buf175, cat_14, buf200, buf202, alias_10, buf204, buf196, buf191, buf186, buf201, buf205, 37824, 40, grid=grid(37824, 40), stream=stream0)
        del alias_10
        del buf186
        del buf196
        del cat_14
        buf206 = empty((1576, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_63.run(buf205, buf206, 1512960, grid=grid(1512960), stream=stream0)
        buf207 = reinterpret_tensor(buf202, (1576, 320), (320, 1), 0); del buf202  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf206, permute_178, out=buf207)
        del permute_178
        buf217 = buf174; del buf174  # reuse
        # Source Nodes: [cur_20], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_64.run(buf217, buf207, primals_24, cat_13, getitem_57, rsqrt_13, 1576, 320, grid=grid(1576), stream=stream0)
        del primals_24
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf220 = aten.convolution_backward(reinterpret_tensor(buf217, (8, 320, 14, 14), (63040, 1, 4480, 320), 320), view_105, primals_99, [320], [1, 1], [1, 1], [1, 1], False, [0, 0], 320, [True, True, False])
        del view_105
        buf221 = buf220[0]
        buf223 = reinterpret_tensor(buf201, (8, 197, 320), (63040, 320, 1), 0); del buf201  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.slice_backward]
        triton_poi_fused_add_slice_backward_65.run(buf217, buf221, buf223, 504320, grid=grid(504320), stream=stream0)
        buf224 = reinterpret_tensor(buf163, (1576, 1280), (1280, 1), 0); del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf223, (1576, 320), (320, 1), 0), permute_184, out=buf224)
        del permute_184
        buf228 = reinterpret_tensor(buf224, (8, 197, 1280), (252160, 1280, 1), 0); del buf224  # reuse
        # Source Nodes: [x_96], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_48.run(buf228, addmm_18, 2017280, grid=grid(2017280), stream=stream0)
        del addmm_18
        buf229 = reinterpret_tensor(buf200, (1576, 320), (320, 1), 0); del buf200  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf228, (1576, 1280), (1280, 1), 0), permute_188, out=buf229)
        del permute_188
        buf239 = reinterpret_tensor(buf175, (8, 197, 320), (63040, 320, 1), 0); del buf175  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_66.run(buf229, primals_22, mul_46, buf223, div_16, buf239, 1576, 320, grid=grid(1576), stream=stream0)
        del div_16
        del primals_22
        buf240 = empty((1576, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf239, (1576, 320), (320, 1), 0), permute_192, out=buf240)
        del permute_192
        buf244 = reinterpret_tensor(buf221, (8, 8, 196, 40), (62720, 7840, 40, 1), 0); del buf221  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul]
        triton_poi_fused_constant_pad_nd_mul_54.run(buf240, slice_62, buf244, 501760, grid=grid(501760), stream=stream0)
        del slice_62
        buf247 = reinterpret_tensor(buf191, (8, 120, 14, 14), (23520, 1, 1680, 120), 0); del buf191  # reuse
        # Source Nodes: [], Original ATen: [aten.slice]
        triton_poi_fused_slice_57.run(buf244, buf247, 960, 196, grid=grid(960, 196), stream=stream0)
        buf248 = empty_strided((120, 13), (1, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_58.run(buf247, buf248, 1560, 121, grid=grid(1560), stream=stream0)
        buf184 = empty((120, ), device='cuda', dtype=torch.float32)
        buf254 = buf184; del buf184  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward]
        triton_per_fused_add_convolution_backward_67.run(buf254, buf183, buf248, 120, 13, grid=grid(120), stream=stream0)
        buf187 = buf185[1]
        del buf185
        buf188 = buf248; del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_68.run(buf181, buf188, 1560, 121, grid=grid(1560), stream=stream0)
        buf246 = buf181; del buf181  # reuse
        # Source Nodes: [], Original ATen: [aten.slice]
        triton_poi_fused_slice_56.run(buf244, buf246, 960, 196, grid=grid(960, 196), stream=stream0)
        buf255 = buf183; del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_58.run(buf246, buf255, 1560, 121, grid=grid(1560), stream=stream0)
        buf189 = empty((120, ), device='cuda', dtype=torch.float32)
        buf261 = buf189; del buf189  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward]
        triton_per_fused_add_convolution_backward_67.run(buf261, buf188, buf255, 120, 13, grid=grid(120), stream=stream0)
        del buf188
        del buf255
        buf192 = buf190[1]
        del buf190
        buf193 = empty_strided((80, 13), (1, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_69.run(buf180, buf193, 1040, 121, grid=grid(1040), stream=stream0)
        buf245 = buf180; del buf180  # reuse
        # Source Nodes: [], Original ATen: [aten.slice]
        triton_poi_fused_slice_55.run(buf244, buf245, 640, 196, grid=grid(640, 196), stream=stream0)
        buf262 = empty_strided((80, 13), (1, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_70.run(buf245, buf262, 1040, 121, grid=grid(1040), stream=stream0)
        buf194 = empty((80, ), device='cuda', dtype=torch.float32)
        buf268 = buf194; del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward]
        triton_per_fused_add_convolution_backward_71.run(buf268, buf193, buf262, 80, 13, grid=grid(80), stream=stream0)
        del buf193
        del buf262
        buf197 = buf195[1]
        del buf195
        buf208 = empty((960, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf206, (960, 1576), (1, 960), 0), view_107, out=buf208)
        del view_107
        buf209 = empty_strided((1, 960, 13), (12480, 1, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_72.run(buf206, buf209, 12480, 122, grid=grid(12480), stream=stream0)
        buf210 = empty((1, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_73.run(buf209, buf210, 960, 13, grid=grid(960), stream=stream0)
        buf213 = reinterpret_tensor(buf177, (320, 13), (1, 320), 0); del buf177  # reuse
        buf215 = buf170; del buf170  # reuse
        # Source Nodes: [cur_20], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_74.run(buf207, cat_13, getitem_57, rsqrt_13, buf213, buf215, 4160, 122, grid=grid(4160), stream=stream0)
        del cat_13
        del getitem_57
        del rsqrt_13
        buf214 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [cur_20], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_sum_47.run(buf213, buf214, 320, 13, grid=grid(320), stream=stream0)
        buf216 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_47.run(buf215, buf216, 320, 13, grid=grid(320), stream=stream0)
        buf218 = buf215; del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_75.run(buf217, buf218, 4160, 121, grid=grid(4160), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf250 = aten.convolution_backward(buf247, getitem_53, primals_107, [120], [1, 1], [3, 3], [1, 1], False, [0, 0], 120, [True, True, False])
        del buf247
        del getitem_53
        del primals_107
        buf251 = buf250[0]
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf257 = aten.convolution_backward(buf246, getitem_52, primals_105, [120], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
        del buf246
        del getitem_52
        del primals_105
        buf258 = buf257[0]
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf264 = aten.convolution_backward(buf245, getitem_51, primals_103, [80], [1, 1], [1, 1], [1, 1], False, [0, 0], 80, [True, True, False])
        del buf245
        del getitem_51
        del primals_103
        buf265 = buf264[0]
        buf269 = reinterpret_tensor(buf217, (8, 8, 197, 40), (63040, 7880, 40, 1), 0); del buf217  # reuse
        # Source Nodes: [], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_59.run(buf240, buf269, 504320, grid=grid(504320), stream=stream0)
        buf271 = reinterpret_tensor(buf207, (64, 197, 40), (7880, 40, 1), 0); del buf207  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf269, (64, 197, 40), (7880, 40, 1), 0), permute_200, out=buf271)
        del permute_200
        buf270 = buf199; del buf199  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_199, reinterpret_tensor(buf269, (64, 197, 40), (7880, 40, 1), 0), out=buf270)
        del permute_199
        buf272 = reinterpret_tensor(buf269, (64, 197, 40), (7880, 40, 1), 0); del buf269  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_201, reinterpret_tensor(buf270, (64, 40, 40), (1600, 40, 1), 0), out=buf272)
        del permute_201
        buf273 = empty((64, 40, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf270, (64, 40, 40), (1600, 40, 1), 0), permute_202, out=buf273)
        del permute_202
        buf274 = buf203; del buf203  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_red_fused__softmax_backward_data_60.run(buf273, alias_11, buf274, 5120, 99, grid=grid(5120), stream=stream0)
        buf275 = buf204; del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_61.run(buf274, buf275, 2560, 2, grid=grid(2560), stream=stream0)
        del buf274
        buf276 = reinterpret_tensor(buf206, (24, 8, 197, 40), (63040, 7880, 40, 1), 0); del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_76.run(buf240, cat_12, buf271, buf273, alias_11, buf275, buf265, buf258, buf251, buf272, buf276, 192, 7880, grid=grid(192, 7880), stream=stream0)
        del alias_11
        del buf240
        del buf251
        del buf258
        del buf265
        del buf271
        del buf275
        del cat_12
        buf277 = reinterpret_tensor(buf205, (1576, 960), (960, 1), 0); del buf205  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_63.run(buf276, buf277, 1512960, grid=grid(1512960), stream=stream0)
        del buf276
        buf278 = reinterpret_tensor(buf273, (1576, 320), (320, 1), 0); del buf273  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf277, permute_205, out=buf278)
        del permute_205
        buf288 = reinterpret_tensor(buf272, (8, 197, 320), (63040, 320, 1), 0); del buf272  # reuse
        # Source Nodes: [cur_16], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_77.run(buf278, primals_20, cat_11, getitem_47, rsqrt_11, buf239, buf288, 1576, 320, grid=grid(1576), stream=stream0)
        del primals_20
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf291 = aten.convolution_backward(reinterpret_tensor(buf288, (8, 320, 14, 14), (63040, 1, 4480, 320), 320), view_85, primals_99, [320], [1, 1], [1, 1], [1, 1], False, [0, 0], 320, [True, True, False])
        del primals_99
        del view_85
        buf292 = buf291[0]
        buf289 = buf213; del buf213  # reuse
        buf300 = empty_strided((320, 13), (1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_layer_norm_backward]
        triton_red_fused_convolution_backward_native_layer_norm_backward_78.run(buf288, buf292, mul_40, buf289, buf300, 4160, 121, grid=grid(4160), stream=stream0)
        buf219 = empty((320, ), device='cuda', dtype=torch.float32)
        buf295 = buf219; del buf219  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward]
        triton_per_fused_add_convolution_backward_79.run(buf295, buf218, buf289, 320, 13, grid=grid(320), stream=stream0)
        buf222 = buf220[1]
        del buf220
        buf225 = empty((320, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf223, (320, 1576), (1, 320), 0), view_103, out=buf225)
        del view_103
        buf226 = reinterpret_tensor(buf289, (1, 320, 13), (4160, 1, 320), 0); del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_46.run(buf223, buf226, 4160, 122, grid=grid(4160), stream=stream0)
        del buf223
        buf227 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_47.run(buf226, buf227, 320, 13, grid=grid(320), stream=stream0)
        buf230 = empty((1280, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf228, (1280, 1576), (1, 1280), 0), view_101, out=buf230)
        del view_101
        buf231 = buf166; del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_49.run(buf228, buf231, 16640, 122, grid=grid(16640), stream=stream0)
        del buf228
        buf232 = empty((1, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_50.run(buf231, buf232, 1280, 13, grid=grid(1280), stream=stream0)
        del buf231
        buf235 = reinterpret_tensor(buf226, (320, 13), (1, 320), 0); del buf226  # reuse
        buf237 = buf218; del buf218  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_52.run(buf229, mul_46, buf235, buf237, 4160, 122, grid=grid(4160), stream=stream0)
        del buf229
        del mul_46
        buf236 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_47.run(buf235, buf236, 320, 13, grid=grid(320), stream=stream0)
        buf238 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_47.run(buf237, buf238, 320, 13, grid=grid(320), stream=stream0)
        buf241 = reinterpret_tensor(buf270, (320, 320), (320, 1), 0); del buf270  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf239, (320, 1576), (1, 320), 0), view_99, out=buf241)
        del view_99
        buf242 = reinterpret_tensor(buf237, (1, 320, 13), (4160, 1, 320), 0); del buf237  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_46.run(buf239, buf242, 4160, 122, grid=grid(4160), stream=stream0)
        del buf239
        buf243 = empty((1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_47.run(buf242, buf243, 320, 13, grid=grid(320), stream=stream0)
        buf252 = buf250[1]
        del buf250
        buf253 = buf187; del buf187  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_80.run(buf253, buf252, 5880, grid=grid(5880), stream=stream0)
        del buf252
        buf259 = buf257[1]
        del buf257
        buf260 = buf192; del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(buf260, buf259, 3000, grid=grid(3000), stream=stream0)
        del buf259
        buf266 = buf264[1]
        del buf264
        buf267 = buf197; del buf197  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_82.run(buf267, buf266, 720, grid=grid(720), stream=stream0)
        del buf266
        buf279 = empty((960, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf277, (960, 1576), (1, 960), 0), view_87, out=buf279)
        del view_87
        buf280 = buf209; del buf209  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_72.run(buf277, buf280, 12480, 122, grid=grid(12480), stream=stream0)
        del buf277
        buf281 = empty((1, 960), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_73.run(buf280, buf281, 960, 13, grid=grid(960), stream=stream0)
        del buf280
        buf284 = reinterpret_tensor(buf242, (320, 13), (1, 320), 0); del buf242  # reuse
        buf286 = buf235; del buf235  # reuse
        # Source Nodes: [cur_16], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_74.run(buf278, cat_11, getitem_47, rsqrt_11, buf284, buf286, 4160, 122, grid=grid(4160), stream=stream0)
        del buf278
        del cat_11
        del getitem_47
        del rsqrt_11
        buf285 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [cur_16], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_sum_47.run(buf284, buf285, 320, 13, grid=grid(320), stream=stream0)
        del buf284
        buf287 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_47.run(buf286, buf287, 320, 13, grid=grid(320), stream=stream0)
        del buf286
        buf293 = buf291[1]
        del buf291
        buf294 = buf222; del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_83.run(buf294, buf293, 2880, grid=grid(2880), stream=stream0)
        del buf293
        buf296 = empty((1, 1, 320), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_84.run(buf288, buf292, buf296, 320, 8, grid=grid(320), stream=stream0)
        buf297 = empty_strided((8, 196, 1, 3), (588, 3, 4704, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_85.run(buf288, buf292, primals_97, buf297, 4704, 107, grid=grid(4704), stream=stream0)
        buf298 = reinterpret_tensor(buf145, (8, 196, 1), (196, 1, 1568), 0); del buf145  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_86.run(buf297, buf298, 1568, 3, grid=grid(1568), stream=stream0)
        buf304 = reinterpret_tensor(buf244, (8, 196, 320), (62720, 320, 1), 0); del buf244  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_87.run(buf288, buf292, primals_97, mul_40, div_18, buf298, buf304, 1568, 320, grid=grid(1568), stream=stream0)
        del div_18
        del mul_40
        del primals_97
        buf301 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_47.run(buf300, buf301, 320, 13, grid=grid(320), stream=stream0)
        buf302 = reinterpret_tensor(buf300, (320, 13), (13, 1), 0); del buf300  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_88.run(buf288, buf292, buf302, 4160, 121, grid=grid(4160), stream=stream0)
        del buf288
        del buf292
        buf303 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_89.run(buf302, buf303, 320, 13, grid=grid(320), stream=stream0)
        buf305 = reinterpret_tensor(buf302, (320, 13), (1, 320), 0); del buf302  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_90.run(buf304, buf305, 4160, 121, grid=grid(4160), stream=stream0)
        buf306 = empty((320, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_sum_47.run(buf305, buf306, 320, 13, grid=grid(320), stream=stream0)
        del buf305
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf307 = aten.convolution_backward(reinterpret_tensor(buf304, (8, 320, 14, 14), (62720, 1, 4480, 320), 0), clone_31, primals_95, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf304
        del clone_31
        del primals_95
        buf308 = buf307[0]
        buf309 = buf307[1]
        del buf307
        buf310 = empty((6280, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.slice_backward, aten.view]
        triton_poi_fused_slice_backward_view_91.run(buf308, buf310, 803840, grid=grid(803840), stream=stream0)
        buf311 = empty((6280, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf310, permute_213, out=buf311)
        del permute_213
        buf312 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf310, (128, 6280), (1, 128), 0), view_81, out=buf312)
        del view_81
        buf313 = empty_strided((1, 128, 50), (6400, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_92.run(buf310, buf313, 6400, 126, grid=grid(6400), stream=stream0)
        buf314 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_93.run(buf313, buf314, 128, 50, grid=grid(128), stream=stream0)
        buf315 = reinterpret_tensor(buf311, (8, 785, 1024), (803840, 1024, 1), 0); del buf311  # reuse
        # Source Nodes: [x_74], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_94.run(buf315, addmm_14, 6430720, grid=grid(6430720), stream=stream0)
        del addmm_14
        buf316 = buf310; del buf310  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf315, (6280, 1024), (1024, 1), 0), permute_217, out=buf316)
        del permute_217
        buf317 = empty((1024, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf315, (1024, 6280), (1, 1024), 0), view_79, out=buf317)
        del view_79
        buf318 = empty_strided((1, 1024, 50), (51200, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_95.run(buf315, buf318, 51200, 126, grid=grid(51200), stream=stream0)
        buf319 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_96.run(buf318, buf319, 1024, 50, grid=grid(1024), stream=stream0)
        buf326 = empty((8, 785, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.slice_backward]
        triton_per_fused_add_native_layer_norm_backward_slice_backward_97.run(buf316, primals_17, mul_35, buf308, div_19, buf326, 6280, 128, grid=grid(6280), stream=stream0)
        del div_19
        del primals_17
        buf322 = reinterpret_tensor(buf313, (128, 50), (1, 128), 0); del buf313  # reuse
        buf324 = empty_strided((128, 50), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_98.run(buf316, mul_35, buf322, buf324, 6400, 126, grid=grid(6400), stream=stream0)
        del mul_35
        buf323 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_93.run(buf322, buf323, 128, 50, grid=grid(128), stream=stream0)
        buf325 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_93.run(buf324, buf325, 128, 50, grid=grid(128), stream=stream0)
        buf327 = buf316; del buf316  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf326, (6280, 128), (128, 1), 0), permute_221, out=buf327)
        del permute_221
        buf328 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf326, (128, 6280), (1, 128), 0), view_77, out=buf328)
        del view_77
        buf329 = reinterpret_tensor(buf324, (1, 128, 50), (6400, 1, 128), 0); del buf324  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_99.run(buf326, buf329, 6400, 126, grid=grid(6400), stream=stream0)
        buf330 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_93.run(buf329, buf330, 128, 50, grid=grid(128), stream=stream0)
        buf331 = reinterpret_tensor(buf308, (8, 8, 784, 16), (100352, 12544, 16, 1), 0); del buf308  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul]
        triton_poi_fused_constant_pad_nd_mul_100.run(buf327, slice_47, buf331, 802816, grid=grid(802816), stream=stream0)
        del slice_47
        buf332 = reinterpret_tensor(buf152, (8, 32, 28, 28), (25088, 1, 896, 32), 0); del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.slice]
        triton_poi_fused_slice_101.run(buf331, buf332, 256, 784, grid=grid(256, 784), stream=stream0)
        buf333 = empty_strided((8, 48, 28, 28), (37632, 1, 1344, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.slice]
        triton_poi_fused_slice_102.run(buf331, buf333, 384, 784, grid=grid(384, 784), stream=stream0)
        buf334 = empty_strided((8, 48, 28, 28), (37632, 1, 1344, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.slice]
        triton_poi_fused_slice_103.run(buf331, buf334, 384, 784, grid=grid(384, 784), stream=stream0)
        del buf331
        buf335 = empty_strided((48, 49), (1, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_104.run(buf334, buf335, 2352, 128, grid=grid(2352), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf337 = aten.convolution_backward(buf334, getitem_41, primals_79, [48], [1, 1], [3, 3], [1, 1], False, [0, 0], 48, [True, True, False])
        del buf334
        del getitem_41
        buf338 = buf337[0]
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf342 = aten.convolution_backward(buf333, getitem_40, primals_77, [48], [1, 1], [2, 2], [1, 1], False, [0, 0], 48, [True, True, False])
        del getitem_40
        buf343 = buf342[0]
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf347 = aten.convolution_backward(buf332, getitem_39, primals_75, [32], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del getitem_39
        buf348 = buf347[0]
        buf350 = empty((8, 8, 785, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_105.run(buf327, buf350, 803840, grid=grid(803840), stream=stream0)
        buf352 = empty((64, 785, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf350, (64, 785, 16), (12560, 16, 1), 0), permute_229, out=buf352)
        del permute_229
        buf351 = empty((64, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_228, reinterpret_tensor(buf350, (64, 785, 16), (12560, 16, 1), 0), out=buf351)
        del permute_228
        buf353 = reinterpret_tensor(buf350, (64, 785, 16), (12560, 16, 1), 0); del buf350  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_230, reinterpret_tensor(buf351, (64, 16, 16), (256, 16, 1), 0), out=buf353)
        del permute_230
        buf354 = empty((64, 16, 785), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf351, (64, 16, 16), (256, 16, 1), 0), permute_231, out=buf354)
        del permute_231
        buf355 = empty_strided((8, 8, 1, 16, 7), (896, 112, 7168, 7, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_red_fused__softmax_backward_data_106.run(buf354, alias_12, buf355, 7168, 113, grid=grid(7168), stream=stream0)
        buf356 = empty_strided((8, 8, 1, 16), (128, 16, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_107.run(buf355, buf356, 1024, 7, grid=grid(1024), stream=stream0)
        buf357 = empty((24, 8, 785, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_108.run(buf327, cat_9, buf352, buf354, alias_12, buf356, buf348, buf343, buf338, buf353, buf357, 150720, 16, grid=grid(150720, 16), stream=stream0)
        del alias_12
        del buf338
        del buf348
        del cat_9
        buf358 = empty((6280, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_109.run(buf357, buf358, 2411520, grid=grid(2411520), stream=stream0)
        buf359 = reinterpret_tensor(buf354, (6280, 128), (128, 1), 0); del buf354  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf358, permute_234, out=buf359)
        del permute_234
        buf369 = buf326; del buf326  # reuse
        # Source Nodes: [cur_12], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_110.run(buf369, buf359, primals_15, cat_8, getitem_35, rsqrt_8, 6280, 128, grid=grid(6280), stream=stream0)
        del primals_15
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf372 = aten.convolution_backward(reinterpret_tensor(buf369, (8, 128, 28, 28), (100480, 1, 3584, 128), 128), view_63, primals_71, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, False])
        del view_63
        buf373 = buf372[0]
        buf375 = reinterpret_tensor(buf353, (8, 785, 128), (100480, 128, 1), 0); del buf353  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.slice_backward]
        triton_poi_fused_add_slice_backward_111.run(buf369, buf373, buf375, 803840, grid=grid(803840), stream=stream0)
        buf376 = reinterpret_tensor(buf315, (6280, 1024), (1024, 1), 0); del buf315  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf375, (6280, 128), (128, 1), 0), permute_240, out=buf376)
        del permute_240
        buf380 = reinterpret_tensor(buf376, (8, 785, 1024), (803840, 1024, 1), 0); del buf376  # reuse
        # Source Nodes: [x_56], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_94.run(buf380, addmm_10, 6430720, grid=grid(6430720), stream=stream0)
        del addmm_10
        buf381 = reinterpret_tensor(buf352, (6280, 128), (128, 1), 0); del buf352  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf380, (6280, 1024), (1024, 1), 0), permute_244, out=buf381)
        del permute_244
        buf391 = reinterpret_tensor(buf327, (8, 785, 128), (100480, 128, 1), 0); del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_112.run(buf381, primals_13, mul_26, buf375, div_21, buf391, 6280, 128, grid=grid(6280), stream=stream0)
        del div_21
        del primals_13
        buf392 = empty((6280, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf391, (6280, 128), (128, 1), 0), permute_248, out=buf392)
        del permute_248
        buf396 = reinterpret_tensor(buf373, (8, 8, 784, 16), (100352, 12544, 16, 1), 0); del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul]
        triton_poi_fused_constant_pad_nd_mul_100.run(buf392, slice_35, buf396, 802816, grid=grid(802816), stream=stream0)
        del slice_35
        buf399 = reinterpret_tensor(buf343, (8, 48, 28, 28), (37632, 1, 1344, 48), 0); del buf343  # reuse
        # Source Nodes: [], Original ATen: [aten.slice]
        triton_poi_fused_slice_103.run(buf396, buf399, 384, 784, grid=grid(384, 784), stream=stream0)
        buf400 = empty_strided((48, 49), (1, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_104.run(buf399, buf400, 2352, 128, grid=grid(2352), stream=stream0)
        buf336 = empty((48, ), device='cuda', dtype=torch.float32)
        buf406 = buf336; del buf336  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward]
        triton_per_fused_add_convolution_backward_113.run(buf406, buf335, buf400, 48, 49, grid=grid(48), stream=stream0)
        buf339 = buf337[1]
        del buf337
        buf340 = buf400; del buf400  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_114.run(buf333, buf340, 2352, 128, grid=grid(2352), stream=stream0)
        buf398 = buf333; del buf333  # reuse
        # Source Nodes: [], Original ATen: [aten.slice]
        triton_poi_fused_slice_102.run(buf396, buf398, 384, 784, grid=grid(384, 784), stream=stream0)
        buf407 = buf335; del buf335  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_104.run(buf398, buf407, 2352, 128, grid=grid(2352), stream=stream0)
        buf341 = empty((48, ), device='cuda', dtype=torch.float32)
        buf413 = buf341; del buf341  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward]
        triton_per_fused_add_convolution_backward_113.run(buf413, buf340, buf407, 48, 49, grid=grid(48), stream=stream0)
        del buf340
        del buf407
        buf344 = buf342[1]
        del buf342
        buf345 = reinterpret_tensor(buf298, (32, 49), (1, 32), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_115.run(buf332, buf345, 1568, 128, grid=grid(1568), stream=stream0)
        buf397 = buf332; del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.slice]
        triton_poi_fused_slice_101.run(buf396, buf397, 256, 784, grid=grid(256, 784), stream=stream0)
        buf414 = empty_strided((32, 49), (1, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_116.run(buf397, buf414, 1568, 128, grid=grid(1568), stream=stream0)
        buf346 = empty((32, ), device='cuda', dtype=torch.float32)
        buf420 = buf346; del buf346  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward]
        triton_per_fused_add_convolution_backward_117.run(buf420, buf345, buf414, 32, 49, grid=grid(32), stream=stream0)
        del buf345
        del buf414
        buf349 = buf347[1]
        del buf347
        buf360 = empty((384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf358, (384, 6280), (1, 384), 0), view_65, out=buf360)
        del view_65
        buf361 = empty_strided((1, 384, 50), (19200, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_118.run(buf358, buf361, 19200, 126, grid=grid(19200), stream=stream0)
        buf362 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_119.run(buf361, buf362, 384, 50, grid=grid(384), stream=stream0)
        buf365 = reinterpret_tensor(buf329, (128, 50), (1, 128), 0); del buf329  # reuse
        buf367 = buf322; del buf322  # reuse
        # Source Nodes: [cur_12], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_120.run(buf359, cat_8, getitem_35, rsqrt_8, buf365, buf367, 6400, 126, grid=grid(6400), stream=stream0)
        del cat_8
        del getitem_35
        del rsqrt_8
        buf366 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [cur_12], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_sum_93.run(buf365, buf366, 128, 50, grid=grid(128), stream=stream0)
        buf368 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_93.run(buf367, buf368, 128, 50, grid=grid(128), stream=stream0)
        buf370 = empty_strided((128, 49), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_121.run(buf369, buf370, 6272, 128, grid=grid(6272), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf402 = aten.convolution_backward(buf399, getitem_31, primals_79, [48], [1, 1], [3, 3], [1, 1], False, [0, 0], 48, [True, True, False])
        del buf399
        del getitem_31
        del primals_79
        buf403 = buf402[0]
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf409 = aten.convolution_backward(buf398, getitem_30, primals_77, [48], [1, 1], [2, 2], [1, 1], False, [0, 0], 48, [True, True, False])
        del buf398
        del getitem_30
        del primals_77
        buf410 = buf409[0]
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf416 = aten.convolution_backward(buf397, getitem_29, primals_75, [32], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False])
        del buf397
        del getitem_29
        del primals_75
        buf417 = buf416[0]
        buf421 = reinterpret_tensor(buf369, (8, 8, 785, 16), (100480, 12560, 16, 1), 0); del buf369  # reuse
        # Source Nodes: [], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_105.run(buf392, buf421, 803840, grid=grid(803840), stream=stream0)
        buf423 = reinterpret_tensor(buf359, (64, 785, 16), (12560, 16, 1), 0); del buf359  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf421, (64, 785, 16), (12560, 16, 1), 0), permute_256, out=buf423)
        del permute_256
        buf422 = buf351; del buf351  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_255, reinterpret_tensor(buf421, (64, 785, 16), (12560, 16, 1), 0), out=buf422)
        del permute_255
        buf424 = reinterpret_tensor(buf421, (64, 785, 16), (12560, 16, 1), 0); del buf421  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_257, reinterpret_tensor(buf422, (64, 16, 16), (256, 16, 1), 0), out=buf424)
        del permute_257
        buf425 = empty((64, 16, 785), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf422, (64, 16, 16), (256, 16, 1), 0), permute_258, out=buf425)
        del permute_258
        buf426 = buf355; del buf355  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_red_fused__softmax_backward_data_106.run(buf425, alias_13, buf426, 7168, 113, grid=grid(7168), stream=stream0)
        buf427 = buf356; del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_107.run(buf426, buf427, 1024, 7, grid=grid(1024), stream=stream0)
        del buf426
        buf428 = reinterpret_tensor(buf358, (24, 8, 785, 16), (100480, 12560, 16, 1), 0); del buf358  # reuse
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_108.run(buf392, cat_7, buf423, buf425, alias_13, buf427, buf417, buf410, buf403, buf424, buf428, 150720, 16, grid=grid(150720, 16), stream=stream0)
        del alias_13
        del buf392
        del buf403
        del buf410
        del buf417
        del buf423
        del cat_7
        buf429 = reinterpret_tensor(buf357, (6280, 384), (384, 1), 0); del buf357  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_109.run(buf428, buf429, 2411520, grid=grid(2411520), stream=stream0)
        del buf428
        buf430 = reinterpret_tensor(buf425, (6280, 128), (128, 1), 0); del buf425  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf429, permute_261, out=buf430)
        del permute_261
        buf440 = reinterpret_tensor(buf424, (8, 785, 128), (100480, 128, 1), 0); del buf424  # reuse
        # Source Nodes: [cur_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_122.run(buf430, primals_11, cat_6, getitem_25, rsqrt_6, buf391, buf440, 6280, 128, grid=grid(6280), stream=stream0)
        del primals_11
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf443 = aten.convolution_backward(reinterpret_tensor(buf440, (8, 128, 28, 28), (100480, 1, 3584, 128), 128), view_43, primals_71, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, False])
        del primals_71
        del view_43
        buf444 = buf443[0]
        buf441 = empty_strided((128, 49), (1, 128), device='cuda', dtype=torch.float32)
        buf451 = empty_strided((128, 49), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_layer_norm_backward]
        triton_red_fused_convolution_backward_native_layer_norm_backward_123.run(buf440, buf444, mul_20, buf441, buf451, 6272, 128, grid=grid(6272), stream=stream0)
        buf371 = empty((128, ), device='cuda', dtype=torch.float32)
        buf447 = buf371; del buf371  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward]
        triton_per_fused_add_convolution_backward_124.run(buf447, buf370, buf441, 128, 49, grid=grid(128), stream=stream0)
        del buf370
        del buf441
        buf374 = buf372[1]
        del buf372
        buf377 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf375, (128, 6280), (1, 128), 0), view_61, out=buf377)
        del view_61
        buf378 = reinterpret_tensor(buf367, (1, 128, 50), (6400, 1, 128), 0); del buf367  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_92.run(buf375, buf378, 6400, 126, grid=grid(6400), stream=stream0)
        del buf375
        buf379 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_93.run(buf378, buf379, 128, 50, grid=grid(128), stream=stream0)
        buf382 = empty((1024, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf380, (1024, 6280), (1, 1024), 0), view_59, out=buf382)
        del view_59
        buf383 = buf318; del buf318  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_95.run(buf380, buf383, 51200, 126, grid=grid(51200), stream=stream0)
        del buf380
        buf384 = reinterpret_tensor(buf427, (1, 1024), (1024, 1), 0); del buf427  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_96.run(buf383, buf384, 1024, 50, grid=grid(1024), stream=stream0)
        del buf383
        buf387 = reinterpret_tensor(buf378, (128, 50), (1, 128), 0); del buf378  # reuse
        buf389 = buf365; del buf365  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_98.run(buf381, mul_26, buf387, buf389, 6400, 126, grid=grid(6400), stream=stream0)
        del buf381
        del mul_26
        buf388 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_93.run(buf387, buf388, 128, 50, grid=grid(128), stream=stream0)
        buf390 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_93.run(buf389, buf390, 128, 50, grid=grid(128), stream=stream0)
        buf393 = reinterpret_tensor(buf422, (128, 128), (128, 1), 0); del buf422  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf391, (128, 6280), (1, 128), 0), view_57, out=buf393)
        del view_57
        buf394 = reinterpret_tensor(buf389, (1, 128, 50), (6400, 1, 128), 0); del buf389  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_92.run(buf391, buf394, 6400, 126, grid=grid(6400), stream=stream0)
        del buf391
        buf395 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_93.run(buf394, buf395, 128, 50, grid=grid(128), stream=stream0)
        buf404 = buf402[1]
        del buf402
        buf405 = buf339; del buf339  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_125.run(buf405, buf404, 2352, grid=grid(2352), stream=stream0)
        del buf404
        buf411 = buf409[1]
        del buf409
        buf412 = buf344; del buf344  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_126.run(buf412, buf411, 1200, grid=grid(1200), stream=stream0)
        del buf411
        buf418 = buf416[1]
        del buf416
        buf419 = buf349; del buf349  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_127.run(buf419, buf418, 288, grid=grid(288), stream=stream0)
        del buf418
        buf431 = empty((384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf429, (384, 6280), (1, 384), 0), view_45, out=buf431)
        del view_45
        buf432 = buf361; del buf361  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_118.run(buf429, buf432, 19200, 126, grid=grid(19200), stream=stream0)
        del buf429
        buf433 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_119.run(buf432, buf433, 384, 50, grid=grid(384), stream=stream0)
        del buf432
        buf436 = reinterpret_tensor(buf394, (128, 50), (1, 128), 0); del buf394  # reuse
        buf438 = buf387; del buf387  # reuse
        # Source Nodes: [cur_8], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_120.run(buf430, cat_6, getitem_25, rsqrt_6, buf436, buf438, 6400, 126, grid=grid(6400), stream=stream0)
        del buf430
        del cat_6
        del getitem_25
        del rsqrt_6
        buf437 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [cur_8], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_sum_93.run(buf436, buf437, 128, 50, grid=grid(128), stream=stream0)
        del buf436
        buf439 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_93.run(buf438, buf439, 128, 50, grid=grid(128), stream=stream0)
        del buf438
        buf445 = buf443[1]
        del buf443
        buf446 = buf374; del buf374  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_128.run(buf446, buf445, 1152, grid=grid(1152), stream=stream0)
        del buf445
        buf448 = empty((1, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_129.run(buf440, buf444, buf448, 128, 8, grid=grid(128), stream=stream0)
        buf455 = reinterpret_tensor(buf396, (8, 784, 128), (100352, 128, 1), 0); del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_130.run(buf440, buf444, primals_69, mul_20, div_23, buf455, 6272, 128, grid=grid(6272), stream=stream0)
        del div_23
        del mul_20
        del primals_69
        buf452 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_131.run(buf451, buf452, 128, 49, grid=grid(128), stream=stream0)
        buf453 = reinterpret_tensor(buf451, (128, 49), (49, 1), 0); del buf451  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_132.run(buf440, buf444, buf453, 6272, 128, grid=grid(6272), stream=stream0)
        del buf440
        del buf444
        buf454 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_133.run(buf453, buf454, 128, 49, grid=grid(128), stream=stream0)
        buf456 = reinterpret_tensor(buf453, (128, 49), (1, 128), 0); del buf453  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_134.run(buf455, buf456, 6272, 128, grid=grid(6272), stream=stream0)
        buf457 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_131.run(buf456, buf457, 128, 49, grid=grid(128), stream=stream0)
        del buf456
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf458 = aten.convolution_backward(reinterpret_tensor(buf455, (8, 128, 28, 28), (100352, 1, 3584, 128), 0), clone_15, primals_67, [128], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf455
        del clone_15
        del primals_67
        buf459 = buf458[0]
        buf460 = buf458[1]
        del buf458
        buf461 = empty((25096, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.slice_backward, aten.view]
        triton_poi_fused_slice_backward_view_135.run(buf459, buf461, 1606144, grid=grid(1606144), stream=stream0)
        buf462 = empty((25096, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf461, permute_269, out=buf462)
        del permute_269
        buf463 = empty((64, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf461, (64, 25096), (1, 64), 0), view_39, out=buf463)
        del view_39
        buf464 = empty_strided((1, 64, 197), (12608, 1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_136.run(buf461, buf464, 12608, 128, grid=grid(12608), stream=stream0)
        buf465 = empty((1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_137.run(buf464, buf465, 64, 197, grid=grid(64), stream=stream0)
        buf466 = reinterpret_tensor(buf462, (8, 3137, 512), (1606144, 512, 1), 0); del buf462  # reuse
        # Source Nodes: [x_34], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_138.run(buf466, addmm_6, 12849152, grid=grid(12849152), stream=stream0)
        del addmm_6
        buf467 = buf461; del buf461  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf466, (25096, 512), (512, 1), 0), permute_273, out=buf467)
        del permute_273
        buf468 = empty((512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf466, (512, 25096), (1, 512), 0), view_37, out=buf468)
        del view_37
        buf469 = empty_strided((1, 512, 197), (100864, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_139.run(buf466, buf469, 100864, 128, grid=grid(100864), stream=stream0)
        buf470 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_140.run(buf469, buf470, 512, 197, grid=grid(512), stream=stream0)
        buf477 = empty((8, 3137, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.slice_backward]
        triton_per_fused_add_native_layer_norm_backward_slice_backward_141.run(buf467, primals_8, mul_15, buf459, div_24, buf477, 25096, 64, grid=grid(25096), stream=stream0)
        del div_24
        del primals_8
        buf473 = reinterpret_tensor(buf464, (64, 197), (1, 64), 0); del buf464  # reuse
        buf475 = empty_strided((64, 197), (1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_142.run(buf467, mul_15, buf473, buf475, 12608, 128, grid=grid(12608), stream=stream0)
        del mul_15
        buf474 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_sum_137.run(buf473, buf474, 64, 197, grid=grid(64), stream=stream0)
        buf476 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_sum_137.run(buf475, buf476, 64, 197, grid=grid(64), stream=stream0)
        buf478 = buf467; del buf467  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf477, (25096, 64), (64, 1), 0), permute_277, out=buf478)
        del permute_277
        buf479 = reinterpret_tensor(buf123, (64, 64), (64, 1), 0); del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf477, (64, 25096), (1, 64), 0), view_35, out=buf479)
        del view_35
        buf480 = reinterpret_tensor(buf475, (1, 64, 197), (12608, 1, 64), 0); del buf475  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_143.run(buf477, buf480, 12608, 128, grid=grid(12608), stream=stream0)
        buf481 = empty((1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_137.run(buf480, buf481, 64, 197, grid=grid(64), stream=stream0)
        buf482 = reinterpret_tensor(buf459, (8, 8, 3136, 8), (200704, 25088, 8, 1), 0); del buf459  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul]
        triton_poi_fused_constant_pad_nd_mul_144.run(buf478, slice_20, buf482, 1605632, grid=grid(1605632), stream=stream0)
        del slice_20
        buf483 = empty_strided((8, 16, 56, 56), (50176, 1, 896, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.slice]
        triton_poi_fused_slice_145.run(buf482, buf483, 128, 3136, grid=grid(128, 3136), stream=stream0)
        buf484 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.slice]
        triton_poi_fused_slice_146.run(buf482, buf484, 192, 3136, grid=grid(192, 3136), stream=stream0)
        buf485 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.slice]
        triton_poi_fused_slice_147.run(buf482, buf485, 192, 3136, grid=grid(192, 3136), stream=stream0)
        del buf482
        buf486 = reinterpret_tensor(buf297, (24, 196), (1, 24), 0); del buf297  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_148.run(buf485, buf486, 4704, 128, grid=grid(4704), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf488 = aten.convolution_backward(buf485, getitem_19, primals_51, [24], [1, 1], [3, 3], [1, 1], False, [0, 0], 24, [True, True, False])
        del buf485
        del getitem_19
        buf489 = buf488[0]
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf493 = aten.convolution_backward(buf484, getitem_18, primals_49, [24], [1, 1], [2, 2], [1, 1], False, [0, 0], 24, [True, True, False])
        del getitem_18
        buf494 = buf493[0]
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf498 = aten.convolution_backward(buf483, getitem_17, primals_47, [16], [1, 1], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False])
        del getitem_17
        buf499 = buf498[0]
        buf501 = empty((8, 8, 3137, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_149.run(buf478, buf501, 1606144, grid=grid(1606144), stream=stream0)
        buf503 = empty((64, 3137, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf501, (64, 3137, 8), (25096, 8, 1), 0), permute_285, out=buf503)
        del permute_285
        buf502 = empty((64, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_284, reinterpret_tensor(buf501, (64, 3137, 8), (25096, 8, 1), 0), out=buf502)
        del permute_284
        buf504 = reinterpret_tensor(buf501, (64, 3137, 8), (25096, 8, 1), 0); del buf501  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_286, reinterpret_tensor(buf502, (64, 8, 8), (64, 8, 1), 0), out=buf504)
        del permute_286
        buf505 = empty((64, 8, 3137), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf502, (64, 8, 8), (64, 8, 1), 0), permute_287, out=buf505)
        del permute_287
        buf506 = empty_strided((8, 8, 1, 8, 25), (1600, 200, 12800, 25, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_red_fused__softmax_backward_data_150.run(buf505, alias_14, buf506, 12800, 126, grid=grid(12800), stream=stream0)
        buf507 = empty_strided((8, 8, 1, 8), (64, 8, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_151.run(buf506, buf507, 512, 25, grid=grid(512), stream=stream0)
        buf508 = empty((24, 8, 3137, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_152.run(buf478, cat_4, buf503, buf505, alias_14, buf507, buf499, buf494, buf489, buf504, buf508, 602304, 8, grid=grid(602304, 8), stream=stream0)
        del alias_14
        del buf489
        del buf499
        del cat_4
        buf509 = empty((25096, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_153.run(buf508, buf509, 4818432, grid=grid(4818432), stream=stream0)
        buf510 = reinterpret_tensor(buf505, (25096, 64), (64, 1), 0); del buf505  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf509, permute_290, out=buf510)
        del permute_290
        buf520 = buf477; del buf477  # reuse
        # Source Nodes: [cur_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_154.run(buf520, buf510, primals_6, cat_3, getitem_13, rsqrt_3, 25096, 64, grid=grid(25096), stream=stream0)
        del primals_6
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf523 = aten.convolution_backward(reinterpret_tensor(buf520, (8, 64, 56, 56), (200768, 1, 3584, 64), 64), view_21, primals_43, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False])
        del view_21
        buf524 = buf523[0]
        buf526 = reinterpret_tensor(buf504, (8, 3137, 64), (200768, 64, 1), 0); del buf504  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.slice_backward]
        triton_poi_fused_add_slice_backward_155.run(buf520, buf524, buf526, 1606144, grid=grid(1606144), stream=stream0)
        buf527 = reinterpret_tensor(buf466, (25096, 512), (512, 1), 0); del buf466  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf526, (25096, 64), (64, 1), 0), permute_296, out=buf527)
        del permute_296
        buf531 = reinterpret_tensor(buf527, (8, 3137, 512), (1606144, 512, 1), 0); del buf527  # reuse
        # Source Nodes: [x_16], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_138.run(buf531, addmm_2, 12849152, grid=grid(12849152), stream=stream0)
        del addmm_2
        buf532 = reinterpret_tensor(buf503, (25096, 64), (64, 1), 0); del buf503  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf531, (25096, 512), (512, 1), 0), permute_300, out=buf532)
        del permute_300
        buf542 = reinterpret_tensor(buf478, (8, 3137, 64), (200768, 64, 1), 0); del buf478  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_156.run(buf532, primals_4, mul_6, buf526, div_26, buf542, 25096, 64, grid=grid(25096), stream=stream0)
        del div_26
        del primals_4
        buf543 = empty((25096, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf542, (25096, 64), (64, 1), 0), permute_304, out=buf543)
        del permute_304
        buf547 = reinterpret_tensor(buf524, (8, 8, 3136, 8), (200704, 25088, 8, 1), 0); del buf524  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd, aten.mul]
        triton_poi_fused_constant_pad_nd_mul_144.run(buf543, slice_8, buf547, 1605632, grid=grid(1605632), stream=stream0)
        del slice_8
        buf550 = reinterpret_tensor(buf494, (8, 24, 56, 56), (75264, 1, 1344, 24), 0); del buf494  # reuse
        # Source Nodes: [], Original ATen: [aten.slice]
        triton_poi_fused_slice_147.run(buf547, buf550, 192, 3136, grid=grid(192, 3136), stream=stream0)
        buf551 = empty_strided((24, 196), (1, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_148.run(buf550, buf551, 4704, 128, grid=grid(4704), stream=stream0)
        buf487 = empty((24, ), device='cuda', dtype=torch.float32)
        buf557 = buf487; del buf487  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward]
        triton_red_fused_add_convolution_backward_157.run(buf557, buf486, buf551, 24, 196, grid=grid(24), stream=stream0)
        buf490 = buf488[1]
        del buf488
        buf491 = buf551; del buf551  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_158.run(buf484, buf491, 4704, 128, grid=grid(4704), stream=stream0)
        buf549 = buf484; del buf484  # reuse
        # Source Nodes: [], Original ATen: [aten.slice]
        triton_poi_fused_slice_146.run(buf547, buf549, 192, 3136, grid=grid(192, 3136), stream=stream0)
        buf558 = buf486; del buf486  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_148.run(buf549, buf558, 4704, 128, grid=grid(4704), stream=stream0)
        buf492 = empty((24, ), device='cuda', dtype=torch.float32)
        buf564 = buf492; del buf492  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward]
        triton_red_fused_add_convolution_backward_157.run(buf564, buf491, buf558, 24, 196, grid=grid(24), stream=stream0)
        del buf491
        del buf558
        buf495 = buf493[1]
        del buf493
        buf496 = empty_strided((16, 196), (1, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_159.run(buf483, buf496, 3136, 128, grid=grid(3136), stream=stream0)
        buf548 = buf483; del buf483  # reuse
        # Source Nodes: [], Original ATen: [aten.slice]
        triton_poi_fused_slice_145.run(buf547, buf548, 128, 3136, grid=grid(128, 3136), stream=stream0)
        buf565 = empty_strided((16, 196), (1, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_160.run(buf548, buf565, 3136, 128, grid=grid(3136), stream=stream0)
        buf497 = empty((16, ), device='cuda', dtype=torch.float32)
        buf571 = buf497; del buf497  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward]
        triton_red_fused_add_convolution_backward_161.run(buf571, buf496, buf565, 16, 196, grid=grid(16), stream=stream0)
        del buf496
        del buf565
        buf500 = buf498[1]
        del buf498
        buf511 = empty((192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf509, (192, 25096), (1, 192), 0), view_23, out=buf511)
        del view_23
        buf512 = empty_strided((1, 192, 197), (37824, 1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_162.run(buf509, buf512, 37824, 128, grid=grid(37824), stream=stream0)
        buf513 = empty((1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_163.run(buf512, buf513, 192, 197, grid=grid(192), stream=stream0)
        buf516 = reinterpret_tensor(buf480, (64, 197), (1, 64), 0); del buf480  # reuse
        buf518 = buf473; del buf473  # reuse
        # Source Nodes: [cur_4], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_164.run(buf510, cat_3, getitem_13, rsqrt_3, buf516, buf518, 12608, 128, grid=grid(12608), stream=stream0)
        del cat_3
        del getitem_13
        del rsqrt_3
        buf517 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [cur_4], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_sum_137.run(buf516, buf517, 64, 197, grid=grid(64), stream=stream0)
        buf519 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_sum_137.run(buf518, buf519, 64, 197, grid=grid(64), stream=stream0)
        buf521 = empty_strided((64, 196), (1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_165.run(buf520, buf521, 12544, 128, grid=grid(12544), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf553 = aten.convolution_backward(buf550, getitem_9, primals_51, [24], [1, 1], [3, 3], [1, 1], False, [0, 0], 24, [True, True, False])
        del buf550
        del getitem_9
        del primals_51
        buf554 = buf553[0]
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf560 = aten.convolution_backward(buf549, getitem_8, primals_49, [24], [1, 1], [2, 2], [1, 1], False, [0, 0], 24, [True, True, False])
        del buf549
        del getitem_8
        del primals_49
        buf561 = buf560[0]
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf567 = aten.convolution_backward(buf548, getitem_7, primals_47, [16], [1, 1], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False])
        del buf548
        del getitem_7
        del primals_47
        buf568 = buf567[0]
        buf572 = reinterpret_tensor(buf520, (8, 8, 3137, 8), (200768, 25096, 8, 1), 0); del buf520  # reuse
        # Source Nodes: [], Original ATen: [aten.clone, aten.mul]
        triton_poi_fused_clone_mul_149.run(buf543, buf572, 1606144, grid=grid(1606144), stream=stream0)
        buf574 = reinterpret_tensor(buf510, (64, 3137, 8), (25096, 8, 1), 0); del buf510  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf572, (64, 3137, 8), (25096, 8, 1), 0), permute_312, out=buf574)
        del permute_312
        buf573 = buf502; del buf502  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_311, reinterpret_tensor(buf572, (64, 3137, 8), (25096, 8, 1), 0), out=buf573)
        del permute_311
        buf575 = reinterpret_tensor(buf572, (64, 3137, 8), (25096, 8, 1), 0); del buf572  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_313, reinterpret_tensor(buf573, (64, 8, 8), (64, 8, 1), 0), out=buf575)
        del permute_313
        buf576 = empty((64, 8, 3137), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf573, (64, 8, 8), (64, 8, 1), 0), permute_314, out=buf576)
        del permute_314
        buf577 = buf506; del buf506  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_red_fused__softmax_backward_data_150.run(buf576, alias_15, buf577, 12800, 126, grid=grid(12800), stream=stream0)
        buf578 = buf507; del buf507  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_151.run(buf577, buf578, 512, 25, grid=grid(512), stream=stream0)
        del buf577
        buf579 = reinterpret_tensor(buf509, (24, 8, 3137, 8), (200768, 25096, 8, 1), 0); del buf509  # reuse
        # Source Nodes: [], Original ATen: [aten.stack]
        triton_poi_fused_stack_152.run(buf543, cat_2, buf574, buf576, alias_15, buf578, buf568, buf561, buf554, buf575, buf579, 602304, 8, grid=grid(602304, 8), stream=stream0)
        del alias_15
        del buf543
        del buf554
        del buf561
        del buf568
        del buf574
        del cat_2
        buf580 = reinterpret_tensor(buf508, (25096, 192), (192, 1), 0); del buf508  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_153.run(buf579, buf580, 4818432, grid=grid(4818432), stream=stream0)
        del buf579
        buf581 = reinterpret_tensor(buf576, (25096, 64), (64, 1), 0); del buf576  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf580, permute_317, out=buf581)
        del permute_317
        buf591 = reinterpret_tensor(buf575, (8, 3137, 64), (200768, 64, 1), 0); del buf575  # reuse
        # Source Nodes: [cur], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_166.run(buf581, primals_2, cat_1, getitem_3, rsqrt_1, buf542, buf591, 25096, 64, grid=grid(25096), stream=stream0)
        del primals_2
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf594 = aten.convolution_backward(reinterpret_tensor(buf591, (8, 64, 56, 56), (200768, 1, 3584, 64), 64), view_1, primals_43, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False])
        del primals_43
        del view_1
        buf595 = buf594[0]
        buf592 = empty_strided((64, 196), (1, 64), device='cuda', dtype=torch.float32)
        buf602 = empty_strided((64, 196), (1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_layer_norm_backward]
        triton_red_fused_convolution_backward_native_layer_norm_backward_167.run(buf591, buf595, mul, buf592, buf602, 12544, 128, grid=grid(12544), stream=stream0)
        buf522 = empty((64, ), device='cuda', dtype=torch.float32)
        buf598 = buf522; del buf522  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward]
        triton_red_fused_add_convolution_backward_168.run(buf598, buf521, buf592, 64, 196, grid=grid(64), stream=stream0)
        del buf521
        del buf592
        buf525 = buf523[1]
        del buf523
        buf528 = empty((64, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf526, (64, 25096), (1, 64), 0), view_19, out=buf528)
        del view_19
        buf529 = reinterpret_tensor(buf518, (1, 64, 197), (12608, 1, 64), 0); del buf518  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_136.run(buf526, buf529, 12608, 128, grid=grid(12608), stream=stream0)
        del buf526
        buf530 = empty((1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_137.run(buf529, buf530, 64, 197, grid=grid(64), stream=stream0)
        buf533 = empty((512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf531, (512, 25096), (1, 512), 0), view_17, out=buf533)
        del view_17
        buf534 = buf469; del buf469  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_139.run(buf531, buf534, 100864, 128, grid=grid(100864), stream=stream0)
        del buf531
        buf535 = reinterpret_tensor(buf578, (1, 512), (512, 1), 0); del buf578  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_140.run(buf534, buf535, 512, 197, grid=grid(512), stream=stream0)
        del buf534
        buf538 = reinterpret_tensor(buf529, (64, 197), (1, 64), 0); del buf529  # reuse
        buf540 = buf516; del buf516  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_142.run(buf532, mul_6, buf538, buf540, 12608, 128, grid=grid(12608), stream=stream0)
        del buf532
        del mul_6
        buf539 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_sum_137.run(buf538, buf539, 64, 197, grid=grid(64), stream=stream0)
        buf541 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_sum_137.run(buf540, buf541, 64, 197, grid=grid(64), stream=stream0)
        buf544 = reinterpret_tensor(buf573, (64, 64), (64, 1), 0); del buf573  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf542, (64, 25096), (1, 64), 0), view_15, out=buf544)
        del view_15
        buf545 = reinterpret_tensor(buf540, (1, 64, 197), (12608, 1, 64), 0); del buf540  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_136.run(buf542, buf545, 12608, 128, grid=grid(12608), stream=stream0)
        del buf542
        buf546 = empty((1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_137.run(buf545, buf546, 64, 197, grid=grid(64), stream=stream0)
        buf555 = buf553[1]
        del buf553
        buf556 = buf490; del buf490  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_169.run(buf556, buf555, 1176, grid=grid(1176), stream=stream0)
        del buf555
        buf562 = buf560[1]
        del buf560
        buf563 = buf495; del buf495  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_170.run(buf563, buf562, 600, grid=grid(600), stream=stream0)
        del buf562
        buf569 = buf567[1]
        del buf567
        buf570 = buf500; del buf500  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_171.run(buf570, buf569, 144, grid=grid(144), stream=stream0)
        del buf569
        buf582 = empty((192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf580, (192, 25096), (1, 192), 0), view_3, out=buf582)
        del view_3
        buf583 = buf512; del buf512  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_162.run(buf580, buf583, 37824, 128, grid=grid(37824), stream=stream0)
        del buf580
        buf584 = empty((1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_163.run(buf583, buf584, 192, 197, grid=grid(192), stream=stream0)
        del buf583
        buf587 = reinterpret_tensor(buf545, (64, 197), (1, 64), 0); del buf545  # reuse
        buf589 = buf538; del buf538  # reuse
        # Source Nodes: [cur], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_164.run(buf581, cat_1, getitem_3, rsqrt_1, buf587, buf589, 12608, 128, grid=grid(12608), stream=stream0)
        del buf581
        del cat_1
        del getitem_3
        del rsqrt_1
        buf588 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [cur], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_sum_137.run(buf587, buf588, 64, 197, grid=grid(64), stream=stream0)
        del buf587
        buf590 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_sum_137.run(buf589, buf590, 64, 197, grid=grid(64), stream=stream0)
        del buf589
        buf596 = buf594[1]
        del buf594
        buf597 = buf525; del buf525  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_172.run(buf597, buf596, 576, grid=grid(576), stream=stream0)
        del buf596
        buf599 = empty((1, 1, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_173.run(buf591, buf595, buf599, 64, 8, grid=grid(64), stream=stream0)
        buf606 = reinterpret_tensor(buf547, (8, 3136, 64), (200704, 64, 1), 0); del buf547  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_174.run(buf591, buf595, primals_41, mul, div_28, buf606, 25088, 64, grid=grid(25088), stream=stream0)
        del div_28
        del mul
        del primals_41
        buf603 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_175.run(buf602, buf603, 64, 196, grid=grid(64), stream=stream0)
        buf604 = reinterpret_tensor(buf602, (64, 196), (196, 1), 0); del buf602  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_176.run(buf591, buf595, buf604, 12544, 128, grid=grid(12544), stream=stream0)
        del buf591
        del buf595
        buf605 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_177.run(buf604, buf605, 64, 196, grid=grid(64), stream=stream0)
        buf607 = reinterpret_tensor(buf604, (64, 196), (1, 64), 0); del buf604  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_178.run(buf606, buf607, 12544, 128, grid=grid(12544), stream=stream0)
        buf608 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_native_layer_norm_backward_175.run(buf607, buf608, 64, 196, grid=grid(64), stream=stream0)
        del buf607
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf609 = aten.convolution_backward(reinterpret_tensor(buf606, (8, 64, 56, 56), (200704, 1, 3584, 64), 0), primals_153, primals_39, [64], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf606
        del primals_153
        del primals_39
        buf610 = buf609[1]
        return (buf599, buf588, buf590, buf539, buf541, buf517, buf519, buf474, buf476, buf448, buf437, buf439, buf388, buf390, buf366, buf368, buf323, buf325, buf296, buf285, buf287, buf236, buf238, buf214, buf216, buf171, buf173, buf144, buf133, buf135, buf85, buf87, buf63, buf65, buf21, buf23, buf7, buf8, buf610, buf608, buf603, buf605, buf597, buf598, reinterpret_tensor(buf582, (192, 64), (64, 1), 0), reinterpret_tensor(buf584, (192, ), (1, ), 0), buf570, buf571, buf563, buf564, buf556, buf557, reinterpret_tensor(buf544, (64, 64), (64, 1), 0), reinterpret_tensor(buf546, (64, ), (1, ), 0), reinterpret_tensor(buf533, (512, 64), (64, 1), 0), reinterpret_tensor(buf535, (512, ), (1, ), 0), reinterpret_tensor(buf528, (64, 512), (512, 1), 0), reinterpret_tensor(buf530, (64, ), (1, ), 0), reinterpret_tensor(buf511, (192, 64), (64, 1), 0), reinterpret_tensor(buf513, (192, ), (1, ), 0), reinterpret_tensor(buf479, (64, 64), (64, 1), 0), reinterpret_tensor(buf481, (64, ), (1, ), 0), reinterpret_tensor(buf468, (512, 64), (64, 1), 0), reinterpret_tensor(buf470, (512, ), (1, ), 0), reinterpret_tensor(buf463, (64, 512), (512, 1), 0), reinterpret_tensor(buf465, (64, ), (1, ), 0), buf460, buf457, buf452, buf454, buf446, buf447, reinterpret_tensor(buf431, (384, 128), (128, 1), 0), reinterpret_tensor(buf433, (384, ), (1, ), 0), buf419, buf420, buf412, buf413, buf405, buf406, reinterpret_tensor(buf393, (128, 128), (128, 1), 0), reinterpret_tensor(buf395, (128, ), (1, ), 0), reinterpret_tensor(buf382, (1024, 128), (128, 1), 0), reinterpret_tensor(buf384, (1024, ), (1, ), 0), reinterpret_tensor(buf377, (128, 1024), (1024, 1), 0), reinterpret_tensor(buf379, (128, ), (1, ), 0), reinterpret_tensor(buf360, (384, 128), (128, 1), 0), reinterpret_tensor(buf362, (384, ), (1, ), 0), reinterpret_tensor(buf328, (128, 128), (128, 1), 0), reinterpret_tensor(buf330, (128, ), (1, ), 0), reinterpret_tensor(buf317, (1024, 128), (128, 1), 0), reinterpret_tensor(buf319, (1024, ), (1, ), 0), reinterpret_tensor(buf312, (128, 1024), (1024, 1), 0), reinterpret_tensor(buf314, (128, ), (1, ), 0), buf309, buf306, buf301, buf303, buf294, buf295, reinterpret_tensor(buf279, (960, 320), (320, 1), 0), reinterpret_tensor(buf281, (960, ), (1, ), 0), buf267, buf268, buf260, buf261, buf253, buf254, reinterpret_tensor(buf241, (320, 320), (320, 1), 0), reinterpret_tensor(buf243, (320, ), (1, ), 0), reinterpret_tensor(buf230, (1280, 320), (320, 1), 0), reinterpret_tensor(buf232, (1280, ), (1, ), 0), reinterpret_tensor(buf225, (320, 1280), (1280, 1), 0), reinterpret_tensor(buf227, (320, ), (1, ), 0), reinterpret_tensor(buf208, (960, 320), (320, 1), 0), reinterpret_tensor(buf210, (960, ), (1, ), 0), reinterpret_tensor(buf176, (320, 320), (320, 1), 0), reinterpret_tensor(buf178, (320, ), (1, ), 0), reinterpret_tensor(buf165, (1280, 320), (320, 1), 0), reinterpret_tensor(buf167, (1280, ), (1, ), 0), reinterpret_tensor(buf160, (320, 1280), (1280, 1), 0), reinterpret_tensor(buf162, (320, ), (1, ), 0), buf157, buf154, buf149, buf151, buf142, buf143, reinterpret_tensor(buf127, (1536, 512), (512, 1), 0), reinterpret_tensor(buf129, (1536, ), (1, ), 0), buf116, buf117, buf109, buf110, buf102, buf103, reinterpret_tensor(buf90, (512, 512), (512, 1), 0), reinterpret_tensor(buf92, (512, ), (1, ), 0), reinterpret_tensor(buf79, (2048, 512), (512, 1), 0), reinterpret_tensor(buf81, (2048, ), (1, ), 0), reinterpret_tensor(buf74, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf76, (512, ), (1, ), 0), reinterpret_tensor(buf57, (1536, 512), (512, 1), 0), reinterpret_tensor(buf59, (1536, ), (1, ), 0), reinterpret_tensor(buf26, (512, 512), (512, 1), 0), reinterpret_tensor(buf28, (512, ), (1, ), 0), reinterpret_tensor(buf15, (2048, 512), (512, 1), 0), reinterpret_tensor(buf17, (2048, ), (1, ), 0), reinterpret_tensor(buf10, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf12, (512, ), (1, ), 0), reinterpret_tensor(buf1, (1000, 512), (512, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((64, 3, 4, 4), (48, 1, 12, 3), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((24, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((24, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((128, 64, 2, 2), (256, 1, 128, 64), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((48, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((48, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((320, 128, 2, 2), (512, 1, 256, 128), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((320, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((512, 320, 2, 2), (1280, 1, 640, 320), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((192, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((192, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
    mul = rand_strided((8, 3136, 64), (200704, 64, 1), device='cuda:0', dtype=torch.float32)
    view_1 = rand_strided((8, 64, 56, 56), (200768, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    cat_1 = rand_strided((8, 3137, 64), (200768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((8, 3137, 1), (3137, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_1 = rand_strided((8, 3137, 1), (3137, 1, 1), device='cuda:0', dtype=torch.float32)
    view_3 = rand_strided((25096, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    slice_8 = rand_strided((8, 8, 3136, 8), (602304, 8, 192, 1), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((8, 16, 56, 56), (602304, 1, 10752, 192), device='cuda:0', dtype=torch.float32)
    getitem_8 = rand_strided((8, 24, 56, 56), (602304, 1, 10752, 192), device='cuda:0', dtype=torch.float32)
    getitem_9 = rand_strided((8, 24, 56, 56), (602304, 1, 10752, 192), device='cuda:0', dtype=torch.float32)
    cat_2 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    view_15 = rand_strided((25096, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    mul_6 = rand_strided((8, 3137, 64), (200768, 64, 1), device='cuda:0', dtype=torch.float32)
    view_17 = rand_strided((25096, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    addmm_2 = rand_strided((25096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_19 = rand_strided((25096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_21 = rand_strided((8, 64, 56, 56), (200768, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    cat_3 = rand_strided((8, 3137, 64), (200768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_13 = rand_strided((8, 3137, 1), (3137, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_3 = rand_strided((8, 3137, 1), (3137, 1, 1), device='cuda:0', dtype=torch.float32)
    view_23 = rand_strided((25096, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    slice_20 = rand_strided((8, 8, 3136, 8), (602304, 8, 192, 1), device='cuda:0', dtype=torch.float32)
    getitem_17 = rand_strided((8, 16, 56, 56), (602304, 1, 10752, 192), device='cuda:0', dtype=torch.float32)
    getitem_18 = rand_strided((8, 24, 56, 56), (602304, 1, 10752, 192), device='cuda:0', dtype=torch.float32)
    getitem_19 = rand_strided((8, 24, 56, 56), (602304, 1, 10752, 192), device='cuda:0', dtype=torch.float32)
    cat_4 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    view_35 = rand_strided((25096, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    mul_15 = rand_strided((8, 3137, 64), (200768, 64, 1), device='cuda:0', dtype=torch.float32)
    view_37 = rand_strided((25096, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    addmm_6 = rand_strided((25096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_39 = rand_strided((25096, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    clone_15 = rand_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    mul_20 = rand_strided((8, 784, 128), (100352, 128, 1), device='cuda:0', dtype=torch.float32)
    view_43 = rand_strided((8, 128, 28, 28), (100480, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    cat_6 = rand_strided((8, 785, 128), (100480, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_25 = rand_strided((8, 785, 1), (785, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_6 = rand_strided((8, 785, 1), (785, 1, 1), device='cuda:0', dtype=torch.float32)
    view_45 = rand_strided((6280, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    slice_35 = rand_strided((8, 8, 784, 16), (301440, 16, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_29 = rand_strided((8, 32, 28, 28), (301440, 1, 10752, 384), device='cuda:0', dtype=torch.float32)
    getitem_30 = rand_strided((8, 48, 28, 28), (301440, 1, 10752, 384), device='cuda:0', dtype=torch.float32)
    getitem_31 = rand_strided((8, 48, 28, 28), (301440, 1, 10752, 384), device='cuda:0', dtype=torch.float32)
    cat_7 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    view_57 = rand_strided((6280, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mul_26 = rand_strided((8, 785, 128), (100480, 128, 1), device='cuda:0', dtype=torch.float32)
    view_59 = rand_strided((6280, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((6280, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_61 = rand_strided((6280, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_63 = rand_strided((8, 128, 28, 28), (100480, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    cat_8 = rand_strided((8, 785, 128), (100480, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_35 = rand_strided((8, 785, 1), (785, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_8 = rand_strided((8, 785, 1), (785, 1, 1), device='cuda:0', dtype=torch.float32)
    view_65 = rand_strided((6280, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    slice_47 = rand_strided((8, 8, 784, 16), (301440, 16, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_39 = rand_strided((8, 32, 28, 28), (301440, 1, 10752, 384), device='cuda:0', dtype=torch.float32)
    getitem_40 = rand_strided((8, 48, 28, 28), (301440, 1, 10752, 384), device='cuda:0', dtype=torch.float32)
    getitem_41 = rand_strided((8, 48, 28, 28), (301440, 1, 10752, 384), device='cuda:0', dtype=torch.float32)
    cat_9 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    view_77 = rand_strided((6280, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mul_35 = rand_strided((8, 785, 128), (100480, 128, 1), device='cuda:0', dtype=torch.float32)
    view_79 = rand_strided((6280, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_14 = rand_strided((6280, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_81 = rand_strided((6280, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    clone_31 = rand_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    mul_40 = rand_strided((8, 196, 320), (62720, 320, 1), device='cuda:0', dtype=torch.float32)
    view_85 = rand_strided((8, 320, 14, 14), (63040, 1, 4480, 320), device='cuda:0', dtype=torch.float32)
    cat_11 = rand_strided((8, 197, 320), (63040, 320, 1), device='cuda:0', dtype=torch.float32)
    getitem_47 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_11 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_87 = rand_strided((1576, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    slice_62 = rand_strided((8, 8, 196, 40), (189120, 40, 960, 1), device='cuda:0', dtype=torch.float32)
    getitem_51 = rand_strided((8, 80, 14, 14), (189120, 1, 13440, 960), device='cuda:0', dtype=torch.float32)
    getitem_52 = rand_strided((8, 120, 14, 14), (189120, 1, 13440, 960), device='cuda:0', dtype=torch.float32)
    getitem_53 = rand_strided((8, 120, 14, 14), (189120, 1, 13440, 960), device='cuda:0', dtype=torch.float32)
    cat_12 = rand_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda:0', dtype=torch.float32)
    view_99 = rand_strided((1576, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    mul_46 = rand_strided((8, 197, 320), (63040, 320, 1), device='cuda:0', dtype=torch.float32)
    view_101 = rand_strided((1576, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    addmm_18 = rand_strided((1576, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    view_103 = rand_strided((1576, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    view_105 = rand_strided((8, 320, 14, 14), (63040, 1, 4480, 320), device='cuda:0', dtype=torch.float32)
    cat_13 = rand_strided((8, 197, 320), (63040, 320, 1), device='cuda:0', dtype=torch.float32)
    getitem_57 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_13 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_107 = rand_strided((1576, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    slice_74 = rand_strided((8, 8, 196, 40), (189120, 40, 960, 1), device='cuda:0', dtype=torch.float32)
    getitem_61 = rand_strided((8, 80, 14, 14), (189120, 1, 13440, 960), device='cuda:0', dtype=torch.float32)
    getitem_62 = rand_strided((8, 120, 14, 14), (189120, 1, 13440, 960), device='cuda:0', dtype=torch.float32)
    getitem_63 = rand_strided((8, 120, 14, 14), (189120, 1, 13440, 960), device='cuda:0', dtype=torch.float32)
    cat_14 = rand_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda:0', dtype=torch.float32)
    view_119 = rand_strided((1576, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    mul_55 = rand_strided((8, 197, 320), (63040, 320, 1), device='cuda:0', dtype=torch.float32)
    view_121 = rand_strided((1576, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((1576, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    view_123 = rand_strided((1576, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    clone_47 = rand_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cuda:0', dtype=torch.float32)
    mul_60 = rand_strided((8, 49, 512), (25088, 512, 1), device='cuda:0', dtype=torch.float32)
    view_127 = rand_strided((8, 512, 7, 7), (25600, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    cat_16 = rand_strided((8, 50, 512), (25600, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_69 = rand_strided((8, 50, 1), (50, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_16 = rand_strided((8, 50, 1), (50, 1, 1), device='cuda:0', dtype=torch.float32)
    view_129 = rand_strided((400, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    slice_89 = rand_strided((8, 8, 49, 64), (76800, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_73 = rand_strided((8, 128, 7, 7), (76800, 1, 10752, 1536), device='cuda:0', dtype=torch.float32)
    getitem_74 = rand_strided((8, 192, 7, 7), (76800, 1, 10752, 1536), device='cuda:0', dtype=torch.float32)
    getitem_75 = rand_strided((8, 192, 7, 7), (76800, 1, 10752, 1536), device='cuda:0', dtype=torch.float32)
    cat_17 = rand_strided((8, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    view_141 = rand_strided((400, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mul_66 = rand_strided((8, 50, 512), (25600, 512, 1), device='cuda:0', dtype=torch.float32)
    view_143 = rand_strided((400, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_26 = rand_strided((400, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_145 = rand_strided((400, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_147 = rand_strided((8, 512, 7, 7), (25600, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    cat_18 = rand_strided((8, 50, 512), (25600, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_79 = rand_strided((8, 50, 1), (50, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_18 = rand_strided((8, 50, 1), (50, 1, 1), device='cuda:0', dtype=torch.float32)
    view_149 = rand_strided((400, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    slice_101 = rand_strided((8, 8, 49, 64), (76800, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_83 = rand_strided((8, 128, 7, 7), (76800, 1, 10752, 1536), device='cuda:0', dtype=torch.float32)
    getitem_84 = rand_strided((8, 192, 7, 7), (76800, 1, 10752, 1536), device='cuda:0', dtype=torch.float32)
    getitem_85 = rand_strided((8, 192, 7, 7), (76800, 1, 10752, 1536), device='cuda:0', dtype=torch.float32)
    cat_19 = rand_strided((8, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    view_161 = rand_strided((400, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mul_75 = rand_strided((8, 50, 512), (25600, 512, 1), device='cuda:0', dtype=torch.float32)
    view_163 = rand_strided((400, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_30 = rand_strided((400, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_165 = rand_strided((400, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    mul_80 = rand_strided((8, 50, 512), (25600, 512, 1), device='cuda:0', dtype=torch.float32)
    clone_64 = rand_strided((8, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_97 = rand_strided((1000, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_8 = rand_strided((8, 50, 1), (50, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_101 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_105 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_9 = rand_strided((8, 50, 1), (50, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_109 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_116 = rand_strided((64, 64, 50), (3200, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_117 = rand_strided((64, 64, 64), (4096, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_118 = rand_strided((64, 50, 64), (3200, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_119 = rand_strided((64, 64, 50), (3200, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_8 = rand_strided((8, 8, 50, 64), (25600, 1, 512, 8), device='cuda:0', dtype=torch.float32)
    permute_122 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_128 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_132 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_11 = rand_strided((8, 50, 1), (50, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_136 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_143 = rand_strided((64, 64, 50), (3200, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_144 = rand_strided((64, 64, 64), (4096, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_145 = rand_strided((64, 50, 64), (3200, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_146 = rand_strided((64, 64, 50), (3200, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_9 = rand_strided((8, 8, 50, 64), (25600, 1, 512, 8), device='cuda:0', dtype=torch.float32)
    permute_149 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_13 = rand_strided((8, 49, 1), (49, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_157 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_161 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_14 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_165 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    permute_172 = rand_strided((64, 40, 197), (7880, 1, 40), device='cuda:0', dtype=torch.float32)
    permute_173 = rand_strided((64, 40, 40), (1600, 1, 40), device='cuda:0', dtype=torch.float32)
    permute_174 = rand_strided((64, 197, 40), (7880, 40, 1), device='cuda:0', dtype=torch.float32)
    permute_175 = rand_strided((64, 40, 197), (7880, 1, 40), device='cuda:0', dtype=torch.float32)
    alias_10 = rand_strided((8, 8, 197, 40), (63040, 1, 320, 8), device='cuda:0', dtype=torch.float32)
    permute_178 = rand_strided((960, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    permute_184 = rand_strided((320, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_188 = rand_strided((1280, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_16 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_192 = rand_strided((320, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    permute_199 = rand_strided((64, 40, 197), (7880, 1, 40), device='cuda:0', dtype=torch.float32)
    permute_200 = rand_strided((64, 40, 40), (1600, 1, 40), device='cuda:0', dtype=torch.float32)
    permute_201 = rand_strided((64, 197, 40), (7880, 40, 1), device='cuda:0', dtype=torch.float32)
    permute_202 = rand_strided((64, 40, 197), (7880, 1, 40), device='cuda:0', dtype=torch.float32)
    alias_11 = rand_strided((8, 8, 197, 40), (63040, 1, 320, 8), device='cuda:0', dtype=torch.float32)
    permute_205 = rand_strided((960, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    div_18 = rand_strided((8, 196, 1), (196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_213 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_217 = rand_strided((1024, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_19 = rand_strided((8, 785, 1), (785, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_221 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_228 = rand_strided((64, 16, 785), (12560, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_229 = rand_strided((64, 16, 16), (256, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_230 = rand_strided((64, 785, 16), (12560, 16, 1), device='cuda:0', dtype=torch.float32)
    permute_231 = rand_strided((64, 16, 785), (12560, 1, 16), device='cuda:0', dtype=torch.float32)
    alias_12 = rand_strided((8, 8, 785, 16), (100480, 1, 128, 8), device='cuda:0', dtype=torch.float32)
    permute_234 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_240 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_244 = rand_strided((1024, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_21 = rand_strided((8, 785, 1), (785, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_248 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_255 = rand_strided((64, 16, 785), (12560, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_256 = rand_strided((64, 16, 16), (256, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_257 = rand_strided((64, 785, 16), (12560, 16, 1), device='cuda:0', dtype=torch.float32)
    permute_258 = rand_strided((64, 16, 785), (12560, 1, 16), device='cuda:0', dtype=torch.float32)
    alias_13 = rand_strided((8, 8, 785, 16), (100480, 1, 128, 8), device='cuda:0', dtype=torch.float32)
    permute_261 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_23 = rand_strided((8, 784, 1), (784, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_269 = rand_strided((64, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_273 = rand_strided((512, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    div_24 = rand_strided((8, 3137, 1), (3137, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_277 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    permute_284 = rand_strided((64, 8, 3137), (25096, 1, 8), device='cuda:0', dtype=torch.float32)
    permute_285 = rand_strided((64, 8, 8), (64, 1, 8), device='cuda:0', dtype=torch.float32)
    permute_286 = rand_strided((64, 3137, 8), (25096, 8, 1), device='cuda:0', dtype=torch.float32)
    permute_287 = rand_strided((64, 8, 3137), (25096, 1, 8), device='cuda:0', dtype=torch.float32)
    alias_14 = rand_strided((8, 8, 3137, 8), (200768, 1, 64, 8), device='cuda:0', dtype=torch.float32)
    permute_290 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    permute_296 = rand_strided((64, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_300 = rand_strided((512, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    div_26 = rand_strided((8, 3137, 1), (3137, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_304 = rand_strided((64, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    permute_311 = rand_strided((64, 8, 3137), (25096, 1, 8), device='cuda:0', dtype=torch.float32)
    permute_312 = rand_strided((64, 8, 8), (64, 1, 8), device='cuda:0', dtype=torch.float32)
    permute_313 = rand_strided((64, 3137, 8), (25096, 8, 1), device='cuda:0', dtype=torch.float32)
    permute_314 = rand_strided((64, 8, 3137), (25096, 1, 8), device='cuda:0', dtype=torch.float32)
    alias_15 = rand_strided((8, 8, 3137, 8), (200768, 1, 64, 8), device='cuda:0', dtype=torch.float32)
    permute_317 = rand_strided((192, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    div_28 = rand_strided((8, 3136, 1), (3136, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_2, primals_4, primals_6, primals_8, primals_11, primals_13, primals_15, primals_17, primals_20, primals_22, primals_24, primals_26, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_47, primals_49, primals_51, primals_67, primals_69, primals_71, primals_75, primals_77, primals_79, primals_95, primals_97, primals_99, primals_103, primals_105, primals_107, primals_123, primals_125, primals_127, primals_131, primals_133, primals_135, primals_153, mul, view_1, cat_1, getitem_3, rsqrt_1, view_3, slice_8, getitem_7, getitem_8, getitem_9, cat_2, view_15, mul_6, view_17, addmm_2, view_19, view_21, cat_3, getitem_13, rsqrt_3, view_23, slice_20, getitem_17, getitem_18, getitem_19, cat_4, view_35, mul_15, view_37, addmm_6, view_39, clone_15, mul_20, view_43, cat_6, getitem_25, rsqrt_6, view_45, slice_35, getitem_29, getitem_30, getitem_31, cat_7, view_57, mul_26, view_59, addmm_10, view_61, view_63, cat_8, getitem_35, rsqrt_8, view_65, slice_47, getitem_39, getitem_40, getitem_41, cat_9, view_77, mul_35, view_79, addmm_14, view_81, clone_31, mul_40, view_85, cat_11, getitem_47, rsqrt_11, view_87, slice_62, getitem_51, getitem_52, getitem_53, cat_12, view_99, mul_46, view_101, addmm_18, view_103, view_105, cat_13, getitem_57, rsqrt_13, view_107, slice_74, getitem_61, getitem_62, getitem_63, cat_14, view_119, mul_55, view_121, addmm_22, view_123, clone_47, mul_60, view_127, cat_16, getitem_69, rsqrt_16, view_129, slice_89, getitem_73, getitem_74, getitem_75, cat_17, view_141, mul_66, view_143, addmm_26, view_145, view_147, cat_18, getitem_79, rsqrt_18, view_149, slice_101, getitem_83, getitem_84, getitem_85, cat_19, view_161, mul_75, view_163, addmm_30, view_165, mul_80, clone_64, permute_97, div_8, permute_101, permute_105, div_9, permute_109, permute_116, permute_117, permute_118, permute_119, alias_8, permute_122, permute_128, permute_132, div_11, permute_136, permute_143, permute_144, permute_145, permute_146, alias_9, permute_149, div_13, permute_157, permute_161, div_14, permute_165, permute_172, permute_173, permute_174, permute_175, alias_10, permute_178, permute_184, permute_188, div_16, permute_192, permute_199, permute_200, permute_201, permute_202, alias_11, permute_205, div_18, permute_213, permute_217, div_19, permute_221, permute_228, permute_229, permute_230, permute_231, alias_12, permute_234, permute_240, permute_244, div_21, permute_248, permute_255, permute_256, permute_257, permute_258, alias_13, permute_261, div_23, permute_269, permute_273, div_24, permute_277, permute_284, permute_285, permute_286, permute_287, alias_14, permute_290, permute_296, permute_300, div_26, permute_304, permute_311, permute_312, permute_313, permute_314, alias_15, permute_317, div_28, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('coat_lite_mini', benchmark_compiled_module)
