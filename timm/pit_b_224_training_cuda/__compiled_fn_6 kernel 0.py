
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


# kernel path: /tmp/torchinductor_youkaichao/n2/cn2hddyxke6uip7ypypmfmqwqlfv4gtknvhhtjd4pkkbbityrtgu.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]

triton_per_fused_native_layer_norm_backward_select_backward_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_select_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 8
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
    tmp2 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp0 = tl.full([1], 0, tl.int32)
    tmp1 = tmp0 == tmp0
    tmp3 = 0.0
    tmp4 = tl.where(tmp1, tmp2, tmp3)
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp12 = tmp6 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/am/cameny52ohgtvxkztnwmawwjr4dxt7szvvteyrttcg25vdm3vrks.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]

triton_per_fused_native_layer_norm_backward_select_backward_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_select_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0 + (1024*r1)), rmask & xmask, other=0.0)
    tmp0 = tl.full([1, 1], 0, tl.int32)
    tmp1 = tmp0 == tmp0
    tmp3 = 0.0
    tmp4 = tl.where(tmp1, tmp2, tmp3)
    tmp6 = tmp4 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ab/cabivrvuc3255kqyrk3wsmrf4bchp6pnyde4aubkfd674m3dpjyq.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]

triton_poi_fused_native_layer_norm_backward_select_backward_slice_backward_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_backward_select_backward_slice_backward_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 532480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 1024) % 65
    x2 = (xindex // 66560)
    x0 = xindex % 1024
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x2), tmp2, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = tmp0 == tmp4
    tmp6 = tl.load(in_ptr1 + (x0 + (1024*x2)), tmp2, eviction_policy='evict_last', other=0.0)
    tmp7 = 0.0
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tl.load(in_ptr2 + (x0), tmp2, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 * tmp9
    tmp11 = 1024.0
    tmp12 = tmp10 * tmp11
    tmp13 = tl.load(in_ptr3 + (x2), tmp2, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 - tmp13
    tmp15 = tl.load(in_ptr4 + (x0 + (1024*x2)), tmp2, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr5 + (x2), tmp2, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp14 - tmp17
    tmp19 = tmp3 * tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp2, tmp19, tmp20)
    tmp22 = tl.where(tmp2, tmp21, tmp7)
    tl.store(out_ptr0 + (x3), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rq/crqwp5xycb76jlyell2h4xlrbsfzoxmukp2tf3enbxkej3oyacrk.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5120
    rnumel = 104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (106496*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3f/c3f776yuipiyz2oqwo4p5gul2hcpsqalehtlvkmlxtcujrhszb6c.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 5
    RBLOCK: tl.constexpr = 8
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


# kernel path: /tmp/torchinductor_youkaichao/sb/csbiovehc3dijnjodz4y4kyyqcv7l3ndnjiwrnydy5qsispint24.py
# Source Nodes: [x_174], Original ATen: [aten.gelu, aten.gelu_backward]
# x_174 => add_92, erf_12, mul_89
triton_poi_fused_gelu_gelu_backward_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2129920
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


# kernel path: /tmp/torchinductor_youkaichao/sv/csvkkdofmex6ahxerqeazyjwlhz5pp4jo5dptbnvwrpsmhc4edmh.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 20480
    rnumel = 104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4096
    x1 = (xindex // 4096)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (4096*r2) + (425984*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ch/cchzznell5mxnlax5vvdc4w3eu5ocuixhjvy4cnfhnkbqjel364y.py
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
    size_hints=[4096, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 5
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (4096*r1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ep/cepmgmmxdvnkgxffme7iolix7svd4qtkjcipo7nghz3alsqyffgx.py
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
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_9', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 520
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 1024.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jt/cjt4aapdihzr7nxdygecxsbjw2jjcwj2ebtfb2w3iu5ycpvzttyb.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5120
    rnumel = 104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (106496*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (1024*r2) + (106496*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp6 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i2/ci2b45djmt45i2riz4bxefjicfiv2io3t7jmhsns2636wayjktqy.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1597440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x5 = (xindex // 66560)
    x6 = xindex
    x0 = xindex % 1024
    x3 = (xindex // 532480)
    x7 = (xindex // 1024) % 520
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
    tmp12 = tl.load(in_ptr1 + ((-532480) + x6), tmp11, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 24, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-1064960) + x6), tmp15, other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tl.where(tmp11, tmp14, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tl.store(out_ptr0 + (x0 + (1024*x3) + (3072*x7)), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ov/cov5mxrzae4c4zulz46rykpbgxsxib6sp36msjpzomelatqmdwgz.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 15360
    rnumel = 104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3072
    x1 = (xindex // 3072)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (3072*r2) + (319488*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m7/cm7k747g3yjgbveppyl3ocejmfisus3yjzt2zkf5wugtiqzdnke6.py
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
    size_hints=[4096, 8],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 5
    RBLOCK: tl.constexpr = 8
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


# kernel path: /tmp/torchinductor_youkaichao/5y/c5yh5t52a2q7dnhfca55p3ly2kg2e5nh3a7op4sfsu2tf3je7oqj.py
# Source Nodes: [getattr_l__mod___transformers_2_blocks___0___norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# getattr_l__mod___transformers_2_blocks___0___norm1 => mul_63, sub_18
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_14', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel):
    xnumel = 520
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask & xmask, other=0.0)
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
    tmp18 = 1024.0
    tmp19 = tmp10 / tmp18
    tmp20 = tmp2 * tmp18
    tmp21 = tmp20 - tmp6
    tmp22 = tmp11 * tmp16
    tmp23 = tmp21 - tmp22
    tmp24 = tmp19 * tmp23
    tmp25 = tmp17 + tmp24
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/su/csudc2lkk5oih5vuqwtcqbf67tvmnuttmkxh22q7i7ggwviaikwe.py
# Source Nodes: [getattr_l__mod___transformers_2_blocks___0___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# getattr_l__mod___transformers_2_blocks___0___norm1 => mul_63, sub_18
triton_red_fused_native_layer_norm_native_layer_norm_backward_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5120
    rnumel = 104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (106496*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (1024*r2) + (106496*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r2 + (104*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr3 + (r2 + (104*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 - tmp2
        tmp5 = tmp3 * tmp4
        tmp6 = tmp0 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
        tmp10 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p6/cp6bpk4r6r4mpug6z2jawarcc4yn6dns67xq6nero4prucqftavd.py
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
    tmp0 = tl.load(in_ptr0 + (x0 + (66560*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mk/cmkmwjw6s7euqu5dvzwwozic3dbdm6nnlm5zoaqs4lsfs72aktk4.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (1024 + x0 + (1024*(r2 % 64)) + (66560*(r2 // 64)) + (133120*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wz/cwzqjzg7kz3t4zmh2pztn3kcig54s7uimbhgl7rlbd7jvtlap3st.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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


# kernel path: /tmp/torchinductor_youkaichao/va/cva3tnk6zmii37gjb6z5a5kpxb2mmhtq7tjwamjdulbdh4rdqtzf.py
# Source Nodes: [], Original ATen: [aten.add, aten.slice_backward]

triton_poi_fused_add_slice_backward_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_slice_backward_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2056
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 257
    x2 = xindex
    y1 = (yindex // 257)
    y3 = yindex
    tmp0 = y0
    tmp1 = tl.full([1, 1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-1) + y0 + (256*x2) + (131072*y1)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = tmp0 < tmp1
    tmp9 = tl.load(in_ptr1 + (x2 + (512*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = tl.where(tmp8, tmp11, tmp6)
    tmp13 = tmp7 + tmp12
    tl.store(out_ptr0 + (x2 + (512*y3)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hf/chfp5f4k7u7or34y2mdarboi5rhxr6o4fi2zarjzrg56pci53yc7.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8704
    rnumel = 121
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
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 2056, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (512*r2) + (61952*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fu/cfuhrlzfgimymx3inmmiwbiy6oyryyy2fi6utfuj44ujiladcsvq.py
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
    size_hints=[512, 32],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 17
    RBLOCK: tl.constexpr = 32
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


# kernel path: /tmp/torchinductor_youkaichao/64/c64k7l64t5oqknie6tiqwmdc2xp2p77kl3gpbz5zy64efvihelf6.py
# Source Nodes: [x_117], Original ATen: [aten.gelu, aten.gelu_backward]
# x_117 => add_63, erf_8, mul_61
triton_poi_fused_gelu_gelu_backward_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4210688
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


# kernel path: /tmp/torchinductor_youkaichao/zv/czvlaznl4nocs6yn42m46am3cnhywmwpu74676sk3t5starb6lmc.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 34816
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 2048)
    x0 = xindex % 2048
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 2056, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (2048*r2) + (247808*x1)), rmask & tmp2, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/d2/cd2y63rhkt2z23awy6g4rr2bzipdcgwcga3yjmurvkulbe4gj63l.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 32],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 17
    RBLOCK: tl.constexpr = 32
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


# kernel path: /tmp/torchinductor_youkaichao/lw/clws4jeowiyefvtfpi63prxadkroefnsawhjukg544k3zih3irv7.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_25', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 2056
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


# kernel path: /tmp/torchinductor_youkaichao/d4/cd4q7m2uub4gzeillqn4fcn74z6vhbg2hembrf7dfjyclrmmsuqz.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8704
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 512)
    x0 = xindex % 512
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 2056, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (512*((r2 + (121*x1)) % 2056))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (512*((r2 + (121*x1)) % 2056))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/vf/cvfytxqlvakul3upaap33benccqwufdtdbvde7mtoytxwpqnyjqt.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3158016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x5 = (xindex // 131584)
    x6 = xindex
    x0 = xindex % 512
    x3 = (xindex // 1052672)
    x7 = (xindex // 512) % 2056
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
    tmp12 = tl.load(in_ptr1 + ((-1052672) + x6), tmp11, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 24, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-2105344) + x6), tmp15, other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tl.where(tmp11, tmp14, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tl.store(out_ptr0 + (x0 + (512*x3) + (1536*x7)), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sy/csy5hzj5yx764q25oyheihbjp6btcrchy4736qhfw6se6jnteaym.py
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 26112
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
        tmp1 = tl.full([1, 1], 2056, tl.int32)
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


# kernel path: /tmp/torchinductor_youkaichao/df/cdfgz5di4ptea3zf4ikphs5q6nxkcukjz7qcyqeqaqtigjyu23kv.py
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
    size_hints=[2048, 32],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 17
    RBLOCK: tl.constexpr = 32
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


# kernel path: /tmp/torchinductor_youkaichao/cd/ccdl72jb6z3gch3yhiaabf3jwghd3fy4nwkzc2zjddpaftceoluy.py
# Source Nodes: [getattr_l__mod___transformers_1_blocks___0___norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# getattr_l__mod___transformers_1_blocks___0___norm1 => mul_21, sub_6
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_30', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel):
    xnumel = 2056
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


# kernel path: /tmp/torchinductor_youkaichao/ch/cchvvxsdfsw5yzbpalmq4shkupirs3myaoyt7f6uwq47apfcppl3.py
# Source Nodes: [getattr_l__mod___transformers_1_blocks___0___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# getattr_l__mod___transformers_1_blocks___0___norm1 => mul_21, sub_6
triton_red_fused_native_layer_norm_native_layer_norm_backward_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8704
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 512)
    x0 = xindex % 512
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 2056, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (512*((r2 + (121*x1)) % 2056))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (512*((r2 + (121*x1)) % 2056))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + ((r2 + (121*x1)) % 2056), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 - tmp5
        tmp7 = tl.load(in_ptr3 + ((r2 + (121*x1)) % 2056), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/64/c64vtpgfbryaxfbfqomvmzdte553d2n5s3x7jmk2lrqusckfn6fk.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x0 + (131584*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/aw/caws2ebf4vebccqgzn5j5w7mwo4jsupzpuo3a7z5anfzkprhpe7f.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (512 + x0 + (512*(r2 % 16)) + (8192*(((r2 + (128*x1)) // 16) % 16)) + (131584*((r2 + (128*x1)) // 256))), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/so/csonf33zttn7df3m5qhkrgclg4oqkpy3t6mcfkyrxuvefg4fmwv2.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_per_fused_convolution_backward_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_backward_34', 'mutated_arg_names': []}
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
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o5/co5eksno4nbes3bug6mehwmnis5msylguvfansml5gl2p73vwijd.py
# Source Nodes: [], Original ATen: [aten.add, aten.slice_backward]

triton_poi_fused_add_slice_backward_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_slice_backward_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7696
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 962
    x2 = xindex
    y1 = (yindex // 962)
    y3 = yindex
    tmp0 = y0
    tmp1 = tl.full([1, 1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-1) + y0 + (961*x2) + (246016*y1)), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = tmp0 < tmp1
    tmp9 = tl.load(in_ptr1 + (x2 + (256*y1)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = tl.where(tmp8, tmp11, tmp6)
    tmp13 = tmp7 + tmp12
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/df/cdfchgd4zssprstgbm5spvit6fxo2zgzhckwypfmffkyr2oget46.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 15616
    rnumel = 127
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
        tmp0 = r2 + (127*x1)
        tmp1 = tl.full([1, 1], 7696, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (256*r2) + (32512*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tq/ctqs7ftktv5eu3cqk3i5xaxa7rydwkn2pwg334ublbu7ykf2jlzm.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 61
    RBLOCK: tl.constexpr = 64
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


# kernel path: /tmp/torchinductor_youkaichao/ka/ckaxjizzwhammqgpnnfygkzqesjmq4ywy5yeo6pnby2tbcts4noo.py
# Source Nodes: [x_36], Original ATen: [aten.gelu, aten.gelu_backward]
# x_36 => add_20, erf_2, mul_19
triton_poi_fused_gelu_gelu_backward_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_38', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7880704
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


# kernel path: /tmp/torchinductor_youkaichao/rj/crjp3c5tlpr5s5mwd2attsdcwt6zanxupryjjtg6lkxp3txlxvbd.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 62464
    rnumel = 127
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
        tmp0 = r2 + (127*x1)
        tmp1 = tl.full([1, 1], 7696, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (1024*r2) + (130048*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cd/ccdatrrll5445gspsckh3ol4mpf5fv54nr3pt32i6ekftclyqbyx.py
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
    size_hints=[1024, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 61
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


# kernel path: /tmp/torchinductor_youkaichao/t5/ct5wvdmelt24i4vfffrcci2jsrtkm26kjqndltixmx62ivwsoutu.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_41', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 7696
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


# kernel path: /tmp/torchinductor_youkaichao/xd/cxdzqmvn7a22tidmyw6cgip5z3q6yv6zuwgdg4jrt4gahpxc6lio.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 15616
    rnumel = 127
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
        tmp0 = r2 + (127*x1)
        tmp1 = tl.full([1, 1], 7696, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (256*((r2 + (127*x1)) % 7696))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (256*((r2 + (127*x1)) % 7696))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/od/codw4p4js65rahpwbpzskc5nanmm5t7tihonykdz6pighb7iint7.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5910528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x5 = (xindex // 246272)
    x6 = xindex
    x0 = xindex % 256
    x3 = (xindex // 1970176)
    x7 = (xindex // 256) % 7696
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
    tmp12 = tl.load(in_ptr1 + ((-1970176) + x6), tmp11, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 24, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-3940352) + x6), tmp15, other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tl.where(tmp11, tmp14, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tl.store(out_ptr0 + (x0 + (256*x3) + (768*x7)), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ek/cekdwnanbghqf64nr2ufgxaccjuadq2ede7jacwprlvljjrvlkqf.py
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
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 46848
    rnumel = 127
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
        tmp0 = r2 + (127*x1)
        tmp1 = tl.full([1, 1], 7696, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (768*r2) + (97536*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/52/c523o2k4bm72jylla4vyeglsy76gytw4ijhjnz2462vkltsftjc2.py
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
    size_hints=[1024, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 61
    RBLOCK: tl.constexpr = 64
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


# kernel path: /tmp/torchinductor_youkaichao/kp/ckpghi3v52kmd562rcovqrn5q4uibtmrk7ylxow3t6ekloptcdik.py
# Source Nodes: [getattr_l__mod___transformers_0_blocks___0___norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# getattr_l__mod___transformers_0_blocks___0___norm1 => mul, sub
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_46', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel):
    xnumel = 7696
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


# kernel path: /tmp/torchinductor_youkaichao/eq/ceq6w3uwnryzl7otgefrkjg3g4iq4sp2ndlktzecfugyj25khnyl.py
# Source Nodes: [getattr_l__mod___transformers_0_blocks___0___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# getattr_l__mod___transformers_0_blocks___0___norm1 => mul, sub
triton_red_fused_native_layer_norm_native_layer_norm_backward_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 15616
    rnumel = 127
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
        tmp0 = r2 + (127*x1)
        tmp1 = tl.full([1, 1], 7696, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (256*((r2 + (127*x1)) % 7696))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (256*((r2 + (127*x1)) % 7696))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + ((r2 + (127*x1)) % 7696), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 - tmp5
        tmp7 = tl.load(in_ptr3 + ((r2 + (127*x1)) % 7696), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/el/cel3leondapeqqr4qr4t27fwpfzpi6zqhhxxauy5gxfnk4k4rbms.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_48', 'mutated_arg_names': []}
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
    tmp0 = tl.load(in_ptr0 + (x0 + (246272*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tx/ctxisw3tzcacshacnkwc5qpiyc62bs6cta7piqigkgmaqa26txri.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 246016
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 961
    x1 = (xindex // 961)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + x1 + (256*x0) + (246272*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b6/cb6uknerig4xfwph3oy2ocondvzf2v6yqeuzekwnmkxgpfghsnyz.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 15616
    rnumel = 127
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
        tmp0 = r2 + (127*x1)
        tmp1 = tl.full([1, 1], 7688, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (256 + x0 + (256*((r2 + (127*x1)) % 961)) + (246272*(((r2 + (127*x1)) // 961) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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
    primals_3, primals_5, primals_11, primals_17, primals_23, primals_29, primals_35, primals_41, primals_45, primals_51, primals_57, primals_63, primals_69, primals_75, primals_81, primals_87, primals_93, primals_99, primals_105, primals_111, primals_117, primals_121, primals_127, primals_133, primals_139, primals_145, primals_151, primals_157, primals_163, primals_169, primals_173, cat, getitem_1, rsqrt, view_1, getitem_2, getitem_3, getitem_4, getitem_6, getitem_7, getitem_8, view_5, mul_2, view_7, addmm_2, view_9, mul_7, view_11, getitem_13, getitem_14, getitem_15, getitem_17, getitem_18, getitem_19, view_15, mul_9, view_17, addmm_6, view_19, mul_14, view_21, getitem_24, getitem_25, getitem_26, getitem_28, getitem_29, getitem_30, view_25, mul_16, view_27, addmm_10, view_29, view_31, view_32, cat_1, getitem_34, rsqrt_6, view_35, getitem_35, getitem_36, getitem_37, getitem_39, getitem_40, getitem_41, view_39, mul_23, view_41, addmm_14, view_43, mul_28, view_45, getitem_46, getitem_47, getitem_48, getitem_50, getitem_51, getitem_52, view_49, mul_30, view_51, addmm_18, view_53, mul_35, view_55, getitem_57, getitem_58, getitem_59, getitem_61, getitem_62, getitem_63, view_59, mul_37, view_61, addmm_22, view_63, mul_42, view_65, getitem_68, getitem_69, getitem_70, getitem_72, getitem_73, getitem_74, view_69, mul_44, view_71, addmm_26, view_73, mul_49, view_75, getitem_79, getitem_80, getitem_81, getitem_83, getitem_84, getitem_85, view_79, mul_51, view_81, addmm_30, view_83, mul_56, view_85, getitem_90, getitem_91, getitem_92, getitem_94, getitem_95, getitem_96, view_89, mul_58, view_91, addmm_34, view_93, view_95, view_96, cat_2, getitem_100, rsqrt_18, view_99, getitem_101, getitem_102, getitem_103, getitem_105, getitem_106, getitem_107, view_103, mul_65, view_105, addmm_38, view_107, mul_70, view_109, getitem_112, getitem_113, getitem_114, getitem_116, getitem_117, getitem_118, view_113, mul_72, view_115, addmm_42, view_117, mul_77, view_119, getitem_123, getitem_124, getitem_125, getitem_127, getitem_128, getitem_129, view_123, mul_79, view_125, addmm_46, view_127, mul_84, view_129, getitem_134, getitem_135, getitem_136, getitem_138, getitem_139, getitem_140, view_133, mul_86, view_135, addmm_50, view_137, mul_91, clone_41, permute_87, div, permute_91, permute_95, div_1, permute_99, alias_13, permute_105, div_2, permute_109, permute_113, div_3, permute_117, alias_14, permute_123, div_4, permute_127, permute_131, div_5, permute_135, alias_15, permute_141, div_6, permute_145, permute_149, div_7, permute_153, alias_16, permute_159, permute_166, permute_169, permute_173, div_9, permute_177, alias_17, permute_183, div_10, permute_187, permute_191, div_11, permute_195, alias_18, permute_201, div_12, permute_205, permute_209, div_13, permute_213, alias_19, permute_219, div_14, permute_223, permute_227, div_15, permute_231, alias_20, permute_237, div_16, permute_241, permute_245, div_17, permute_249, alias_21, permute_255, div_18, permute_259, permute_263, div_19, permute_267, alias_22, permute_273, permute_280, permute_283, permute_287, div_21, permute_291, alias_23, permute_297, div_22, permute_301, permute_305, div_23, permute_309, alias_24, permute_315, div_24, permute_319, permute_323, div_25, permute_327, alias_25, permute_333, tangents_1 = args
    args.clear()
    assert_size_stride(primals_3, (256, 3, 14, 14), (588, 196, 14, 1))
    assert_size_stride(primals_5, (256, ), (1, ))
    assert_size_stride(primals_11, (256, ), (1, ))
    assert_size_stride(primals_17, (256, ), (1, ))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_29, (256, ), (1, ))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_41, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_45, (512, ), (1, ))
    assert_size_stride(primals_51, (512, ), (1, ))
    assert_size_stride(primals_57, (512, ), (1, ))
    assert_size_stride(primals_63, (512, ), (1, ))
    assert_size_stride(primals_69, (512, ), (1, ))
    assert_size_stride(primals_75, (512, ), (1, ))
    assert_size_stride(primals_81, (512, ), (1, ))
    assert_size_stride(primals_87, (512, ), (1, ))
    assert_size_stride(primals_93, (512, ), (1, ))
    assert_size_stride(primals_99, (512, ), (1, ))
    assert_size_stride(primals_105, (512, ), (1, ))
    assert_size_stride(primals_111, (512, ), (1, ))
    assert_size_stride(primals_117, (1024, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_121, (1024, ), (1, ))
    assert_size_stride(primals_127, (1024, ), (1, ))
    assert_size_stride(primals_133, (1024, ), (1, ))
    assert_size_stride(primals_139, (1024, ), (1, ))
    assert_size_stride(primals_145, (1024, ), (1, ))
    assert_size_stride(primals_151, (1024, ), (1, ))
    assert_size_stride(primals_157, (1024, ), (1, ))
    assert_size_stride(primals_163, (1024, ), (1, ))
    assert_size_stride(primals_169, (1024, ), (1, ))
    assert_size_stride(primals_173, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(cat, (8, 962, 256), (246272, 256, 1))
    assert_size_stride(getitem_1, (8, 962, 1), (962, 1, 1))
    assert_size_stride(rsqrt, (8, 962, 1), (962, 1, 1))
    assert_size_stride(view_1, (7696, 256), (256, 1))
    assert_size_stride(getitem_2, (8, 4, 962, 64), (738816, 64, 768, 1))
    assert_size_stride(getitem_3, (8, 4, 962, 64), (738816, 64, 768, 1))
    assert_size_stride(getitem_4, (8, 4, 962, 64), (738816, 64, 768, 1))
    assert_size_stride(getitem_6, (8, 4, 992), (3968, 992, 1))
    assert_size_stride(getitem_7, (), ())
    assert_size_stride(getitem_8, (), ())
    assert_size_stride(view_5, (7696, 256), (256, 1))
    assert_size_stride(mul_2, (8, 962, 256), (246272, 256, 1))
    assert_size_stride(view_7, (7696, 256), (256, 1))
    assert_size_stride(addmm_2, (7696, 1024), (1024, 1))
    assert_size_stride(view_9, (7696, 1024), (1024, 1))
    assert_size_stride(mul_7, (8, 962, 256), (246272, 256, 1))
    assert_size_stride(view_11, (7696, 256), (256, 1))
    assert_size_stride(getitem_13, (8, 4, 962, 64), (738816, 64, 768, 1))
    assert_size_stride(getitem_14, (8, 4, 962, 64), (738816, 64, 768, 1))
    assert_size_stride(getitem_15, (8, 4, 962, 64), (738816, 64, 768, 1))
    assert_size_stride(getitem_17, (8, 4, 992), (3968, 992, 1))
    assert_size_stride(getitem_18, (), ())
    assert_size_stride(getitem_19, (), ())
    assert_size_stride(view_15, (7696, 256), (256, 1))
    assert_size_stride(mul_9, (8, 962, 256), (246272, 256, 1))
    assert_size_stride(view_17, (7696, 256), (256, 1))
    assert_size_stride(addmm_6, (7696, 1024), (1024, 1))
    assert_size_stride(view_19, (7696, 1024), (1024, 1))
    assert_size_stride(mul_14, (8, 962, 256), (246272, 256, 1))
    assert_size_stride(view_21, (7696, 256), (256, 1))
    assert_size_stride(getitem_24, (8, 4, 962, 64), (738816, 64, 768, 1))
    assert_size_stride(getitem_25, (8, 4, 962, 64), (738816, 64, 768, 1))
    assert_size_stride(getitem_26, (8, 4, 962, 64), (738816, 64, 768, 1))
    assert_size_stride(getitem_28, (8, 4, 992), (3968, 992, 1))
    assert_size_stride(getitem_29, (), ())
    assert_size_stride(getitem_30, (), ())
    assert_size_stride(view_25, (7696, 256), (256, 1))
    assert_size_stride(mul_16, (8, 962, 256), (246272, 256, 1))
    assert_size_stride(view_27, (7696, 256), (256, 1))
    assert_size_stride(addmm_10, (7696, 1024), (1024, 1))
    assert_size_stride(view_29, (7696, 1024), (1024, 1))
    assert_size_stride(view_31, (8, 256, 31, 31), (246272, 1, 7936, 256))
    assert_size_stride(view_32, (8, 256), (246272, 1))
    assert_size_stride(cat_1, (8, 257, 512), (131584, 512, 1))
    assert_size_stride(getitem_34, (8, 257, 1), (257, 1, 1))
    assert_size_stride(rsqrt_6, (8, 257, 1), (257, 1, 1))
    assert_size_stride(view_35, (2056, 512), (512, 1))
    assert_size_stride(getitem_35, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_36, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_37, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_39, (8, 8, 288), (2304, 288, 1))
    assert_size_stride(getitem_40, (), ())
    assert_size_stride(getitem_41, (), ())
    assert_size_stride(view_39, (2056, 512), (512, 1))
    assert_size_stride(mul_23, (8, 257, 512), (131584, 512, 1))
    assert_size_stride(view_41, (2056, 512), (512, 1))
    assert_size_stride(addmm_14, (2056, 2048), (2048, 1))
    assert_size_stride(view_43, (2056, 2048), (2048, 1))
    assert_size_stride(mul_28, (8, 257, 512), (131584, 512, 1))
    assert_size_stride(view_45, (2056, 512), (512, 1))
    assert_size_stride(getitem_46, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_47, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_48, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_50, (8, 8, 288), (2304, 288, 1))
    assert_size_stride(getitem_51, (), ())
    assert_size_stride(getitem_52, (), ())
    assert_size_stride(view_49, (2056, 512), (512, 1))
    assert_size_stride(mul_30, (8, 257, 512), (131584, 512, 1))
    assert_size_stride(view_51, (2056, 512), (512, 1))
    assert_size_stride(addmm_18, (2056, 2048), (2048, 1))
    assert_size_stride(view_53, (2056, 2048), (2048, 1))
    assert_size_stride(mul_35, (8, 257, 512), (131584, 512, 1))
    assert_size_stride(view_55, (2056, 512), (512, 1))
    assert_size_stride(getitem_57, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_58, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_59, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_61, (8, 8, 288), (2304, 288, 1))
    assert_size_stride(getitem_62, (), ())
    assert_size_stride(getitem_63, (), ())
    assert_size_stride(view_59, (2056, 512), (512, 1))
    assert_size_stride(mul_37, (8, 257, 512), (131584, 512, 1))
    assert_size_stride(view_61, (2056, 512), (512, 1))
    assert_size_stride(addmm_22, (2056, 2048), (2048, 1))
    assert_size_stride(view_63, (2056, 2048), (2048, 1))
    assert_size_stride(mul_42, (8, 257, 512), (131584, 512, 1))
    assert_size_stride(view_65, (2056, 512), (512, 1))
    assert_size_stride(getitem_68, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_69, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_70, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_72, (8, 8, 288), (2304, 288, 1))
    assert_size_stride(getitem_73, (), ())
    assert_size_stride(getitem_74, (), ())
    assert_size_stride(view_69, (2056, 512), (512, 1))
    assert_size_stride(mul_44, (8, 257, 512), (131584, 512, 1))
    assert_size_stride(view_71, (2056, 512), (512, 1))
    assert_size_stride(addmm_26, (2056, 2048), (2048, 1))
    assert_size_stride(view_73, (2056, 2048), (2048, 1))
    assert_size_stride(mul_49, (8, 257, 512), (131584, 512, 1))
    assert_size_stride(view_75, (2056, 512), (512, 1))
    assert_size_stride(getitem_79, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_80, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_81, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_83, (8, 8, 288), (2304, 288, 1))
    assert_size_stride(getitem_84, (), ())
    assert_size_stride(getitem_85, (), ())
    assert_size_stride(view_79, (2056, 512), (512, 1))
    assert_size_stride(mul_51, (8, 257, 512), (131584, 512, 1))
    assert_size_stride(view_81, (2056, 512), (512, 1))
    assert_size_stride(addmm_30, (2056, 2048), (2048, 1))
    assert_size_stride(view_83, (2056, 2048), (2048, 1))
    assert_size_stride(mul_56, (8, 257, 512), (131584, 512, 1))
    assert_size_stride(view_85, (2056, 512), (512, 1))
    assert_size_stride(getitem_90, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_91, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_92, (8, 8, 257, 64), (394752, 64, 1536, 1))
    assert_size_stride(getitem_94, (8, 8, 288), (2304, 288, 1))
    assert_size_stride(getitem_95, (), ())
    assert_size_stride(getitem_96, (), ())
    assert_size_stride(view_89, (2056, 512), (512, 1))
    assert_size_stride(mul_58, (8, 257, 512), (131584, 512, 1))
    assert_size_stride(view_91, (2056, 512), (512, 1))
    assert_size_stride(addmm_34, (2056, 2048), (2048, 1))
    assert_size_stride(view_93, (2056, 2048), (2048, 1))
    assert_size_stride(view_95, (8, 512, 16, 16), (131584, 1, 8192, 512))
    assert_size_stride(view_96, (8, 512), (131584, 1))
    assert_size_stride(cat_2, (8, 65, 1024), (66560, 1024, 1))
    assert_size_stride(getitem_100, (8, 65, 1), (65, 1, 1))
    assert_size_stride(rsqrt_18, (8, 65, 1), (65, 1, 1))
    assert_size_stride(view_99, (520, 1024), (1024, 1))
    assert_size_stride(getitem_101, (8, 16, 65, 64), (199680, 64, 3072, 1))
    assert_size_stride(getitem_102, (8, 16, 65, 64), (199680, 64, 3072, 1))
    assert_size_stride(getitem_103, (8, 16, 65, 64), (199680, 64, 3072, 1))
    assert_size_stride(getitem_105, (8, 16, 96), (1536, 96, 1))
    assert_size_stride(getitem_106, (), ())
    assert_size_stride(getitem_107, (), ())
    assert_size_stride(view_103, (520, 1024), (1024, 1))
    assert_size_stride(mul_65, (8, 65, 1024), (66560, 1024, 1))
    assert_size_stride(view_105, (520, 1024), (1024, 1))
    assert_size_stride(addmm_38, (520, 4096), (4096, 1))
    assert_size_stride(view_107, (520, 4096), (4096, 1))
    assert_size_stride(mul_70, (8, 65, 1024), (66560, 1024, 1))
    assert_size_stride(view_109, (520, 1024), (1024, 1))
    assert_size_stride(getitem_112, (8, 16, 65, 64), (199680, 64, 3072, 1))
    assert_size_stride(getitem_113, (8, 16, 65, 64), (199680, 64, 3072, 1))
    assert_size_stride(getitem_114, (8, 16, 65, 64), (199680, 64, 3072, 1))
    assert_size_stride(getitem_116, (8, 16, 96), (1536, 96, 1))
    assert_size_stride(getitem_117, (), ())
    assert_size_stride(getitem_118, (), ())
    assert_size_stride(view_113, (520, 1024), (1024, 1))
    assert_size_stride(mul_72, (8, 65, 1024), (66560, 1024, 1))
    assert_size_stride(view_115, (520, 1024), (1024, 1))
    assert_size_stride(addmm_42, (520, 4096), (4096, 1))
    assert_size_stride(view_117, (520, 4096), (4096, 1))
    assert_size_stride(mul_77, (8, 65, 1024), (66560, 1024, 1))
    assert_size_stride(view_119, (520, 1024), (1024, 1))
    assert_size_stride(getitem_123, (8, 16, 65, 64), (199680, 64, 3072, 1))
    assert_size_stride(getitem_124, (8, 16, 65, 64), (199680, 64, 3072, 1))
    assert_size_stride(getitem_125, (8, 16, 65, 64), (199680, 64, 3072, 1))
    assert_size_stride(getitem_127, (8, 16, 96), (1536, 96, 1))
    assert_size_stride(getitem_128, (), ())
    assert_size_stride(getitem_129, (), ())
    assert_size_stride(view_123, (520, 1024), (1024, 1))
    assert_size_stride(mul_79, (8, 65, 1024), (66560, 1024, 1))
    assert_size_stride(view_125, (520, 1024), (1024, 1))
    assert_size_stride(addmm_46, (520, 4096), (4096, 1))
    assert_size_stride(view_127, (520, 4096), (4096, 1))
    assert_size_stride(mul_84, (8, 65, 1024), (66560, 1024, 1))
    assert_size_stride(view_129, (520, 1024), (1024, 1))
    assert_size_stride(getitem_134, (8, 16, 65, 64), (199680, 64, 3072, 1))
    assert_size_stride(getitem_135, (8, 16, 65, 64), (199680, 64, 3072, 1))
    assert_size_stride(getitem_136, (8, 16, 65, 64), (199680, 64, 3072, 1))
    assert_size_stride(getitem_138, (8, 16, 96), (1536, 96, 1))
    assert_size_stride(getitem_139, (), ())
    assert_size_stride(getitem_140, (), ())
    assert_size_stride(view_133, (520, 1024), (1024, 1))
    assert_size_stride(mul_86, (8, 65, 1024), (66560, 1024, 1))
    assert_size_stride(view_135, (520, 1024), (1024, 1))
    assert_size_stride(addmm_50, (520, 4096), (4096, 1))
    assert_size_stride(view_137, (520, 4096), (4096, 1))
    assert_size_stride(mul_91, (8, 1, 1024), (1024, 1024, 1))
    assert_size_stride(clone_41, (8, 1024), (1024, 1))
    assert_size_stride(permute_87, (1000, 1024), (1024, 1))
    assert_size_stride(div, (8, 1, 1), (1, 1, 1))
    assert_size_stride(permute_91, (1024, 4096), (4096, 1))
    assert_size_stride(permute_95, (4096, 1024), (1024, 1))
    assert_size_stride(div_1, (8, 65, 1), (65, 1, 1))
    assert_size_stride(permute_99, (1024, 1024), (1024, 1))
    assert_size_stride(alias_13, (8, 16, 65, 64), (66560, 64, 1024, 1))
    assert_size_stride(permute_105, (3072, 1024), (1024, 1))
    assert_size_stride(div_2, (8, 65, 1), (65, 1, 1))
    assert_size_stride(permute_109, (1024, 4096), (4096, 1))
    assert_size_stride(permute_113, (4096, 1024), (1024, 1))
    assert_size_stride(div_3, (8, 65, 1), (65, 1, 1))
    assert_size_stride(permute_117, (1024, 1024), (1024, 1))
    assert_size_stride(alias_14, (8, 16, 65, 64), (66560, 64, 1024, 1))
    assert_size_stride(permute_123, (3072, 1024), (1024, 1))
    assert_size_stride(div_4, (8, 65, 1), (65, 1, 1))
    assert_size_stride(permute_127, (1024, 4096), (4096, 1))
    assert_size_stride(permute_131, (4096, 1024), (1024, 1))
    assert_size_stride(div_5, (8, 65, 1), (65, 1, 1))
    assert_size_stride(permute_135, (1024, 1024), (1024, 1))
    assert_size_stride(alias_15, (8, 16, 65, 64), (66560, 64, 1024, 1))
    assert_size_stride(permute_141, (3072, 1024), (1024, 1))
    assert_size_stride(div_6, (8, 65, 1), (65, 1, 1))
    assert_size_stride(permute_145, (1024, 4096), (4096, 1))
    assert_size_stride(permute_149, (4096, 1024), (1024, 1))
    assert_size_stride(div_7, (8, 65, 1), (65, 1, 1))
    assert_size_stride(permute_153, (1024, 1024), (1024, 1))
    assert_size_stride(alias_16, (8, 16, 65, 64), (66560, 64, 1024, 1))
    assert_size_stride(permute_159, (3072, 1024), (1024, 1))
    assert_size_stride(permute_166, (1024, 512), (512, 1))
    assert_size_stride(permute_169, (512, 2048), (2048, 1))
    assert_size_stride(permute_173, (2048, 512), (512, 1))
    assert_size_stride(div_9, (8, 257, 1), (257, 1, 1))
    assert_size_stride(permute_177, (512, 512), (512, 1))
    assert_size_stride(alias_17, (8, 8, 257, 64), (131584, 64, 512, 1))
    assert_size_stride(permute_183, (1536, 512), (512, 1))
    assert_size_stride(div_10, (8, 257, 1), (257, 1, 1))
    assert_size_stride(permute_187, (512, 2048), (2048, 1))
    assert_size_stride(permute_191, (2048, 512), (512, 1))
    assert_size_stride(div_11, (8, 257, 1), (257, 1, 1))
    assert_size_stride(permute_195, (512, 512), (512, 1))
    assert_size_stride(alias_18, (8, 8, 257, 64), (131584, 64, 512, 1))
    assert_size_stride(permute_201, (1536, 512), (512, 1))
    assert_size_stride(div_12, (8, 257, 1), (257, 1, 1))
    assert_size_stride(permute_205, (512, 2048), (2048, 1))
    assert_size_stride(permute_209, (2048, 512), (512, 1))
    assert_size_stride(div_13, (8, 257, 1), (257, 1, 1))
    assert_size_stride(permute_213, (512, 512), (512, 1))
    assert_size_stride(alias_19, (8, 8, 257, 64), (131584, 64, 512, 1))
    assert_size_stride(permute_219, (1536, 512), (512, 1))
    assert_size_stride(div_14, (8, 257, 1), (257, 1, 1))
    assert_size_stride(permute_223, (512, 2048), (2048, 1))
    assert_size_stride(permute_227, (2048, 512), (512, 1))
    assert_size_stride(div_15, (8, 257, 1), (257, 1, 1))
    assert_size_stride(permute_231, (512, 512), (512, 1))
    assert_size_stride(alias_20, (8, 8, 257, 64), (131584, 64, 512, 1))
    assert_size_stride(permute_237, (1536, 512), (512, 1))
    assert_size_stride(div_16, (8, 257, 1), (257, 1, 1))
    assert_size_stride(permute_241, (512, 2048), (2048, 1))
    assert_size_stride(permute_245, (2048, 512), (512, 1))
    assert_size_stride(div_17, (8, 257, 1), (257, 1, 1))
    assert_size_stride(permute_249, (512, 512), (512, 1))
    assert_size_stride(alias_21, (8, 8, 257, 64), (131584, 64, 512, 1))
    assert_size_stride(permute_255, (1536, 512), (512, 1))
    assert_size_stride(div_18, (8, 257, 1), (257, 1, 1))
    assert_size_stride(permute_259, (512, 2048), (2048, 1))
    assert_size_stride(permute_263, (2048, 512), (512, 1))
    assert_size_stride(div_19, (8, 257, 1), (257, 1, 1))
    assert_size_stride(permute_267, (512, 512), (512, 1))
    assert_size_stride(alias_22, (8, 8, 257, 64), (131584, 64, 512, 1))
    assert_size_stride(permute_273, (1536, 512), (512, 1))
    assert_size_stride(permute_280, (512, 256), (256, 1))
    assert_size_stride(permute_283, (256, 1024), (1024, 1))
    assert_size_stride(permute_287, (1024, 256), (256, 1))
    assert_size_stride(div_21, (8, 962, 1), (962, 1, 1))
    assert_size_stride(permute_291, (256, 256), (256, 1))
    assert_size_stride(alias_23, (8, 4, 962, 64), (246272, 64, 256, 1))
    assert_size_stride(permute_297, (768, 256), (256, 1))
    assert_size_stride(div_22, (8, 962, 1), (962, 1, 1))
    assert_size_stride(permute_301, (256, 1024), (1024, 1))
    assert_size_stride(permute_305, (1024, 256), (256, 1))
    assert_size_stride(div_23, (8, 962, 1), (962, 1, 1))
    assert_size_stride(permute_309, (256, 256), (256, 1))
    assert_size_stride(alias_24, (8, 4, 962, 64), (246272, 64, 256, 1))
    assert_size_stride(permute_315, (768, 256), (256, 1))
    assert_size_stride(div_24, (8, 962, 1), (962, 1, 1))
    assert_size_stride(permute_319, (256, 1024), (1024, 1))
    assert_size_stride(permute_323, (1024, 256), (256, 1))
    assert_size_stride(div_25, (8, 962, 1), (962, 1, 1))
    assert_size_stride(permute_327, (256, 256), (256, 1))
    assert_size_stride(alias_25, (8, 4, 962, 64), (246272, 64, 256, 1))
    assert_size_stride(permute_333, (768, 256), (256, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_87, out=buf0)
        del permute_87
        buf1 = empty((1000, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_41, out=buf1)
        del clone_41
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf3 = empty_strided((8, 1, 1), (1, 8, 8), device='cuda', dtype=torch.float32)
        buf4 = empty_strided((8, 1, 1), (1, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_1.run(buf0, primals_169, mul_91, buf3, buf4, 8, 1024, grid=grid(8), stream=stream0)
        buf5 = empty((1024, ), device='cuda', dtype=torch.float32)
        buf6 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward]
        triton_per_fused_native_layer_norm_backward_select_backward_2.run(buf0, mul_91, buf5, buf6, 1024, 8, grid=grid(1024), stream=stream0)
        buf7 = empty((8, 65, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_poi_fused_native_layer_norm_backward_select_backward_slice_backward_3.run(div, buf0, primals_169, buf3, mul_91, buf4, buf7, 532480, grid=grid(532480), stream=stream0)
        del buf3
        del buf4
        del div
        del mul_91
        del primals_169
        buf8 = empty((520, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf7, (520, 1024), (1024, 1), 0), permute_91, out=buf8)
        del permute_91
        buf9 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf7, (1024, 520), (1, 1024), 0), view_137, out=buf9)
        del view_137
        buf10 = empty_strided((1, 1024, 5), (5120, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf7, buf10, 5120, 104, grid=grid(5120), stream=stream0)
        buf11 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf10, buf11, 1024, 5, grid=grid(1024), stream=stream0)
        buf12 = reinterpret_tensor(buf8, (8, 65, 4096), (266240, 4096, 1), 0); del buf8  # reuse
        # Source Nodes: [x_174], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf12, addmm_50, 2129920, grid=grid(2129920), stream=stream0)
        del addmm_50
        buf13 = empty((520, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf12, (520, 4096), (4096, 1), 0), permute_95, out=buf13)
        del permute_95
        buf14 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf12, (4096, 520), (1, 4096), 0), view_135, out=buf14)
        del view_135
        buf15 = empty_strided((1, 4096, 5), (20480, 1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf12, buf15, 20480, 104, grid=grid(20480), stream=stream0)
        buf16 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf15, buf16, 4096, 5, grid=grid(4096), stream=stream0)
        buf23 = buf7; del buf7  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf23, buf13, primals_163, mul_86, div_1, 520, 1024, grid=grid(520), stream=stream0)
        del div_1
        del primals_163
        buf19 = reinterpret_tensor(buf10, (1024, 5), (1, 1024), 0); del buf10  # reuse
        buf21 = empty_strided((1024, 5), (1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf13, mul_86, buf19, buf21, 5120, 104, grid=grid(5120), stream=stream0)
        del mul_86
        buf20 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_5.run(buf19, buf20, 1024, 5, grid=grid(1024), stream=stream0)
        buf22 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_5.run(buf21, buf22, 1024, 5, grid=grid(1024), stream=stream0)
        buf24 = buf13; del buf13  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (520, 1024), (1024, 1), 0), permute_99, out=buf24)
        del permute_99
        buf25 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (1024, 520), (1, 1024), 0), view_133, out=buf25)
        del view_133
        buf26 = reinterpret_tensor(buf21, (1, 1024, 5), (5120, 1, 1024), 0); del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf23, buf26, 5120, 104, grid=grid(5120), stream=stream0)
        buf27 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf26, buf27, 1024, 5, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf28 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf24, (8, 16, 65, 64), (66560, 64, 1024, 1), 0), getitem_134, getitem_135, getitem_136, None, alias_13, getitem_138, getitem_139, getitem_140, 0.0, [True, True, True, False])
        del alias_13
        del buf24
        del getitem_134
        del getitem_135
        del getitem_136
        del getitem_138
        del getitem_139
        del getitem_140
        buf29 = buf28[0]
        buf30 = buf28[1]
        buf31 = buf28[2]
        del buf28
        buf32 = empty((8, 65, 3, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf29, buf30, buf31, buf32, 1597440, grid=grid(1597440), stream=stream0)
        del buf29
        del buf30
        buf33 = reinterpret_tensor(buf31, (520, 1024), (1024, 1), 0); del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf32, (520, 3072), (3072, 1), 0), permute_105, out=buf33)
        del permute_105
        buf34 = empty((3072, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf32, (3072, 520), (1, 3072), 0), view_129, out=buf34)
        del view_129
        buf35 = empty_strided((1, 3072, 5), (15360, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf32, buf35, 15360, 104, grid=grid(15360), stream=stream0)
        buf36 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf35, buf36, 3072, 5, grid=grid(3072), stream=stream0)
        buf43 = buf23; del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf43, buf33, primals_157, mul_84, div_2, 520, 1024, grid=grid(520), stream=stream0)
        del div_2
        del primals_157
        buf39 = reinterpret_tensor(buf26, (1024, 5), (1, 1024), 0); del buf26  # reuse
        buf41 = buf19; del buf19  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf33, mul_84, buf39, buf41, 5120, 104, grid=grid(5120), stream=stream0)
        del mul_84
        buf40 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_5.run(buf39, buf40, 1024, 5, grid=grid(1024), stream=stream0)
        buf42 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_5.run(buf41, buf42, 1024, 5, grid=grid(1024), stream=stream0)
        buf44 = reinterpret_tensor(buf12, (520, 4096), (4096, 1), 0); del buf12  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf43, (520, 1024), (1024, 1), 0), permute_109, out=buf44)
        del permute_109
        buf45 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf43, (1024, 520), (1, 1024), 0), view_127, out=buf45)
        del view_127
        buf46 = reinterpret_tensor(buf41, (1, 1024, 5), (5120, 1, 1024), 0); del buf41  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf43, buf46, 5120, 104, grid=grid(5120), stream=stream0)
        buf47 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf46, buf47, 1024, 5, grid=grid(1024), stream=stream0)
        buf48 = reinterpret_tensor(buf44, (8, 65, 4096), (266240, 4096, 1), 0); del buf44  # reuse
        # Source Nodes: [x_162], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf48, addmm_46, 2129920, grid=grid(2129920), stream=stream0)
        del addmm_46
        buf49 = buf33; del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf48, (520, 4096), (4096, 1), 0), permute_113, out=buf49)
        del permute_113
        buf50 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf48, (4096, 520), (1, 4096), 0), view_125, out=buf50)
        del view_125
        buf51 = buf15; del buf15  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf48, buf51, 20480, 104, grid=grid(20480), stream=stream0)
        buf52 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf51, buf52, 4096, 5, grid=grid(4096), stream=stream0)
        buf59 = buf43; del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf59, buf49, primals_151, mul_79, div_3, 520, 1024, grid=grid(520), stream=stream0)
        del div_3
        del primals_151
        buf55 = reinterpret_tensor(buf46, (1024, 5), (1, 1024), 0); del buf46  # reuse
        buf57 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf49, mul_79, buf55, buf57, 5120, 104, grid=grid(5120), stream=stream0)
        del mul_79
        buf56 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_5.run(buf55, buf56, 1024, 5, grid=grid(1024), stream=stream0)
        buf58 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_5.run(buf57, buf58, 1024, 5, grid=grid(1024), stream=stream0)
        buf60 = buf49; del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf59, (520, 1024), (1024, 1), 0), permute_117, out=buf60)
        del permute_117
        buf61 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf59, (1024, 520), (1, 1024), 0), view_123, out=buf61)
        del view_123
        buf62 = reinterpret_tensor(buf57, (1, 1024, 5), (5120, 1, 1024), 0); del buf57  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf59, buf62, 5120, 104, grid=grid(5120), stream=stream0)
        buf63 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf62, buf63, 1024, 5, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf64 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf60, (8, 16, 65, 64), (66560, 64, 1024, 1), 0), getitem_123, getitem_124, getitem_125, None, alias_14, getitem_127, getitem_128, getitem_129, 0.0, [True, True, True, False])
        del alias_14
        del buf60
        del getitem_123
        del getitem_124
        del getitem_125
        del getitem_127
        del getitem_128
        del getitem_129
        buf65 = buf64[0]
        buf66 = buf64[1]
        buf67 = buf64[2]
        del buf64
        buf68 = buf32; del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf65, buf66, buf67, buf68, 1597440, grid=grid(1597440), stream=stream0)
        del buf65
        del buf66
        buf69 = reinterpret_tensor(buf67, (520, 1024), (1024, 1), 0); del buf67  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf68, (520, 3072), (3072, 1), 0), permute_123, out=buf69)
        del permute_123
        buf70 = empty((3072, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf68, (3072, 520), (1, 3072), 0), view_119, out=buf70)
        del view_119
        buf71 = buf35; del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf68, buf71, 15360, 104, grid=grid(15360), stream=stream0)
        buf72 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf71, buf72, 3072, 5, grid=grid(3072), stream=stream0)
        buf79 = buf59; del buf59  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf79, buf69, primals_145, mul_77, div_4, 520, 1024, grid=grid(520), stream=stream0)
        del div_4
        del primals_145
        buf75 = reinterpret_tensor(buf62, (1024, 5), (1, 1024), 0); del buf62  # reuse
        buf77 = buf55; del buf55  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf69, mul_77, buf75, buf77, 5120, 104, grid=grid(5120), stream=stream0)
        del mul_77
        buf76 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_5.run(buf75, buf76, 1024, 5, grid=grid(1024), stream=stream0)
        buf78 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_5.run(buf77, buf78, 1024, 5, grid=grid(1024), stream=stream0)
        buf80 = reinterpret_tensor(buf48, (520, 4096), (4096, 1), 0); del buf48  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf79, (520, 1024), (1024, 1), 0), permute_127, out=buf80)
        del permute_127
        buf81 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf79, (1024, 520), (1, 1024), 0), view_117, out=buf81)
        del view_117
        buf82 = reinterpret_tensor(buf77, (1, 1024, 5), (5120, 1, 1024), 0); del buf77  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf79, buf82, 5120, 104, grid=grid(5120), stream=stream0)
        buf83 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf82, buf83, 1024, 5, grid=grid(1024), stream=stream0)
        buf84 = reinterpret_tensor(buf80, (8, 65, 4096), (266240, 4096, 1), 0); del buf80  # reuse
        # Source Nodes: [x_150], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf84, addmm_42, 2129920, grid=grid(2129920), stream=stream0)
        del addmm_42
        buf85 = buf69; del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf84, (520, 4096), (4096, 1), 0), permute_131, out=buf85)
        del permute_131
        buf86 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf84, (4096, 520), (1, 4096), 0), view_115, out=buf86)
        del view_115
        buf87 = buf51; del buf51  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf84, buf87, 20480, 104, grid=grid(20480), stream=stream0)
        buf88 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf87, buf88, 4096, 5, grid=grid(4096), stream=stream0)
        buf95 = buf79; del buf79  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf95, buf85, primals_139, mul_72, div_5, 520, 1024, grid=grid(520), stream=stream0)
        del div_5
        del primals_139
        buf91 = reinterpret_tensor(buf82, (1024, 5), (1, 1024), 0); del buf82  # reuse
        buf93 = buf75; del buf75  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf85, mul_72, buf91, buf93, 5120, 104, grid=grid(5120), stream=stream0)
        del mul_72
        buf92 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_5.run(buf91, buf92, 1024, 5, grid=grid(1024), stream=stream0)
        buf94 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_5.run(buf93, buf94, 1024, 5, grid=grid(1024), stream=stream0)
        buf96 = buf85; del buf85  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf95, (520, 1024), (1024, 1), 0), permute_135, out=buf96)
        del permute_135
        buf97 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf95, (1024, 520), (1, 1024), 0), view_113, out=buf97)
        del view_113
        buf98 = reinterpret_tensor(buf93, (1, 1024, 5), (5120, 1, 1024), 0); del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf95, buf98, 5120, 104, grid=grid(5120), stream=stream0)
        buf99 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf98, buf99, 1024, 5, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf100 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf96, (8, 16, 65, 64), (66560, 64, 1024, 1), 0), getitem_112, getitem_113, getitem_114, None, alias_15, getitem_116, getitem_117, getitem_118, 0.0, [True, True, True, False])
        del alias_15
        del buf96
        del getitem_112
        del getitem_113
        del getitem_114
        del getitem_116
        del getitem_117
        del getitem_118
        buf101 = buf100[0]
        buf102 = buf100[1]
        buf103 = buf100[2]
        del buf100
        buf104 = buf68; del buf68  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf101, buf102, buf103, buf104, 1597440, grid=grid(1597440), stream=stream0)
        del buf101
        del buf102
        buf105 = reinterpret_tensor(buf103, (520, 1024), (1024, 1), 0); del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf104, (520, 3072), (3072, 1), 0), permute_141, out=buf105)
        del permute_141
        buf106 = empty((3072, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf104, (3072, 520), (1, 3072), 0), view_109, out=buf106)
        del view_109
        buf107 = buf71; del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf104, buf107, 15360, 104, grid=grid(15360), stream=stream0)
        buf108 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf107, buf108, 3072, 5, grid=grid(3072), stream=stream0)
        buf115 = buf95; del buf95  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf115, buf105, primals_133, mul_70, div_6, 520, 1024, grid=grid(520), stream=stream0)
        del div_6
        del primals_133
        buf111 = reinterpret_tensor(buf98, (1024, 5), (1, 1024), 0); del buf98  # reuse
        buf113 = buf91; del buf91  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf105, mul_70, buf111, buf113, 5120, 104, grid=grid(5120), stream=stream0)
        del mul_70
        buf112 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_5.run(buf111, buf112, 1024, 5, grid=grid(1024), stream=stream0)
        buf114 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_5.run(buf113, buf114, 1024, 5, grid=grid(1024), stream=stream0)
        buf116 = reinterpret_tensor(buf84, (520, 4096), (4096, 1), 0); del buf84  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf115, (520, 1024), (1024, 1), 0), permute_145, out=buf116)
        del permute_145
        buf117 = empty((1024, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf115, (1024, 520), (1, 1024), 0), view_107, out=buf117)
        del view_107
        buf118 = reinterpret_tensor(buf113, (1, 1024, 5), (5120, 1, 1024), 0); del buf113  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf115, buf118, 5120, 104, grid=grid(5120), stream=stream0)
        buf119 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf118, buf119, 1024, 5, grid=grid(1024), stream=stream0)
        buf120 = reinterpret_tensor(buf116, (8, 65, 4096), (266240, 4096, 1), 0); del buf116  # reuse
        # Source Nodes: [x_138], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf120, addmm_38, 2129920, grid=grid(2129920), stream=stream0)
        del addmm_38
        buf121 = buf105; del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf120, (520, 4096), (4096, 1), 0), permute_149, out=buf121)
        del permute_149
        buf122 = empty((4096, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf120, (4096, 520), (1, 4096), 0), view_105, out=buf122)
        del view_105
        buf123 = buf87; del buf87  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf120, buf123, 20480, 104, grid=grid(20480), stream=stream0)
        del buf120
        buf124 = empty((1, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf123, buf124, 4096, 5, grid=grid(4096), stream=stream0)
        del buf123
        buf131 = buf115; del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_9.run(buf131, buf121, primals_127, mul_65, div_7, 520, 1024, grid=grid(520), stream=stream0)
        del div_7
        del primals_127
        buf127 = reinterpret_tensor(buf118, (1024, 5), (1, 1024), 0); del buf118  # reuse
        buf129 = buf111; del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_10.run(buf121, mul_65, buf127, buf129, 5120, 104, grid=grid(5120), stream=stream0)
        del mul_65
        buf128 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_5.run(buf127, buf128, 1024, 5, grid=grid(1024), stream=stream0)
        buf130 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_5.run(buf129, buf130, 1024, 5, grid=grid(1024), stream=stream0)
        buf132 = buf121; del buf121  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf131, (520, 1024), (1024, 1), 0), permute_153, out=buf132)
        del permute_153
        buf133 = empty((1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf131, (1024, 520), (1, 1024), 0), view_103, out=buf133)
        del view_103
        buf134 = reinterpret_tensor(buf129, (1, 1024, 5), (5120, 1, 1024), 0); del buf129  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf131, buf134, 5120, 104, grid=grid(5120), stream=stream0)
        buf135 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf134, buf135, 1024, 5, grid=grid(1024), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf136 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf132, (8, 16, 65, 64), (66560, 64, 1024, 1), 0), getitem_101, getitem_102, getitem_103, None, alias_16, getitem_105, getitem_106, getitem_107, 0.0, [True, True, True, False])
        del alias_16
        del buf132
        del getitem_101
        del getitem_102
        del getitem_103
        del getitem_105
        del getitem_106
        del getitem_107
        buf137 = buf136[0]
        buf138 = buf136[1]
        buf139 = buf136[2]
        del buf136
        buf140 = buf104; del buf104  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf137, buf138, buf139, buf140, 1597440, grid=grid(1597440), stream=stream0)
        del buf137
        del buf138
        buf141 = reinterpret_tensor(buf139, (520, 1024), (1024, 1), 0); del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf140, (520, 3072), (3072, 1), 0), permute_159, out=buf141)
        del permute_159
        buf142 = empty((3072, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf140, (3072, 520), (1, 3072), 0), view_99, out=buf142)
        del view_99
        buf143 = buf107; del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf140, buf143, 15360, 104, grid=grid(15360), stream=stream0)
        del buf140
        buf144 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf143, buf144, 3072, 5, grid=grid(3072), stream=stream0)
        del buf143
        buf151 = buf131; del buf131  # reuse
        # Source Nodes: [getattr_l__mod___transformers_2_blocks___0___norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_14.run(buf151, buf141, primals_121, cat_2, getitem_100, rsqrt_18, 520, 1024, grid=grid(520), stream=stream0)
        del primals_121
        buf147 = reinterpret_tensor(buf134, (1024, 5), (1, 1024), 0); del buf134  # reuse
        buf149 = buf127; del buf127  # reuse
        # Source Nodes: [getattr_l__mod___transformers_2_blocks___0___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_15.run(buf141, cat_2, getitem_100, rsqrt_18, buf147, buf149, 5120, 104, grid=grid(5120), stream=stream0)
        del buf141
        del cat_2
        del getitem_100
        del rsqrt_18
        buf148 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_2_blocks___0___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_sum_5.run(buf147, buf148, 1024, 5, grid=grid(1024), stream=stream0)
        del buf147
        buf150 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_5.run(buf149, buf150, 1024, 5, grid=grid(1024), stream=stream0)
        del buf149
        buf152 = empty((1, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_16.run(buf151, buf152, 1024, 8, grid=grid(1024), stream=stream0)
        buf153 = empty((1024, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf151, (1024, 8), (1, 66560), 0), view_96, out=buf153)
        del view_96
        buf154 = empty((8, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf151, (8, 1024), (66560, 1), 0), permute_166, out=buf154)
        del permute_166
        buf155 = empty_strided((1024, 4), (1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_17.run(buf151, buf155, 4096, 128, grid=grid(4096), stream=stream0)
        buf156 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_18.run(buf155, buf156, 1024, 4, grid=grid(1024), stream=stream0)
        del buf155
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf157 = aten.convolution_backward(reinterpret_tensor(buf151, (8, 1024, 8, 8), (66560, 1, 8192, 1024), 1024), view_95, primals_117, [1024], [2, 2], [1, 1], [1, 1], False, [0, 0], 512, [True, True, False])
        del buf151
        del primals_117
        del view_95
        buf158 = buf157[0]
        buf159 = buf157[1]
        del buf157
        buf160 = empty((8, 257, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.slice_backward]
        triton_poi_fused_add_slice_backward_19.run(buf158, buf154, buf160, 2056, 512, grid=grid(2056, 512), stream=stream0)
        del buf154
        buf161 = empty((2056, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf160, (2056, 512), (512, 1), 0), permute_169, out=buf161)
        del permute_169
        buf162 = reinterpret_tensor(buf158, (512, 2048), (2048, 1), 0); del buf158  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf160, (512, 2056), (1, 512), 0), view_93, out=buf162)
        del view_93
        buf163 = empty_strided((1, 512, 17), (8704, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_20.run(buf160, buf163, 8704, 121, grid=grid(8704), stream=stream0)
        buf164 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_21.run(buf163, buf164, 512, 17, grid=grid(512), stream=stream0)
        buf165 = reinterpret_tensor(buf161, (8, 257, 2048), (526336, 2048, 1), 0); del buf161  # reuse
        # Source Nodes: [x_117], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_22.run(buf165, addmm_34, 4210688, grid=grid(4210688), stream=stream0)
        del addmm_34
        buf166 = empty((2056, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (2056, 2048), (2048, 1), 0), permute_173, out=buf166)
        del permute_173
        buf167 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (2048, 2056), (1, 2048), 0), view_91, out=buf167)
        del view_91
        buf168 = empty_strided((1, 2048, 17), (34816, 1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf165, buf168, 34816, 121, grid=grid(34816), stream=stream0)
        buf169 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf168, buf169, 2048, 17, grid=grid(2048), stream=stream0)
        buf176 = buf160; del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_25.run(buf176, buf166, primals_111, mul_58, div_9, 2056, 512, grid=grid(2056), stream=stream0)
        del div_9
        del primals_111
        buf172 = reinterpret_tensor(buf163, (512, 17), (1, 512), 0); del buf163  # reuse
        buf174 = empty_strided((512, 17), (1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_26.run(buf166, mul_58, buf172, buf174, 8704, 121, grid=grid(8704), stream=stream0)
        del mul_58
        buf173 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_21.run(buf172, buf173, 512, 17, grid=grid(512), stream=stream0)
        buf175 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_21.run(buf174, buf175, 512, 17, grid=grid(512), stream=stream0)
        buf177 = buf166; del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf176, (2056, 512), (512, 1), 0), permute_177, out=buf177)
        del permute_177
        buf178 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf176, (512, 2056), (1, 512), 0), view_89, out=buf178)
        del view_89
        buf179 = reinterpret_tensor(buf174, (1, 512, 17), (8704, 1, 512), 0); del buf174  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_20.run(buf176, buf179, 8704, 121, grid=grid(8704), stream=stream0)
        buf180 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_21.run(buf179, buf180, 512, 17, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf181 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf177, (8, 8, 257, 64), (131584, 64, 512, 1), 0), getitem_90, getitem_91, getitem_92, None, alias_17, getitem_94, getitem_95, getitem_96, 0.0, [True, True, True, False])
        del alias_17
        del buf177
        del getitem_90
        del getitem_91
        del getitem_92
        del getitem_94
        del getitem_95
        del getitem_96
        buf182 = buf181[0]
        buf183 = buf181[1]
        buf184 = buf181[2]
        del buf181
        buf185 = empty((8, 257, 3, 8, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf182, buf183, buf184, buf185, 3158016, grid=grid(3158016), stream=stream0)
        del buf182
        del buf183
        buf186 = reinterpret_tensor(buf184, (2056, 512), (512, 1), 0); del buf184  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf185, (2056, 1536), (1536, 1), 0), permute_183, out=buf186)
        del permute_183
        buf187 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf185, (1536, 2056), (1, 1536), 0), view_85, out=buf187)
        del view_85
        buf188 = empty_strided((1, 1536, 17), (26112, 1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf185, buf188, 26112, 121, grid=grid(26112), stream=stream0)
        buf189 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_29.run(buf188, buf189, 1536, 17, grid=grid(1536), stream=stream0)
        buf196 = buf176; del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_25.run(buf196, buf186, primals_105, mul_56, div_10, 2056, 512, grid=grid(2056), stream=stream0)
        del div_10
        del primals_105
        buf192 = reinterpret_tensor(buf179, (512, 17), (1, 512), 0); del buf179  # reuse
        buf194 = buf172; del buf172  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_26.run(buf186, mul_56, buf192, buf194, 8704, 121, grid=grid(8704), stream=stream0)
        del mul_56
        buf193 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_21.run(buf192, buf193, 512, 17, grid=grid(512), stream=stream0)
        buf195 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_21.run(buf194, buf195, 512, 17, grid=grid(512), stream=stream0)
        buf197 = reinterpret_tensor(buf165, (2056, 2048), (2048, 1), 0); del buf165  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf196, (2056, 512), (512, 1), 0), permute_187, out=buf197)
        del permute_187
        buf198 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf196, (512, 2056), (1, 512), 0), view_83, out=buf198)
        del view_83
        buf199 = reinterpret_tensor(buf194, (1, 512, 17), (8704, 1, 512), 0); del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_20.run(buf196, buf199, 8704, 121, grid=grid(8704), stream=stream0)
        buf200 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_21.run(buf199, buf200, 512, 17, grid=grid(512), stream=stream0)
        buf201 = reinterpret_tensor(buf197, (8, 257, 2048), (526336, 2048, 1), 0); del buf197  # reuse
        # Source Nodes: [x_105], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_22.run(buf201, addmm_30, 4210688, grid=grid(4210688), stream=stream0)
        del addmm_30
        buf202 = buf186; del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf201, (2056, 2048), (2048, 1), 0), permute_191, out=buf202)
        del permute_191
        buf203 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf201, (2048, 2056), (1, 2048), 0), view_81, out=buf203)
        del view_81
        buf204 = buf168; del buf168  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf201, buf204, 34816, 121, grid=grid(34816), stream=stream0)
        buf205 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf204, buf205, 2048, 17, grid=grid(2048), stream=stream0)
        buf212 = buf196; del buf196  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_25.run(buf212, buf202, primals_99, mul_51, div_11, 2056, 512, grid=grid(2056), stream=stream0)
        del div_11
        del primals_99
        buf208 = reinterpret_tensor(buf199, (512, 17), (1, 512), 0); del buf199  # reuse
        buf210 = buf192; del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_26.run(buf202, mul_51, buf208, buf210, 8704, 121, grid=grid(8704), stream=stream0)
        del mul_51
        buf209 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_21.run(buf208, buf209, 512, 17, grid=grid(512), stream=stream0)
        buf211 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_21.run(buf210, buf211, 512, 17, grid=grid(512), stream=stream0)
        buf213 = buf202; del buf202  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf212, (2056, 512), (512, 1), 0), permute_195, out=buf213)
        del permute_195
        buf214 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf212, (512, 2056), (1, 512), 0), view_79, out=buf214)
        del view_79
        buf215 = reinterpret_tensor(buf210, (1, 512, 17), (8704, 1, 512), 0); del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_20.run(buf212, buf215, 8704, 121, grid=grid(8704), stream=stream0)
        buf216 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_21.run(buf215, buf216, 512, 17, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf217 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf213, (8, 8, 257, 64), (131584, 64, 512, 1), 0), getitem_79, getitem_80, getitem_81, None, alias_18, getitem_83, getitem_84, getitem_85, 0.0, [True, True, True, False])
        del alias_18
        del buf213
        del getitem_79
        del getitem_80
        del getitem_81
        del getitem_83
        del getitem_84
        del getitem_85
        buf218 = buf217[0]
        buf219 = buf217[1]
        buf220 = buf217[2]
        del buf217
        buf221 = buf185; del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf218, buf219, buf220, buf221, 3158016, grid=grid(3158016), stream=stream0)
        del buf218
        del buf219
        buf222 = reinterpret_tensor(buf220, (2056, 512), (512, 1), 0); del buf220  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf221, (2056, 1536), (1536, 1), 0), permute_201, out=buf222)
        del permute_201
        buf223 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf221, (1536, 2056), (1, 1536), 0), view_75, out=buf223)
        del view_75
        buf224 = buf188; del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf221, buf224, 26112, 121, grid=grid(26112), stream=stream0)
        buf225 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_29.run(buf224, buf225, 1536, 17, grid=grid(1536), stream=stream0)
        buf232 = buf212; del buf212  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_25.run(buf232, buf222, primals_93, mul_49, div_12, 2056, 512, grid=grid(2056), stream=stream0)
        del div_12
        del primals_93
        buf228 = reinterpret_tensor(buf215, (512, 17), (1, 512), 0); del buf215  # reuse
        buf230 = buf208; del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_26.run(buf222, mul_49, buf228, buf230, 8704, 121, grid=grid(8704), stream=stream0)
        del mul_49
        buf229 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_21.run(buf228, buf229, 512, 17, grid=grid(512), stream=stream0)
        buf231 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_21.run(buf230, buf231, 512, 17, grid=grid(512), stream=stream0)
        buf233 = reinterpret_tensor(buf201, (2056, 2048), (2048, 1), 0); del buf201  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf232, (2056, 512), (512, 1), 0), permute_205, out=buf233)
        del permute_205
        buf234 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf232, (512, 2056), (1, 512), 0), view_73, out=buf234)
        del view_73
        buf235 = reinterpret_tensor(buf230, (1, 512, 17), (8704, 1, 512), 0); del buf230  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_20.run(buf232, buf235, 8704, 121, grid=grid(8704), stream=stream0)
        buf236 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_21.run(buf235, buf236, 512, 17, grid=grid(512), stream=stream0)
        buf237 = reinterpret_tensor(buf233, (8, 257, 2048), (526336, 2048, 1), 0); del buf233  # reuse
        # Source Nodes: [x_93], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_22.run(buf237, addmm_26, 4210688, grid=grid(4210688), stream=stream0)
        del addmm_26
        buf238 = buf222; del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf237, (2056, 2048), (2048, 1), 0), permute_209, out=buf238)
        del permute_209
        buf239 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf237, (2048, 2056), (1, 2048), 0), view_71, out=buf239)
        del view_71
        buf240 = buf204; del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf237, buf240, 34816, 121, grid=grid(34816), stream=stream0)
        buf241 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf240, buf241, 2048, 17, grid=grid(2048), stream=stream0)
        buf248 = buf232; del buf232  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_25.run(buf248, buf238, primals_87, mul_44, div_13, 2056, 512, grid=grid(2056), stream=stream0)
        del div_13
        del primals_87
        buf244 = reinterpret_tensor(buf235, (512, 17), (1, 512), 0); del buf235  # reuse
        buf246 = buf228; del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_26.run(buf238, mul_44, buf244, buf246, 8704, 121, grid=grid(8704), stream=stream0)
        del mul_44
        buf245 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_21.run(buf244, buf245, 512, 17, grid=grid(512), stream=stream0)
        buf247 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_21.run(buf246, buf247, 512, 17, grid=grid(512), stream=stream0)
        buf249 = buf238; del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf248, (2056, 512), (512, 1), 0), permute_213, out=buf249)
        del permute_213
        buf250 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf248, (512, 2056), (1, 512), 0), view_69, out=buf250)
        del view_69
        buf251 = reinterpret_tensor(buf246, (1, 512, 17), (8704, 1, 512), 0); del buf246  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_20.run(buf248, buf251, 8704, 121, grid=grid(8704), stream=stream0)
        buf252 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_21.run(buf251, buf252, 512, 17, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf253 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf249, (8, 8, 257, 64), (131584, 64, 512, 1), 0), getitem_68, getitem_69, getitem_70, None, alias_19, getitem_72, getitem_73, getitem_74, 0.0, [True, True, True, False])
        del alias_19
        del buf249
        del getitem_68
        del getitem_69
        del getitem_70
        del getitem_72
        del getitem_73
        del getitem_74
        buf254 = buf253[0]
        buf255 = buf253[1]
        buf256 = buf253[2]
        del buf253
        buf257 = buf221; del buf221  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf254, buf255, buf256, buf257, 3158016, grid=grid(3158016), stream=stream0)
        del buf254
        del buf255
        buf258 = reinterpret_tensor(buf256, (2056, 512), (512, 1), 0); del buf256  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf257, (2056, 1536), (1536, 1), 0), permute_219, out=buf258)
        del permute_219
        buf259 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf257, (1536, 2056), (1, 1536), 0), view_65, out=buf259)
        del view_65
        buf260 = buf224; del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf257, buf260, 26112, 121, grid=grid(26112), stream=stream0)
        buf261 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_29.run(buf260, buf261, 1536, 17, grid=grid(1536), stream=stream0)
        buf268 = buf248; del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_25.run(buf268, buf258, primals_81, mul_42, div_14, 2056, 512, grid=grid(2056), stream=stream0)
        del div_14
        del primals_81
        buf264 = reinterpret_tensor(buf251, (512, 17), (1, 512), 0); del buf251  # reuse
        buf266 = buf244; del buf244  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_26.run(buf258, mul_42, buf264, buf266, 8704, 121, grid=grid(8704), stream=stream0)
        del mul_42
        buf265 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_21.run(buf264, buf265, 512, 17, grid=grid(512), stream=stream0)
        buf267 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_21.run(buf266, buf267, 512, 17, grid=grid(512), stream=stream0)
        buf269 = reinterpret_tensor(buf237, (2056, 2048), (2048, 1), 0); del buf237  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf268, (2056, 512), (512, 1), 0), permute_223, out=buf269)
        del permute_223
        buf270 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf268, (512, 2056), (1, 512), 0), view_63, out=buf270)
        del view_63
        buf271 = reinterpret_tensor(buf266, (1, 512, 17), (8704, 1, 512), 0); del buf266  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_20.run(buf268, buf271, 8704, 121, grid=grid(8704), stream=stream0)
        buf272 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_21.run(buf271, buf272, 512, 17, grid=grid(512), stream=stream0)
        buf273 = reinterpret_tensor(buf269, (8, 257, 2048), (526336, 2048, 1), 0); del buf269  # reuse
        # Source Nodes: [x_81], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_22.run(buf273, addmm_22, 4210688, grid=grid(4210688), stream=stream0)
        del addmm_22
        buf274 = buf258; del buf258  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf273, (2056, 2048), (2048, 1), 0), permute_227, out=buf274)
        del permute_227
        buf275 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf273, (2048, 2056), (1, 2048), 0), view_61, out=buf275)
        del view_61
        buf276 = buf240; del buf240  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf273, buf276, 34816, 121, grid=grid(34816), stream=stream0)
        buf277 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf276, buf277, 2048, 17, grid=grid(2048), stream=stream0)
        buf284 = buf268; del buf268  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_25.run(buf284, buf274, primals_75, mul_37, div_15, 2056, 512, grid=grid(2056), stream=stream0)
        del div_15
        del primals_75
        buf280 = reinterpret_tensor(buf271, (512, 17), (1, 512), 0); del buf271  # reuse
        buf282 = buf264; del buf264  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_26.run(buf274, mul_37, buf280, buf282, 8704, 121, grid=grid(8704), stream=stream0)
        del mul_37
        buf281 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_21.run(buf280, buf281, 512, 17, grid=grid(512), stream=stream0)
        buf283 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_21.run(buf282, buf283, 512, 17, grid=grid(512), stream=stream0)
        buf285 = buf274; del buf274  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf284, (2056, 512), (512, 1), 0), permute_231, out=buf285)
        del permute_231
        buf286 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf284, (512, 2056), (1, 512), 0), view_59, out=buf286)
        del view_59
        buf287 = reinterpret_tensor(buf282, (1, 512, 17), (8704, 1, 512), 0); del buf282  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_20.run(buf284, buf287, 8704, 121, grid=grid(8704), stream=stream0)
        buf288 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_21.run(buf287, buf288, 512, 17, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf289 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf285, (8, 8, 257, 64), (131584, 64, 512, 1), 0), getitem_57, getitem_58, getitem_59, None, alias_20, getitem_61, getitem_62, getitem_63, 0.0, [True, True, True, False])
        del alias_20
        del buf285
        del getitem_57
        del getitem_58
        del getitem_59
        del getitem_61
        del getitem_62
        del getitem_63
        buf290 = buf289[0]
        buf291 = buf289[1]
        buf292 = buf289[2]
        del buf289
        buf293 = buf257; del buf257  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf290, buf291, buf292, buf293, 3158016, grid=grid(3158016), stream=stream0)
        del buf290
        del buf291
        buf294 = reinterpret_tensor(buf292, (2056, 512), (512, 1), 0); del buf292  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf293, (2056, 1536), (1536, 1), 0), permute_237, out=buf294)
        del permute_237
        buf295 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf293, (1536, 2056), (1, 1536), 0), view_55, out=buf295)
        del view_55
        buf296 = buf260; del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf293, buf296, 26112, 121, grid=grid(26112), stream=stream0)
        buf297 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_29.run(buf296, buf297, 1536, 17, grid=grid(1536), stream=stream0)
        buf304 = buf284; del buf284  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_25.run(buf304, buf294, primals_69, mul_35, div_16, 2056, 512, grid=grid(2056), stream=stream0)
        del div_16
        del primals_69
        buf300 = reinterpret_tensor(buf287, (512, 17), (1, 512), 0); del buf287  # reuse
        buf302 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_26.run(buf294, mul_35, buf300, buf302, 8704, 121, grid=grid(8704), stream=stream0)
        del mul_35
        buf301 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_21.run(buf300, buf301, 512, 17, grid=grid(512), stream=stream0)
        buf303 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_21.run(buf302, buf303, 512, 17, grid=grid(512), stream=stream0)
        buf305 = reinterpret_tensor(buf273, (2056, 2048), (2048, 1), 0); del buf273  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (2056, 512), (512, 1), 0), permute_241, out=buf305)
        del permute_241
        buf306 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (512, 2056), (1, 512), 0), view_53, out=buf306)
        del view_53
        buf307 = reinterpret_tensor(buf302, (1, 512, 17), (8704, 1, 512), 0); del buf302  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_20.run(buf304, buf307, 8704, 121, grid=grid(8704), stream=stream0)
        buf308 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_21.run(buf307, buf308, 512, 17, grid=grid(512), stream=stream0)
        buf309 = reinterpret_tensor(buf305, (8, 257, 2048), (526336, 2048, 1), 0); del buf305  # reuse
        # Source Nodes: [x_69], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_22.run(buf309, addmm_18, 4210688, grid=grid(4210688), stream=stream0)
        del addmm_18
        buf310 = buf294; del buf294  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf309, (2056, 2048), (2048, 1), 0), permute_245, out=buf310)
        del permute_245
        buf311 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf309, (2048, 2056), (1, 2048), 0), view_51, out=buf311)
        del view_51
        buf312 = buf276; del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf309, buf312, 34816, 121, grid=grid(34816), stream=stream0)
        buf313 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf312, buf313, 2048, 17, grid=grid(2048), stream=stream0)
        buf320 = buf304; del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_25.run(buf320, buf310, primals_63, mul_30, div_17, 2056, 512, grid=grid(2056), stream=stream0)
        del div_17
        del primals_63
        buf316 = reinterpret_tensor(buf307, (512, 17), (1, 512), 0); del buf307  # reuse
        buf318 = buf300; del buf300  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_26.run(buf310, mul_30, buf316, buf318, 8704, 121, grid=grid(8704), stream=stream0)
        del mul_30
        buf317 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_21.run(buf316, buf317, 512, 17, grid=grid(512), stream=stream0)
        buf319 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_21.run(buf318, buf319, 512, 17, grid=grid(512), stream=stream0)
        buf321 = buf310; del buf310  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf320, (2056, 512), (512, 1), 0), permute_249, out=buf321)
        del permute_249
        buf322 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf320, (512, 2056), (1, 512), 0), view_49, out=buf322)
        del view_49
        buf323 = reinterpret_tensor(buf318, (1, 512, 17), (8704, 1, 512), 0); del buf318  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_20.run(buf320, buf323, 8704, 121, grid=grid(8704), stream=stream0)
        buf324 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_21.run(buf323, buf324, 512, 17, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf325 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf321, (8, 8, 257, 64), (131584, 64, 512, 1), 0), getitem_46, getitem_47, getitem_48, None, alias_21, getitem_50, getitem_51, getitem_52, 0.0, [True, True, True, False])
        del alias_21
        del buf321
        del getitem_46
        del getitem_47
        del getitem_48
        del getitem_50
        del getitem_51
        del getitem_52
        buf326 = buf325[0]
        buf327 = buf325[1]
        buf328 = buf325[2]
        del buf325
        buf329 = buf293; del buf293  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf326, buf327, buf328, buf329, 3158016, grid=grid(3158016), stream=stream0)
        del buf326
        del buf327
        buf330 = reinterpret_tensor(buf328, (2056, 512), (512, 1), 0); del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf329, (2056, 1536), (1536, 1), 0), permute_255, out=buf330)
        del permute_255
        buf331 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf329, (1536, 2056), (1, 1536), 0), view_45, out=buf331)
        del view_45
        buf332 = buf296; del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf329, buf332, 26112, 121, grid=grid(26112), stream=stream0)
        buf333 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_29.run(buf332, buf333, 1536, 17, grid=grid(1536), stream=stream0)
        buf340 = buf320; del buf320  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_25.run(buf340, buf330, primals_57, mul_28, div_18, 2056, 512, grid=grid(2056), stream=stream0)
        del div_18
        del primals_57
        buf336 = reinterpret_tensor(buf323, (512, 17), (1, 512), 0); del buf323  # reuse
        buf338 = buf316; del buf316  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_26.run(buf330, mul_28, buf336, buf338, 8704, 121, grid=grid(8704), stream=stream0)
        del mul_28
        buf337 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_21.run(buf336, buf337, 512, 17, grid=grid(512), stream=stream0)
        buf339 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_21.run(buf338, buf339, 512, 17, grid=grid(512), stream=stream0)
        buf341 = reinterpret_tensor(buf309, (2056, 2048), (2048, 1), 0); del buf309  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf340, (2056, 512), (512, 1), 0), permute_259, out=buf341)
        del permute_259
        buf342 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf340, (512, 2056), (1, 512), 0), view_43, out=buf342)
        del view_43
        buf343 = reinterpret_tensor(buf338, (1, 512, 17), (8704, 1, 512), 0); del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_20.run(buf340, buf343, 8704, 121, grid=grid(8704), stream=stream0)
        buf344 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_21.run(buf343, buf344, 512, 17, grid=grid(512), stream=stream0)
        buf345 = reinterpret_tensor(buf341, (8, 257, 2048), (526336, 2048, 1), 0); del buf341  # reuse
        # Source Nodes: [x_57], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_22.run(buf345, addmm_14, 4210688, grid=grid(4210688), stream=stream0)
        del addmm_14
        buf346 = buf330; del buf330  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf345, (2056, 2048), (2048, 1), 0), permute_263, out=buf346)
        del permute_263
        buf347 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf345, (2048, 2056), (1, 2048), 0), view_41, out=buf347)
        del view_41
        buf348 = buf312; del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_23.run(buf345, buf348, 34816, 121, grid=grid(34816), stream=stream0)
        del buf345
        buf349 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_24.run(buf348, buf349, 2048, 17, grid=grid(2048), stream=stream0)
        del buf348
        buf356 = buf340; del buf340  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_25.run(buf356, buf346, primals_51, mul_23, div_19, 2056, 512, grid=grid(2056), stream=stream0)
        del div_19
        del primals_51
        buf352 = reinterpret_tensor(buf343, (512, 17), (1, 512), 0); del buf343  # reuse
        buf354 = buf336; del buf336  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_26.run(buf346, mul_23, buf352, buf354, 8704, 121, grid=grid(8704), stream=stream0)
        del mul_23
        buf353 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_21.run(buf352, buf353, 512, 17, grid=grid(512), stream=stream0)
        buf355 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_21.run(buf354, buf355, 512, 17, grid=grid(512), stream=stream0)
        buf357 = buf346; del buf346  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf356, (2056, 512), (512, 1), 0), permute_267, out=buf357)
        del permute_267
        buf358 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf356, (512, 2056), (1, 512), 0), view_39, out=buf358)
        del view_39
        buf359 = reinterpret_tensor(buf354, (1, 512, 17), (8704, 1, 512), 0); del buf354  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_20.run(buf356, buf359, 8704, 121, grid=grid(8704), stream=stream0)
        buf360 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_21.run(buf359, buf360, 512, 17, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf361 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf357, (8, 8, 257, 64), (131584, 64, 512, 1), 0), getitem_35, getitem_36, getitem_37, None, alias_22, getitem_39, getitem_40, getitem_41, 0.0, [True, True, True, False])
        del alias_22
        del buf357
        del getitem_35
        del getitem_36
        del getitem_37
        del getitem_39
        del getitem_40
        del getitem_41
        buf362 = buf361[0]
        buf363 = buf361[1]
        buf364 = buf361[2]
        del buf361
        buf365 = buf329; del buf329  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf362, buf363, buf364, buf365, 3158016, grid=grid(3158016), stream=stream0)
        del buf362
        del buf363
        buf366 = reinterpret_tensor(buf364, (2056, 512), (512, 1), 0); del buf364  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf365, (2056, 1536), (1536, 1), 0), permute_273, out=buf366)
        del permute_273
        buf367 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf365, (1536, 2056), (1, 1536), 0), view_35, out=buf367)
        del view_35
        buf368 = buf332; del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf365, buf368, 26112, 121, grid=grid(26112), stream=stream0)
        del buf365
        buf369 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_29.run(buf368, buf369, 1536, 17, grid=grid(1536), stream=stream0)
        del buf368
        buf376 = buf356; del buf356  # reuse
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___0___norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_30.run(buf376, buf366, primals_45, cat_1, getitem_34, rsqrt_6, 2056, 512, grid=grid(2056), stream=stream0)
        del primals_45
        buf372 = reinterpret_tensor(buf359, (512, 17), (1, 512), 0); del buf359  # reuse
        buf374 = buf352; del buf352  # reuse
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___0___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_31.run(buf366, cat_1, getitem_34, rsqrt_6, buf372, buf374, 8704, 121, grid=grid(8704), stream=stream0)
        del buf366
        del cat_1
        del getitem_34
        del rsqrt_6
        buf373 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_1_blocks___0___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_sum_21.run(buf372, buf373, 512, 17, grid=grid(512), stream=stream0)
        del buf372
        buf375 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_21.run(buf374, buf375, 512, 17, grid=grid(512), stream=stream0)
        del buf374
        buf377 = empty((1, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_32.run(buf376, buf377, 512, 8, grid=grid(512), stream=stream0)
        buf378 = empty((512, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf376, (512, 8), (1, 131584), 0), view_32, out=buf378)
        del view_32
        buf379 = empty((8, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf376, (8, 512), (131584, 1), 0), permute_280, out=buf379)
        del permute_280
        buf380 = reinterpret_tensor(buf0, (512, 16), (1, 512), 0); del buf0  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_33.run(buf376, buf380, 8192, 128, grid=grid(8192), stream=stream0)
        buf381 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_convolution_backward_34.run(buf380, buf381, 512, 16, grid=grid(512), stream=stream0)
        del buf380
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf382 = aten.convolution_backward(reinterpret_tensor(buf376, (8, 512, 16, 16), (131584, 1, 8192, 512), 512), view_31, primals_41, [512], [2, 2], [1, 1], [1, 1], False, [0, 0], 256, [True, True, False])
        del buf376
        del primals_41
        del view_31
        buf383 = buf382[0]
        buf384 = buf382[1]
        del buf382
        buf385 = empty((8, 962, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.slice_backward]
        triton_poi_fused_add_slice_backward_35.run(buf383, buf379, buf385, 7696, 256, grid=grid(7696, 256), stream=stream0)
        del buf379
        del buf383
        buf386 = empty((7696, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf385, (7696, 256), (256, 1), 0), permute_283, out=buf386)
        del permute_283
        buf387 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf385, (256, 7696), (1, 256), 0), view_29, out=buf387)
        del view_29
        buf388 = empty_strided((1, 256, 61), (15616, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf385, buf388, 15616, 127, grid=grid(15616), stream=stream0)
        buf389 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_37.run(buf388, buf389, 256, 61, grid=grid(256), stream=stream0)
        buf390 = reinterpret_tensor(buf386, (8, 962, 1024), (985088, 1024, 1), 0); del buf386  # reuse
        # Source Nodes: [x_36], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_38.run(buf390, addmm_10, 7880704, grid=grid(7880704), stream=stream0)
        del addmm_10
        buf391 = empty((7696, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf390, (7696, 1024), (1024, 1), 0), permute_287, out=buf391)
        del permute_287
        buf392 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf390, (1024, 7696), (1, 1024), 0), view_27, out=buf392)
        del view_27
        buf393 = empty_strided((1, 1024, 61), (62464, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf390, buf393, 62464, 127, grid=grid(62464), stream=stream0)
        buf394 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf393, buf394, 1024, 61, grid=grid(1024), stream=stream0)
        buf401 = buf385; del buf385  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_41.run(buf401, buf391, primals_35, mul_16, div_21, 7696, 256, grid=grid(7696), stream=stream0)
        del div_21
        del primals_35
        buf397 = reinterpret_tensor(buf388, (256, 61), (1, 256), 0); del buf388  # reuse
        buf399 = empty_strided((256, 61), (1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_42.run(buf391, mul_16, buf397, buf399, 15616, 127, grid=grid(15616), stream=stream0)
        del mul_16
        buf398 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_37.run(buf397, buf398, 256, 61, grid=grid(256), stream=stream0)
        buf400 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_37.run(buf399, buf400, 256, 61, grid=grid(256), stream=stream0)
        buf402 = buf391; del buf391  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf401, (7696, 256), (256, 1), 0), permute_291, out=buf402)
        del permute_291
        buf403 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf401, (256, 7696), (1, 256), 0), view_25, out=buf403)
        del view_25
        buf404 = reinterpret_tensor(buf399, (1, 256, 61), (15616, 1, 256), 0); del buf399  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf401, buf404, 15616, 127, grid=grid(15616), stream=stream0)
        buf405 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_37.run(buf404, buf405, 256, 61, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf406 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf402, (8, 4, 962, 64), (246272, 64, 256, 1), 0), getitem_24, getitem_25, getitem_26, None, alias_23, getitem_28, getitem_29, getitem_30, 0.0, [True, True, True, False])
        del alias_23
        del buf402
        del getitem_24
        del getitem_25
        del getitem_26
        del getitem_28
        del getitem_29
        del getitem_30
        buf407 = buf406[0]
        buf408 = buf406[1]
        buf409 = buf406[2]
        del buf406
        buf410 = empty((8, 962, 3, 4, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_43.run(buf407, buf408, buf409, buf410, 5910528, grid=grid(5910528), stream=stream0)
        del buf407
        del buf408
        buf411 = reinterpret_tensor(buf409, (7696, 256), (256, 1), 0); del buf409  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf410, (7696, 768), (768, 1), 0), permute_297, out=buf411)
        del permute_297
        buf412 = empty((768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf410, (768, 7696), (1, 768), 0), view_21, out=buf412)
        del view_21
        buf413 = empty_strided((1, 768, 61), (46848, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_44.run(buf410, buf413, 46848, 127, grid=grid(46848), stream=stream0)
        buf414 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_45.run(buf413, buf414, 768, 61, grid=grid(768), stream=stream0)
        buf421 = buf401; del buf401  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_41.run(buf421, buf411, primals_29, mul_14, div_22, 7696, 256, grid=grid(7696), stream=stream0)
        del div_22
        del primals_29
        buf417 = reinterpret_tensor(buf404, (256, 61), (1, 256), 0); del buf404  # reuse
        buf419 = buf397; del buf397  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_42.run(buf411, mul_14, buf417, buf419, 15616, 127, grid=grid(15616), stream=stream0)
        del mul_14
        buf418 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_37.run(buf417, buf418, 256, 61, grid=grid(256), stream=stream0)
        buf420 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_37.run(buf419, buf420, 256, 61, grid=grid(256), stream=stream0)
        buf422 = reinterpret_tensor(buf390, (7696, 1024), (1024, 1), 0); del buf390  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf421, (7696, 256), (256, 1), 0), permute_301, out=buf422)
        del permute_301
        buf423 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf421, (256, 7696), (1, 256), 0), view_19, out=buf423)
        del view_19
        buf424 = reinterpret_tensor(buf419, (1, 256, 61), (15616, 1, 256), 0); del buf419  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf421, buf424, 15616, 127, grid=grid(15616), stream=stream0)
        buf425 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_37.run(buf424, buf425, 256, 61, grid=grid(256), stream=stream0)
        buf426 = reinterpret_tensor(buf422, (8, 962, 1024), (985088, 1024, 1), 0); del buf422  # reuse
        # Source Nodes: [x_24], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_38.run(buf426, addmm_6, 7880704, grid=grid(7880704), stream=stream0)
        del addmm_6
        buf427 = buf411; del buf411  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf426, (7696, 1024), (1024, 1), 0), permute_305, out=buf427)
        del permute_305
        buf428 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf426, (1024, 7696), (1, 1024), 0), view_17, out=buf428)
        del view_17
        buf429 = buf393; del buf393  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf426, buf429, 62464, 127, grid=grid(62464), stream=stream0)
        buf430 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf429, buf430, 1024, 61, grid=grid(1024), stream=stream0)
        buf437 = buf421; del buf421  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_41.run(buf437, buf427, primals_23, mul_9, div_23, 7696, 256, grid=grid(7696), stream=stream0)
        del div_23
        del primals_23
        buf433 = reinterpret_tensor(buf424, (256, 61), (1, 256), 0); del buf424  # reuse
        buf435 = buf417; del buf417  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_42.run(buf427, mul_9, buf433, buf435, 15616, 127, grid=grid(15616), stream=stream0)
        del mul_9
        buf434 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_37.run(buf433, buf434, 256, 61, grid=grid(256), stream=stream0)
        buf436 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_37.run(buf435, buf436, 256, 61, grid=grid(256), stream=stream0)
        buf438 = buf427; del buf427  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf437, (7696, 256), (256, 1), 0), permute_309, out=buf438)
        del permute_309
        buf439 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf437, (256, 7696), (1, 256), 0), view_15, out=buf439)
        del view_15
        buf440 = reinterpret_tensor(buf435, (1, 256, 61), (15616, 1, 256), 0); del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf437, buf440, 15616, 127, grid=grid(15616), stream=stream0)
        buf441 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_37.run(buf440, buf441, 256, 61, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf442 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf438, (8, 4, 962, 64), (246272, 64, 256, 1), 0), getitem_13, getitem_14, getitem_15, None, alias_24, getitem_17, getitem_18, getitem_19, 0.0, [True, True, True, False])
        del alias_24
        del buf438
        del getitem_13
        del getitem_14
        del getitem_15
        del getitem_17
        del getitem_18
        del getitem_19
        buf443 = buf442[0]
        buf444 = buf442[1]
        buf445 = buf442[2]
        del buf442
        buf446 = buf410; del buf410  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_43.run(buf443, buf444, buf445, buf446, 5910528, grid=grid(5910528), stream=stream0)
        del buf443
        del buf444
        buf447 = reinterpret_tensor(buf445, (7696, 256), (256, 1), 0); del buf445  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf446, (7696, 768), (768, 1), 0), permute_315, out=buf447)
        del permute_315
        buf448 = empty((768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf446, (768, 7696), (1, 768), 0), view_11, out=buf448)
        del view_11
        buf449 = buf413; del buf413  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_44.run(buf446, buf449, 46848, 127, grid=grid(46848), stream=stream0)
        buf450 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_45.run(buf449, buf450, 768, 61, grid=grid(768), stream=stream0)
        buf457 = buf437; del buf437  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_41.run(buf457, buf447, primals_17, mul_7, div_24, 7696, 256, grid=grid(7696), stream=stream0)
        del div_24
        del primals_17
        buf453 = reinterpret_tensor(buf440, (256, 61), (1, 256), 0); del buf440  # reuse
        buf455 = buf433; del buf433  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_42.run(buf447, mul_7, buf453, buf455, 15616, 127, grid=grid(15616), stream=stream0)
        del mul_7
        buf454 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_37.run(buf453, buf454, 256, 61, grid=grid(256), stream=stream0)
        buf456 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_37.run(buf455, buf456, 256, 61, grid=grid(256), stream=stream0)
        buf458 = reinterpret_tensor(buf426, (7696, 1024), (1024, 1), 0); del buf426  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf457, (7696, 256), (256, 1), 0), permute_319, out=buf458)
        del permute_319
        buf459 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf457, (256, 7696), (1, 256), 0), view_9, out=buf459)
        del view_9
        buf460 = reinterpret_tensor(buf455, (1, 256, 61), (15616, 1, 256), 0); del buf455  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf457, buf460, 15616, 127, grid=grid(15616), stream=stream0)
        buf461 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_37.run(buf460, buf461, 256, 61, grid=grid(256), stream=stream0)
        buf462 = reinterpret_tensor(buf458, (8, 962, 1024), (985088, 1024, 1), 0); del buf458  # reuse
        # Source Nodes: [x_12], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_38.run(buf462, addmm_2, 7880704, grid=grid(7880704), stream=stream0)
        del addmm_2
        buf463 = buf447; del buf447  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf462, (7696, 1024), (1024, 1), 0), permute_323, out=buf463)
        del permute_323
        buf464 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf462, (1024, 7696), (1, 1024), 0), view_7, out=buf464)
        del view_7
        buf465 = buf429; del buf429  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf462, buf465, 62464, 127, grid=grid(62464), stream=stream0)
        del buf462
        buf466 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf465, buf466, 1024, 61, grid=grid(1024), stream=stream0)
        del buf465
        buf473 = buf457; del buf457  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_41.run(buf473, buf463, primals_11, mul_2, div_25, 7696, 256, grid=grid(7696), stream=stream0)
        del div_25
        del primals_11
        buf469 = reinterpret_tensor(buf460, (256, 61), (1, 256), 0); del buf460  # reuse
        buf471 = buf453; del buf453  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_42.run(buf463, mul_2, buf469, buf471, 15616, 127, grid=grid(15616), stream=stream0)
        del mul_2
        buf470 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_37.run(buf469, buf470, 256, 61, grid=grid(256), stream=stream0)
        buf472 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_37.run(buf471, buf472, 256, 61, grid=grid(256), stream=stream0)
        buf474 = buf463; del buf463  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf473, (7696, 256), (256, 1), 0), permute_327, out=buf474)
        del permute_327
        buf475 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf473, (256, 7696), (1, 256), 0), view_5, out=buf475)
        del view_5
        buf476 = reinterpret_tensor(buf471, (1, 256, 61), (15616, 1, 256), 0); del buf471  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf473, buf476, 15616, 127, grid=grid(15616), stream=stream0)
        buf477 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_37.run(buf476, buf477, 256, 61, grid=grid(256), stream=stream0)
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf478 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf474, (8, 4, 962, 64), (246272, 64, 256, 1), 0), getitem_2, getitem_3, getitem_4, None, alias_25, getitem_6, getitem_7, getitem_8, 0.0, [True, True, True, False])
        del alias_25
        del buf474
        del getitem_2
        del getitem_3
        del getitem_4
        del getitem_6
        del getitem_7
        del getitem_8
        buf479 = buf478[0]
        buf480 = buf478[1]
        buf481 = buf478[2]
        del buf478
        buf482 = buf446; del buf446  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_43.run(buf479, buf480, buf481, buf482, 5910528, grid=grid(5910528), stream=stream0)
        del buf479
        del buf480
        buf483 = reinterpret_tensor(buf481, (7696, 256), (256, 1), 0); del buf481  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf482, (7696, 768), (768, 1), 0), permute_333, out=buf483)
        del permute_333
        buf484 = empty((768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf482, (768, 7696), (1, 768), 0), view_1, out=buf484)
        del view_1
        buf485 = buf449; del buf449  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_44.run(buf482, buf485, 46848, 127, grid=grid(46848), stream=stream0)
        del buf482
        buf486 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_45.run(buf485, buf486, 768, 61, grid=grid(768), stream=stream0)
        del buf485
        buf493 = buf473; del buf473  # reuse
        # Source Nodes: [getattr_l__mod___transformers_0_blocks___0___norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_46.run(buf493, buf483, primals_5, cat, getitem_1, rsqrt, 7696, 256, grid=grid(7696), stream=stream0)
        del primals_5
        buf489 = reinterpret_tensor(buf476, (256, 61), (1, 256), 0); del buf476  # reuse
        buf491 = buf469; del buf469  # reuse
        # Source Nodes: [getattr_l__mod___transformers_0_blocks___0___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_native_layer_norm_backward_47.run(buf483, cat, getitem_1, rsqrt, buf489, buf491, 15616, 127, grid=grid(15616), stream=stream0)
        del buf483
        del cat
        del getitem_1
        del rsqrt
        buf490 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___transformers_0_blocks___0___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_sum_37.run(buf489, buf490, 256, 61, grid=grid(256), stream=stream0)
        del buf489
        buf492 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_37.run(buf491, buf492, 256, 61, grid=grid(256), stream=stream0)
        buf494 = empty((1, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_48.run(buf493, buf494, 256, 8, grid=grid(256), stream=stream0)
        buf495 = empty((1, 256, 31, 31), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_49.run(buf493, buf495, 246016, 8, grid=grid(246016), stream=stream0)
        buf496 = buf491; del buf491  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_50.run(buf493, buf496, 15616, 127, grid=grid(15616), stream=stream0)
        buf497 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_sum_37.run(buf496, buf497, 256, 61, grid=grid(256), stream=stream0)
        del buf496
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf498 = aten.convolution_backward(reinterpret_tensor(buf493, (8, 256, 31, 31), (246272, 1, 7936, 256), 256), primals_173, primals_3, [256], [7, 7], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf493
        del primals_173
        del primals_3
        buf499 = buf498[1]
        return (buf495, buf494, buf499, buf497, buf490, buf492, reinterpret_tensor(buf484, (768, 256), (256, 1), 0), reinterpret_tensor(buf486, (768, ), (1, ), 0), reinterpret_tensor(buf475, (256, 256), (256, 1), 0), reinterpret_tensor(buf477, (256, ), (1, ), 0), buf470, buf472, reinterpret_tensor(buf464, (1024, 256), (256, 1), 0), reinterpret_tensor(buf466, (1024, ), (1, ), 0), reinterpret_tensor(buf459, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf461, (256, ), (1, ), 0), buf454, buf456, reinterpret_tensor(buf448, (768, 256), (256, 1), 0), reinterpret_tensor(buf450, (768, ), (1, ), 0), reinterpret_tensor(buf439, (256, 256), (256, 1), 0), reinterpret_tensor(buf441, (256, ), (1, ), 0), buf434, buf436, reinterpret_tensor(buf428, (1024, 256), (256, 1), 0), reinterpret_tensor(buf430, (1024, ), (1, ), 0), reinterpret_tensor(buf423, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf425, (256, ), (1, ), 0), buf418, buf420, reinterpret_tensor(buf412, (768, 256), (256, 1), 0), reinterpret_tensor(buf414, (768, ), (1, ), 0), reinterpret_tensor(buf403, (256, 256), (256, 1), 0), reinterpret_tensor(buf405, (256, ), (1, ), 0), buf398, buf400, reinterpret_tensor(buf392, (1024, 256), (256, 1), 0), reinterpret_tensor(buf394, (1024, ), (1, ), 0), reinterpret_tensor(buf387, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf389, (256, ), (1, ), 0), buf384, buf381, reinterpret_tensor(buf378, (512, 256), (256, 1), 0), reinterpret_tensor(buf377, (512, ), (1, ), 0), buf373, buf375, reinterpret_tensor(buf367, (1536, 512), (512, 1), 0), reinterpret_tensor(buf369, (1536, ), (1, ), 0), reinterpret_tensor(buf358, (512, 512), (512, 1), 0), reinterpret_tensor(buf360, (512, ), (1, ), 0), buf353, buf355, reinterpret_tensor(buf347, (2048, 512), (512, 1), 0), reinterpret_tensor(buf349, (2048, ), (1, ), 0), reinterpret_tensor(buf342, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf344, (512, ), (1, ), 0), buf337, buf339, reinterpret_tensor(buf331, (1536, 512), (512, 1), 0), reinterpret_tensor(buf333, (1536, ), (1, ), 0), reinterpret_tensor(buf322, (512, 512), (512, 1), 0), reinterpret_tensor(buf324, (512, ), (1, ), 0), buf317, buf319, reinterpret_tensor(buf311, (2048, 512), (512, 1), 0), reinterpret_tensor(buf313, (2048, ), (1, ), 0), reinterpret_tensor(buf306, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf308, (512, ), (1, ), 0), buf301, buf303, reinterpret_tensor(buf295, (1536, 512), (512, 1), 0), reinterpret_tensor(buf297, (1536, ), (1, ), 0), reinterpret_tensor(buf286, (512, 512), (512, 1), 0), reinterpret_tensor(buf288, (512, ), (1, ), 0), buf281, buf283, reinterpret_tensor(buf275, (2048, 512), (512, 1), 0), reinterpret_tensor(buf277, (2048, ), (1, ), 0), reinterpret_tensor(buf270, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf272, (512, ), (1, ), 0), buf265, buf267, reinterpret_tensor(buf259, (1536, 512), (512, 1), 0), reinterpret_tensor(buf261, (1536, ), (1, ), 0), reinterpret_tensor(buf250, (512, 512), (512, 1), 0), reinterpret_tensor(buf252, (512, ), (1, ), 0), buf245, buf247, reinterpret_tensor(buf239, (2048, 512), (512, 1), 0), reinterpret_tensor(buf241, (2048, ), (1, ), 0), reinterpret_tensor(buf234, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf236, (512, ), (1, ), 0), buf229, buf231, reinterpret_tensor(buf223, (1536, 512), (512, 1), 0), reinterpret_tensor(buf225, (1536, ), (1, ), 0), reinterpret_tensor(buf214, (512, 512), (512, 1), 0), reinterpret_tensor(buf216, (512, ), (1, ), 0), buf209, buf211, reinterpret_tensor(buf203, (2048, 512), (512, 1), 0), reinterpret_tensor(buf205, (2048, ), (1, ), 0), reinterpret_tensor(buf198, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf200, (512, ), (1, ), 0), buf193, buf195, reinterpret_tensor(buf187, (1536, 512), (512, 1), 0), reinterpret_tensor(buf189, (1536, ), (1, ), 0), reinterpret_tensor(buf178, (512, 512), (512, 1), 0), reinterpret_tensor(buf180, (512, ), (1, ), 0), buf173, buf175, reinterpret_tensor(buf167, (2048, 512), (512, 1), 0), reinterpret_tensor(buf169, (2048, ), (1, ), 0), reinterpret_tensor(buf162, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf164, (512, ), (1, ), 0), buf159, buf156, reinterpret_tensor(buf153, (1024, 512), (512, 1), 0), reinterpret_tensor(buf152, (1024, ), (1, ), 0), buf148, buf150, reinterpret_tensor(buf142, (3072, 1024), (1024, 1), 0), reinterpret_tensor(buf144, (3072, ), (1, ), 0), reinterpret_tensor(buf133, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf135, (1024, ), (1, ), 0), buf128, buf130, reinterpret_tensor(buf122, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf124, (4096, ), (1, ), 0), reinterpret_tensor(buf117, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf119, (1024, ), (1, ), 0), buf112, buf114, reinterpret_tensor(buf106, (3072, 1024), (1024, 1), 0), reinterpret_tensor(buf108, (3072, ), (1, ), 0), reinterpret_tensor(buf97, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf99, (1024, ), (1, ), 0), buf92, buf94, reinterpret_tensor(buf86, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf88, (4096, ), (1, ), 0), reinterpret_tensor(buf81, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf83, (1024, ), (1, ), 0), buf76, buf78, reinterpret_tensor(buf70, (3072, 1024), (1024, 1), 0), reinterpret_tensor(buf72, (3072, ), (1, ), 0), reinterpret_tensor(buf61, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf63, (1024, ), (1, ), 0), buf56, buf58, reinterpret_tensor(buf50, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf52, (4096, ), (1, ), 0), reinterpret_tensor(buf45, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf47, (1024, ), (1, ), 0), buf40, buf42, reinterpret_tensor(buf34, (3072, 1024), (1024, 1), 0), reinterpret_tensor(buf36, (3072, ), (1, ), 0), reinterpret_tensor(buf25, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf27, (1024, ), (1, ), 0), buf20, buf22, reinterpret_tensor(buf14, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf16, (4096, ), (1, ), 0), reinterpret_tensor(buf9, (1024, 4096), (4096, 1), 0), reinterpret_tensor(buf11, (1024, ), (1, ), 0), buf5, buf6, reinterpret_tensor(buf1, (1000, 1024), (1024, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_3 = rand_strided((256, 3, 14, 14), (588, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((1024, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    cat = rand_strided((8, 962, 256), (246272, 256, 1), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((8, 962, 1), (962, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt = rand_strided((8, 962, 1), (962, 1, 1), device='cuda:0', dtype=torch.float32)
    view_1 = rand_strided((7696, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    getitem_2 = rand_strided((8, 4, 962, 64), (738816, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((8, 4, 962, 64), (738816, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_4 = rand_strided((8, 4, 962, 64), (738816, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_6 = rand_strided((8, 4, 992), (3968, 992, 1), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_8 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_5 = rand_strided((7696, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mul_2 = rand_strided((8, 962, 256), (246272, 256, 1), device='cuda:0', dtype=torch.float32)
    view_7 = rand_strided((7696, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_2 = rand_strided((7696, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_9 = rand_strided((7696, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_7 = rand_strided((8, 962, 256), (246272, 256, 1), device='cuda:0', dtype=torch.float32)
    view_11 = rand_strided((7696, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    getitem_13 = rand_strided((8, 4, 962, 64), (738816, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_14 = rand_strided((8, 4, 962, 64), (738816, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_15 = rand_strided((8, 4, 962, 64), (738816, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_17 = rand_strided((8, 4, 992), (3968, 992, 1), device='cuda:0', dtype=torch.float32)
    getitem_18 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_19 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_15 = rand_strided((7696, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mul_9 = rand_strided((8, 962, 256), (246272, 256, 1), device='cuda:0', dtype=torch.float32)
    view_17 = rand_strided((7696, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_6 = rand_strided((7696, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_19 = rand_strided((7696, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_14 = rand_strided((8, 962, 256), (246272, 256, 1), device='cuda:0', dtype=torch.float32)
    view_21 = rand_strided((7696, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    getitem_24 = rand_strided((8, 4, 962, 64), (738816, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_25 = rand_strided((8, 4, 962, 64), (738816, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_26 = rand_strided((8, 4, 962, 64), (738816, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_28 = rand_strided((8, 4, 992), (3968, 992, 1), device='cuda:0', dtype=torch.float32)
    getitem_29 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_30 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_25 = rand_strided((7696, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mul_16 = rand_strided((8, 962, 256), (246272, 256, 1), device='cuda:0', dtype=torch.float32)
    view_27 = rand_strided((7696, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((7696, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_29 = rand_strided((7696, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_31 = rand_strided((8, 256, 31, 31), (246272, 1, 7936, 256), device='cuda:0', dtype=torch.float32)
    view_32 = rand_strided((8, 256), (246272, 1), device='cuda:0', dtype=torch.float32)
    cat_1 = rand_strided((8, 257, 512), (131584, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_34 = rand_strided((8, 257, 1), (257, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_6 = rand_strided((8, 257, 1), (257, 1, 1), device='cuda:0', dtype=torch.float32)
    view_35 = rand_strided((2056, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_35 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_36 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_37 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_39 = rand_strided((8, 8, 288), (2304, 288, 1), device='cuda:0', dtype=torch.float32)
    getitem_40 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_41 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_39 = rand_strided((2056, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mul_23 = rand_strided((8, 257, 512), (131584, 512, 1), device='cuda:0', dtype=torch.float32)
    view_41 = rand_strided((2056, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_14 = rand_strided((2056, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_43 = rand_strided((2056, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    mul_28 = rand_strided((8, 257, 512), (131584, 512, 1), device='cuda:0', dtype=torch.float32)
    view_45 = rand_strided((2056, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_46 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_47 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_48 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_50 = rand_strided((8, 8, 288), (2304, 288, 1), device='cuda:0', dtype=torch.float32)
    getitem_51 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_52 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_49 = rand_strided((2056, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mul_30 = rand_strided((8, 257, 512), (131584, 512, 1), device='cuda:0', dtype=torch.float32)
    view_51 = rand_strided((2056, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_18 = rand_strided((2056, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_53 = rand_strided((2056, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    mul_35 = rand_strided((8, 257, 512), (131584, 512, 1), device='cuda:0', dtype=torch.float32)
    view_55 = rand_strided((2056, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_57 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_58 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_59 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_61 = rand_strided((8, 8, 288), (2304, 288, 1), device='cuda:0', dtype=torch.float32)
    getitem_62 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_63 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_59 = rand_strided((2056, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mul_37 = rand_strided((8, 257, 512), (131584, 512, 1), device='cuda:0', dtype=torch.float32)
    view_61 = rand_strided((2056, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((2056, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_63 = rand_strided((2056, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    mul_42 = rand_strided((8, 257, 512), (131584, 512, 1), device='cuda:0', dtype=torch.float32)
    view_65 = rand_strided((2056, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_68 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_69 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_70 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_72 = rand_strided((8, 8, 288), (2304, 288, 1), device='cuda:0', dtype=torch.float32)
    getitem_73 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_74 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_69 = rand_strided((2056, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mul_44 = rand_strided((8, 257, 512), (131584, 512, 1), device='cuda:0', dtype=torch.float32)
    view_71 = rand_strided((2056, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_26 = rand_strided((2056, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_73 = rand_strided((2056, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    mul_49 = rand_strided((8, 257, 512), (131584, 512, 1), device='cuda:0', dtype=torch.float32)
    view_75 = rand_strided((2056, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_79 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_80 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_81 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_83 = rand_strided((8, 8, 288), (2304, 288, 1), device='cuda:0', dtype=torch.float32)
    getitem_84 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_85 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_79 = rand_strided((2056, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mul_51 = rand_strided((8, 257, 512), (131584, 512, 1), device='cuda:0', dtype=torch.float32)
    view_81 = rand_strided((2056, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_30 = rand_strided((2056, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_83 = rand_strided((2056, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    mul_56 = rand_strided((8, 257, 512), (131584, 512, 1), device='cuda:0', dtype=torch.float32)
    view_85 = rand_strided((2056, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    getitem_90 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_91 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_92 = rand_strided((8, 8, 257, 64), (394752, 64, 1536, 1), device='cuda:0', dtype=torch.float32)
    getitem_94 = rand_strided((8, 8, 288), (2304, 288, 1), device='cuda:0', dtype=torch.float32)
    getitem_95 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_96 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_89 = rand_strided((2056, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mul_58 = rand_strided((8, 257, 512), (131584, 512, 1), device='cuda:0', dtype=torch.float32)
    view_91 = rand_strided((2056, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((2056, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_93 = rand_strided((2056, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_95 = rand_strided((8, 512, 16, 16), (131584, 1, 8192, 512), device='cuda:0', dtype=torch.float32)
    view_96 = rand_strided((8, 512), (131584, 1), device='cuda:0', dtype=torch.float32)
    cat_2 = rand_strided((8, 65, 1024), (66560, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_100 = rand_strided((8, 65, 1), (65, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_18 = rand_strided((8, 65, 1), (65, 1, 1), device='cuda:0', dtype=torch.float32)
    view_99 = rand_strided((520, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_101 = rand_strided((8, 16, 65, 64), (199680, 64, 3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_102 = rand_strided((8, 16, 65, 64), (199680, 64, 3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_103 = rand_strided((8, 16, 65, 64), (199680, 64, 3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_105 = rand_strided((8, 16, 96), (1536, 96, 1), device='cuda:0', dtype=torch.float32)
    getitem_106 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_107 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_103 = rand_strided((520, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_65 = rand_strided((8, 65, 1024), (66560, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_105 = rand_strided((520, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_38 = rand_strided((520, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_107 = rand_strided((520, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_70 = rand_strided((8, 65, 1024), (66560, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_109 = rand_strided((520, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_112 = rand_strided((8, 16, 65, 64), (199680, 64, 3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_113 = rand_strided((8, 16, 65, 64), (199680, 64, 3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_114 = rand_strided((8, 16, 65, 64), (199680, 64, 3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_116 = rand_strided((8, 16, 96), (1536, 96, 1), device='cuda:0', dtype=torch.float32)
    getitem_117 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_118 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_113 = rand_strided((520, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_72 = rand_strided((8, 65, 1024), (66560, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_115 = rand_strided((520, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_42 = rand_strided((520, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_117 = rand_strided((520, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_77 = rand_strided((8, 65, 1024), (66560, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_119 = rand_strided((520, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_123 = rand_strided((8, 16, 65, 64), (199680, 64, 3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_124 = rand_strided((8, 16, 65, 64), (199680, 64, 3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_125 = rand_strided((8, 16, 65, 64), (199680, 64, 3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_127 = rand_strided((8, 16, 96), (1536, 96, 1), device='cuda:0', dtype=torch.float32)
    getitem_128 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_129 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_123 = rand_strided((520, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_79 = rand_strided((8, 65, 1024), (66560, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_125 = rand_strided((520, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_46 = rand_strided((520, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_127 = rand_strided((520, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_84 = rand_strided((8, 65, 1024), (66560, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_129 = rand_strided((520, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_134 = rand_strided((8, 16, 65, 64), (199680, 64, 3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_135 = rand_strided((8, 16, 65, 64), (199680, 64, 3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_136 = rand_strided((8, 16, 65, 64), (199680, 64, 3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_138 = rand_strided((8, 16, 96), (1536, 96, 1), device='cuda:0', dtype=torch.float32)
    getitem_139 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_140 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_133 = rand_strided((520, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mul_86 = rand_strided((8, 65, 1024), (66560, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_135 = rand_strided((520, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    addmm_50 = rand_strided((520, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    view_137 = rand_strided((520, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    mul_91 = rand_strided((8, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.float32)
    clone_41 = rand_strided((8, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_87 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_91 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_95 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_1 = rand_strided((8, 65, 1), (65, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_99 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    alias_13 = rand_strided((8, 16, 65, 64), (66560, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_105 = rand_strided((3072, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_2 = rand_strided((8, 65, 1), (65, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_109 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_113 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((8, 65, 1), (65, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_117 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    alias_14 = rand_strided((8, 16, 65, 64), (66560, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_123 = rand_strided((3072, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_4 = rand_strided((8, 65, 1), (65, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_127 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_131 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_5 = rand_strided((8, 65, 1), (65, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_135 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    alias_15 = rand_strided((8, 16, 65, 64), (66560, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_141 = rand_strided((3072, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_6 = rand_strided((8, 65, 1), (65, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_145 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    permute_149 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    div_7 = rand_strided((8, 65, 1), (65, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_153 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    alias_16 = rand_strided((8, 16, 65, 64), (66560, 64, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_159 = rand_strided((3072, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_166 = rand_strided((1024, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_169 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_173 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_9 = rand_strided((8, 257, 1), (257, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_177 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    alias_17 = rand_strided((8, 8, 257, 64), (131584, 64, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_183 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_10 = rand_strided((8, 257, 1), (257, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_187 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_191 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_11 = rand_strided((8, 257, 1), (257, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_195 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    alias_18 = rand_strided((8, 8, 257, 64), (131584, 64, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_201 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_12 = rand_strided((8, 257, 1), (257, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_205 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_209 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_13 = rand_strided((8, 257, 1), (257, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_213 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    alias_19 = rand_strided((8, 8, 257, 64), (131584, 64, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_219 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_14 = rand_strided((8, 257, 1), (257, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_223 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_227 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_15 = rand_strided((8, 257, 1), (257, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_231 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    alias_20 = rand_strided((8, 8, 257, 64), (131584, 64, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_237 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_16 = rand_strided((8, 257, 1), (257, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_241 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_245 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_17 = rand_strided((8, 257, 1), (257, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_249 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    alias_21 = rand_strided((8, 8, 257, 64), (131584, 64, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_255 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_18 = rand_strided((8, 257, 1), (257, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_259 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_263 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_19 = rand_strided((8, 257, 1), (257, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_267 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    alias_22 = rand_strided((8, 8, 257, 64), (131584, 64, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_273 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_280 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_283 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_287 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_21 = rand_strided((8, 962, 1), (962, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_291 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    alias_23 = rand_strided((8, 4, 962, 64), (246272, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    permute_297 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_22 = rand_strided((8, 962, 1), (962, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_301 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_305 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_23 = rand_strided((8, 962, 1), (962, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_309 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    alias_24 = rand_strided((8, 4, 962, 64), (246272, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    permute_315 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_24 = rand_strided((8, 962, 1), (962, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_319 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_323 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_25 = rand_strided((8, 962, 1), (962, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_327 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    alias_25 = rand_strided((8, 4, 962, 64), (246272, 64, 256, 1), device='cuda:0', dtype=torch.float32)
    permute_333 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_3, primals_5, primals_11, primals_17, primals_23, primals_29, primals_35, primals_41, primals_45, primals_51, primals_57, primals_63, primals_69, primals_75, primals_81, primals_87, primals_93, primals_99, primals_105, primals_111, primals_117, primals_121, primals_127, primals_133, primals_139, primals_145, primals_151, primals_157, primals_163, primals_169, primals_173, cat, getitem_1, rsqrt, view_1, getitem_2, getitem_3, getitem_4, getitem_6, getitem_7, getitem_8, view_5, mul_2, view_7, addmm_2, view_9, mul_7, view_11, getitem_13, getitem_14, getitem_15, getitem_17, getitem_18, getitem_19, view_15, mul_9, view_17, addmm_6, view_19, mul_14, view_21, getitem_24, getitem_25, getitem_26, getitem_28, getitem_29, getitem_30, view_25, mul_16, view_27, addmm_10, view_29, view_31, view_32, cat_1, getitem_34, rsqrt_6, view_35, getitem_35, getitem_36, getitem_37, getitem_39, getitem_40, getitem_41, view_39, mul_23, view_41, addmm_14, view_43, mul_28, view_45, getitem_46, getitem_47, getitem_48, getitem_50, getitem_51, getitem_52, view_49, mul_30, view_51, addmm_18, view_53, mul_35, view_55, getitem_57, getitem_58, getitem_59, getitem_61, getitem_62, getitem_63, view_59, mul_37, view_61, addmm_22, view_63, mul_42, view_65, getitem_68, getitem_69, getitem_70, getitem_72, getitem_73, getitem_74, view_69, mul_44, view_71, addmm_26, view_73, mul_49, view_75, getitem_79, getitem_80, getitem_81, getitem_83, getitem_84, getitem_85, view_79, mul_51, view_81, addmm_30, view_83, mul_56, view_85, getitem_90, getitem_91, getitem_92, getitem_94, getitem_95, getitem_96, view_89, mul_58, view_91, addmm_34, view_93, view_95, view_96, cat_2, getitem_100, rsqrt_18, view_99, getitem_101, getitem_102, getitem_103, getitem_105, getitem_106, getitem_107, view_103, mul_65, view_105, addmm_38, view_107, mul_70, view_109, getitem_112, getitem_113, getitem_114, getitem_116, getitem_117, getitem_118, view_113, mul_72, view_115, addmm_42, view_117, mul_77, view_119, getitem_123, getitem_124, getitem_125, getitem_127, getitem_128, getitem_129, view_123, mul_79, view_125, addmm_46, view_127, mul_84, view_129, getitem_134, getitem_135, getitem_136, getitem_138, getitem_139, getitem_140, view_133, mul_86, view_135, addmm_50, view_137, mul_91, clone_41, permute_87, div, permute_91, permute_95, div_1, permute_99, alias_13, permute_105, div_2, permute_109, permute_113, div_3, permute_117, alias_14, permute_123, div_4, permute_127, permute_131, div_5, permute_135, alias_15, permute_141, div_6, permute_145, permute_149, div_7, permute_153, alias_16, permute_159, permute_166, permute_169, permute_173, div_9, permute_177, alias_17, permute_183, div_10, permute_187, permute_191, div_11, permute_195, alias_18, permute_201, div_12, permute_205, permute_209, div_13, permute_213, alias_19, permute_219, div_14, permute_223, permute_227, div_15, permute_231, alias_20, permute_237, div_16, permute_241, permute_245, div_17, permute_249, alias_21, permute_255, div_18, permute_259, permute_263, div_19, permute_267, alias_22, permute_273, permute_280, permute_283, permute_287, div_21, permute_291, alias_23, permute_297, div_22, permute_301, permute_305, div_23, permute_309, alias_24, permute_315, div_24, permute_319, permute_323, div_25, permute_327, alias_25, permute_333, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('pit_b_224', benchmark_compiled_module)
