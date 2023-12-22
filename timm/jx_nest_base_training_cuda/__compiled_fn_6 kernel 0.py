
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


# kernel path: /tmp/torchinductor_youkaichao/oh/cohbqpddcsk7versxdxusiuwduqzfysbefq3rycszlhhd2vuamun.py
# Source Nodes: [], Original ATen: [aten.as_strided_scatter]

triton_poi_fused_as_strided_scatter_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_as_strided_scatter_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mz/cmzkyah25cxtjv5ormchytnvykglsngnqnoh5jy2j6erejncsjrk.py
# Source Nodes: [], Original ATen: [aten.as_strided_scatter]

triton_poi_fused_as_strided_scatter_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_as_strided_scatter_2', 'mutated_arg_names': ['out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2e/c2ekg4as6akj4swo2gdlphc5x3m6ribqvj6nnz2vt52unw7teeuj.py
# Source Nodes: [div__45], Original ATen: [aten.div, aten.mul, aten.native_layer_norm_backward]
# div__45 => div_69
triton_red_fused_div_mul_native_layer_norm_backward_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_mul_native_layer_norm_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 512
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
        tmp0 = tl.load(in_ptr0 + (r2 + (512*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r2 + (512*x3)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tmp26 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp14 = tl.load(in_ptr0 + (r2 + (512*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tl.load(in_ptr2 + (r2 + (512*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = 196.0
        tmp16 = tmp14 / tmp15
        tmp18 = tmp16 * tmp17
        tmp19 = 512.0
        tmp20 = tmp18 * tmp19
        tmp21 = tmp20 - tmp6
        tmp23 = tmp22 * tmp11
        tmp24 = tmp21 - tmp23
        tmp25 = tmp13 * tmp24
        tmp27 = 0.5
        tmp28 = tmp26 / tmp27
        tmp29 = tmp25 * tmp28
        tl.store(out_ptr2 + (r2 + (512*x3)), tmp25, rmask & xmask)
        tl.store(out_ptr3 + (r2 + (512*x3)), tmp29, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/be/cbe2ojerr6jugg3sks7txni6om33234s7z5esxjk4iyikfjcl6zk.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6656
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 512)
    x0 = xindex % 512
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (512*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 196.0
        tmp5 = tmp3 / tmp4
        tmp6 = tl.load(in_ptr1 + (x0 + (512*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tmp5 * tmp6
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gc/cgc23vpd3hdlrw762kufxuj4jqqw35arsheohcl4ed4rxlwoxptr.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 13
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


# kernel path: /tmp/torchinductor_youkaichao/e2/ce277wlgmprvgl3bgqdlndfo4zwzbiwsawo5acd2psu2ccgs4wk4.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
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
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = 196.0
        tmp2 = tmp0 / tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zb/czb4zk6qozpn5gf3blwj6mey3a7rwvq6sybeahvfa3xrsr2bkwpg.py
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
    xnumel = 6656
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
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (512*r2) + (61952*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y6/cy6njsmhafdkbrazb34zzejfxkhojduyjk35s5djdpwdy3x36vsj.py
# Source Nodes: [x_371], Original ATen: [aten.gelu, aten.gelu_backward]
# x_371 => add_173, erf_23, mul_263
triton_poi_fused_gelu_gelu_backward_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
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


# kernel path: /tmp/torchinductor_youkaichao/jl/cjlhrj4di25lhuw5gwdqt5q5axfkkiou7rbhgskvzapjscxh4del.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 26624
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
        tmp1 = tl.full([1, 1], 1568, tl.int32)
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


# kernel path: /tmp/torchinductor_youkaichao/7c/c7csgzzcqniqs22kbif27akxgazz2iggtxeobc7oaoygmgbh63ax.py
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
    size_hints=[2048, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 13
    RBLOCK: tl.constexpr = 16
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


# kernel path: /tmp/torchinductor_youkaichao/ph/cphenfurq3tvkp4qgcfze7ndss6lypzqqbu7zfk6b4mst5wpywtm.py
# Source Nodes: [div__44], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__44 => div_68
triton_per_fused_add_div_mul_native_layer_norm_backward_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_11', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
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
    x3 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
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
    tmp23 = 0.5
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l5/cl55sia5aja66m56zptmzluaq7stc2f6t5w6ohauzctwlaplnvap.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (512*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp6 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/73/c73fdmuy4yyik5k4dcp43lxbit22u2n5o6of3nlu7chblels2jox.py
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
    size_hints=[128, 8192], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 6272
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (16*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (6272*y3)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/en/cenoddn7aqmzdidebb3hchamdylb5uaig676uxv3tkjxu7mjdume.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_per_fused__softmax_backward_data_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
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
    tmp1 = tl.load(in_ptr1 + (r1 + (196*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tmp1 * tmp6
    tmp8 = tmp2 - tmp7
    tl.store(out_ptr1 + (r1 + (196*x0)), tmp8, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4q/c4qm2ei4c55hfma6hinyl36e7cogvthcwlbs2tmkdhgzj2xlqq5i.py
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
    size_hints=[131072, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 75264
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y6 = (yindex // 3136)
    x4 = xindex
    y7 = yindex
    y0 = yindex % 196
    y8 = (yindex // 196)
    y1 = (yindex // 196) % 16
    y2 = (yindex // 3136) % 8
    y3 = (yindex // 25088)
    tmp0 = y6
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + (32*y7)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = 0.42044820762685725
    tmp7 = tmp5 * tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 16, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr1 + ((-802816) + y0 + (196*x4) + (6272*y8)), tmp13 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp14 * tmp6
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp13, tmp15, tmp16)
    tmp18 = tmp0 >= tmp11
    tmp19 = tl.full([1, 1], 24, tl.int64)
    tmp20 = tmp0 < tmp19
    tmp21 = tl.load(in_ptr2 + ((-1605632) + x4 + (32*y7)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp18, tmp21, tmp22)
    tmp24 = tl.where(tmp13, tmp17, tmp23)
    tmp25 = tl.where(tmp4, tmp9, tmp24)
    tl.store(out_ptr0 + (x4 + (32*y1) + (512*y3) + (1536*y0) + (301056*y2)), tmp25, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rw/crwiryjaz6oizp2qomunmip2h4r4ufqpo3r7q7yugenohbzelqmc.py
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


# kernel path: /tmp/torchinductor_youkaichao/uu/cuu73titlmdsecfdvipjqumjhsnyrgwkwb6setlfgss3tjbamxku.py
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
    size_hints=[2048, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_17', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/tw/ctwlgf3ofef5ekv4kwbejcr6jvyoalt76lxk6yl2scebughgds2j.py
# Source Nodes: [div__43], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__43 => div_66
triton_per_fused_add_div_mul_native_layer_norm_backward_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_18', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
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
    x3 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
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
    tmp23 = 0.52173912525177
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oh/cohckpr4esitikcsfdkzhoipgg6yk2rxmw4ocua4pptywn2uo66r.py
# Source Nodes: [div__41], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__41 => div_63
triton_per_fused_add_div_mul_native_layer_norm_backward_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_19', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
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
    x3 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
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
    tmp23 = 0.54347825050354
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/am/camgjdi55nxxpyl4m3id7flpfqiryl7ia4rwy22fnbatyehtwmsv.py
# Source Nodes: [div__39], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__39 => div_60
triton_per_fused_add_div_mul_native_layer_norm_backward_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_20', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
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
    x3 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
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
    tmp23 = 0.5652174055576324
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ti/cti2bweepqoj4nb2ji7guufasebrepqpuxm4y6sai5qzrwznj2wx.py
# Source Nodes: [div__37], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__37 => div_57
triton_per_fused_add_div_mul_native_layer_norm_backward_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_21', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
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
    x3 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
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
    tmp23 = 0.5869565308094025
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nf/cnfitrjxy7exxlbhys3otup4vqfpdp57tafx3njzbrdgksymp4ov.py
# Source Nodes: [div__35], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__35 => div_54
triton_per_fused_add_div_mul_native_layer_norm_backward_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_22', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
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
    x3 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
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
    tmp23 = 0.6086956560611725
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6l/c6l5rg43zb5ix727asject4jboh7mopl2i6qwic2qxt2fj6yzxjq.py
# Source Nodes: [div__33], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__33 => div_51
triton_per_fused_add_div_mul_native_layer_norm_backward_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_23', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
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
    x3 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
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
    tmp23 = 0.6304347813129425
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/75/c75uxjgqiz72sw7vz64izt36paqdx766j3hcizumbwxqzu4mxhbf.py
# Source Nodes: [div__31], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__31 => div_48
triton_per_fused_add_div_mul_native_layer_norm_backward_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_24', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
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
    x3 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
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
    tmp23 = 0.6521739065647125
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zt/cztyqxlr4peztagup5tf3edzx35ymi75xpdrwvyehl5z7knkcsto.py
# Source Nodes: [div__29], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__29 => div_45
triton_per_fused_add_div_mul_native_layer_norm_backward_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_25', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
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
    x3 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
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
    tmp23 = 0.6739130616188049
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wg/cwgzrd6p6gptuinugfxze2hmfm7fkx2dwcgvglibefzujhnswhes.py
# Source Nodes: [div__27], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__27 => div_42
triton_per_fused_add_div_mul_native_layer_norm_backward_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_26', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
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
    x3 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
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
    tmp23 = 0.695652186870575
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dv/cdvhg67ntur22g2gp755qrvyolutn6tyiqln3c72k7b6vkk6mwq7.py
# Source Nodes: [div__25], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__25 => div_39
triton_per_fused_add_div_mul_native_layer_norm_backward_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_27', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
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
    x3 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
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
    tmp23 = 0.717391312122345
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dj/cdjzvd2u3niee4yr2xzx7ib7frm3wkq4j26fpd3fqufssbvipnwl.py
# Source Nodes: [div__23], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__23 => div_36
triton_per_fused_add_div_mul_native_layer_norm_backward_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_28', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
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
    x3 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
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
    tmp23 = 0.739130437374115
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eb/cebaoqp3ecehrb6qukdlybsj2tn2zghuyco77gsaq3z65w26ube7.py
# Source Nodes: [div__21], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__21 => div_33
triton_per_fused_add_div_mul_native_layer_norm_backward_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_29', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
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
    x3 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
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
    tmp23 = 0.760869562625885
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5b/c5bwgpgmalrvwv2rbjmz34jk2zjdj5joieatmx3odl5kkyxkrqro.py
# Source Nodes: [div__19], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__19 => div_30
triton_per_fused_add_div_mul_native_layer_norm_backward_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_30', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
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
    x3 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
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
    tmp23 = 0.782608687877655
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mj/cmj5ldw3uhkacrlbonson5pfe32dw7me6gsnpedlm2ejgyhnbx73.py
# Source Nodes: [div__17], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__17 => div_27
triton_per_fused_add_div_mul_native_layer_norm_backward_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_31', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
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
    x3 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
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
    tmp23 = 0.8043478280305862
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oy/coykurvax5aw7nsileqskbezzq4qhkbojm7hujiws5p7b4sukeis.py
# Source Nodes: [div__15], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__15 => div_24
triton_per_fused_add_div_mul_native_layer_norm_backward_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_32', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
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
    x3 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
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
    tmp23 = 0.8260869532823563
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gb/cgbhhp6ocyqet63sj2onvdfxv32o2wwe3xx26wapeugfc5uqcqcq.py
# Source Nodes: [div__13], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__13 => div_21
triton_per_fused_add_div_mul_native_layer_norm_backward_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_33', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
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
    x3 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
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
    tmp23 = 0.8478260785341263
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7b/c7bc3azjnqe75sz7k4ibfy5dspmbmnfnyfv2bfgo7ceqtr2qukjr.py
# Source Nodes: [div__11], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__11 => div_18
triton_per_fused_add_div_mul_native_layer_norm_backward_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_34', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
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
    x3 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
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
    tmp23 = 0.8695652186870575
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wc/cwcb5dssg4v4ybiscc4eoq7gujav3rrh3sd4is6vfu2ygwedwxmk.py
# Source Nodes: [div__9], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__9 => div_15
triton_per_fused_add_div_mul_native_layer_norm_backward_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_35', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
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
    x3 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
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
    tmp23 = 0.8913043439388275
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2d/c2die7o7pefarmpa5tfbkkmr23bg2fzgzyu4e2oro5zr5wz2oeu7.py
# Source Nodes: [div__7], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__7 => div_12
triton_per_fused_add_div_mul_native_layer_norm_backward_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_36', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
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
    x3 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
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
    tmp23 = 0.9130434766411781
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zk/czk7jxdwefcqayrmdywrufo4vkaaojao5dr6jgvjgit3jsapizrc.py
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
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_37', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 1568
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


# kernel path: /tmp/torchinductor_youkaichao/br/cbr3svh7m6vuwtoqwmulw55a7gvctglr6vv2hswu2yppftm6hwgh.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[131072, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (100352*r1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/s4/cs4biiwbuxm2dv3lwv2p2fq672w6jjpxha7vh36tb5zdfzljyusd.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
    xnumel = 6272
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x1 = (xindex // 28) % 28
    x0 = xindex % 28
    r3 = rindex
    x2 = (xindex // 784)
    x4 = xindex
    tmp9 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr2 + (r3 + (512*x4)), rmask & xmask, other=0.0)
    tmp21 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 29, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (r3 + (512*x0) + (14848*x1) + (430592*x2)), rmask & tmp5 & xmask, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp10 = tmp8 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = tmp10 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp22 = 512.0
    tmp23 = tmp10 * tmp22
    tmp24 = tmp23 - tmp14
    tmp25 = tmp15 * tmp20
    tmp26 = tmp24 - tmp25
    tmp27 = tmp21 * tmp26
    tl.store(out_ptr2 + (r3 + (512*x4)), tmp27, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qg/cqg5eszu7mtxpqa2abzl4gpywbejn36y4atqu5fl2hakvxyzijon.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 512)
    x0 = xindex % 512
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp9 = tl.load(in_ptr1 + (x0 + (512*r2) + (65536*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp0 = ((r2 + (128*x1)) // 28) % 28
        tmp1 = tl.full([1, 1], 29, tl.int64)
        tmp2 = tmp0 < tmp1
        tmp3 = (r2 + (128*x1)) % 28
        tmp4 = tmp3 < tmp1
        tmp5 = tmp2 & tmp4
        tmp6 = tl.load(in_ptr0 + (x0 + (512*((r2 + (128*x1)) % 28)) + (14848*(((r2 + (128*x1)) // 28) % 28)) + (430592*((r2 + (128*x1)) // 784))), rmask & tmp5 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp10 = tmp8 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
        tmp14 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/np/cnppmdauynhulwzd2c7hcedo75iwwguxab7zhf7zm4lhx6fm7pq4.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 49
    RBLOCK: tl.constexpr = 64
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


# kernel path: /tmp/torchinductor_youkaichao/mu/cmut5d2u5vtleqk4pkcpc7mpw6x4ri3sxf6qmpqllzbf53prhrzi.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
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
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v2/cv2chmathejipdboqspy4lhos3sygd2d7dd6bkwbi7q2b6muvi36.py
# Source Nodes: [div__5], Original ATen: [aten.div, aten.mul]
# div__5 => div_9
triton_poi_fused_div_mul_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_mul_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196) % 4
    y2 = (yindex // 784)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + ((14*(y1 % 2)) + (28*(y0 // 14)) + (392*(y1 // 2)) + (784*x3) + (200704*y2) + (y0 % 14)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y2), ymask, eviction_policy='evict_last')
    tmp2 = 0.9347826093435287
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x3 + (256*y4)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/va/cva4gjiyjvnlmwcrha33sixkqmqsia3uepggjodssed6d7xpm72s.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_44', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zo/czo4bz2m4x3f4ctbrn66gr4fuqsmckorcz3hbnvg4uixudid3v3n.py
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
    size_hints=[256, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 49
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


# kernel path: /tmp/torchinductor_youkaichao/ue/cuebr7k2yrr7uqa2ard5hndrktb7dwggv7isamzdghovhma75mwi.py
# Source Nodes: [x_75], Original ATen: [aten.gelu, aten.gelu_backward]
# x_75 => add_30, erf_3, mul_41
triton_poi_fused_gelu_gelu_backward_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_46', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
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


# kernel path: /tmp/torchinductor_youkaichao/vv/cvv633bv6pwpgtheyd5o4yyslqda7x7lezqstodyhty2ttmntjdr.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
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
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (131072*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fm/cfmldvquticeehlp57ong45rmyjmuk6dgbbtmbo2kkrimhpd473k.py
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
    size_hints=[1024, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_48', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 49
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


# kernel path: /tmp/torchinductor_youkaichao/7x/c7x3p6xfhbrelnur4xtvyrb7vdqkdek35tpg4ho56yjthkgy4jnm.py
# Source Nodes: [div__4], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__4 => div_8
triton_per_fused_add_div_mul_native_layer_norm_backward_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 6272
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
    x2 = xindex % 196
    x3 = (xindex // 196) % 4
    x4 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + ((14*(x3 % 2)) + (28*(x2 // 14)) + (392*(x3 // 2)) + (784*r1) + (200704*x4) + (x2 % 14)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (x4), xmask, eviction_policy='evict_last')
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
    tmp23 = 0.9347826093435287
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (256*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tc/ctclhnfxboeyovd2zytqqejjzqnr6slfi7gnw7sxbfh5yn2dvgjz.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/bu/cbukgqdv75g2mwikbjjzecruizfea4onioqxihdbd6w6p7jk4e75.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64, 32768], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 25088
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 8
    y1 = (yindex // 8)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (8*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (25088*y3)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pz/cpzimjyezaz7oy2xonyg57yspy3xewwfyp6d6q27xxoxjwjcoykw.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_per_fused__softmax_backward_data_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[65536, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_52', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
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
    tmp1 = tl.load(in_ptr1 + (r1 + (196*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tmp1 * tmp6
    tmp8 = tmp2 - tmp7
    tl.store(out_ptr1 + (r1 + (196*x0)), tmp8, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dk/cdktsqm4osofggpnjqsefkq2e5uz6i6swa6g3wxdqdl74xl65uto.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_53 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_53', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 150528
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y7 = (yindex // 6272)
    x5 = xindex
    y8 = yindex
    y0 = yindex % 196
    y9 = (yindex // 196)
    y10 = yindex % 784
    y2 = (yindex // 784) % 8
    y3 = (yindex // 6272) % 8
    y4 = (yindex // 50176)
    tmp0 = y7
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x5 + (32*y8)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = 0.42044820762685725
    tmp7 = tmp5 * tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 16, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr1 + ((-1605632) + y0 + (196*x5) + (6272*y9)), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp14 * tmp6
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp13, tmp15, tmp16)
    tmp18 = tmp0 >= tmp11
    tmp19 = tl.full([1, 1], 24, tl.int64)
    tmp20 = tmp0 < tmp19
    tmp21 = tl.load(in_ptr2 + ((-3211264) + x5 + (32*y8)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp18, tmp21, tmp22)
    tmp24 = tl.where(tmp13, tmp17, tmp23)
    tmp25 = tl.where(tmp4, tmp9, tmp24)
    tl.store(out_ptr0 + (x5 + (32*y2) + (256*y4) + (768*y10) + (602112*y3)), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3t/c3totkcrqezdmkxrad7wtyk6r6k7jxuiyj6zy4tipphaneci2pmu.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_54', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 37632
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z3/cz3jekaupj3icrvlcens6p2n2ytpd5cjhcu47kl5n3ctc46anq3i.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_55', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 49
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


# kernel path: /tmp/torchinductor_youkaichao/sp/cspflndm4rctjk3gizqdiklhhfqjchn6iwf6c6cxjex4l47qgvgh.py
# Source Nodes: [div__3], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div__3 => div_6
triton_per_fused_add_div_mul_native_layer_norm_backward_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_56', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 6272
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
    x3 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (256*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
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
    tmp23 = 0.9565217383205891
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(in_out_ptr0 + (r1 + (256*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bc/cbckhxw45adnllkrj5g4zdwbeahcfju2hkeg3fuu5sdeatgczcql.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_57', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 6272
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


# kernel path: /tmp/torchinductor_youkaichao/l4/cl4yqe5ymra24u4zewjuebsw43lpqka7m5zepgd63ejxtd763fvr.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_58', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (200704*r1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wn/cwn5cspptcakgizbllpokto5fnb45nhxckablxaaja5rprr4lgcm.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_59', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3584
    x1 = (xindex // 3584) % 2
    x2 = (xindex // 7168) % 14
    x3 = (xindex // 100352)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3584*x2) + (50176*x1) + (100352*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lq/clq3j7otbiqd26xj3fx6t5i6n2rviu2w2ldf4jblm52whi4fr65l.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_60 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_60', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 56) % 56
    x0 = xindex % 56
    x2 = (xindex // 3136)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp9 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr2 + (r3 + (256*x4)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x1
        tmp1 = tl.full([1, 1], 57, tl.int64)
        tmp2 = tmp0 < tmp1
        tmp3 = x0
        tmp4 = tmp3 < tmp1
        tmp5 = tmp2 & tmp4
        tmp6 = tl.load(in_ptr0 + (r3 + (256*x0) + (14592*x1) + (831744*x2)), rmask & tmp5 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp10 = tmp8 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
        tmp15 = tmp10 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tmp19 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp29 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp34 = tl.load(in_ptr2 + (r3 + (256*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp20 = x1
        tmp21 = tl.full([1, 1], 57, tl.int64)
        tmp22 = tmp20 < tmp21
        tmp23 = x0
        tmp24 = tmp23 < tmp21
        tmp25 = tmp22 & tmp24
        tmp26 = tl.load(in_ptr0 + (r3 + (256*x0) + (14592*x1) + (831744*x2)), rmask & tmp25 & xmask, eviction_policy='evict_first', other=0.0)
        tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
        tmp28 = tl.where(tmp25, tmp26, tmp27)
        tmp30 = tmp28 * tmp29
        tmp31 = 256.0
        tmp32 = tmp30 * tmp31
        tmp33 = tmp32 - tmp12
        tmp35 = tmp34 * tmp17
        tmp36 = tmp33 - tmp35
        tmp37 = tmp19 * tmp36
        tl.store(out_ptr2 + (r3 + (256*x4)), tmp37, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xm/cxmqzqtnevnygaabekac4mdxm3ij5zrcm6iqk3n6s7knemm4yqn4.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_61 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_61', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 256)
    x0 = xindex % 256
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp9 = tl.load(in_ptr1 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp0 = ((r2 + (128*x1)) // 56) % 56
        tmp1 = tl.full([1, 1], 57, tl.int64)
        tmp2 = tmp0 < tmp1
        tmp3 = (r2 + (128*x1)) % 56
        tmp4 = tmp3 < tmp1
        tmp5 = tmp2 & tmp4
        tmp6 = tl.load(in_ptr0 + (x0 + (256*((r2 + (128*x1)) % 56)) + (14592*(((r2 + (128*x1)) // 56) % 56)) + (831744*((r2 + (128*x1)) // 3136))), rmask & tmp5 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp10 = tmp8 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
        tmp14 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nt/cntj3z5wyelkwaethsltcknzr6nfcm3q6ifmxfbk3ddisl7ipxrn.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
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
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sg/csg3duzspobz7gj6bi2zqzbmwc5fqgvrl347lg2nanabvw7xeblr.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_63', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 50176
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2o/c2ofack3xpjuae3vxoo2oowzn6xwhjqg6mtc73q7hqduhzxj5wmw.py
# Source Nodes: [div__1], Original ATen: [aten.div, aten.mul]
# div__1 => div_3
triton_poi_fused_div_mul_64 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_mul_64', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196) % 16
    y2 = (yindex // 3136)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + ((14*(y1 % 4)) + (56*(y0 // 14)) + (784*(y1 // 4)) + (3136*x3) + (401408*y2) + (y0 % 14)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y2), ymask, eviction_policy='evict_last')
    tmp2 = 0.9782608691602945
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x3 + (128*y4)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6o/c6ozelbvrvp2tjll2fjxctojey5k6vtoaeuoolv3hvhdqkcrh3tu.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_65 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_65', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
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
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5l/c5lqbqg24ixg5xjl2dujndwjbzaqzopgfgj44c2oqtuso2qhdw4e.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_66 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_66', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o7/co7qtjs4cabv5qe4msbbmekgoqmpvonbiaju7xapi6tfnfrubgxv.py
# Source Nodes: [x_31], Original ATen: [aten.gelu, aten.gelu_backward]
# x_31 => add_13, erf_1, mul_17
triton_poi_fused_gelu_gelu_backward_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_67', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
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


# kernel path: /tmp/torchinductor_youkaichao/aj/cajyjwdmmxqvjurfw323vdqordg5vpx37sqchojjvrvmkhbcduor.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_68 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_68', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 100352
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
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (65536*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6q/c6qzyatkkjm2yeyfzoetqx7g5ag5ed6bkzrsviexuhquen3ctdii.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_69 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_69', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
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
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ci/cciheumvj32erzps3b7a5nxwev7wvnbexsdne6dfgo3wzsfw6i37.py
# Source Nodes: [div_], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
# div_ => div_2
triton_per_fused_add_div_mul_native_layer_norm_backward_70 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_native_layer_norm_backward_70', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 196
    x3 = (xindex // 196) % 16
    x4 = (xindex // 3136)
    tmp0 = tl.load(in_ptr0 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (128*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + ((14*(x3 % 4)) + (56*(x2 // 14)) + (784*(x3 // 4)) + (3136*r1) + (401408*x4) + (x2 % 14)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (x4), xmask, eviction_policy='evict_last')
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
    tmp23 = 0.9782608691602945
    tmp24 = tmp22 / tmp23
    tmp25 = tmp21 * tmp24
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (128*x0)), tmp25, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bh/cbhtpa4zlfw57zvnv7mzgww5y55dxujsr4myxz7pod4q35uggcpn.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_71 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_71', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (128*r2) + (16384*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/pv/cpvzflwpfgd4oxmrhe4xygpn7zemrbzydmchsfsmg2aomzy4ojp3.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_72 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 131072], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_72', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 100352
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 4
    y1 = (yindex // 4)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (4*x2) + (401408*y1)), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (100352*y3)), tmp0, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w4/cw4s6cdc7lyxc4rcpjug4j6hkapfyqffsesdbukwffw5wv6o3zv6.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data]

triton_per_fused__softmax_backward_data_73 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[131072, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_73', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (196*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (196*x0)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tmp1 * tmp6
    tmp8 = tmp2 - tmp7
    tl.store(out_ptr1 + (r1 + (196*x0)), tmp8, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yf/cyforji3l5gy7foli3m5kpgm4or7xks5g5xrbsuftgq5c2u3a2ye.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_74 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_74', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 301056
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y7 = (yindex // 12544)
    x5 = xindex
    y8 = yindex
    y0 = yindex % 196
    y9 = (yindex // 196)
    y10 = yindex % 3136
    y2 = (yindex // 3136) % 4
    y3 = (yindex // 12544) % 8
    y4 = (yindex // 100352)
    tmp0 = y7
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x5 + (32*y8)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = 0.42044820762685725
    tmp7 = tmp5 * tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 16, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr1 + ((-3211264) + y0 + (196*x5) + (6272*y9)), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp14 * tmp6
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp13, tmp15, tmp16)
    tmp18 = tmp0 >= tmp11
    tmp19 = tl.full([1, 1], 24, tl.int64)
    tmp20 = tmp0 < tmp19
    tmp21 = tl.load(in_ptr2 + ((-6422528) + x5 + (32*y8)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp18, tmp21, tmp22)
    tmp24 = tl.where(tmp13, tmp17, tmp23)
    tmp25 = tl.where(tmp4, tmp9, tmp24)
    tl.store(out_ptr0 + (x5 + (32*y2) + (128*y4) + (384*y10) + (1204224*y3)), tmp25, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zx/czxs3fuw5ushnagy6pprnz2i565dpucjfsqo3xcuat2lzhjgysft.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_75 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_75', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 75264
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
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (49152*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mq/cmqg2jgcrawjnu6zr6xkqx5rqrt36vmue724oevdffw4ehevsbh7.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_76 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_76', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
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
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xl/cxl6pycg2dmtcybyulbjdjqtpzrj54n62msdelusx2v3eb7p2wqw.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_77 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_77', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
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


# kernel path: /tmp/torchinductor_youkaichao/dy/cdyikee6rvsn36wc25dkqyhazwwrljbrhjzpkj4rykfb2mdteoa2.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_78 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_78', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 25088
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


# kernel path: /tmp/torchinductor_youkaichao/nq/cnqhh26lbgts2qea4gcdhqm6sbhrhkzw44gkw6alujkkuiux3xvf.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_79 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_79', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (401408*r1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6y/c6yd24v2pejja2mhy2aclgfk6zv2ar3pycu65g7v4rwtxwgod3kg.py
# Source Nodes: [], Original ATen: [aten.permute]

triton_poi_fused_permute_80 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_permute_80', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 56
    x2 = (xindex // 7168) % 56
    x3 = (xindex // 401408)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*(x1 % 14)) + (1792*(x2 % 14)) + (25088*(x1 // 14)) + (100352*(x2 // 14)) + (401408*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_2, primals_4, primals_6, primals_8, primals_10, primals_13, primals_15, primals_17, primals_19, primals_21, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_38, primals_40, primals_42, primals_44, primals_46, primals_48, primals_50, primals_52, primals_54, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_106, primals_124, primals_142, primals_306, mul, view_2, view_12, mul_4, view_14, addmm_2, view_16, mul_9, view_18, view_28, bernoulli, mul_14, view_30, addmm_6, view_32, bernoulli_1, permute_17, mul_20, constant_pad_nd, getitem_17, mul_22, view_38, view_48, bernoulli_2, mul_27, view_50, addmm_10, view_52, bernoulli_3, mul_33, view_54, view_64, bernoulli_4, mul_38, view_66, addmm_14, view_68, bernoulli_5, permute_37, mul_44, constant_pad_nd_1, getitem_35, mul_46, view_74, view_84, bernoulli_6, mul_51, view_86, addmm_18, view_88, bernoulli_7, mul_57, view_90, view_100, bernoulli_8, mul_62, view_102, addmm_22, view_104, bernoulli_9, mul_68, view_106, view_116, bernoulli_10, mul_73, view_118, addmm_26, view_120, bernoulli_11, mul_79, view_122, view_132, bernoulli_12, mul_84, view_134, addmm_30, view_136, bernoulli_13, mul_90, view_138, view_148, bernoulli_14, mul_95, view_150, addmm_34, view_152, bernoulli_15, mul_101, view_154, view_164, bernoulli_16, mul_106, view_166, addmm_38, view_168, bernoulli_17, mul_112, view_170, view_180, bernoulli_18, mul_117, view_182, addmm_42, view_184, bernoulli_19, mul_123, view_186, view_196, bernoulli_20, mul_128, view_198, addmm_46, view_200, bernoulli_21, mul_134, view_202, view_212, bernoulli_22, mul_139, view_214, addmm_50, view_216, bernoulli_23, mul_145, view_218, view_228, bernoulli_24, mul_150, view_230, addmm_54, view_232, bernoulli_25, mul_156, view_234, view_244, bernoulli_26, mul_161, view_246, addmm_58, view_248, bernoulli_27, mul_167, view_250, view_260, bernoulli_28, mul_172, view_262, addmm_62, view_264, bernoulli_29, mul_178, view_266, view_276, bernoulli_30, mul_183, view_278, addmm_66, view_280, bernoulli_31, mul_189, view_282, view_292, bernoulli_32, mul_194, view_294, addmm_70, view_296, bernoulli_33, mul_200, view_298, view_308, bernoulli_34, mul_205, view_310, addmm_74, view_312, bernoulli_35, mul_211, view_314, view_324, bernoulli_36, mul_216, view_326, addmm_78, view_328, bernoulli_37, mul_222, view_330, view_340, bernoulli_38, mul_227, view_342, addmm_82, view_344, bernoulli_39, mul_233, view_346, view_356, bernoulli_40, mul_238, view_358, addmm_86, view_360, bernoulli_41, mul_244, view_362, view_372, bernoulli_42, mul_249, view_374, addmm_90, view_376, bernoulli_43, mul_255, view_378, view_388, bernoulli_44, mul_260, view_390, addmm_94, view_392, bernoulli_45, mul_266, clone_174, permute_187, div_71, permute_195, permute_199, div_72, permute_203, permute_208, permute_209, alias_24, permute_210, permute_211, permute_214, div_73, permute_218, permute_222, div_74, permute_226, permute_231, permute_232, alias_25, permute_233, permute_234, permute_237, div_75, permute_241, permute_245, div_76, permute_249, permute_254, permute_255, alias_26, permute_256, permute_257, permute_260, div_77, permute_264, permute_268, div_78, permute_272, permute_277, permute_278, alias_27, permute_279, permute_280, permute_283, div_79, permute_287, permute_291, div_80, permute_295, permute_300, permute_301, alias_28, permute_302, permute_303, permute_306, div_81, permute_310, permute_314, div_82, permute_318, permute_323, permute_324, alias_29, permute_325, permute_326, permute_329, div_83, permute_333, permute_337, div_84, permute_341, permute_346, permute_347, alias_30, permute_348, permute_349, permute_352, div_85, permute_356, permute_360, div_86, permute_364, permute_369, permute_370, alias_31, permute_371, permute_372, permute_375, div_87, permute_379, permute_383, div_88, permute_387, permute_392, permute_393, alias_32, permute_394, permute_395, permute_398, div_89, permute_402, permute_406, div_90, permute_410, permute_415, permute_416, alias_33, permute_417, permute_418, permute_421, div_91, permute_425, permute_429, div_92, permute_433, permute_438, permute_439, alias_34, permute_440, permute_441, permute_444, div_93, permute_448, permute_452, div_94, permute_456, permute_461, permute_462, alias_35, permute_463, permute_464, permute_467, div_95, permute_471, permute_475, div_96, permute_479, permute_484, permute_485, alias_36, permute_486, permute_487, permute_490, div_97, permute_494, permute_498, div_98, permute_502, permute_507, permute_508, alias_37, permute_509, permute_510, permute_513, div_99, permute_517, permute_521, div_100, permute_525, permute_530, permute_531, alias_38, permute_532, permute_533, permute_536, div_101, permute_540, permute_544, div_102, permute_548, permute_553, permute_554, alias_39, permute_555, permute_556, permute_559, div_103, permute_563, permute_567, div_104, permute_571, permute_576, permute_577, alias_40, permute_578, permute_579, permute_582, div_105, permute_586, permute_590, div_106, permute_594, permute_599, permute_600, alias_41, permute_601, permute_602, permute_605, div_107, permute_609, permute_613, div_108, permute_617, permute_622, permute_623, alias_42, permute_624, permute_625, permute_628, div_109, permute_632, permute_636, div_110, permute_640, permute_645, permute_646, alias_43, permute_647, permute_648, permute_651, div_111, div_112, permute_661, permute_665, div_113, permute_669, permute_674, permute_675, alias_44, permute_676, permute_677, permute_680, div_114, permute_684, permute_688, div_115, permute_692, permute_697, permute_698, alias_45, permute_699, permute_700, permute_703, div_116, div_117, permute_713, permute_717, div_118, permute_721, permute_726, permute_727, alias_46, permute_728, permute_729, permute_732, div_119, permute_736, permute_740, div_120, permute_744, permute_749, permute_750, alias_47, permute_751, permute_752, permute_755, div_121, tangents_1 = args
    args.clear()
    assert_size_stride(primals_2, (128, ), (1, ))
    assert_size_stride(primals_4, (128, ), (1, ))
    assert_size_stride(primals_6, (128, ), (1, ))
    assert_size_stride(primals_8, (128, ), (1, ))
    assert_size_stride(primals_10, (256, ), (1, ))
    assert_size_stride(primals_13, (256, ), (1, ))
    assert_size_stride(primals_15, (256, ), (1, ))
    assert_size_stride(primals_17, (256, ), (1, ))
    assert_size_stride(primals_19, (256, ), (1, ))
    assert_size_stride(primals_21, (512, ), (1, ))
    assert_size_stride(primals_24, (512, ), (1, ))
    assert_size_stride(primals_26, (512, ), (1, ))
    assert_size_stride(primals_28, (512, ), (1, ))
    assert_size_stride(primals_30, (512, ), (1, ))
    assert_size_stride(primals_32, (512, ), (1, ))
    assert_size_stride(primals_34, (512, ), (1, ))
    assert_size_stride(primals_36, (512, ), (1, ))
    assert_size_stride(primals_38, (512, ), (1, ))
    assert_size_stride(primals_40, (512, ), (1, ))
    assert_size_stride(primals_42, (512, ), (1, ))
    assert_size_stride(primals_44, (512, ), (1, ))
    assert_size_stride(primals_46, (512, ), (1, ))
    assert_size_stride(primals_48, (512, ), (1, ))
    assert_size_stride(primals_50, (512, ), (1, ))
    assert_size_stride(primals_52, (512, ), (1, ))
    assert_size_stride(primals_54, (512, ), (1, ))
    assert_size_stride(primals_56, (512, ), (1, ))
    assert_size_stride(primals_58, (512, ), (1, ))
    assert_size_stride(primals_60, (512, ), (1, ))
    assert_size_stride(primals_62, (512, ), (1, ))
    assert_size_stride(primals_64, (512, ), (1, ))
    assert_size_stride(primals_66, (512, ), (1, ))
    assert_size_stride(primals_68, (512, ), (1, ))
    assert_size_stride(primals_70, (512, ), (1, ))
    assert_size_stride(primals_72, (512, ), (1, ))
    assert_size_stride(primals_74, (512, ), (1, ))
    assert_size_stride(primals_76, (512, ), (1, ))
    assert_size_stride(primals_78, (512, ), (1, ))
    assert_size_stride(primals_80, (512, ), (1, ))
    assert_size_stride(primals_82, (512, ), (1, ))
    assert_size_stride(primals_84, (512, ), (1, ))
    assert_size_stride(primals_86, (512, ), (1, ))
    assert_size_stride(primals_88, (512, ), (1, ))
    assert_size_stride(primals_90, (512, ), (1, ))
    assert_size_stride(primals_92, (512, ), (1, ))
    assert_size_stride(primals_94, (512, ), (1, ))
    assert_size_stride(primals_96, (512, ), (1, ))
    assert_size_stride(primals_98, (512, ), (1, ))
    assert_size_stride(primals_100, (512, ), (1, ))
    assert_size_stride(primals_102, (512, ), (1, ))
    assert_size_stride(primals_104, (512, ), (1, ))
    assert_size_stride(primals_106, (128, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(primals_124, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_142, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_306, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(mul, (8, 16, 196, 128), (401408, 25088, 128, 1))
    assert_size_stride(view_2, (25088, 128), (128, 1))
    assert_size_stride(view_12, (25088, 128), (128, 1))
    assert_size_stride(mul_4, (8, 16, 196, 128), (401408, 25088, 128, 1))
    assert_size_stride(view_14, (25088, 128), (128, 1))
    assert_size_stride(addmm_2, (25088, 512), (512, 1))
    assert_size_stride(view_16, (25088, 512), (512, 1))
    assert_size_stride(mul_9, (8, 16, 196, 128), (401408, 25088, 128, 1))
    assert_size_stride(view_18, (25088, 128), (128, 1))
    assert_size_stride(view_28, (25088, 128), (128, 1))
    assert_size_stride(bernoulli, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_14, (8, 16, 196, 128), (401408, 25088, 128, 1))
    assert_size_stride(view_30, (25088, 128), (128, 1))
    assert_size_stride(addmm_6, (25088, 512), (512, 1))
    assert_size_stride(view_32, (25088, 512), (512, 1))
    assert_size_stride(bernoulli_1, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(permute_17, (8, 128, 56, 56), (401408, 1, 7168, 128))
    assert_size_stride(mul_20, (8, 56, 56, 256), (802816, 14336, 256, 1))
    assert_size_stride(constant_pad_nd, (8, 256, 57, 57), (831744, 1, 14592, 256))
    assert_size_stride(getitem_17, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(mul_22, (8, 4, 196, 256), (200704, 50176, 256, 1))
    assert_size_stride(view_38, (6272, 256), (256, 1))
    assert_size_stride(view_48, (6272, 256), (256, 1))
    assert_size_stride(bernoulli_2, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_27, (8, 4, 196, 256), (200704, 50176, 256, 1))
    assert_size_stride(view_50, (6272, 256), (256, 1))
    assert_size_stride(addmm_10, (6272, 1024), (1024, 1))
    assert_size_stride(view_52, (6272, 1024), (1024, 1))
    assert_size_stride(bernoulli_3, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_33, (8, 4, 196, 256), (200704, 50176, 256, 1))
    assert_size_stride(view_54, (6272, 256), (256, 1))
    assert_size_stride(view_64, (6272, 256), (256, 1))
    assert_size_stride(bernoulli_4, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_38, (8, 4, 196, 256), (200704, 50176, 256, 1))
    assert_size_stride(view_66, (6272, 256), (256, 1))
    assert_size_stride(addmm_14, (6272, 1024), (1024, 1))
    assert_size_stride(view_68, (6272, 1024), (1024, 1))
    assert_size_stride(bernoulli_5, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(permute_37, (8, 256, 28, 28), (200704, 1, 7168, 256))
    assert_size_stride(mul_44, (8, 28, 28, 512), (401408, 14336, 512, 1))
    assert_size_stride(constant_pad_nd_1, (8, 512, 29, 29), (430592, 1, 14848, 512))
    assert_size_stride(getitem_35, (8, 512, 14, 14), (100352, 1, 7168, 512))
    assert_size_stride(mul_46, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_74, (1568, 512), (512, 1))
    assert_size_stride(view_84, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_6, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_51, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_86, (1568, 512), (512, 1))
    assert_size_stride(addmm_18, (1568, 2048), (2048, 1))
    assert_size_stride(view_88, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_7, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_57, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_90, (1568, 512), (512, 1))
    assert_size_stride(view_100, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_8, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_62, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_102, (1568, 512), (512, 1))
    assert_size_stride(addmm_22, (1568, 2048), (2048, 1))
    assert_size_stride(view_104, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_9, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_68, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_106, (1568, 512), (512, 1))
    assert_size_stride(view_116, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_10, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_73, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_118, (1568, 512), (512, 1))
    assert_size_stride(addmm_26, (1568, 2048), (2048, 1))
    assert_size_stride(view_120, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_11, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_79, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_122, (1568, 512), (512, 1))
    assert_size_stride(view_132, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_12, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_84, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_134, (1568, 512), (512, 1))
    assert_size_stride(addmm_30, (1568, 2048), (2048, 1))
    assert_size_stride(view_136, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_13, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_90, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_138, (1568, 512), (512, 1))
    assert_size_stride(view_148, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_14, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_95, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_150, (1568, 512), (512, 1))
    assert_size_stride(addmm_34, (1568, 2048), (2048, 1))
    assert_size_stride(view_152, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_15, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_101, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_154, (1568, 512), (512, 1))
    assert_size_stride(view_164, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_16, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_106, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_166, (1568, 512), (512, 1))
    assert_size_stride(addmm_38, (1568, 2048), (2048, 1))
    assert_size_stride(view_168, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_17, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_112, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_170, (1568, 512), (512, 1))
    assert_size_stride(view_180, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_18, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_117, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_182, (1568, 512), (512, 1))
    assert_size_stride(addmm_42, (1568, 2048), (2048, 1))
    assert_size_stride(view_184, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_19, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_123, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_186, (1568, 512), (512, 1))
    assert_size_stride(view_196, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_20, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_128, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_198, (1568, 512), (512, 1))
    assert_size_stride(addmm_46, (1568, 2048), (2048, 1))
    assert_size_stride(view_200, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_21, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_134, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_202, (1568, 512), (512, 1))
    assert_size_stride(view_212, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_22, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_139, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_214, (1568, 512), (512, 1))
    assert_size_stride(addmm_50, (1568, 2048), (2048, 1))
    assert_size_stride(view_216, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_23, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_145, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_218, (1568, 512), (512, 1))
    assert_size_stride(view_228, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_24, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_150, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_230, (1568, 512), (512, 1))
    assert_size_stride(addmm_54, (1568, 2048), (2048, 1))
    assert_size_stride(view_232, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_25, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_156, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_234, (1568, 512), (512, 1))
    assert_size_stride(view_244, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_26, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_161, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_246, (1568, 512), (512, 1))
    assert_size_stride(addmm_58, (1568, 2048), (2048, 1))
    assert_size_stride(view_248, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_27, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_167, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_250, (1568, 512), (512, 1))
    assert_size_stride(view_260, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_28, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_172, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_262, (1568, 512), (512, 1))
    assert_size_stride(addmm_62, (1568, 2048), (2048, 1))
    assert_size_stride(view_264, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_29, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_178, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_266, (1568, 512), (512, 1))
    assert_size_stride(view_276, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_30, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_183, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_278, (1568, 512), (512, 1))
    assert_size_stride(addmm_66, (1568, 2048), (2048, 1))
    assert_size_stride(view_280, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_31, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_189, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_282, (1568, 512), (512, 1))
    assert_size_stride(view_292, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_32, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_194, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_294, (1568, 512), (512, 1))
    assert_size_stride(addmm_70, (1568, 2048), (2048, 1))
    assert_size_stride(view_296, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_33, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_200, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_298, (1568, 512), (512, 1))
    assert_size_stride(view_308, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_34, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_205, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_310, (1568, 512), (512, 1))
    assert_size_stride(addmm_74, (1568, 2048), (2048, 1))
    assert_size_stride(view_312, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_35, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_211, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_314, (1568, 512), (512, 1))
    assert_size_stride(view_324, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_36, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_216, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_326, (1568, 512), (512, 1))
    assert_size_stride(addmm_78, (1568, 2048), (2048, 1))
    assert_size_stride(view_328, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_37, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_222, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_330, (1568, 512), (512, 1))
    assert_size_stride(view_340, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_38, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_227, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_342, (1568, 512), (512, 1))
    assert_size_stride(addmm_82, (1568, 2048), (2048, 1))
    assert_size_stride(view_344, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_39, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_233, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_346, (1568, 512), (512, 1))
    assert_size_stride(view_356, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_40, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_238, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_358, (1568, 512), (512, 1))
    assert_size_stride(addmm_86, (1568, 2048), (2048, 1))
    assert_size_stride(view_360, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_41, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_244, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_362, (1568, 512), (512, 1))
    assert_size_stride(view_372, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_42, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_249, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_374, (1568, 512), (512, 1))
    assert_size_stride(addmm_90, (1568, 2048), (2048, 1))
    assert_size_stride(view_376, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_43, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_255, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_378, (1568, 512), (512, 1))
    assert_size_stride(view_388, (1568, 512), (512, 1))
    assert_size_stride(bernoulli_44, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_260, (8, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(view_390, (1568, 512), (512, 1))
    assert_size_stride(addmm_94, (1568, 2048), (2048, 1))
    assert_size_stride(view_392, (1568, 2048), (2048, 1))
    assert_size_stride(bernoulli_45, (8, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(mul_266, (8, 14, 14, 512), (100352, 7168, 512, 1))
    assert_size_stride(clone_174, (8, 512), (512, 1))
    assert_size_stride(permute_187, (1000, 512), (512, 1))
    assert_size_stride(div_71, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_195, (512, 2048), (2048, 1))
    assert_size_stride(permute_199, (2048, 512), (512, 1))
    assert_size_stride(div_72, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_203, (512, 512), (512, 1))
    assert_size_stride(permute_208, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_209, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_24, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1))
    assert_size_stride(permute_210, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_211, (128, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_214, (1536, 512), (512, 1))
    assert_size_stride(div_73, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_218, (512, 2048), (2048, 1))
    assert_size_stride(permute_222, (2048, 512), (512, 1))
    assert_size_stride(div_74, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_226, (512, 512), (512, 1))
    assert_size_stride(permute_231, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_232, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_25, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1))
    assert_size_stride(permute_233, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_234, (128, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_237, (1536, 512), (512, 1))
    assert_size_stride(div_75, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_241, (512, 2048), (2048, 1))
    assert_size_stride(permute_245, (2048, 512), (512, 1))
    assert_size_stride(div_76, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_249, (512, 512), (512, 1))
    assert_size_stride(permute_254, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_255, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_26, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1))
    assert_size_stride(permute_256, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_257, (128, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_260, (1536, 512), (512, 1))
    assert_size_stride(div_77, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_264, (512, 2048), (2048, 1))
    assert_size_stride(permute_268, (2048, 512), (512, 1))
    assert_size_stride(div_78, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_272, (512, 512), (512, 1))
    assert_size_stride(permute_277, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_278, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_27, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1))
    assert_size_stride(permute_279, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_280, (128, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_283, (1536, 512), (512, 1))
    assert_size_stride(div_79, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_287, (512, 2048), (2048, 1))
    assert_size_stride(permute_291, (2048, 512), (512, 1))
    assert_size_stride(div_80, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_295, (512, 512), (512, 1))
    assert_size_stride(permute_300, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_301, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_28, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1))
    assert_size_stride(permute_302, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_303, (128, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_306, (1536, 512), (512, 1))
    assert_size_stride(div_81, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_310, (512, 2048), (2048, 1))
    assert_size_stride(permute_314, (2048, 512), (512, 1))
    assert_size_stride(div_82, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_318, (512, 512), (512, 1))
    assert_size_stride(permute_323, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_324, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_29, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1))
    assert_size_stride(permute_325, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_326, (128, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_329, (1536, 512), (512, 1))
    assert_size_stride(div_83, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_333, (512, 2048), (2048, 1))
    assert_size_stride(permute_337, (2048, 512), (512, 1))
    assert_size_stride(div_84, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_341, (512, 512), (512, 1))
    assert_size_stride(permute_346, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_347, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_30, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1))
    assert_size_stride(permute_348, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_349, (128, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_352, (1536, 512), (512, 1))
    assert_size_stride(div_85, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_356, (512, 2048), (2048, 1))
    assert_size_stride(permute_360, (2048, 512), (512, 1))
    assert_size_stride(div_86, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_364, (512, 512), (512, 1))
    assert_size_stride(permute_369, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_370, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_31, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1))
    assert_size_stride(permute_371, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_372, (128, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_375, (1536, 512), (512, 1))
    assert_size_stride(div_87, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_379, (512, 2048), (2048, 1))
    assert_size_stride(permute_383, (2048, 512), (512, 1))
    assert_size_stride(div_88, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_387, (512, 512), (512, 1))
    assert_size_stride(permute_392, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_393, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_32, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1))
    assert_size_stride(permute_394, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_395, (128, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_398, (1536, 512), (512, 1))
    assert_size_stride(div_89, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_402, (512, 2048), (2048, 1))
    assert_size_stride(permute_406, (2048, 512), (512, 1))
    assert_size_stride(div_90, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_410, (512, 512), (512, 1))
    assert_size_stride(permute_415, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_416, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_33, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1))
    assert_size_stride(permute_417, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_418, (128, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_421, (1536, 512), (512, 1))
    assert_size_stride(div_91, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_425, (512, 2048), (2048, 1))
    assert_size_stride(permute_429, (2048, 512), (512, 1))
    assert_size_stride(div_92, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_433, (512, 512), (512, 1))
    assert_size_stride(permute_438, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_439, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_34, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1))
    assert_size_stride(permute_440, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_441, (128, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_444, (1536, 512), (512, 1))
    assert_size_stride(div_93, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_448, (512, 2048), (2048, 1))
    assert_size_stride(permute_452, (2048, 512), (512, 1))
    assert_size_stride(div_94, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_456, (512, 512), (512, 1))
    assert_size_stride(permute_461, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_462, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_35, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1))
    assert_size_stride(permute_463, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_464, (128, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_467, (1536, 512), (512, 1))
    assert_size_stride(div_95, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_471, (512, 2048), (2048, 1))
    assert_size_stride(permute_475, (2048, 512), (512, 1))
    assert_size_stride(div_96, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_479, (512, 512), (512, 1))
    assert_size_stride(permute_484, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_485, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_36, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1))
    assert_size_stride(permute_486, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_487, (128, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_490, (1536, 512), (512, 1))
    assert_size_stride(div_97, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_494, (512, 2048), (2048, 1))
    assert_size_stride(permute_498, (2048, 512), (512, 1))
    assert_size_stride(div_98, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_502, (512, 512), (512, 1))
    assert_size_stride(permute_507, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_508, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_37, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1))
    assert_size_stride(permute_509, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_510, (128, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_513, (1536, 512), (512, 1))
    assert_size_stride(div_99, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_517, (512, 2048), (2048, 1))
    assert_size_stride(permute_521, (2048, 512), (512, 1))
    assert_size_stride(div_100, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_525, (512, 512), (512, 1))
    assert_size_stride(permute_530, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_531, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_38, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1))
    assert_size_stride(permute_532, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_533, (128, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_536, (1536, 512), (512, 1))
    assert_size_stride(div_101, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_540, (512, 2048), (2048, 1))
    assert_size_stride(permute_544, (2048, 512), (512, 1))
    assert_size_stride(div_102, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_548, (512, 512), (512, 1))
    assert_size_stride(permute_553, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_554, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_39, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1))
    assert_size_stride(permute_555, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_556, (128, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_559, (1536, 512), (512, 1))
    assert_size_stride(div_103, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_563, (512, 2048), (2048, 1))
    assert_size_stride(permute_567, (2048, 512), (512, 1))
    assert_size_stride(div_104, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_571, (512, 512), (512, 1))
    assert_size_stride(permute_576, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_577, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_40, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1))
    assert_size_stride(permute_578, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_579, (128, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_582, (1536, 512), (512, 1))
    assert_size_stride(div_105, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_586, (512, 2048), (2048, 1))
    assert_size_stride(permute_590, (2048, 512), (512, 1))
    assert_size_stride(div_106, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_594, (512, 512), (512, 1))
    assert_size_stride(permute_599, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_600, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_41, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1))
    assert_size_stride(permute_601, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_602, (128, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_605, (1536, 512), (512, 1))
    assert_size_stride(div_107, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_609, (512, 2048), (2048, 1))
    assert_size_stride(permute_613, (2048, 512), (512, 1))
    assert_size_stride(div_108, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_617, (512, 512), (512, 1))
    assert_size_stride(permute_622, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_623, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_42, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1))
    assert_size_stride(permute_624, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_625, (128, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_628, (1536, 512), (512, 1))
    assert_size_stride(div_109, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_632, (512, 2048), (2048, 1))
    assert_size_stride(permute_636, (2048, 512), (512, 1))
    assert_size_stride(div_110, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(permute_640, (512, 512), (512, 1))
    assert_size_stride(permute_645, (128, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_646, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_43, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1))
    assert_size_stride(permute_647, (128, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_648, (128, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_651, (1536, 512), (512, 1))
    assert_size_stride(div_111, (8, 1, 196, 1), (196, 196, 1, 1))
    assert_size_stride(div_112, (8, 28, 28, 1), (784, 28, 1, 1))
    assert_size_stride(permute_661, (256, 1024), (1024, 1))
    assert_size_stride(permute_665, (1024, 256), (256, 1))
    assert_size_stride(div_113, (8, 4, 196, 1), (784, 196, 1, 1))
    assert_size_stride(permute_669, (256, 256), (256, 1))
    assert_size_stride(permute_674, (256, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_675, (256, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_44, (8, 8, 4, 196, 196), (1229312, 153664, 38416, 196, 1))
    assert_size_stride(permute_676, (256, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_677, (256, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_680, (768, 256), (256, 1))
    assert_size_stride(div_114, (8, 4, 196, 1), (784, 196, 1, 1))
    assert_size_stride(permute_684, (256, 1024), (1024, 1))
    assert_size_stride(permute_688, (1024, 256), (256, 1))
    assert_size_stride(div_115, (8, 4, 196, 1), (784, 196, 1, 1))
    assert_size_stride(permute_692, (256, 256), (256, 1))
    assert_size_stride(permute_697, (256, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_698, (256, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_45, (8, 8, 4, 196, 196), (1229312, 153664, 38416, 196, 1))
    assert_size_stride(permute_699, (256, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_700, (256, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_703, (768, 256), (256, 1))
    assert_size_stride(div_116, (8, 4, 196, 1), (784, 196, 1, 1))
    assert_size_stride(div_117, (8, 56, 56, 1), (3136, 56, 1, 1))
    assert_size_stride(permute_713, (128, 512), (512, 1))
    assert_size_stride(permute_717, (512, 128), (128, 1))
    assert_size_stride(div_118, (8, 16, 196, 1), (3136, 196, 1, 1))
    assert_size_stride(permute_721, (128, 128), (128, 1))
    assert_size_stride(permute_726, (512, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_727, (512, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_46, (8, 4, 16, 196, 196), (2458624, 614656, 38416, 196, 1))
    assert_size_stride(permute_728, (512, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_729, (512, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_732, (384, 128), (128, 1))
    assert_size_stride(div_119, (8, 16, 196, 1), (3136, 196, 1, 1))
    assert_size_stride(permute_736, (128, 512), (512, 1))
    assert_size_stride(permute_740, (512, 128), (128, 1))
    assert_size_stride(div_120, (8, 16, 196, 1), (3136, 196, 1, 1))
    assert_size_stride(permute_744, (128, 128), (128, 1))
    assert_size_stride(permute_749, (512, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_750, (512, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_47, (8, 4, 16, 196, 196), (2458624, 614656, 38416, 196, 1))
    assert_size_stride(permute_751, (512, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_752, (512, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_755, (384, 128), (128, 1))
    assert_size_stride(div_121, (8, 16, 196, 1), (3136, 196, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_187, out=buf0)
        del permute_187
        buf1 = empty((1000, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone_174, out=buf1)
        del clone_174
        buf2 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_sum_0.run(tangents_1, buf2, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf4 = empty((4096, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_1.run(buf4, 4096, grid=grid(4096), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_2.run(buf0, buf4, 4096, grid=grid(4096), stream=stream0)
        del buf0
        buf11 = empty((8, 14, 14, 512), device='cuda', dtype=torch.float32)
        buf12 = empty((8, 1, 196, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__45], Original ATen: [aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_red_fused_div_mul_native_layer_norm_backward_3.run(buf4, primals_104, mul_266, div_71, bernoulli_45, buf11, buf12, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_45
        del div_71
        del primals_104
        buf8 = empty_strided((512, 13), (1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_4.run(buf4, mul_266, buf8, 6656, 121, grid=grid(6656), stream=stream0)
        del mul_266
        buf9 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_5.run(buf8, buf9, 512, 13, grid=grid(512), stream=stream0)
        buf10 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_6.run(buf4, buf10, 512, 1568, grid=grid(512), stream=stream0)
        del buf4
        buf13 = empty((1568, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf12, (1568, 512), (512, 1), 0), permute_195, out=buf13)
        del permute_195
        buf14 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf12, (512, 1568), (1, 512), 0), view_392, out=buf14)
        del view_392
        buf15 = reinterpret_tensor(buf8, (1, 512, 13), (6656, 1, 512), 0); del buf8  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf12, buf15, 6656, 121, grid=grid(6656), stream=stream0)
        buf16 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf15, buf16, 512, 13, grid=grid(512), stream=stream0)
        buf17 = reinterpret_tensor(buf13, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf13  # reuse
        # Source Nodes: [x_371], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_8.run(buf17, addmm_94, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_94
        buf18 = reinterpret_tensor(buf12, (1568, 512), (512, 1), 0); del buf12  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (1568, 2048), (2048, 1), 0), permute_199, out=buf18)
        del permute_199
        buf19 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (2048, 1568), (1, 2048), 0), view_390, out=buf19)
        del view_390
        buf20 = empty_strided((1, 2048, 13), (26624, 1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf17, buf20, 26624, 121, grid=grid(26624), stream=stream0)
        buf21 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf20, buf21, 2048, 13, grid=grid(2048), stream=stream0)
        buf26 = reinterpret_tensor(buf11, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf11  # reuse
        buf27 = empty((8, 1, 196, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__44], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_11.run(buf26, buf18, primals_102, mul_260, div_72, bernoulli_44, buf27, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_44
        del div_72
        del primals_102
        buf24 = empty((512, ), device='cuda', dtype=torch.float32)
        buf25 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf18, mul_260, buf24, buf25, 512, 1568, grid=grid(512), stream=stream0)
        del mul_260
        buf28 = buf18; del buf18  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (1568, 512), (512, 1), 0), permute_203, out=buf28)
        del permute_203
        buf29 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (512, 1568), (1, 512), 0), view_388, out=buf29)
        del view_388
        buf30 = buf15; del buf15  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf27, buf30, 6656, 121, grid=grid(6656), stream=stream0)
        buf31 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf30, buf31, 512, 13, grid=grid(512), stream=stream0)
        buf32 = reinterpret_tensor(buf27, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf28, buf32, 128, 6272, grid=grid(128, 6272), stream=stream0)
        buf33 = reinterpret_tensor(buf28, (128, 196, 32), (6272, 32, 1), 0); del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_208, reinterpret_tensor(buf32, (128, 196, 32), (6272, 32, 1), 0), out=buf33)
        del permute_208
        buf34 = empty((128, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf32, (128, 196, 32), (6272, 32, 1), 0), permute_209, out=buf34)
        del permute_209
        buf36 = empty((8, 16, 1, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_14.run(buf34, alias_24, buf36, 25088, 196, grid=grid(25088), stream=stream0)
        del alias_24
        buf37 = reinterpret_tensor(buf32, (128, 32, 196), (6272, 196, 1), 0); del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_210, reinterpret_tensor(buf36, (128, 196, 196), (38416, 196, 1), 0), out=buf37)
        del permute_210
        buf38 = empty((128, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf36, (128, 196, 196), (38416, 196, 1), 0), permute_211, out=buf38)
        del permute_211
        buf39 = empty((8, 1, 196, 3, 16, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf38, buf37, buf33, buf39, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf40 = reinterpret_tensor(buf38, (1568, 512), (512, 1), 0); del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (1568, 1536), (1536, 1), 0), permute_214, out=buf40)
        del permute_214
        buf41 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (1536, 1568), (1, 1536), 0), view_378, out=buf41)
        del view_378
        buf42 = empty_strided((1, 1536, 13), (19968, 1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf39, buf42, 19968, 121, grid=grid(19968), stream=stream0)
        buf43 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf42, buf43, 1536, 13, grid=grid(1536), stream=stream0)
        buf48 = buf26; del buf26  # reuse
        buf49 = reinterpret_tensor(buf37, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf37  # reuse
        # Source Nodes: [div__43], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_18.run(buf48, buf40, primals_100, mul_255, div_73, bernoulli_43, buf49, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_43
        del div_73
        del primals_100
        buf46 = empty((512, ), device='cuda', dtype=torch.float32)
        buf47 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf40, mul_255, buf46, buf47, 512, 1568, grid=grid(512), stream=stream0)
        del mul_255
        buf50 = reinterpret_tensor(buf17, (1568, 2048), (2048, 1), 0); del buf17  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf49, (1568, 512), (512, 1), 0), permute_218, out=buf50)
        del permute_218
        buf51 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf49, (512, 1568), (1, 512), 0), view_376, out=buf51)
        del view_376
        buf52 = buf30; del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf49, buf52, 6656, 121, grid=grid(6656), stream=stream0)
        buf53 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf52, buf53, 512, 13, grid=grid(512), stream=stream0)
        buf54 = reinterpret_tensor(buf50, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf50  # reuse
        # Source Nodes: [x_357], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_8.run(buf54, addmm_90, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_90
        buf55 = reinterpret_tensor(buf49, (1568, 512), (512, 1), 0); del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf54, (1568, 2048), (2048, 1), 0), permute_222, out=buf55)
        del permute_222
        buf56 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf54, (2048, 1568), (1, 2048), 0), view_374, out=buf56)
        del view_374
        buf57 = buf20; del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf54, buf57, 26624, 121, grid=grid(26624), stream=stream0)
        buf58 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf57, buf58, 2048, 13, grid=grid(2048), stream=stream0)
        buf63 = buf48; del buf48  # reuse
        buf64 = reinterpret_tensor(buf40, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf40  # reuse
        # Source Nodes: [div__42], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_18.run(buf63, buf55, primals_98, mul_249, div_74, bernoulli_42, buf64, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_42
        del div_74
        del primals_98
        buf61 = empty((512, ), device='cuda', dtype=torch.float32)
        buf62 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf55, mul_249, buf61, buf62, 512, 1568, grid=grid(512), stream=stream0)
        del mul_249
        buf65 = buf55; del buf55  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (1568, 512), (512, 1), 0), permute_226, out=buf65)
        del permute_226
        buf66 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (512, 1568), (1, 512), 0), view_372, out=buf66)
        del view_372
        buf67 = buf52; del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf64, buf67, 6656, 121, grid=grid(6656), stream=stream0)
        buf68 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf67, buf68, 512, 13, grid=grid(512), stream=stream0)
        buf69 = reinterpret_tensor(buf64, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf65, buf69, 128, 6272, grid=grid(128, 6272), stream=stream0)
        buf70 = reinterpret_tensor(buf65, (128, 196, 32), (6272, 32, 1), 0); del buf65  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_231, reinterpret_tensor(buf69, (128, 196, 32), (6272, 32, 1), 0), out=buf70)
        del permute_231
        buf71 = reinterpret_tensor(buf36, (128, 196, 196), (38416, 196, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf69, (128, 196, 32), (6272, 32, 1), 0), permute_232, out=buf71)
        del permute_232
        buf73 = reinterpret_tensor(buf34, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_14.run(buf71, alias_25, buf73, 25088, 196, grid=grid(25088), stream=stream0)
        del alias_25
        buf74 = reinterpret_tensor(buf69, (128, 32, 196), (6272, 196, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_233, reinterpret_tensor(buf73, (128, 196, 196), (38416, 196, 1), 0), out=buf74)
        del permute_233
        buf75 = buf33; del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf73, (128, 196, 196), (38416, 196, 1), 0), permute_234, out=buf75)
        del permute_234
        buf76 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf75, buf74, buf70, buf76, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf77 = reinterpret_tensor(buf75, (1568, 512), (512, 1), 0); del buf75  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf76, (1568, 1536), (1536, 1), 0), permute_237, out=buf77)
        del permute_237
        buf78 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf76, (1536, 1568), (1, 1536), 0), view_362, out=buf78)
        del view_362
        buf79 = buf42; del buf42  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf76, buf79, 19968, 121, grid=grid(19968), stream=stream0)
        buf80 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf79, buf80, 1536, 13, grid=grid(1536), stream=stream0)
        buf85 = buf63; del buf63  # reuse
        buf86 = reinterpret_tensor(buf74, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf74  # reuse
        # Source Nodes: [div__41], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_19.run(buf85, buf77, primals_96, mul_244, div_75, bernoulli_41, buf86, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_41
        del div_75
        del primals_96
        buf83 = empty((512, ), device='cuda', dtype=torch.float32)
        buf84 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf77, mul_244, buf83, buf84, 512, 1568, grid=grid(512), stream=stream0)
        del mul_244
        buf87 = reinterpret_tensor(buf54, (1568, 2048), (2048, 1), 0); del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf86, (1568, 512), (512, 1), 0), permute_241, out=buf87)
        del permute_241
        buf88 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf86, (512, 1568), (1, 512), 0), view_360, out=buf88)
        del view_360
        buf89 = buf67; del buf67  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf86, buf89, 6656, 121, grid=grid(6656), stream=stream0)
        buf90 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf89, buf90, 512, 13, grid=grid(512), stream=stream0)
        buf91 = reinterpret_tensor(buf87, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf87  # reuse
        # Source Nodes: [x_343], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_8.run(buf91, addmm_86, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_86
        buf92 = reinterpret_tensor(buf86, (1568, 512), (512, 1), 0); del buf86  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf91, (1568, 2048), (2048, 1), 0), permute_245, out=buf92)
        del permute_245
        buf93 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf91, (2048, 1568), (1, 2048), 0), view_358, out=buf93)
        del view_358
        buf94 = buf57; del buf57  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf91, buf94, 26624, 121, grid=grid(26624), stream=stream0)
        buf95 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf94, buf95, 2048, 13, grid=grid(2048), stream=stream0)
        buf100 = buf85; del buf85  # reuse
        buf101 = reinterpret_tensor(buf77, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf77  # reuse
        # Source Nodes: [div__40], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_19.run(buf100, buf92, primals_94, mul_238, div_76, bernoulli_40, buf101, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_40
        del div_76
        del primals_94
        buf98 = empty((512, ), device='cuda', dtype=torch.float32)
        buf99 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf92, mul_238, buf98, buf99, 512, 1568, grid=grid(512), stream=stream0)
        del mul_238
        buf102 = buf92; del buf92  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf101, (1568, 512), (512, 1), 0), permute_249, out=buf102)
        del permute_249
        buf103 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf101, (512, 1568), (1, 512), 0), view_356, out=buf103)
        del view_356
        buf104 = buf89; del buf89  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf101, buf104, 6656, 121, grid=grid(6656), stream=stream0)
        buf105 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf104, buf105, 512, 13, grid=grid(512), stream=stream0)
        buf106 = reinterpret_tensor(buf101, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf101  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf102, buf106, 128, 6272, grid=grid(128, 6272), stream=stream0)
        buf107 = reinterpret_tensor(buf102, (128, 196, 32), (6272, 32, 1), 0); del buf102  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_254, reinterpret_tensor(buf106, (128, 196, 32), (6272, 32, 1), 0), out=buf107)
        del permute_254
        buf108 = reinterpret_tensor(buf73, (128, 196, 196), (38416, 196, 1), 0); del buf73  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf106, (128, 196, 32), (6272, 32, 1), 0), permute_255, out=buf108)
        del permute_255
        buf110 = reinterpret_tensor(buf71, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_14.run(buf108, alias_26, buf110, 25088, 196, grid=grid(25088), stream=stream0)
        del alias_26
        buf111 = reinterpret_tensor(buf106, (128, 32, 196), (6272, 196, 1), 0); del buf106  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_256, reinterpret_tensor(buf110, (128, 196, 196), (38416, 196, 1), 0), out=buf111)
        del permute_256
        buf112 = buf70; del buf70  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf110, (128, 196, 196), (38416, 196, 1), 0), permute_257, out=buf112)
        del permute_257
        buf113 = buf76; del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf112, buf111, buf107, buf113, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf114 = reinterpret_tensor(buf112, (1568, 512), (512, 1), 0); del buf112  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf113, (1568, 1536), (1536, 1), 0), permute_260, out=buf114)
        del permute_260
        buf115 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf113, (1536, 1568), (1, 1536), 0), view_346, out=buf115)
        del view_346
        buf116 = buf79; del buf79  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf113, buf116, 19968, 121, grid=grid(19968), stream=stream0)
        buf117 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf116, buf117, 1536, 13, grid=grid(1536), stream=stream0)
        buf122 = buf100; del buf100  # reuse
        buf123 = reinterpret_tensor(buf111, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf111  # reuse
        # Source Nodes: [div__39], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_20.run(buf122, buf114, primals_92, mul_233, div_77, bernoulli_39, buf123, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_39
        del div_77
        del primals_92
        buf120 = empty((512, ), device='cuda', dtype=torch.float32)
        buf121 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf114, mul_233, buf120, buf121, 512, 1568, grid=grid(512), stream=stream0)
        del mul_233
        buf124 = reinterpret_tensor(buf91, (1568, 2048), (2048, 1), 0); del buf91  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf123, (1568, 512), (512, 1), 0), permute_264, out=buf124)
        del permute_264
        buf125 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf123, (512, 1568), (1, 512), 0), view_344, out=buf125)
        del view_344
        buf126 = buf104; del buf104  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf123, buf126, 6656, 121, grid=grid(6656), stream=stream0)
        buf127 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf126, buf127, 512, 13, grid=grid(512), stream=stream0)
        buf128 = reinterpret_tensor(buf124, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf124  # reuse
        # Source Nodes: [x_329], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_8.run(buf128, addmm_82, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_82
        buf129 = reinterpret_tensor(buf123, (1568, 512), (512, 1), 0); del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf128, (1568, 2048), (2048, 1), 0), permute_268, out=buf129)
        del permute_268
        buf130 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf128, (2048, 1568), (1, 2048), 0), view_342, out=buf130)
        del view_342
        buf131 = buf94; del buf94  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf128, buf131, 26624, 121, grid=grid(26624), stream=stream0)
        buf132 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf131, buf132, 2048, 13, grid=grid(2048), stream=stream0)
        buf137 = buf122; del buf122  # reuse
        buf138 = reinterpret_tensor(buf114, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf114  # reuse
        # Source Nodes: [div__38], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_20.run(buf137, buf129, primals_90, mul_227, div_78, bernoulli_38, buf138, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_38
        del div_78
        del primals_90
        buf135 = empty((512, ), device='cuda', dtype=torch.float32)
        buf136 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf129, mul_227, buf135, buf136, 512, 1568, grid=grid(512), stream=stream0)
        del mul_227
        buf139 = buf129; del buf129  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf138, (1568, 512), (512, 1), 0), permute_272, out=buf139)
        del permute_272
        buf140 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf138, (512, 1568), (1, 512), 0), view_340, out=buf140)
        del view_340
        buf141 = buf126; del buf126  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf138, buf141, 6656, 121, grid=grid(6656), stream=stream0)
        buf142 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf141, buf142, 512, 13, grid=grid(512), stream=stream0)
        buf143 = reinterpret_tensor(buf138, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf139, buf143, 128, 6272, grid=grid(128, 6272), stream=stream0)
        buf144 = reinterpret_tensor(buf139, (128, 196, 32), (6272, 32, 1), 0); del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_277, reinterpret_tensor(buf143, (128, 196, 32), (6272, 32, 1), 0), out=buf144)
        del permute_277
        buf145 = reinterpret_tensor(buf110, (128, 196, 196), (38416, 196, 1), 0); del buf110  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf143, (128, 196, 32), (6272, 32, 1), 0), permute_278, out=buf145)
        del permute_278
        buf147 = reinterpret_tensor(buf108, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf108  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_14.run(buf145, alias_27, buf147, 25088, 196, grid=grid(25088), stream=stream0)
        del alias_27
        buf148 = reinterpret_tensor(buf143, (128, 32, 196), (6272, 196, 1), 0); del buf143  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_279, reinterpret_tensor(buf147, (128, 196, 196), (38416, 196, 1), 0), out=buf148)
        del permute_279
        buf149 = buf107; del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf147, (128, 196, 196), (38416, 196, 1), 0), permute_280, out=buf149)
        del permute_280
        buf150 = buf113; del buf113  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf149, buf148, buf144, buf150, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf151 = reinterpret_tensor(buf149, (1568, 512), (512, 1), 0); del buf149  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf150, (1568, 1536), (1536, 1), 0), permute_283, out=buf151)
        del permute_283
        buf152 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf150, (1536, 1568), (1, 1536), 0), view_330, out=buf152)
        del view_330
        buf153 = buf116; del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf150, buf153, 19968, 121, grid=grid(19968), stream=stream0)
        buf154 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf153, buf154, 1536, 13, grid=grid(1536), stream=stream0)
        buf159 = buf137; del buf137  # reuse
        buf160 = reinterpret_tensor(buf148, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf148  # reuse
        # Source Nodes: [div__37], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_21.run(buf159, buf151, primals_88, mul_222, div_79, bernoulli_37, buf160, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_37
        del div_79
        del primals_88
        buf157 = empty((512, ), device='cuda', dtype=torch.float32)
        buf158 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf151, mul_222, buf157, buf158, 512, 1568, grid=grid(512), stream=stream0)
        del mul_222
        buf161 = reinterpret_tensor(buf128, (1568, 2048), (2048, 1), 0); del buf128  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf160, (1568, 512), (512, 1), 0), permute_287, out=buf161)
        del permute_287
        buf162 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf160, (512, 1568), (1, 512), 0), view_328, out=buf162)
        del view_328
        buf163 = buf141; del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf160, buf163, 6656, 121, grid=grid(6656), stream=stream0)
        buf164 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf163, buf164, 512, 13, grid=grid(512), stream=stream0)
        buf165 = reinterpret_tensor(buf161, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf161  # reuse
        # Source Nodes: [x_315], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_8.run(buf165, addmm_78, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_78
        buf166 = reinterpret_tensor(buf160, (1568, 512), (512, 1), 0); del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (1568, 2048), (2048, 1), 0), permute_291, out=buf166)
        del permute_291
        buf167 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf165, (2048, 1568), (1, 2048), 0), view_326, out=buf167)
        del view_326
        buf168 = buf131; del buf131  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf165, buf168, 26624, 121, grid=grid(26624), stream=stream0)
        buf169 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf168, buf169, 2048, 13, grid=grid(2048), stream=stream0)
        buf174 = buf159; del buf159  # reuse
        buf175 = reinterpret_tensor(buf151, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf151  # reuse
        # Source Nodes: [div__36], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_21.run(buf174, buf166, primals_86, mul_216, div_80, bernoulli_36, buf175, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_36
        del div_80
        del primals_86
        buf172 = empty((512, ), device='cuda', dtype=torch.float32)
        buf173 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf166, mul_216, buf172, buf173, 512, 1568, grid=grid(512), stream=stream0)
        del mul_216
        buf176 = buf166; del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf175, (1568, 512), (512, 1), 0), permute_295, out=buf176)
        del permute_295
        buf177 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf175, (512, 1568), (1, 512), 0), view_324, out=buf177)
        del view_324
        buf178 = buf163; del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf175, buf178, 6656, 121, grid=grid(6656), stream=stream0)
        buf179 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf178, buf179, 512, 13, grid=grid(512), stream=stream0)
        buf180 = reinterpret_tensor(buf175, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf175  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf176, buf180, 128, 6272, grid=grid(128, 6272), stream=stream0)
        buf181 = reinterpret_tensor(buf176, (128, 196, 32), (6272, 32, 1), 0); del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_300, reinterpret_tensor(buf180, (128, 196, 32), (6272, 32, 1), 0), out=buf181)
        del permute_300
        buf182 = reinterpret_tensor(buf147, (128, 196, 196), (38416, 196, 1), 0); del buf147  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf180, (128, 196, 32), (6272, 32, 1), 0), permute_301, out=buf182)
        del permute_301
        buf184 = reinterpret_tensor(buf145, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf145  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_14.run(buf182, alias_28, buf184, 25088, 196, grid=grid(25088), stream=stream0)
        del alias_28
        buf185 = reinterpret_tensor(buf180, (128, 32, 196), (6272, 196, 1), 0); del buf180  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_302, reinterpret_tensor(buf184, (128, 196, 196), (38416, 196, 1), 0), out=buf185)
        del permute_302
        buf186 = buf144; del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf184, (128, 196, 196), (38416, 196, 1), 0), permute_303, out=buf186)
        del permute_303
        buf187 = buf150; del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf186, buf185, buf181, buf187, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf188 = reinterpret_tensor(buf186, (1568, 512), (512, 1), 0); del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf187, (1568, 1536), (1536, 1), 0), permute_306, out=buf188)
        del permute_306
        buf189 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf187, (1536, 1568), (1, 1536), 0), view_314, out=buf189)
        del view_314
        buf190 = buf153; del buf153  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf187, buf190, 19968, 121, grid=grid(19968), stream=stream0)
        buf191 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf190, buf191, 1536, 13, grid=grid(1536), stream=stream0)
        buf196 = buf174; del buf174  # reuse
        buf197 = reinterpret_tensor(buf185, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf185  # reuse
        # Source Nodes: [div__35], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_22.run(buf196, buf188, primals_84, mul_211, div_81, bernoulli_35, buf197, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_35
        del div_81
        del primals_84
        buf194 = empty((512, ), device='cuda', dtype=torch.float32)
        buf195 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf188, mul_211, buf194, buf195, 512, 1568, grid=grid(512), stream=stream0)
        del mul_211
        buf198 = reinterpret_tensor(buf165, (1568, 2048), (2048, 1), 0); del buf165  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf197, (1568, 512), (512, 1), 0), permute_310, out=buf198)
        del permute_310
        buf199 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf197, (512, 1568), (1, 512), 0), view_312, out=buf199)
        del view_312
        buf200 = buf178; del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf197, buf200, 6656, 121, grid=grid(6656), stream=stream0)
        buf201 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf200, buf201, 512, 13, grid=grid(512), stream=stream0)
        buf202 = reinterpret_tensor(buf198, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf198  # reuse
        # Source Nodes: [x_301], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_8.run(buf202, addmm_74, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_74
        buf203 = reinterpret_tensor(buf197, (1568, 512), (512, 1), 0); del buf197  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf202, (1568, 2048), (2048, 1), 0), permute_314, out=buf203)
        del permute_314
        buf204 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf202, (2048, 1568), (1, 2048), 0), view_310, out=buf204)
        del view_310
        buf205 = buf168; del buf168  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf202, buf205, 26624, 121, grid=grid(26624), stream=stream0)
        buf206 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf205, buf206, 2048, 13, grid=grid(2048), stream=stream0)
        buf211 = buf196; del buf196  # reuse
        buf212 = reinterpret_tensor(buf188, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf188  # reuse
        # Source Nodes: [div__34], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_22.run(buf211, buf203, primals_82, mul_205, div_82, bernoulli_34, buf212, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_34
        del div_82
        del primals_82
        buf209 = empty((512, ), device='cuda', dtype=torch.float32)
        buf210 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf203, mul_205, buf209, buf210, 512, 1568, grid=grid(512), stream=stream0)
        del mul_205
        buf213 = buf203; del buf203  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf212, (1568, 512), (512, 1), 0), permute_318, out=buf213)
        del permute_318
        buf214 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf212, (512, 1568), (1, 512), 0), view_308, out=buf214)
        del view_308
        buf215 = buf200; del buf200  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf212, buf215, 6656, 121, grid=grid(6656), stream=stream0)
        buf216 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf215, buf216, 512, 13, grid=grid(512), stream=stream0)
        buf217 = reinterpret_tensor(buf212, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf212  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf213, buf217, 128, 6272, grid=grid(128, 6272), stream=stream0)
        buf218 = reinterpret_tensor(buf213, (128, 196, 32), (6272, 32, 1), 0); del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_323, reinterpret_tensor(buf217, (128, 196, 32), (6272, 32, 1), 0), out=buf218)
        del permute_323
        buf219 = reinterpret_tensor(buf184, (128, 196, 196), (38416, 196, 1), 0); del buf184  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf217, (128, 196, 32), (6272, 32, 1), 0), permute_324, out=buf219)
        del permute_324
        buf221 = reinterpret_tensor(buf182, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf182  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_14.run(buf219, alias_29, buf221, 25088, 196, grid=grid(25088), stream=stream0)
        del alias_29
        buf222 = reinterpret_tensor(buf217, (128, 32, 196), (6272, 196, 1), 0); del buf217  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_325, reinterpret_tensor(buf221, (128, 196, 196), (38416, 196, 1), 0), out=buf222)
        del permute_325
        buf223 = buf181; del buf181  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf221, (128, 196, 196), (38416, 196, 1), 0), permute_326, out=buf223)
        del permute_326
        buf224 = buf187; del buf187  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf223, buf222, buf218, buf224, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf225 = reinterpret_tensor(buf223, (1568, 512), (512, 1), 0); del buf223  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf224, (1568, 1536), (1536, 1), 0), permute_329, out=buf225)
        del permute_329
        buf226 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf224, (1536, 1568), (1, 1536), 0), view_298, out=buf226)
        del view_298
        buf227 = buf190; del buf190  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf224, buf227, 19968, 121, grid=grid(19968), stream=stream0)
        buf228 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf227, buf228, 1536, 13, grid=grid(1536), stream=stream0)
        buf233 = buf211; del buf211  # reuse
        buf234 = reinterpret_tensor(buf222, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf222  # reuse
        # Source Nodes: [div__33], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_23.run(buf233, buf225, primals_80, mul_200, div_83, bernoulli_33, buf234, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_33
        del div_83
        del primals_80
        buf231 = empty((512, ), device='cuda', dtype=torch.float32)
        buf232 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf225, mul_200, buf231, buf232, 512, 1568, grid=grid(512), stream=stream0)
        del mul_200
        buf235 = reinterpret_tensor(buf202, (1568, 2048), (2048, 1), 0); del buf202  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf234, (1568, 512), (512, 1), 0), permute_333, out=buf235)
        del permute_333
        buf236 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf234, (512, 1568), (1, 512), 0), view_296, out=buf236)
        del view_296
        buf237 = buf215; del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf234, buf237, 6656, 121, grid=grid(6656), stream=stream0)
        buf238 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf237, buf238, 512, 13, grid=grid(512), stream=stream0)
        buf239 = reinterpret_tensor(buf235, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf235  # reuse
        # Source Nodes: [x_287], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_8.run(buf239, addmm_70, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_70
        buf240 = reinterpret_tensor(buf234, (1568, 512), (512, 1), 0); del buf234  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf239, (1568, 2048), (2048, 1), 0), permute_337, out=buf240)
        del permute_337
        buf241 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf239, (2048, 1568), (1, 2048), 0), view_294, out=buf241)
        del view_294
        buf242 = buf205; del buf205  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf239, buf242, 26624, 121, grid=grid(26624), stream=stream0)
        buf243 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf242, buf243, 2048, 13, grid=grid(2048), stream=stream0)
        buf248 = buf233; del buf233  # reuse
        buf249 = reinterpret_tensor(buf225, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf225  # reuse
        # Source Nodes: [div__32], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_23.run(buf248, buf240, primals_78, mul_194, div_84, bernoulli_32, buf249, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_32
        del div_84
        del primals_78
        buf246 = empty((512, ), device='cuda', dtype=torch.float32)
        buf247 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf240, mul_194, buf246, buf247, 512, 1568, grid=grid(512), stream=stream0)
        del mul_194
        buf250 = buf240; del buf240  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf249, (1568, 512), (512, 1), 0), permute_341, out=buf250)
        del permute_341
        buf251 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf249, (512, 1568), (1, 512), 0), view_292, out=buf251)
        del view_292
        buf252 = buf237; del buf237  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf249, buf252, 6656, 121, grid=grid(6656), stream=stream0)
        buf253 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf252, buf253, 512, 13, grid=grid(512), stream=stream0)
        buf254 = reinterpret_tensor(buf249, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf249  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf250, buf254, 128, 6272, grid=grid(128, 6272), stream=stream0)
        buf255 = reinterpret_tensor(buf250, (128, 196, 32), (6272, 32, 1), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_346, reinterpret_tensor(buf254, (128, 196, 32), (6272, 32, 1), 0), out=buf255)
        del permute_346
        buf256 = reinterpret_tensor(buf221, (128, 196, 196), (38416, 196, 1), 0); del buf221  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf254, (128, 196, 32), (6272, 32, 1), 0), permute_347, out=buf256)
        del permute_347
        buf258 = reinterpret_tensor(buf219, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf219  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_14.run(buf256, alias_30, buf258, 25088, 196, grid=grid(25088), stream=stream0)
        del alias_30
        buf259 = reinterpret_tensor(buf254, (128, 32, 196), (6272, 196, 1), 0); del buf254  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_348, reinterpret_tensor(buf258, (128, 196, 196), (38416, 196, 1), 0), out=buf259)
        del permute_348
        buf260 = buf218; del buf218  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf258, (128, 196, 196), (38416, 196, 1), 0), permute_349, out=buf260)
        del permute_349
        buf261 = buf224; del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf260, buf259, buf255, buf261, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf262 = reinterpret_tensor(buf260, (1568, 512), (512, 1), 0); del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf261, (1568, 1536), (1536, 1), 0), permute_352, out=buf262)
        del permute_352
        buf263 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf261, (1536, 1568), (1, 1536), 0), view_282, out=buf263)
        del view_282
        buf264 = buf227; del buf227  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf261, buf264, 19968, 121, grid=grid(19968), stream=stream0)
        buf265 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf264, buf265, 1536, 13, grid=grid(1536), stream=stream0)
        buf270 = buf248; del buf248  # reuse
        buf271 = reinterpret_tensor(buf259, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf259  # reuse
        # Source Nodes: [div__31], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_24.run(buf270, buf262, primals_76, mul_189, div_85, bernoulli_31, buf271, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_31
        del div_85
        del primals_76
        buf268 = empty((512, ), device='cuda', dtype=torch.float32)
        buf269 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf262, mul_189, buf268, buf269, 512, 1568, grid=grid(512), stream=stream0)
        del mul_189
        buf272 = reinterpret_tensor(buf239, (1568, 2048), (2048, 1), 0); del buf239  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf271, (1568, 512), (512, 1), 0), permute_356, out=buf272)
        del permute_356
        buf273 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf271, (512, 1568), (1, 512), 0), view_280, out=buf273)
        del view_280
        buf274 = buf252; del buf252  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf271, buf274, 6656, 121, grid=grid(6656), stream=stream0)
        buf275 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf274, buf275, 512, 13, grid=grid(512), stream=stream0)
        buf276 = reinterpret_tensor(buf272, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf272  # reuse
        # Source Nodes: [x_273], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_8.run(buf276, addmm_66, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_66
        buf277 = reinterpret_tensor(buf271, (1568, 512), (512, 1), 0); del buf271  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf276, (1568, 2048), (2048, 1), 0), permute_360, out=buf277)
        del permute_360
        buf278 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf276, (2048, 1568), (1, 2048), 0), view_278, out=buf278)
        del view_278
        buf279 = buf242; del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf276, buf279, 26624, 121, grid=grid(26624), stream=stream0)
        buf280 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf279, buf280, 2048, 13, grid=grid(2048), stream=stream0)
        buf285 = buf270; del buf270  # reuse
        buf286 = reinterpret_tensor(buf262, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf262  # reuse
        # Source Nodes: [div__30], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_24.run(buf285, buf277, primals_74, mul_183, div_86, bernoulli_30, buf286, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_30
        del div_86
        del primals_74
        buf283 = empty((512, ), device='cuda', dtype=torch.float32)
        buf284 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf277, mul_183, buf283, buf284, 512, 1568, grid=grid(512), stream=stream0)
        del mul_183
        buf287 = buf277; del buf277  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf286, (1568, 512), (512, 1), 0), permute_364, out=buf287)
        del permute_364
        buf288 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf286, (512, 1568), (1, 512), 0), view_276, out=buf288)
        del view_276
        buf289 = buf274; del buf274  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf286, buf289, 6656, 121, grid=grid(6656), stream=stream0)
        buf290 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf289, buf290, 512, 13, grid=grid(512), stream=stream0)
        buf291 = reinterpret_tensor(buf286, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf286  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf287, buf291, 128, 6272, grid=grid(128, 6272), stream=stream0)
        buf292 = reinterpret_tensor(buf287, (128, 196, 32), (6272, 32, 1), 0); del buf287  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_369, reinterpret_tensor(buf291, (128, 196, 32), (6272, 32, 1), 0), out=buf292)
        del permute_369
        buf293 = reinterpret_tensor(buf258, (128, 196, 196), (38416, 196, 1), 0); del buf258  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf291, (128, 196, 32), (6272, 32, 1), 0), permute_370, out=buf293)
        del permute_370
        buf295 = reinterpret_tensor(buf256, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf256  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_14.run(buf293, alias_31, buf295, 25088, 196, grid=grid(25088), stream=stream0)
        del alias_31
        buf296 = reinterpret_tensor(buf291, (128, 32, 196), (6272, 196, 1), 0); del buf291  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_371, reinterpret_tensor(buf295, (128, 196, 196), (38416, 196, 1), 0), out=buf296)
        del permute_371
        buf297 = buf255; del buf255  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf295, (128, 196, 196), (38416, 196, 1), 0), permute_372, out=buf297)
        del permute_372
        buf298 = buf261; del buf261  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf297, buf296, buf292, buf298, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf299 = reinterpret_tensor(buf297, (1568, 512), (512, 1), 0); del buf297  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf298, (1568, 1536), (1536, 1), 0), permute_375, out=buf299)
        del permute_375
        buf300 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf298, (1536, 1568), (1, 1536), 0), view_266, out=buf300)
        del view_266
        buf301 = buf264; del buf264  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf298, buf301, 19968, 121, grid=grid(19968), stream=stream0)
        buf302 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf301, buf302, 1536, 13, grid=grid(1536), stream=stream0)
        buf307 = buf285; del buf285  # reuse
        buf308 = reinterpret_tensor(buf296, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf296  # reuse
        # Source Nodes: [div__29], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_25.run(buf307, buf299, primals_72, mul_178, div_87, bernoulli_29, buf308, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_29
        del div_87
        del primals_72
        buf305 = empty((512, ), device='cuda', dtype=torch.float32)
        buf306 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf299, mul_178, buf305, buf306, 512, 1568, grid=grid(512), stream=stream0)
        del mul_178
        buf309 = reinterpret_tensor(buf276, (1568, 2048), (2048, 1), 0); del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf308, (1568, 512), (512, 1), 0), permute_379, out=buf309)
        del permute_379
        buf310 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf308, (512, 1568), (1, 512), 0), view_264, out=buf310)
        del view_264
        buf311 = buf289; del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf308, buf311, 6656, 121, grid=grid(6656), stream=stream0)
        buf312 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf311, buf312, 512, 13, grid=grid(512), stream=stream0)
        buf313 = reinterpret_tensor(buf309, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf309  # reuse
        # Source Nodes: [x_259], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_8.run(buf313, addmm_62, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_62
        buf314 = reinterpret_tensor(buf308, (1568, 512), (512, 1), 0); del buf308  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf313, (1568, 2048), (2048, 1), 0), permute_383, out=buf314)
        del permute_383
        buf315 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf313, (2048, 1568), (1, 2048), 0), view_262, out=buf315)
        del view_262
        buf316 = buf279; del buf279  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf313, buf316, 26624, 121, grid=grid(26624), stream=stream0)
        buf317 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf316, buf317, 2048, 13, grid=grid(2048), stream=stream0)
        buf322 = buf307; del buf307  # reuse
        buf323 = reinterpret_tensor(buf299, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf299  # reuse
        # Source Nodes: [div__28], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_25.run(buf322, buf314, primals_70, mul_172, div_88, bernoulli_28, buf323, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_28
        del div_88
        del primals_70
        buf320 = empty((512, ), device='cuda', dtype=torch.float32)
        buf321 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf314, mul_172, buf320, buf321, 512, 1568, grid=grid(512), stream=stream0)
        del mul_172
        buf324 = buf314; del buf314  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf323, (1568, 512), (512, 1), 0), permute_387, out=buf324)
        del permute_387
        buf325 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf323, (512, 1568), (1, 512), 0), view_260, out=buf325)
        del view_260
        buf326 = buf311; del buf311  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf323, buf326, 6656, 121, grid=grid(6656), stream=stream0)
        buf327 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf326, buf327, 512, 13, grid=grid(512), stream=stream0)
        buf328 = reinterpret_tensor(buf323, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf323  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf324, buf328, 128, 6272, grid=grid(128, 6272), stream=stream0)
        buf329 = reinterpret_tensor(buf324, (128, 196, 32), (6272, 32, 1), 0); del buf324  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_392, reinterpret_tensor(buf328, (128, 196, 32), (6272, 32, 1), 0), out=buf329)
        del permute_392
        buf330 = reinterpret_tensor(buf295, (128, 196, 196), (38416, 196, 1), 0); del buf295  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf328, (128, 196, 32), (6272, 32, 1), 0), permute_393, out=buf330)
        del permute_393
        buf332 = reinterpret_tensor(buf293, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf293  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_14.run(buf330, alias_32, buf332, 25088, 196, grid=grid(25088), stream=stream0)
        del alias_32
        buf333 = reinterpret_tensor(buf328, (128, 32, 196), (6272, 196, 1), 0); del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_394, reinterpret_tensor(buf332, (128, 196, 196), (38416, 196, 1), 0), out=buf333)
        del permute_394
        buf334 = buf292; del buf292  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf332, (128, 196, 196), (38416, 196, 1), 0), permute_395, out=buf334)
        del permute_395
        buf335 = buf298; del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf334, buf333, buf329, buf335, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf336 = reinterpret_tensor(buf334, (1568, 512), (512, 1), 0); del buf334  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf335, (1568, 1536), (1536, 1), 0), permute_398, out=buf336)
        del permute_398
        buf337 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf335, (1536, 1568), (1, 1536), 0), view_250, out=buf337)
        del view_250
        buf338 = buf301; del buf301  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf335, buf338, 19968, 121, grid=grid(19968), stream=stream0)
        buf339 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf338, buf339, 1536, 13, grid=grid(1536), stream=stream0)
        buf344 = buf322; del buf322  # reuse
        buf345 = reinterpret_tensor(buf333, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf333  # reuse
        # Source Nodes: [div__27], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_26.run(buf344, buf336, primals_68, mul_167, div_89, bernoulli_27, buf345, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_27
        del div_89
        del primals_68
        buf342 = empty((512, ), device='cuda', dtype=torch.float32)
        buf343 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf336, mul_167, buf342, buf343, 512, 1568, grid=grid(512), stream=stream0)
        del mul_167
        buf346 = reinterpret_tensor(buf313, (1568, 2048), (2048, 1), 0); del buf313  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf345, (1568, 512), (512, 1), 0), permute_402, out=buf346)
        del permute_402
        buf347 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf345, (512, 1568), (1, 512), 0), view_248, out=buf347)
        del view_248
        buf348 = buf326; del buf326  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf345, buf348, 6656, 121, grid=grid(6656), stream=stream0)
        buf349 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf348, buf349, 512, 13, grid=grid(512), stream=stream0)
        buf350 = reinterpret_tensor(buf346, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf346  # reuse
        # Source Nodes: [x_245], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_8.run(buf350, addmm_58, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_58
        buf351 = reinterpret_tensor(buf345, (1568, 512), (512, 1), 0); del buf345  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf350, (1568, 2048), (2048, 1), 0), permute_406, out=buf351)
        del permute_406
        buf352 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf350, (2048, 1568), (1, 2048), 0), view_246, out=buf352)
        del view_246
        buf353 = buf316; del buf316  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf350, buf353, 26624, 121, grid=grid(26624), stream=stream0)
        buf354 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf353, buf354, 2048, 13, grid=grid(2048), stream=stream0)
        buf359 = buf344; del buf344  # reuse
        buf360 = reinterpret_tensor(buf336, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf336  # reuse
        # Source Nodes: [div__26], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_26.run(buf359, buf351, primals_66, mul_161, div_90, bernoulli_26, buf360, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_26
        del div_90
        del primals_66
        buf357 = empty((512, ), device='cuda', dtype=torch.float32)
        buf358 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf351, mul_161, buf357, buf358, 512, 1568, grid=grid(512), stream=stream0)
        del mul_161
        buf361 = buf351; del buf351  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf360, (1568, 512), (512, 1), 0), permute_410, out=buf361)
        del permute_410
        buf362 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf360, (512, 1568), (1, 512), 0), view_244, out=buf362)
        del view_244
        buf363 = buf348; del buf348  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf360, buf363, 6656, 121, grid=grid(6656), stream=stream0)
        buf364 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf363, buf364, 512, 13, grid=grid(512), stream=stream0)
        buf365 = reinterpret_tensor(buf360, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf360  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf361, buf365, 128, 6272, grid=grid(128, 6272), stream=stream0)
        buf366 = reinterpret_tensor(buf361, (128, 196, 32), (6272, 32, 1), 0); del buf361  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_415, reinterpret_tensor(buf365, (128, 196, 32), (6272, 32, 1), 0), out=buf366)
        del permute_415
        buf367 = reinterpret_tensor(buf332, (128, 196, 196), (38416, 196, 1), 0); del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf365, (128, 196, 32), (6272, 32, 1), 0), permute_416, out=buf367)
        del permute_416
        buf369 = reinterpret_tensor(buf330, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf330  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_14.run(buf367, alias_33, buf369, 25088, 196, grid=grid(25088), stream=stream0)
        del alias_33
        buf370 = reinterpret_tensor(buf365, (128, 32, 196), (6272, 196, 1), 0); del buf365  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_417, reinterpret_tensor(buf369, (128, 196, 196), (38416, 196, 1), 0), out=buf370)
        del permute_417
        buf371 = buf329; del buf329  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf369, (128, 196, 196), (38416, 196, 1), 0), permute_418, out=buf371)
        del permute_418
        buf372 = buf335; del buf335  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf371, buf370, buf366, buf372, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf373 = reinterpret_tensor(buf371, (1568, 512), (512, 1), 0); del buf371  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf372, (1568, 1536), (1536, 1), 0), permute_421, out=buf373)
        del permute_421
        buf374 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf372, (1536, 1568), (1, 1536), 0), view_234, out=buf374)
        del view_234
        buf375 = buf338; del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf372, buf375, 19968, 121, grid=grid(19968), stream=stream0)
        buf376 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf375, buf376, 1536, 13, grid=grid(1536), stream=stream0)
        buf381 = buf359; del buf359  # reuse
        buf382 = reinterpret_tensor(buf370, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf370  # reuse
        # Source Nodes: [div__25], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_27.run(buf381, buf373, primals_64, mul_156, div_91, bernoulli_25, buf382, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_25
        del div_91
        del primals_64
        buf379 = empty((512, ), device='cuda', dtype=torch.float32)
        buf380 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf373, mul_156, buf379, buf380, 512, 1568, grid=grid(512), stream=stream0)
        del mul_156
        buf383 = reinterpret_tensor(buf350, (1568, 2048), (2048, 1), 0); del buf350  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf382, (1568, 512), (512, 1), 0), permute_425, out=buf383)
        del permute_425
        buf384 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf382, (512, 1568), (1, 512), 0), view_232, out=buf384)
        del view_232
        buf385 = buf363; del buf363  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf382, buf385, 6656, 121, grid=grid(6656), stream=stream0)
        buf386 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf385, buf386, 512, 13, grid=grid(512), stream=stream0)
        buf387 = reinterpret_tensor(buf383, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf383  # reuse
        # Source Nodes: [x_231], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_8.run(buf387, addmm_54, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_54
        buf388 = reinterpret_tensor(buf382, (1568, 512), (512, 1), 0); del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf387, (1568, 2048), (2048, 1), 0), permute_429, out=buf388)
        del permute_429
        buf389 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf387, (2048, 1568), (1, 2048), 0), view_230, out=buf389)
        del view_230
        buf390 = buf353; del buf353  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf387, buf390, 26624, 121, grid=grid(26624), stream=stream0)
        buf391 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf390, buf391, 2048, 13, grid=grid(2048), stream=stream0)
        buf396 = buf381; del buf381  # reuse
        buf397 = reinterpret_tensor(buf373, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf373  # reuse
        # Source Nodes: [div__24], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_27.run(buf396, buf388, primals_62, mul_150, div_92, bernoulli_24, buf397, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_24
        del div_92
        del primals_62
        buf394 = empty((512, ), device='cuda', dtype=torch.float32)
        buf395 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf388, mul_150, buf394, buf395, 512, 1568, grid=grid(512), stream=stream0)
        del mul_150
        buf398 = buf388; del buf388  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf397, (1568, 512), (512, 1), 0), permute_433, out=buf398)
        del permute_433
        buf399 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf397, (512, 1568), (1, 512), 0), view_228, out=buf399)
        del view_228
        buf400 = buf385; del buf385  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf397, buf400, 6656, 121, grid=grid(6656), stream=stream0)
        buf401 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf400, buf401, 512, 13, grid=grid(512), stream=stream0)
        buf402 = reinterpret_tensor(buf397, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf397  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf398, buf402, 128, 6272, grid=grid(128, 6272), stream=stream0)
        buf403 = reinterpret_tensor(buf398, (128, 196, 32), (6272, 32, 1), 0); del buf398  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_438, reinterpret_tensor(buf402, (128, 196, 32), (6272, 32, 1), 0), out=buf403)
        del permute_438
        buf404 = reinterpret_tensor(buf369, (128, 196, 196), (38416, 196, 1), 0); del buf369  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf402, (128, 196, 32), (6272, 32, 1), 0), permute_439, out=buf404)
        del permute_439
        buf406 = reinterpret_tensor(buf367, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf367  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_14.run(buf404, alias_34, buf406, 25088, 196, grid=grid(25088), stream=stream0)
        del alias_34
        buf407 = reinterpret_tensor(buf402, (128, 32, 196), (6272, 196, 1), 0); del buf402  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_440, reinterpret_tensor(buf406, (128, 196, 196), (38416, 196, 1), 0), out=buf407)
        del permute_440
        buf408 = buf366; del buf366  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf406, (128, 196, 196), (38416, 196, 1), 0), permute_441, out=buf408)
        del permute_441
        buf409 = buf372; del buf372  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf408, buf407, buf403, buf409, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf410 = reinterpret_tensor(buf408, (1568, 512), (512, 1), 0); del buf408  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf409, (1568, 1536), (1536, 1), 0), permute_444, out=buf410)
        del permute_444
        buf411 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf409, (1536, 1568), (1, 1536), 0), view_218, out=buf411)
        del view_218
        buf412 = buf375; del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf409, buf412, 19968, 121, grid=grid(19968), stream=stream0)
        buf413 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf412, buf413, 1536, 13, grid=grid(1536), stream=stream0)
        buf418 = buf396; del buf396  # reuse
        buf419 = reinterpret_tensor(buf407, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf407  # reuse
        # Source Nodes: [div__23], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_28.run(buf418, buf410, primals_60, mul_145, div_93, bernoulli_23, buf419, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_23
        del div_93
        del primals_60
        buf416 = empty((512, ), device='cuda', dtype=torch.float32)
        buf417 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf410, mul_145, buf416, buf417, 512, 1568, grid=grid(512), stream=stream0)
        del mul_145
        buf420 = reinterpret_tensor(buf387, (1568, 2048), (2048, 1), 0); del buf387  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf419, (1568, 512), (512, 1), 0), permute_448, out=buf420)
        del permute_448
        buf421 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf419, (512, 1568), (1, 512), 0), view_216, out=buf421)
        del view_216
        buf422 = buf400; del buf400  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf419, buf422, 6656, 121, grid=grid(6656), stream=stream0)
        buf423 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf422, buf423, 512, 13, grid=grid(512), stream=stream0)
        buf424 = reinterpret_tensor(buf420, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf420  # reuse
        # Source Nodes: [x_217], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_8.run(buf424, addmm_50, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_50
        buf425 = reinterpret_tensor(buf419, (1568, 512), (512, 1), 0); del buf419  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf424, (1568, 2048), (2048, 1), 0), permute_452, out=buf425)
        del permute_452
        buf426 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf424, (2048, 1568), (1, 2048), 0), view_214, out=buf426)
        del view_214
        buf427 = buf390; del buf390  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf424, buf427, 26624, 121, grid=grid(26624), stream=stream0)
        buf428 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf427, buf428, 2048, 13, grid=grid(2048), stream=stream0)
        buf433 = buf418; del buf418  # reuse
        buf434 = reinterpret_tensor(buf410, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf410  # reuse
        # Source Nodes: [div__22], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_28.run(buf433, buf425, primals_58, mul_139, div_94, bernoulli_22, buf434, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_22
        del div_94
        del primals_58
        buf431 = empty((512, ), device='cuda', dtype=torch.float32)
        buf432 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf425, mul_139, buf431, buf432, 512, 1568, grid=grid(512), stream=stream0)
        del mul_139
        buf435 = buf425; del buf425  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf434, (1568, 512), (512, 1), 0), permute_456, out=buf435)
        del permute_456
        buf436 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf434, (512, 1568), (1, 512), 0), view_212, out=buf436)
        del view_212
        buf437 = buf422; del buf422  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf434, buf437, 6656, 121, grid=grid(6656), stream=stream0)
        buf438 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf437, buf438, 512, 13, grid=grid(512), stream=stream0)
        buf439 = reinterpret_tensor(buf434, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf434  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf435, buf439, 128, 6272, grid=grid(128, 6272), stream=stream0)
        buf440 = reinterpret_tensor(buf435, (128, 196, 32), (6272, 32, 1), 0); del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_461, reinterpret_tensor(buf439, (128, 196, 32), (6272, 32, 1), 0), out=buf440)
        del permute_461
        buf441 = reinterpret_tensor(buf406, (128, 196, 196), (38416, 196, 1), 0); del buf406  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf439, (128, 196, 32), (6272, 32, 1), 0), permute_462, out=buf441)
        del permute_462
        buf443 = reinterpret_tensor(buf404, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf404  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_14.run(buf441, alias_35, buf443, 25088, 196, grid=grid(25088), stream=stream0)
        del alias_35
        buf444 = reinterpret_tensor(buf439, (128, 32, 196), (6272, 196, 1), 0); del buf439  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_463, reinterpret_tensor(buf443, (128, 196, 196), (38416, 196, 1), 0), out=buf444)
        del permute_463
        buf445 = buf403; del buf403  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf443, (128, 196, 196), (38416, 196, 1), 0), permute_464, out=buf445)
        del permute_464
        buf446 = buf409; del buf409  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf445, buf444, buf440, buf446, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf447 = reinterpret_tensor(buf445, (1568, 512), (512, 1), 0); del buf445  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf446, (1568, 1536), (1536, 1), 0), permute_467, out=buf447)
        del permute_467
        buf448 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf446, (1536, 1568), (1, 1536), 0), view_202, out=buf448)
        del view_202
        buf449 = buf412; del buf412  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf446, buf449, 19968, 121, grid=grid(19968), stream=stream0)
        buf450 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf449, buf450, 1536, 13, grid=grid(1536), stream=stream0)
        buf455 = buf433; del buf433  # reuse
        buf456 = reinterpret_tensor(buf444, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf444  # reuse
        # Source Nodes: [div__21], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_29.run(buf455, buf447, primals_56, mul_134, div_95, bernoulli_21, buf456, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_21
        del div_95
        del primals_56
        buf453 = empty((512, ), device='cuda', dtype=torch.float32)
        buf454 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf447, mul_134, buf453, buf454, 512, 1568, grid=grid(512), stream=stream0)
        del mul_134
        buf457 = reinterpret_tensor(buf424, (1568, 2048), (2048, 1), 0); del buf424  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf456, (1568, 512), (512, 1), 0), permute_471, out=buf457)
        del permute_471
        buf458 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf456, (512, 1568), (1, 512), 0), view_200, out=buf458)
        del view_200
        buf459 = buf437; del buf437  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf456, buf459, 6656, 121, grid=grid(6656), stream=stream0)
        buf460 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf459, buf460, 512, 13, grid=grid(512), stream=stream0)
        buf461 = reinterpret_tensor(buf457, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf457  # reuse
        # Source Nodes: [x_203], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_8.run(buf461, addmm_46, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_46
        buf462 = reinterpret_tensor(buf456, (1568, 512), (512, 1), 0); del buf456  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf461, (1568, 2048), (2048, 1), 0), permute_475, out=buf462)
        del permute_475
        buf463 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf461, (2048, 1568), (1, 2048), 0), view_198, out=buf463)
        del view_198
        buf464 = buf427; del buf427  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf461, buf464, 26624, 121, grid=grid(26624), stream=stream0)
        buf465 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf464, buf465, 2048, 13, grid=grid(2048), stream=stream0)
        buf470 = buf455; del buf455  # reuse
        buf471 = reinterpret_tensor(buf447, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf447  # reuse
        # Source Nodes: [div__20], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_29.run(buf470, buf462, primals_54, mul_128, div_96, bernoulli_20, buf471, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_20
        del div_96
        del primals_54
        buf468 = empty((512, ), device='cuda', dtype=torch.float32)
        buf469 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf462, mul_128, buf468, buf469, 512, 1568, grid=grid(512), stream=stream0)
        del mul_128
        buf472 = buf462; del buf462  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf471, (1568, 512), (512, 1), 0), permute_479, out=buf472)
        del permute_479
        buf473 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf471, (512, 1568), (1, 512), 0), view_196, out=buf473)
        del view_196
        buf474 = buf459; del buf459  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf471, buf474, 6656, 121, grid=grid(6656), stream=stream0)
        buf475 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf474, buf475, 512, 13, grid=grid(512), stream=stream0)
        buf476 = reinterpret_tensor(buf471, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf471  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf472, buf476, 128, 6272, grid=grid(128, 6272), stream=stream0)
        buf477 = reinterpret_tensor(buf472, (128, 196, 32), (6272, 32, 1), 0); del buf472  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_484, reinterpret_tensor(buf476, (128, 196, 32), (6272, 32, 1), 0), out=buf477)
        del permute_484
        buf478 = reinterpret_tensor(buf443, (128, 196, 196), (38416, 196, 1), 0); del buf443  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf476, (128, 196, 32), (6272, 32, 1), 0), permute_485, out=buf478)
        del permute_485
        buf480 = reinterpret_tensor(buf441, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf441  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_14.run(buf478, alias_36, buf480, 25088, 196, grid=grid(25088), stream=stream0)
        del alias_36
        buf481 = reinterpret_tensor(buf476, (128, 32, 196), (6272, 196, 1), 0); del buf476  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_486, reinterpret_tensor(buf480, (128, 196, 196), (38416, 196, 1), 0), out=buf481)
        del permute_486
        buf482 = buf440; del buf440  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf480, (128, 196, 196), (38416, 196, 1), 0), permute_487, out=buf482)
        del permute_487
        buf483 = buf446; del buf446  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf482, buf481, buf477, buf483, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf484 = reinterpret_tensor(buf482, (1568, 512), (512, 1), 0); del buf482  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf483, (1568, 1536), (1536, 1), 0), permute_490, out=buf484)
        del permute_490
        buf485 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf483, (1536, 1568), (1, 1536), 0), view_186, out=buf485)
        del view_186
        buf486 = buf449; del buf449  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf483, buf486, 19968, 121, grid=grid(19968), stream=stream0)
        buf487 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf486, buf487, 1536, 13, grid=grid(1536), stream=stream0)
        buf492 = buf470; del buf470  # reuse
        buf493 = reinterpret_tensor(buf481, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf481  # reuse
        # Source Nodes: [div__19], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_30.run(buf492, buf484, primals_52, mul_123, div_97, bernoulli_19, buf493, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_19
        del div_97
        del primals_52
        buf490 = empty((512, ), device='cuda', dtype=torch.float32)
        buf491 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf484, mul_123, buf490, buf491, 512, 1568, grid=grid(512), stream=stream0)
        del mul_123
        buf494 = reinterpret_tensor(buf461, (1568, 2048), (2048, 1), 0); del buf461  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf493, (1568, 512), (512, 1), 0), permute_494, out=buf494)
        del permute_494
        buf495 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf493, (512, 1568), (1, 512), 0), view_184, out=buf495)
        del view_184
        buf496 = buf474; del buf474  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf493, buf496, 6656, 121, grid=grid(6656), stream=stream0)
        buf497 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf496, buf497, 512, 13, grid=grid(512), stream=stream0)
        buf498 = reinterpret_tensor(buf494, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf494  # reuse
        # Source Nodes: [x_189], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_8.run(buf498, addmm_42, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_42
        buf499 = reinterpret_tensor(buf493, (1568, 512), (512, 1), 0); del buf493  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf498, (1568, 2048), (2048, 1), 0), permute_498, out=buf499)
        del permute_498
        buf500 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf498, (2048, 1568), (1, 2048), 0), view_182, out=buf500)
        del view_182
        buf501 = buf464; del buf464  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf498, buf501, 26624, 121, grid=grid(26624), stream=stream0)
        buf502 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf501, buf502, 2048, 13, grid=grid(2048), stream=stream0)
        buf507 = buf492; del buf492  # reuse
        buf508 = reinterpret_tensor(buf484, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf484  # reuse
        # Source Nodes: [div__18], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_30.run(buf507, buf499, primals_50, mul_117, div_98, bernoulli_18, buf508, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_18
        del div_98
        del primals_50
        buf505 = empty((512, ), device='cuda', dtype=torch.float32)
        buf506 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf499, mul_117, buf505, buf506, 512, 1568, grid=grid(512), stream=stream0)
        del mul_117
        buf509 = buf499; del buf499  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf508, (1568, 512), (512, 1), 0), permute_502, out=buf509)
        del permute_502
        buf510 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf508, (512, 1568), (1, 512), 0), view_180, out=buf510)
        del view_180
        buf511 = buf496; del buf496  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf508, buf511, 6656, 121, grid=grid(6656), stream=stream0)
        buf512 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf511, buf512, 512, 13, grid=grid(512), stream=stream0)
        buf513 = reinterpret_tensor(buf508, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf508  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf509, buf513, 128, 6272, grid=grid(128, 6272), stream=stream0)
        buf514 = reinterpret_tensor(buf509, (128, 196, 32), (6272, 32, 1), 0); del buf509  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_507, reinterpret_tensor(buf513, (128, 196, 32), (6272, 32, 1), 0), out=buf514)
        del permute_507
        buf515 = reinterpret_tensor(buf480, (128, 196, 196), (38416, 196, 1), 0); del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf513, (128, 196, 32), (6272, 32, 1), 0), permute_508, out=buf515)
        del permute_508
        buf517 = reinterpret_tensor(buf478, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf478  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_14.run(buf515, alias_37, buf517, 25088, 196, grid=grid(25088), stream=stream0)
        del alias_37
        buf518 = reinterpret_tensor(buf513, (128, 32, 196), (6272, 196, 1), 0); del buf513  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_509, reinterpret_tensor(buf517, (128, 196, 196), (38416, 196, 1), 0), out=buf518)
        del permute_509
        buf519 = buf477; del buf477  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf517, (128, 196, 196), (38416, 196, 1), 0), permute_510, out=buf519)
        del permute_510
        buf520 = buf483; del buf483  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf519, buf518, buf514, buf520, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf521 = reinterpret_tensor(buf519, (1568, 512), (512, 1), 0); del buf519  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf520, (1568, 1536), (1536, 1), 0), permute_513, out=buf521)
        del permute_513
        buf522 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf520, (1536, 1568), (1, 1536), 0), view_170, out=buf522)
        del view_170
        buf523 = buf486; del buf486  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf520, buf523, 19968, 121, grid=grid(19968), stream=stream0)
        buf524 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf523, buf524, 1536, 13, grid=grid(1536), stream=stream0)
        buf529 = buf507; del buf507  # reuse
        buf530 = reinterpret_tensor(buf518, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf518  # reuse
        # Source Nodes: [div__17], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_31.run(buf529, buf521, primals_48, mul_112, div_99, bernoulli_17, buf530, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_17
        del div_99
        del primals_48
        buf527 = empty((512, ), device='cuda', dtype=torch.float32)
        buf528 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf521, mul_112, buf527, buf528, 512, 1568, grid=grid(512), stream=stream0)
        del mul_112
        buf531 = reinterpret_tensor(buf498, (1568, 2048), (2048, 1), 0); del buf498  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf530, (1568, 512), (512, 1), 0), permute_517, out=buf531)
        del permute_517
        buf532 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf530, (512, 1568), (1, 512), 0), view_168, out=buf532)
        del view_168
        buf533 = buf511; del buf511  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf530, buf533, 6656, 121, grid=grid(6656), stream=stream0)
        buf534 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf533, buf534, 512, 13, grid=grid(512), stream=stream0)
        buf535 = reinterpret_tensor(buf531, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf531  # reuse
        # Source Nodes: [x_175], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_8.run(buf535, addmm_38, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_38
        buf536 = reinterpret_tensor(buf530, (1568, 512), (512, 1), 0); del buf530  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf535, (1568, 2048), (2048, 1), 0), permute_521, out=buf536)
        del permute_521
        buf537 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf535, (2048, 1568), (1, 2048), 0), view_166, out=buf537)
        del view_166
        buf538 = buf501; del buf501  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf535, buf538, 26624, 121, grid=grid(26624), stream=stream0)
        buf539 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf538, buf539, 2048, 13, grid=grid(2048), stream=stream0)
        buf544 = buf529; del buf529  # reuse
        buf545 = reinterpret_tensor(buf521, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf521  # reuse
        # Source Nodes: [div__16], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_31.run(buf544, buf536, primals_46, mul_106, div_100, bernoulli_16, buf545, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_16
        del div_100
        del primals_46
        buf542 = empty((512, ), device='cuda', dtype=torch.float32)
        buf543 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf536, mul_106, buf542, buf543, 512, 1568, grid=grid(512), stream=stream0)
        del mul_106
        buf546 = buf536; del buf536  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf545, (1568, 512), (512, 1), 0), permute_525, out=buf546)
        del permute_525
        buf547 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf545, (512, 1568), (1, 512), 0), view_164, out=buf547)
        del view_164
        buf548 = buf533; del buf533  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf545, buf548, 6656, 121, grid=grid(6656), stream=stream0)
        buf549 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf548, buf549, 512, 13, grid=grid(512), stream=stream0)
        buf550 = reinterpret_tensor(buf545, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf545  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf546, buf550, 128, 6272, grid=grid(128, 6272), stream=stream0)
        buf551 = reinterpret_tensor(buf546, (128, 196, 32), (6272, 32, 1), 0); del buf546  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_530, reinterpret_tensor(buf550, (128, 196, 32), (6272, 32, 1), 0), out=buf551)
        del permute_530
        buf552 = reinterpret_tensor(buf517, (128, 196, 196), (38416, 196, 1), 0); del buf517  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf550, (128, 196, 32), (6272, 32, 1), 0), permute_531, out=buf552)
        del permute_531
        buf554 = reinterpret_tensor(buf515, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf515  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_14.run(buf552, alias_38, buf554, 25088, 196, grid=grid(25088), stream=stream0)
        del alias_38
        buf555 = reinterpret_tensor(buf550, (128, 32, 196), (6272, 196, 1), 0); del buf550  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_532, reinterpret_tensor(buf554, (128, 196, 196), (38416, 196, 1), 0), out=buf555)
        del permute_532
        buf556 = buf514; del buf514  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf554, (128, 196, 196), (38416, 196, 1), 0), permute_533, out=buf556)
        del permute_533
        buf557 = buf520; del buf520  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf556, buf555, buf551, buf557, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf558 = reinterpret_tensor(buf556, (1568, 512), (512, 1), 0); del buf556  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf557, (1568, 1536), (1536, 1), 0), permute_536, out=buf558)
        del permute_536
        buf559 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf557, (1536, 1568), (1, 1536), 0), view_154, out=buf559)
        del view_154
        buf560 = buf523; del buf523  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf557, buf560, 19968, 121, grid=grid(19968), stream=stream0)
        buf561 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf560, buf561, 1536, 13, grid=grid(1536), stream=stream0)
        buf566 = buf544; del buf544  # reuse
        buf567 = reinterpret_tensor(buf555, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf555  # reuse
        # Source Nodes: [div__15], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_32.run(buf566, buf558, primals_44, mul_101, div_101, bernoulli_15, buf567, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_15
        del div_101
        del primals_44
        buf564 = empty((512, ), device='cuda', dtype=torch.float32)
        buf565 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf558, mul_101, buf564, buf565, 512, 1568, grid=grid(512), stream=stream0)
        del mul_101
        buf568 = reinterpret_tensor(buf535, (1568, 2048), (2048, 1), 0); del buf535  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf567, (1568, 512), (512, 1), 0), permute_540, out=buf568)
        del permute_540
        buf569 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf567, (512, 1568), (1, 512), 0), view_152, out=buf569)
        del view_152
        buf570 = buf548; del buf548  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf567, buf570, 6656, 121, grid=grid(6656), stream=stream0)
        buf571 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf570, buf571, 512, 13, grid=grid(512), stream=stream0)
        buf572 = reinterpret_tensor(buf568, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf568  # reuse
        # Source Nodes: [x_161], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_8.run(buf572, addmm_34, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_34
        buf573 = reinterpret_tensor(buf567, (1568, 512), (512, 1), 0); del buf567  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf572, (1568, 2048), (2048, 1), 0), permute_544, out=buf573)
        del permute_544
        buf574 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf572, (2048, 1568), (1, 2048), 0), view_150, out=buf574)
        del view_150
        buf575 = buf538; del buf538  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf572, buf575, 26624, 121, grid=grid(26624), stream=stream0)
        buf576 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf575, buf576, 2048, 13, grid=grid(2048), stream=stream0)
        buf581 = buf566; del buf566  # reuse
        buf582 = reinterpret_tensor(buf558, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf558  # reuse
        # Source Nodes: [div__14], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_32.run(buf581, buf573, primals_42, mul_95, div_102, bernoulli_14, buf582, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_14
        del div_102
        del primals_42
        buf579 = empty((512, ), device='cuda', dtype=torch.float32)
        buf580 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf573, mul_95, buf579, buf580, 512, 1568, grid=grid(512), stream=stream0)
        del mul_95
        buf583 = buf573; del buf573  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf582, (1568, 512), (512, 1), 0), permute_548, out=buf583)
        del permute_548
        buf584 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf582, (512, 1568), (1, 512), 0), view_148, out=buf584)
        del view_148
        buf585 = buf570; del buf570  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf582, buf585, 6656, 121, grid=grid(6656), stream=stream0)
        buf586 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf585, buf586, 512, 13, grid=grid(512), stream=stream0)
        buf587 = reinterpret_tensor(buf582, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf582  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf583, buf587, 128, 6272, grid=grid(128, 6272), stream=stream0)
        buf588 = reinterpret_tensor(buf583, (128, 196, 32), (6272, 32, 1), 0); del buf583  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_553, reinterpret_tensor(buf587, (128, 196, 32), (6272, 32, 1), 0), out=buf588)
        del permute_553
        buf589 = reinterpret_tensor(buf554, (128, 196, 196), (38416, 196, 1), 0); del buf554  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf587, (128, 196, 32), (6272, 32, 1), 0), permute_554, out=buf589)
        del permute_554
        buf591 = reinterpret_tensor(buf552, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf552  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_14.run(buf589, alias_39, buf591, 25088, 196, grid=grid(25088), stream=stream0)
        del alias_39
        buf592 = reinterpret_tensor(buf587, (128, 32, 196), (6272, 196, 1), 0); del buf587  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_555, reinterpret_tensor(buf591, (128, 196, 196), (38416, 196, 1), 0), out=buf592)
        del permute_555
        buf593 = buf551; del buf551  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf591, (128, 196, 196), (38416, 196, 1), 0), permute_556, out=buf593)
        del permute_556
        buf594 = buf557; del buf557  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf593, buf592, buf588, buf594, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf595 = reinterpret_tensor(buf593, (1568, 512), (512, 1), 0); del buf593  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf594, (1568, 1536), (1536, 1), 0), permute_559, out=buf595)
        del permute_559
        buf596 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf594, (1536, 1568), (1, 1536), 0), view_138, out=buf596)
        del view_138
        buf597 = buf560; del buf560  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf594, buf597, 19968, 121, grid=grid(19968), stream=stream0)
        buf598 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf597, buf598, 1536, 13, grid=grid(1536), stream=stream0)
        buf603 = buf581; del buf581  # reuse
        buf604 = reinterpret_tensor(buf592, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf592  # reuse
        # Source Nodes: [div__13], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_33.run(buf603, buf595, primals_40, mul_90, div_103, bernoulli_13, buf604, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_13
        del div_103
        del primals_40
        buf601 = empty((512, ), device='cuda', dtype=torch.float32)
        buf602 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf595, mul_90, buf601, buf602, 512, 1568, grid=grid(512), stream=stream0)
        del mul_90
        buf605 = reinterpret_tensor(buf572, (1568, 2048), (2048, 1), 0); del buf572  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf604, (1568, 512), (512, 1), 0), permute_563, out=buf605)
        del permute_563
        buf606 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf604, (512, 1568), (1, 512), 0), view_136, out=buf606)
        del view_136
        buf607 = buf585; del buf585  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf604, buf607, 6656, 121, grid=grid(6656), stream=stream0)
        buf608 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf607, buf608, 512, 13, grid=grid(512), stream=stream0)
        buf609 = reinterpret_tensor(buf605, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf605  # reuse
        # Source Nodes: [x_147], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_8.run(buf609, addmm_30, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_30
        buf610 = reinterpret_tensor(buf604, (1568, 512), (512, 1), 0); del buf604  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf609, (1568, 2048), (2048, 1), 0), permute_567, out=buf610)
        del permute_567
        buf611 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf609, (2048, 1568), (1, 2048), 0), view_134, out=buf611)
        del view_134
        buf612 = buf575; del buf575  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf609, buf612, 26624, 121, grid=grid(26624), stream=stream0)
        buf613 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf612, buf613, 2048, 13, grid=grid(2048), stream=stream0)
        buf618 = buf603; del buf603  # reuse
        buf619 = reinterpret_tensor(buf595, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf595  # reuse
        # Source Nodes: [div__12], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_33.run(buf618, buf610, primals_38, mul_84, div_104, bernoulli_12, buf619, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_12
        del div_104
        del primals_38
        buf616 = empty((512, ), device='cuda', dtype=torch.float32)
        buf617 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf610, mul_84, buf616, buf617, 512, 1568, grid=grid(512), stream=stream0)
        del mul_84
        buf620 = buf610; del buf610  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf619, (1568, 512), (512, 1), 0), permute_571, out=buf620)
        del permute_571
        buf621 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf619, (512, 1568), (1, 512), 0), view_132, out=buf621)
        del view_132
        buf622 = buf607; del buf607  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf619, buf622, 6656, 121, grid=grid(6656), stream=stream0)
        buf623 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf622, buf623, 512, 13, grid=grid(512), stream=stream0)
        buf624 = reinterpret_tensor(buf619, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf619  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf620, buf624, 128, 6272, grid=grid(128, 6272), stream=stream0)
        buf625 = reinterpret_tensor(buf620, (128, 196, 32), (6272, 32, 1), 0); del buf620  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_576, reinterpret_tensor(buf624, (128, 196, 32), (6272, 32, 1), 0), out=buf625)
        del permute_576
        buf626 = reinterpret_tensor(buf591, (128, 196, 196), (38416, 196, 1), 0); del buf591  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf624, (128, 196, 32), (6272, 32, 1), 0), permute_577, out=buf626)
        del permute_577
        buf628 = reinterpret_tensor(buf589, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf589  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_14.run(buf626, alias_40, buf628, 25088, 196, grid=grid(25088), stream=stream0)
        del alias_40
        buf629 = reinterpret_tensor(buf624, (128, 32, 196), (6272, 196, 1), 0); del buf624  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_578, reinterpret_tensor(buf628, (128, 196, 196), (38416, 196, 1), 0), out=buf629)
        del permute_578
        buf630 = buf588; del buf588  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf628, (128, 196, 196), (38416, 196, 1), 0), permute_579, out=buf630)
        del permute_579
        buf631 = buf594; del buf594  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf630, buf629, buf625, buf631, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf632 = reinterpret_tensor(buf630, (1568, 512), (512, 1), 0); del buf630  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf631, (1568, 1536), (1536, 1), 0), permute_582, out=buf632)
        del permute_582
        buf633 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf631, (1536, 1568), (1, 1536), 0), view_122, out=buf633)
        del view_122
        buf634 = buf597; del buf597  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf631, buf634, 19968, 121, grid=grid(19968), stream=stream0)
        buf635 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf634, buf635, 1536, 13, grid=grid(1536), stream=stream0)
        buf640 = buf618; del buf618  # reuse
        buf641 = reinterpret_tensor(buf629, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf629  # reuse
        # Source Nodes: [div__11], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_34.run(buf640, buf632, primals_36, mul_79, div_105, bernoulli_11, buf641, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_11
        del div_105
        del primals_36
        buf638 = empty((512, ), device='cuda', dtype=torch.float32)
        buf639 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf632, mul_79, buf638, buf639, 512, 1568, grid=grid(512), stream=stream0)
        del mul_79
        buf642 = reinterpret_tensor(buf609, (1568, 2048), (2048, 1), 0); del buf609  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf641, (1568, 512), (512, 1), 0), permute_586, out=buf642)
        del permute_586
        buf643 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf641, (512, 1568), (1, 512), 0), view_120, out=buf643)
        del view_120
        buf644 = buf622; del buf622  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf641, buf644, 6656, 121, grid=grid(6656), stream=stream0)
        buf645 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf644, buf645, 512, 13, grid=grid(512), stream=stream0)
        buf646 = reinterpret_tensor(buf642, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf642  # reuse
        # Source Nodes: [x_133], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_8.run(buf646, addmm_26, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_26
        buf647 = reinterpret_tensor(buf641, (1568, 512), (512, 1), 0); del buf641  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf646, (1568, 2048), (2048, 1), 0), permute_590, out=buf647)
        del permute_590
        buf648 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf646, (2048, 1568), (1, 2048), 0), view_118, out=buf648)
        del view_118
        buf649 = buf612; del buf612  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf646, buf649, 26624, 121, grid=grid(26624), stream=stream0)
        buf650 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf649, buf650, 2048, 13, grid=grid(2048), stream=stream0)
        buf655 = buf640; del buf640  # reuse
        buf656 = reinterpret_tensor(buf632, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf632  # reuse
        # Source Nodes: [div__10], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_34.run(buf655, buf647, primals_34, mul_73, div_106, bernoulli_10, buf656, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_10
        del div_106
        del primals_34
        buf653 = empty((512, ), device='cuda', dtype=torch.float32)
        buf654 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf647, mul_73, buf653, buf654, 512, 1568, grid=grid(512), stream=stream0)
        del mul_73
        buf657 = buf647; del buf647  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf656, (1568, 512), (512, 1), 0), permute_594, out=buf657)
        del permute_594
        buf658 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf656, (512, 1568), (1, 512), 0), view_116, out=buf658)
        del view_116
        buf659 = buf644; del buf644  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf656, buf659, 6656, 121, grid=grid(6656), stream=stream0)
        buf660 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf659, buf660, 512, 13, grid=grid(512), stream=stream0)
        buf661 = reinterpret_tensor(buf656, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf656  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf657, buf661, 128, 6272, grid=grid(128, 6272), stream=stream0)
        buf662 = reinterpret_tensor(buf657, (128, 196, 32), (6272, 32, 1), 0); del buf657  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_599, reinterpret_tensor(buf661, (128, 196, 32), (6272, 32, 1), 0), out=buf662)
        del permute_599
        buf663 = reinterpret_tensor(buf628, (128, 196, 196), (38416, 196, 1), 0); del buf628  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf661, (128, 196, 32), (6272, 32, 1), 0), permute_600, out=buf663)
        del permute_600
        buf665 = reinterpret_tensor(buf626, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf626  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_14.run(buf663, alias_41, buf665, 25088, 196, grid=grid(25088), stream=stream0)
        del alias_41
        buf666 = reinterpret_tensor(buf661, (128, 32, 196), (6272, 196, 1), 0); del buf661  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_601, reinterpret_tensor(buf665, (128, 196, 196), (38416, 196, 1), 0), out=buf666)
        del permute_601
        buf667 = buf625; del buf625  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf665, (128, 196, 196), (38416, 196, 1), 0), permute_602, out=buf667)
        del permute_602
        buf668 = buf631; del buf631  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf667, buf666, buf662, buf668, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf669 = reinterpret_tensor(buf667, (1568, 512), (512, 1), 0); del buf667  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf668, (1568, 1536), (1536, 1), 0), permute_605, out=buf669)
        del permute_605
        buf670 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf668, (1536, 1568), (1, 1536), 0), view_106, out=buf670)
        del view_106
        buf671 = buf634; del buf634  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf668, buf671, 19968, 121, grid=grid(19968), stream=stream0)
        buf672 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf671, buf672, 1536, 13, grid=grid(1536), stream=stream0)
        buf677 = buf655; del buf655  # reuse
        buf678 = reinterpret_tensor(buf666, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf666  # reuse
        # Source Nodes: [div__9], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_35.run(buf677, buf669, primals_32, mul_68, div_107, bernoulli_9, buf678, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_9
        del div_107
        del primals_32
        buf675 = empty((512, ), device='cuda', dtype=torch.float32)
        buf676 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf669, mul_68, buf675, buf676, 512, 1568, grid=grid(512), stream=stream0)
        del mul_68
        buf679 = reinterpret_tensor(buf646, (1568, 2048), (2048, 1), 0); del buf646  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf678, (1568, 512), (512, 1), 0), permute_609, out=buf679)
        del permute_609
        buf680 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf678, (512, 1568), (1, 512), 0), view_104, out=buf680)
        del view_104
        buf681 = buf659; del buf659  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf678, buf681, 6656, 121, grid=grid(6656), stream=stream0)
        buf682 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf681, buf682, 512, 13, grid=grid(512), stream=stream0)
        buf683 = reinterpret_tensor(buf679, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf679  # reuse
        # Source Nodes: [x_119], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_8.run(buf683, addmm_22, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_22
        buf684 = reinterpret_tensor(buf678, (1568, 512), (512, 1), 0); del buf678  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf683, (1568, 2048), (2048, 1), 0), permute_613, out=buf684)
        del permute_613
        buf685 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf683, (2048, 1568), (1, 2048), 0), view_102, out=buf685)
        del view_102
        buf686 = buf649; del buf649  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf683, buf686, 26624, 121, grid=grid(26624), stream=stream0)
        buf687 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf686, buf687, 2048, 13, grid=grid(2048), stream=stream0)
        buf692 = buf677; del buf677  # reuse
        buf693 = reinterpret_tensor(buf669, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf669  # reuse
        # Source Nodes: [div__8], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_35.run(buf692, buf684, primals_30, mul_62, div_108, bernoulli_8, buf693, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_8
        del div_108
        del primals_30
        buf690 = empty((512, ), device='cuda', dtype=torch.float32)
        buf691 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf684, mul_62, buf690, buf691, 512, 1568, grid=grid(512), stream=stream0)
        del mul_62
        buf694 = buf684; del buf684  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf693, (1568, 512), (512, 1), 0), permute_617, out=buf694)
        del permute_617
        buf695 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf693, (512, 1568), (1, 512), 0), view_100, out=buf695)
        del view_100
        buf696 = buf681; del buf681  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf693, buf696, 6656, 121, grid=grid(6656), stream=stream0)
        buf697 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf696, buf697, 512, 13, grid=grid(512), stream=stream0)
        buf698 = reinterpret_tensor(buf693, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf693  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf694, buf698, 128, 6272, grid=grid(128, 6272), stream=stream0)
        buf699 = reinterpret_tensor(buf694, (128, 196, 32), (6272, 32, 1), 0); del buf694  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_622, reinterpret_tensor(buf698, (128, 196, 32), (6272, 32, 1), 0), out=buf699)
        del permute_622
        buf700 = reinterpret_tensor(buf665, (128, 196, 196), (38416, 196, 1), 0); del buf665  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf698, (128, 196, 32), (6272, 32, 1), 0), permute_623, out=buf700)
        del permute_623
        buf702 = reinterpret_tensor(buf663, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf663  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_14.run(buf700, alias_42, buf702, 25088, 196, grid=grid(25088), stream=stream0)
        del alias_42
        buf703 = reinterpret_tensor(buf698, (128, 32, 196), (6272, 196, 1), 0); del buf698  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_624, reinterpret_tensor(buf702, (128, 196, 196), (38416, 196, 1), 0), out=buf703)
        del permute_624
        buf704 = buf662; del buf662  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf702, (128, 196, 196), (38416, 196, 1), 0), permute_625, out=buf704)
        del permute_625
        buf705 = buf668; del buf668  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf704, buf703, buf699, buf705, 75264, 32, grid=grid(75264, 32), stream=stream0)
        buf706 = reinterpret_tensor(buf704, (1568, 512), (512, 1), 0); del buf704  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf705, (1568, 1536), (1536, 1), 0), permute_628, out=buf706)
        del permute_628
        buf707 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf705, (1536, 1568), (1, 1536), 0), view_90, out=buf707)
        del view_90
        buf708 = buf671; del buf671  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf705, buf708, 19968, 121, grid=grid(19968), stream=stream0)
        buf709 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf708, buf709, 1536, 13, grid=grid(1536), stream=stream0)
        buf714 = buf692; del buf692  # reuse
        buf715 = reinterpret_tensor(buf703, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf703  # reuse
        # Source Nodes: [div__7], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_36.run(buf714, buf706, primals_28, mul_57, div_109, bernoulli_7, buf715, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_7
        del div_109
        del primals_28
        buf712 = empty((512, ), device='cuda', dtype=torch.float32)
        buf713 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf706, mul_57, buf712, buf713, 512, 1568, grid=grid(512), stream=stream0)
        del mul_57
        buf716 = reinterpret_tensor(buf683, (1568, 2048), (2048, 1), 0); del buf683  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf715, (1568, 512), (512, 1), 0), permute_632, out=buf716)
        del permute_632
        buf717 = empty((512, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf715, (512, 1568), (1, 512), 0), view_88, out=buf717)
        del view_88
        buf718 = buf696; del buf696  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf715, buf718, 6656, 121, grid=grid(6656), stream=stream0)
        buf719 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf718, buf719, 512, 13, grid=grid(512), stream=stream0)
        buf720 = reinterpret_tensor(buf716, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf716  # reuse
        # Source Nodes: [x_105], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_8.run(buf720, addmm_18, 3211264, grid=grid(3211264), stream=stream0)
        del addmm_18
        buf721 = reinterpret_tensor(buf715, (1568, 512), (512, 1), 0); del buf715  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf720, (1568, 2048), (2048, 1), 0), permute_636, out=buf721)
        del permute_636
        buf722 = empty((2048, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf720, (2048, 1568), (1, 2048), 0), view_86, out=buf722)
        del view_86
        buf723 = buf686; del buf686  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_9.run(buf720, buf723, 26624, 121, grid=grid(26624), stream=stream0)
        buf724 = empty((1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf723, buf724, 2048, 13, grid=grid(2048), stream=stream0)
        del buf723
        buf729 = buf714; del buf714  # reuse
        buf730 = reinterpret_tensor(buf706, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf706  # reuse
        # Source Nodes: [div__6], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_36.run(buf729, buf721, primals_26, mul_51, div_110, bernoulli_6, buf730, 1568, 512, grid=grid(1568), stream=stream0)
        del bernoulli_6
        del div_110
        del primals_26
        buf727 = empty((512, ), device='cuda', dtype=torch.float32)
        buf728 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf721, mul_51, buf727, buf728, 512, 1568, grid=grid(512), stream=stream0)
        del mul_51
        buf731 = buf721; del buf721  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf730, (1568, 512), (512, 1), 0), permute_640, out=buf731)
        del permute_640
        buf732 = empty((512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf730, (512, 1568), (1, 512), 0), view_84, out=buf732)
        del view_84
        buf733 = buf718; del buf718  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf730, buf733, 6656, 121, grid=grid(6656), stream=stream0)
        buf734 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_5.run(buf733, buf734, 512, 13, grid=grid(512), stream=stream0)
        del buf733
        buf735 = reinterpret_tensor(buf730, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf730  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf731, buf735, 128, 6272, grid=grid(128, 6272), stream=stream0)
        buf736 = reinterpret_tensor(buf731, (128, 196, 32), (6272, 32, 1), 0); del buf731  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_645, reinterpret_tensor(buf735, (128, 196, 32), (6272, 32, 1), 0), out=buf736)
        del permute_645
        buf737 = reinterpret_tensor(buf702, (128, 196, 196), (38416, 196, 1), 0); del buf702  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf735, (128, 196, 32), (6272, 32, 1), 0), permute_646, out=buf737)
        del permute_646
        buf739 = reinterpret_tensor(buf700, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf700  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_14.run(buf737, alias_43, buf739, 25088, 196, grid=grid(25088), stream=stream0)
        del alias_43
        del buf737
        buf740 = reinterpret_tensor(buf735, (128, 32, 196), (6272, 196, 1), 0); del buf735  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_647, reinterpret_tensor(buf739, (128, 196, 196), (38416, 196, 1), 0), out=buf740)
        del permute_647
        buf741 = buf699; del buf699  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf739, (128, 196, 196), (38416, 196, 1), 0), permute_648, out=buf741)
        del buf739
        del permute_648
        buf742 = buf705; del buf705  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_15.run(buf741, buf740, buf736, buf742, 75264, 32, grid=grid(75264, 32), stream=stream0)
        del buf736
        del buf740
        buf743 = reinterpret_tensor(buf741, (1568, 512), (512, 1), 0); del buf741  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf742, (1568, 1536), (1536, 1), 0), permute_651, out=buf743)
        del permute_651
        buf744 = empty((1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf742, (1536, 1568), (1, 1536), 0), view_74, out=buf744)
        del view_74
        buf745 = buf708; del buf708  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_16.run(buf742, buf745, 19968, 121, grid=grid(19968), stream=stream0)
        del buf742
        buf746 = empty((1, 1536), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_17.run(buf745, buf746, 1536, 13, grid=grid(1536), stream=stream0)
        del buf745
        buf751 = reinterpret_tensor(buf729, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf729  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_37.run(buf751, buf743, primals_24, mul_46, div_111, 1568, 512, grid=grid(1568), stream=stream0)
        del div_111
        del primals_24
        buf749 = empty((512, ), device='cuda', dtype=torch.float32)
        buf750 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_12.run(buf743, mul_46, buf749, buf750, 512, 1568, grid=grid(512), stream=stream0)
        del buf743
        del mul_46
        buf752 = empty((1, 1, 196, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_38.run(buf751, buf752, 100352, 8, grid=grid(100352), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
        buf753 = aten.max_pool2d_with_indices_backward(reinterpret_tensor(buf751, (8, 512, 14, 14), (100352, 1, 7168, 512), 0), constant_pad_nd_1, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_35)
        del buf751
        del constant_pad_nd_1
        del getitem_35
        buf754 = buf753
        del buf753
        buf761 = reinterpret_tensor(buf720, (8, 28, 28, 512), (401408, 14336, 512, 1), 0); del buf720  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_39.run(buf754, primals_21, mul_44, div_112, buf761, 6272, 512, grid=grid(6272), stream=stream0)
        del div_112
        del primals_21
        buf757 = empty_strided((512, 49), (1, 512), device='cuda', dtype=torch.float32)
        buf759 = empty_strided((512, 49), (1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_40.run(buf754, mul_44, buf757, buf759, 25088, 128, grid=grid(25088), stream=stream0)
        del buf754
        del mul_44
        buf758 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_41.run(buf757, buf758, 512, 49, grid=grid(512), stream=stream0)
        buf760 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_41.run(buf759, buf760, 512, 49, grid=grid(512), stream=stream0)
        buf762 = buf759; del buf759  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_42.run(buf761, buf762, 25088, 128, grid=grid(25088), stream=stream0)
        buf763 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_native_layer_norm_backward_41.run(buf762, buf763, 512, 49, grid=grid(512), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf764 = aten.convolution_backward(reinterpret_tensor(buf761, (8, 512, 28, 28), (401408, 1, 14336, 512), 0), permute_37, primals_142, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del permute_37
        del primals_142
        buf765 = buf764[0]
        buf766 = buf764[1]
        del buf764
        buf767 = empty((8, 4, 196, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__5], Original ATen: [aten.div, aten.mul]
        triton_poi_fused_div_mul_43.run(buf765, bernoulli_5, buf767, 6272, 256, grid=grid(6272, 256), stream=stream0)
        del bernoulli_5
        buf768 = empty((6272, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf767, (6272, 256), (256, 1), 0), permute_661, out=buf768)
        del permute_661
        buf769 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf767, (256, 6272), (1, 256), 0), view_68, out=buf769)
        del view_68
        buf770 = empty_strided((1, 256, 49), (12544, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_44.run(buf767, buf770, 12544, 128, grid=grid(12544), stream=stream0)
        buf771 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_45.run(buf770, buf771, 256, 49, grid=grid(256), stream=stream0)
        buf772 = reinterpret_tensor(buf768, (8, 4, 196, 1024), (802816, 200704, 1024, 1), 0); del buf768  # reuse
        # Source Nodes: [x_75], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_46.run(buf772, addmm_14, 6422528, grid=grid(6422528), stream=stream0)
        del addmm_14
        buf773 = reinterpret_tensor(buf767, (6272, 256), (256, 1), 0); del buf767  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf772, (6272, 1024), (1024, 1), 0), permute_665, out=buf773)
        del permute_665
        buf774 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf772, (1024, 6272), (1, 1024), 0), view_66, out=buf774)
        del view_66
        buf775 = empty_strided((1, 1024, 49), (50176, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_47.run(buf772, buf775, 50176, 128, grid=grid(50176), stream=stream0)
        buf776 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_48.run(buf775, buf776, 1024, 49, grid=grid(1024), stream=stream0)
        buf783 = empty((8, 4, 196, 256), device='cuda', dtype=torch.float32)
        buf784 = empty((8, 4, 196, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [div__4], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_49.run(buf773, primals_19, mul_38, buf765, div_113, bernoulli_4, buf783, buf784, 6272, 256, grid=grid(6272), stream=stream0)
        del bernoulli_4
        del div_113
        del primals_19
        buf779 = reinterpret_tensor(buf770, (256, 49), (1, 256), 0); del buf770  # reuse
        buf781 = empty_strided((256, 49), (1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_50.run(buf773, mul_38, buf779, buf781, 12544, 128, grid=grid(12544), stream=stream0)
        del mul_38
        buf780 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_45.run(buf779, buf780, 256, 49, grid=grid(256), stream=stream0)
        buf782 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_45.run(buf781, buf782, 256, 49, grid=grid(256), stream=stream0)
        buf785 = buf773; del buf773  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf784, (6272, 256), (256, 1), 0), permute_669, out=buf785)
        del permute_669
        buf786 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf784, (256, 6272), (1, 256), 0), view_64, out=buf786)
        del view_64
        buf787 = reinterpret_tensor(buf781, (1, 256, 49), (12544, 1, 256), 0); del buf781  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_44.run(buf784, buf787, 12544, 128, grid=grid(12544), stream=stream0)
        buf788 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_45.run(buf787, buf788, 256, 49, grid=grid(256), stream=stream0)
        buf789 = reinterpret_tensor(buf784, (8, 8, 4, 196, 32), (200704, 25088, 6272, 32, 1), 0); del buf784  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_51.run(buf785, buf789, 64, 25088, grid=grid(64, 25088), stream=stream0)
        buf790 = reinterpret_tensor(buf785, (256, 196, 32), (6272, 32, 1), 0); del buf785  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_674, reinterpret_tensor(buf789, (256, 196, 32), (6272, 32, 1), 0), out=buf790)
        del permute_674
        buf791 = empty((256, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf789, (256, 196, 32), (6272, 32, 1), 0), permute_675, out=buf791)
        del permute_675
        buf793 = empty((8, 8, 4, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_52.run(buf791, alias_44, buf793, 50176, 196, grid=grid(50176), stream=stream0)
        del alias_44
        buf794 = reinterpret_tensor(buf789, (256, 32, 196), (6272, 196, 1), 0); del buf789  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_676, reinterpret_tensor(buf793, (256, 196, 196), (38416, 196, 1), 0), out=buf794)
        del permute_676
        buf795 = reinterpret_tensor(buf765, (256, 196, 32), (6272, 32, 1), 0); del buf765  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf793, (256, 196, 196), (38416, 196, 1), 0), permute_677, out=buf795)
        del permute_677
        buf796 = empty((8, 4, 196, 3, 8, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_53.run(buf795, buf794, buf790, buf796, 150528, 32, grid=grid(150528, 32), stream=stream0)
        buf797 = reinterpret_tensor(buf795, (6272, 256), (256, 1), 0); del buf795  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf796, (6272, 768), (768, 1), 0), permute_680, out=buf797)
        del permute_680
        buf798 = empty((768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf796, (768, 6272), (1, 768), 0), view_54, out=buf798)
        del view_54
        buf799 = empty_strided((1, 768, 49), (37632, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_54.run(buf796, buf799, 37632, 128, grid=grid(37632), stream=stream0)
        buf800 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_55.run(buf799, buf800, 768, 49, grid=grid(768), stream=stream0)
        buf807 = buf783; del buf783  # reuse
        buf808 = reinterpret_tensor(buf794, (8, 4, 196, 256), (200704, 50176, 256, 1), 0); del buf794  # reuse
        # Source Nodes: [div__3], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_56.run(buf807, buf797, primals_17, mul_33, div_114, bernoulli_3, buf808, 6272, 256, grid=grid(6272), stream=stream0)
        del bernoulli_3
        del div_114
        del primals_17
        buf803 = reinterpret_tensor(buf787, (256, 49), (1, 256), 0); del buf787  # reuse
        buf805 = buf779; del buf779  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_50.run(buf797, mul_33, buf803, buf805, 12544, 128, grid=grid(12544), stream=stream0)
        del mul_33
        buf804 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_45.run(buf803, buf804, 256, 49, grid=grid(256), stream=stream0)
        buf806 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_45.run(buf805, buf806, 256, 49, grid=grid(256), stream=stream0)
        buf809 = reinterpret_tensor(buf772, (6272, 1024), (1024, 1), 0); del buf772  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf808, (6272, 256), (256, 1), 0), permute_684, out=buf809)
        del permute_684
        buf810 = empty((256, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf808, (256, 6272), (1, 256), 0), view_52, out=buf810)
        del view_52
        buf811 = reinterpret_tensor(buf805, (1, 256, 49), (12544, 1, 256), 0); del buf805  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_44.run(buf808, buf811, 12544, 128, grid=grid(12544), stream=stream0)
        buf812 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_45.run(buf811, buf812, 256, 49, grid=grid(256), stream=stream0)
        buf813 = reinterpret_tensor(buf809, (8, 4, 196, 1024), (802816, 200704, 1024, 1), 0); del buf809  # reuse
        # Source Nodes: [x_61], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_46.run(buf813, addmm_10, 6422528, grid=grid(6422528), stream=stream0)
        del addmm_10
        buf814 = reinterpret_tensor(buf808, (6272, 256), (256, 1), 0); del buf808  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf813, (6272, 1024), (1024, 1), 0), permute_688, out=buf814)
        del permute_688
        buf815 = empty((1024, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf813, (1024, 6272), (1, 1024), 0), view_50, out=buf815)
        del view_50
        buf816 = buf775; del buf775  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_47.run(buf813, buf816, 50176, 128, grid=grid(50176), stream=stream0)
        buf817 = empty((1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_48.run(buf816, buf817, 1024, 49, grid=grid(1024), stream=stream0)
        buf824 = buf807; del buf807  # reuse
        buf825 = reinterpret_tensor(buf797, (8, 4, 196, 256), (200704, 50176, 256, 1), 0); del buf797  # reuse
        # Source Nodes: [div__2], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_56.run(buf824, buf814, primals_15, mul_27, div_115, bernoulli_2, buf825, 6272, 256, grid=grid(6272), stream=stream0)
        del bernoulli_2
        del div_115
        del primals_15
        buf820 = reinterpret_tensor(buf811, (256, 49), (1, 256), 0); del buf811  # reuse
        buf822 = buf803; del buf803  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_50.run(buf814, mul_27, buf820, buf822, 12544, 128, grid=grid(12544), stream=stream0)
        del mul_27
        buf821 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_45.run(buf820, buf821, 256, 49, grid=grid(256), stream=stream0)
        buf823 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_45.run(buf822, buf823, 256, 49, grid=grid(256), stream=stream0)
        buf826 = buf814; del buf814  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf825, (6272, 256), (256, 1), 0), permute_692, out=buf826)
        del permute_692
        buf827 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf825, (256, 6272), (1, 256), 0), view_48, out=buf827)
        del view_48
        buf828 = reinterpret_tensor(buf822, (1, 256, 49), (12544, 1, 256), 0); del buf822  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_44.run(buf825, buf828, 12544, 128, grid=grid(12544), stream=stream0)
        buf829 = empty((1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_45.run(buf828, buf829, 256, 49, grid=grid(256), stream=stream0)
        buf830 = reinterpret_tensor(buf825, (8, 8, 4, 196, 32), (200704, 25088, 6272, 32, 1), 0); del buf825  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_51.run(buf826, buf830, 64, 25088, grid=grid(64, 25088), stream=stream0)
        buf831 = reinterpret_tensor(buf826, (256, 196, 32), (6272, 32, 1), 0); del buf826  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_697, reinterpret_tensor(buf830, (256, 196, 32), (6272, 32, 1), 0), out=buf831)
        del permute_697
        buf832 = reinterpret_tensor(buf793, (256, 196, 196), (38416, 196, 1), 0); del buf793  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf830, (256, 196, 32), (6272, 32, 1), 0), permute_698, out=buf832)
        del permute_698
        buf834 = reinterpret_tensor(buf791, (8, 8, 4, 196, 196), (1229312, 153664, 38416, 196, 1), 0); del buf791  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_52.run(buf832, alias_45, buf834, 50176, 196, grid=grid(50176), stream=stream0)
        del alias_45
        del buf832
        buf835 = reinterpret_tensor(buf830, (256, 32, 196), (6272, 196, 1), 0); del buf830  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_699, reinterpret_tensor(buf834, (256, 196, 196), (38416, 196, 1), 0), out=buf835)
        del permute_699
        buf836 = buf790; del buf790  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf834, (256, 196, 196), (38416, 196, 1), 0), permute_700, out=buf836)
        del buf834
        del permute_700
        buf837 = buf796; del buf796  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_53.run(buf836, buf835, buf831, buf837, 150528, 32, grid=grid(150528, 32), stream=stream0)
        del buf831
        del buf835
        buf838 = reinterpret_tensor(buf836, (6272, 256), (256, 1), 0); del buf836  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf837, (6272, 768), (768, 1), 0), permute_703, out=buf838)
        del permute_703
        buf839 = empty((768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf837, (768, 6272), (1, 768), 0), view_38, out=buf839)
        del view_38
        buf840 = buf799; del buf799  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_54.run(buf837, buf840, 37632, 128, grid=grid(37632), stream=stream0)
        del buf837
        buf841 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_55.run(buf840, buf841, 768, 49, grid=grid(768), stream=stream0)
        del buf840
        buf848 = buf824; del buf824  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_57.run(buf848, buf838, primals_13, mul_22, div_116, 6272, 256, grid=grid(6272), stream=stream0)
        del div_116
        del primals_13
        buf844 = reinterpret_tensor(buf828, (256, 49), (1, 256), 0); del buf828  # reuse
        buf846 = buf820; del buf820  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_50.run(buf838, mul_22, buf844, buf846, 12544, 128, grid=grid(12544), stream=stream0)
        del mul_22
        buf845 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_45.run(buf844, buf845, 256, 49, grid=grid(256), stream=stream0)
        del buf844
        buf847 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_45.run(buf846, buf847, 256, 49, grid=grid(256), stream=stream0)
        del buf846
        buf849 = empty((1, 4, 196, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_58.run(buf848, buf849, 200704, 8, grid=grid(200704), stream=stream0)
        buf850 = reinterpret_tensor(buf838, (8, 2, 14, 2, 14, 256), (200704, 100352, 7168, 3584, 256, 1), 0); del buf838  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_59.run(buf848, buf850, 1605632, grid=grid(1605632), stream=stream0)
        del buf848
        # Source Nodes: [], Original ATen: [aten.max_pool2d_with_indices_backward]
        buf851 = aten.max_pool2d_with_indices_backward(reinterpret_tensor(buf850, (8, 256, 28, 28), (200704, 1, 7168, 256), 0), constant_pad_nd, [3, 3], [2, 2], [0, 0], [1, 1], False, getitem_17)
        del buf850
        del constant_pad_nd
        del getitem_17
        buf852 = buf851
        del buf851
        buf859 = reinterpret_tensor(buf813, (8, 56, 56, 256), (802816, 14336, 256, 1), 0); del buf813  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_60.run(buf852, primals_10, mul_20, div_117, buf859, 25088, 256, grid=grid(25088), stream=stream0)
        del div_117
        del primals_10
        buf855 = reinterpret_tensor(buf816, (256, 196), (1, 256), 0); del buf816  # reuse
        buf857 = empty_strided((256, 196), (1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_61.run(buf852, mul_20, buf855, buf857, 50176, 128, grid=grid(50176), stream=stream0)
        del buf852
        del mul_20
        buf856 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_62.run(buf855, buf856, 256, 196, grid=grid(256), stream=stream0)
        del buf855
        buf858 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_62.run(buf857, buf858, 256, 196, grid=grid(256), stream=stream0)
        buf860 = buf857; del buf857  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_63.run(buf859, buf860, 50176, 128, grid=grid(50176), stream=stream0)
        buf861 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_native_layer_norm_backward_62.run(buf860, buf861, 256, 196, grid=grid(256), stream=stream0)
        del buf860
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf862 = aten.convolution_backward(reinterpret_tensor(buf859, (8, 256, 56, 56), (802816, 1, 14336, 256), 0), permute_17, primals_124, [256], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf859
        del permute_17
        del primals_124
        buf863 = buf862[0]
        buf864 = buf862[1]
        del buf862
        buf865 = reinterpret_tensor(buf761, (8, 16, 196, 128), (401408, 25088, 128, 1), 0); del buf761  # reuse
        # Source Nodes: [div__1], Original ATen: [aten.div, aten.mul]
        triton_poi_fused_div_mul_64.run(buf863, bernoulli_1, buf865, 25088, 128, grid=grid(25088, 128), stream=stream0)
        del bernoulli_1
        buf866 = empty((25088, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf865, (25088, 128), (128, 1), 0), permute_713, out=buf866)
        del permute_713
        buf867 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf865, (128, 25088), (1, 128), 0), view_32, out=buf867)
        del view_32
        buf868 = reinterpret_tensor(buf762, (1, 128, 196), (25088, 1, 128), 0); del buf762  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_65.run(buf865, buf868, 25088, 128, grid=grid(25088), stream=stream0)
        buf869 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_66.run(buf868, buf869, 128, 196, grid=grid(128), stream=stream0)
        buf870 = reinterpret_tensor(buf866, (8, 16, 196, 512), (1605632, 100352, 512, 1), 0); del buf866  # reuse
        # Source Nodes: [x_31], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_67.run(buf870, addmm_6, 12845056, grid=grid(12845056), stream=stream0)
        del addmm_6
        buf871 = reinterpret_tensor(buf865, (25088, 128), (128, 1), 0); del buf865  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf870, (25088, 512), (512, 1), 0), permute_717, out=buf871)
        del permute_717
        buf872 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf870, (512, 25088), (1, 512), 0), view_30, out=buf872)
        del view_30
        buf873 = empty_strided((1, 512, 196), (100352, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_68.run(buf870, buf873, 100352, 128, grid=grid(100352), stream=stream0)
        buf874 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_69.run(buf873, buf874, 512, 196, grid=grid(512), stream=stream0)
        buf881 = empty((8, 16, 196, 128), device='cuda', dtype=torch.float32)
        buf882 = empty((8, 16, 196, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [div_], Original ATen: [aten.add, aten.div, aten.mul, aten.native_layer_norm_backward]
        triton_per_fused_add_div_mul_native_layer_norm_backward_70.run(buf871, primals_8, mul_14, buf863, div_118, bernoulli, buf881, buf882, 25088, 128, grid=grid(25088), stream=stream0)
        del bernoulli
        del div_118
        del primals_8
        buf877 = reinterpret_tensor(buf868, (128, 196), (1, 128), 0); del buf868  # reuse
        buf879 = reinterpret_tensor(buf757, (128, 196), (1, 128), 0); del buf757  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_71.run(buf871, mul_14, buf877, buf879, 25088, 128, grid=grid(25088), stream=stream0)
        del mul_14
        buf878 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_sum_66.run(buf877, buf878, 128, 196, grid=grid(128), stream=stream0)
        buf880 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_sum_66.run(buf879, buf880, 128, 196, grid=grid(128), stream=stream0)
        buf883 = buf871; del buf871  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf882, (25088, 128), (128, 1), 0), permute_721, out=buf883)
        del permute_721
        buf884 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf882, (128, 25088), (1, 128), 0), view_28, out=buf884)
        del view_28
        buf885 = reinterpret_tensor(buf879, (1, 128, 196), (25088, 1, 128), 0); del buf879  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_65.run(buf882, buf885, 25088, 128, grid=grid(25088), stream=stream0)
        buf886 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_66.run(buf885, buf886, 128, 196, grid=grid(128), stream=stream0)
        buf887 = reinterpret_tensor(buf882, (8, 4, 16, 196, 32), (401408, 100352, 6272, 32, 1), 0); del buf882  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_72.run(buf883, buf887, 32, 100352, grid=grid(32, 100352), stream=stream0)
        buf888 = reinterpret_tensor(buf883, (512, 196, 32), (6272, 32, 1), 0); del buf883  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_726, reinterpret_tensor(buf887, (512, 196, 32), (6272, 32, 1), 0), out=buf888)
        del permute_726
        buf889 = empty((512, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf887, (512, 196, 32), (6272, 32, 1), 0), permute_727, out=buf889)
        del permute_727
        buf891 = empty((8, 4, 16, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_73.run(buf889, alias_46, buf891, 100352, 196, grid=grid(100352), stream=stream0)
        del alias_46
        buf892 = reinterpret_tensor(buf887, (512, 32, 196), (6272, 196, 1), 0); del buf887  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_728, reinterpret_tensor(buf891, (512, 196, 196), (38416, 196, 1), 0), out=buf892)
        del permute_728
        buf893 = reinterpret_tensor(buf863, (512, 196, 32), (6272, 32, 1), 0); del buf863  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf891, (512, 196, 196), (38416, 196, 1), 0), permute_729, out=buf893)
        del permute_729
        buf894 = empty((8, 16, 196, 3, 4, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_74.run(buf893, buf892, buf888, buf894, 301056, 32, grid=grid(301056, 32), stream=stream0)
        buf895 = reinterpret_tensor(buf893, (25088, 128), (128, 1), 0); del buf893  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf894, (25088, 384), (384, 1), 0), permute_732, out=buf895)
        del permute_732
        buf896 = empty((384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf894, (384, 25088), (1, 384), 0), view_18, out=buf896)
        del view_18
        buf897 = empty_strided((1, 384, 196), (75264, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_75.run(buf894, buf897, 75264, 128, grid=grid(75264), stream=stream0)
        buf898 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_76.run(buf897, buf898, 384, 196, grid=grid(384), stream=stream0)
        buf905 = buf881; del buf881  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_77.run(buf905, buf895, primals_6, mul_9, div_119, 25088, 128, grid=grid(25088), stream=stream0)
        del div_119
        del primals_6
        buf901 = reinterpret_tensor(buf885, (128, 196), (1, 128), 0); del buf885  # reuse
        buf903 = buf877; del buf877  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_71.run(buf895, mul_9, buf901, buf903, 25088, 128, grid=grid(25088), stream=stream0)
        del mul_9
        buf902 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_sum_66.run(buf901, buf902, 128, 196, grid=grid(128), stream=stream0)
        buf904 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_sum_66.run(buf903, buf904, 128, 196, grid=grid(128), stream=stream0)
        buf906 = reinterpret_tensor(buf870, (25088, 512), (512, 1), 0); del buf870  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf905, (25088, 128), (128, 1), 0), permute_736, out=buf906)
        del permute_736
        buf907 = empty((128, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf905, (128, 25088), (1, 128), 0), view_16, out=buf907)
        del view_16
        buf908 = reinterpret_tensor(buf903, (1, 128, 196), (25088, 1, 128), 0); del buf903  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_78.run(buf905, buf908, 25088, 128, grid=grid(25088), stream=stream0)
        buf909 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_66.run(buf908, buf909, 128, 196, grid=grid(128), stream=stream0)
        buf910 = reinterpret_tensor(buf906, (8, 16, 196, 512), (1605632, 100352, 512, 1), 0); del buf906  # reuse
        # Source Nodes: [x_17], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_67.run(buf910, addmm_2, 12845056, grid=grid(12845056), stream=stream0)
        del addmm_2
        buf911 = buf895; del buf895  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf910, (25088, 512), (512, 1), 0), permute_740, out=buf911)
        del permute_740
        buf912 = empty((512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf910, (512, 25088), (1, 512), 0), view_14, out=buf912)
        del view_14
        buf913 = buf873; del buf873  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_68.run(buf910, buf913, 100352, 128, grid=grid(100352), stream=stream0)
        del buf910
        buf914 = empty((1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_69.run(buf913, buf914, 512, 196, grid=grid(512), stream=stream0)
        del buf913
        buf921 = buf905; del buf905  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_77.run(buf921, buf911, primals_4, mul_4, div_120, 25088, 128, grid=grid(25088), stream=stream0)
        del div_120
        del primals_4
        buf917 = reinterpret_tensor(buf908, (128, 196), (1, 128), 0); del buf908  # reuse
        buf919 = buf901; del buf901  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_71.run(buf911, mul_4, buf917, buf919, 25088, 128, grid=grid(25088), stream=stream0)
        del mul_4
        buf918 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_sum_66.run(buf917, buf918, 128, 196, grid=grid(128), stream=stream0)
        buf920 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_sum_66.run(buf919, buf920, 128, 196, grid=grid(128), stream=stream0)
        buf922 = buf911; del buf911  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf921, (25088, 128), (128, 1), 0), permute_744, out=buf922)
        del permute_744
        buf923 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf921, (128, 25088), (1, 128), 0), view_12, out=buf923)
        del view_12
        buf924 = reinterpret_tensor(buf919, (1, 128, 196), (25088, 1, 128), 0); del buf919  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_78.run(buf921, buf924, 25088, 128, grid=grid(25088), stream=stream0)
        buf925 = empty((1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_66.run(buf924, buf925, 128, 196, grid=grid(128), stream=stream0)
        buf926 = reinterpret_tensor(buf892, (8, 4, 16, 196, 32), (401408, 100352, 6272, 32, 1), 0); del buf892  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_72.run(buf922, buf926, 32, 100352, grid=grid(32, 100352), stream=stream0)
        buf927 = reinterpret_tensor(buf922, (512, 196, 32), (6272, 32, 1), 0); del buf922  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_749, reinterpret_tensor(buf926, (512, 196, 32), (6272, 32, 1), 0), out=buf927)
        del permute_749
        buf928 = reinterpret_tensor(buf891, (512, 196, 196), (38416, 196, 1), 0); del buf891  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf926, (512, 196, 32), (6272, 32, 1), 0), permute_750, out=buf928)
        del permute_750
        buf930 = reinterpret_tensor(buf889, (8, 4, 16, 196, 196), (2458624, 614656, 38416, 196, 1), 0); del buf889  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_73.run(buf928, alias_47, buf930, 100352, 196, grid=grid(100352), stream=stream0)
        del alias_47
        del buf928
        buf931 = reinterpret_tensor(buf926, (512, 32, 196), (6272, 196, 1), 0); del buf926  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_751, reinterpret_tensor(buf930, (512, 196, 196), (38416, 196, 1), 0), out=buf931)
        del permute_751
        buf932 = buf888; del buf888  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf930, (512, 196, 196), (38416, 196, 1), 0), permute_752, out=buf932)
        del buf930
        del permute_752
        buf933 = buf894; del buf894  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_74.run(buf932, buf931, buf927, buf933, 301056, 32, grid=grid(301056, 32), stream=stream0)
        del buf927
        del buf931
        buf934 = reinterpret_tensor(buf932, (25088, 128), (128, 1), 0); del buf932  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf933, (25088, 384), (384, 1), 0), permute_755, out=buf934)
        del permute_755
        buf935 = empty((384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf933, (384, 25088), (1, 384), 0), view_2, out=buf935)
        del view_2
        buf936 = buf897; del buf897  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_75.run(buf933, buf936, 75264, 128, grid=grid(75264), stream=stream0)
        del buf933
        buf937 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_76.run(buf936, buf937, 384, 196, grid=grid(384), stream=stream0)
        del buf936
        buf944 = buf921; del buf921  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_77.run(buf944, buf934, primals_2, mul, div_121, 25088, 128, grid=grid(25088), stream=stream0)
        del div_121
        del primals_2
        buf940 = reinterpret_tensor(buf924, (128, 196), (1, 128), 0); del buf924  # reuse
        buf942 = buf917; del buf917  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_71.run(buf934, mul, buf940, buf942, 25088, 128, grid=grid(25088), stream=stream0)
        del mul
        buf941 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_sum_66.run(buf940, buf941, 128, 196, grid=grid(128), stream=stream0)
        del buf940
        buf943 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_sum_66.run(buf942, buf943, 128, 196, grid=grid(128), stream=stream0)
        buf945 = empty((1, 16, 196, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_79.run(buf944, buf945, 401408, 8, grid=grid(401408), stream=stream0)
        buf946 = reinterpret_tensor(buf934, (8, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf934  # reuse
        # Source Nodes: [], Original ATen: [aten.permute]
        triton_poi_fused_permute_80.run(buf944, buf946, 3211264, grid=grid(3211264), stream=stream0)
        del buf944
        buf947 = buf942; del buf942  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_sum_78.run(buf946, buf947, 25088, 128, grid=grid(25088), stream=stream0)
        buf948 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_sum_66.run(buf947, buf948, 128, 196, grid=grid(128), stream=stream0)
        del buf947
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf949 = aten.convolution_backward(buf946, primals_306, primals_106, [128], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf946
        del primals_106
        del primals_306
        buf950 = buf949[1]
        return (buf945, buf941, buf943, buf918, buf920, buf902, buf904, buf878, buf880, buf856, buf858, buf849, buf845, buf847, buf821, buf823, buf804, buf806, buf780, buf782, buf758, buf760, buf752, buf749, buf750, buf727, buf728, buf712, buf713, buf690, buf691, buf675, buf676, buf653, buf654, buf638, buf639, buf616, buf617, buf601, buf602, buf579, buf580, buf564, buf565, buf542, buf543, buf527, buf528, buf505, buf506, buf490, buf491, buf468, buf469, buf453, buf454, buf431, buf432, buf416, buf417, buf394, buf395, buf379, buf380, buf357, buf358, buf342, buf343, buf320, buf321, buf305, buf306, buf283, buf284, buf268, buf269, buf246, buf247, buf231, buf232, buf209, buf210, buf194, buf195, buf172, buf173, buf157, buf158, buf135, buf136, buf120, buf121, buf98, buf99, buf83, buf84, buf61, buf62, buf46, buf47, buf24, buf25, buf9, buf10, buf950, buf948, reinterpret_tensor(buf935, (384, 128), (128, 1), 0), reinterpret_tensor(buf937, (384, ), (1, ), 0), reinterpret_tensor(buf923, (128, 128), (128, 1), 0), reinterpret_tensor(buf925, (128, ), (1, ), 0), reinterpret_tensor(buf912, (512, 128), (128, 1), 0), reinterpret_tensor(buf914, (512, ), (1, ), 0), reinterpret_tensor(buf907, (128, 512), (512, 1), 0), reinterpret_tensor(buf909, (128, ), (1, ), 0), reinterpret_tensor(buf896, (384, 128), (128, 1), 0), reinterpret_tensor(buf898, (384, ), (1, ), 0), reinterpret_tensor(buf884, (128, 128), (128, 1), 0), reinterpret_tensor(buf886, (128, ), (1, ), 0), reinterpret_tensor(buf872, (512, 128), (128, 1), 0), reinterpret_tensor(buf874, (512, ), (1, ), 0), reinterpret_tensor(buf867, (128, 512), (512, 1), 0), reinterpret_tensor(buf869, (128, ), (1, ), 0), buf864, buf861, reinterpret_tensor(buf839, (768, 256), (256, 1), 0), reinterpret_tensor(buf841, (768, ), (1, ), 0), reinterpret_tensor(buf827, (256, 256), (256, 1), 0), reinterpret_tensor(buf829, (256, ), (1, ), 0), reinterpret_tensor(buf815, (1024, 256), (256, 1), 0), reinterpret_tensor(buf817, (1024, ), (1, ), 0), reinterpret_tensor(buf810, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf812, (256, ), (1, ), 0), reinterpret_tensor(buf798, (768, 256), (256, 1), 0), reinterpret_tensor(buf800, (768, ), (1, ), 0), reinterpret_tensor(buf786, (256, 256), (256, 1), 0), reinterpret_tensor(buf788, (256, ), (1, ), 0), reinterpret_tensor(buf774, (1024, 256), (256, 1), 0), reinterpret_tensor(buf776, (1024, ), (1, ), 0), reinterpret_tensor(buf769, (256, 1024), (1024, 1), 0), reinterpret_tensor(buf771, (256, ), (1, ), 0), buf766, buf763, reinterpret_tensor(buf744, (1536, 512), (512, 1), 0), reinterpret_tensor(buf746, (1536, ), (1, ), 0), reinterpret_tensor(buf732, (512, 512), (512, 1), 0), reinterpret_tensor(buf734, (512, ), (1, ), 0), reinterpret_tensor(buf722, (2048, 512), (512, 1), 0), reinterpret_tensor(buf724, (2048, ), (1, ), 0), reinterpret_tensor(buf717, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf719, (512, ), (1, ), 0), reinterpret_tensor(buf707, (1536, 512), (512, 1), 0), reinterpret_tensor(buf709, (1536, ), (1, ), 0), reinterpret_tensor(buf695, (512, 512), (512, 1), 0), reinterpret_tensor(buf697, (512, ), (1, ), 0), reinterpret_tensor(buf685, (2048, 512), (512, 1), 0), reinterpret_tensor(buf687, (2048, ), (1, ), 0), reinterpret_tensor(buf680, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf682, (512, ), (1, ), 0), reinterpret_tensor(buf670, (1536, 512), (512, 1), 0), reinterpret_tensor(buf672, (1536, ), (1, ), 0), reinterpret_tensor(buf658, (512, 512), (512, 1), 0), reinterpret_tensor(buf660, (512, ), (1, ), 0), reinterpret_tensor(buf648, (2048, 512), (512, 1), 0), reinterpret_tensor(buf650, (2048, ), (1, ), 0), reinterpret_tensor(buf643, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf645, (512, ), (1, ), 0), reinterpret_tensor(buf633, (1536, 512), (512, 1), 0), reinterpret_tensor(buf635, (1536, ), (1, ), 0), reinterpret_tensor(buf621, (512, 512), (512, 1), 0), reinterpret_tensor(buf623, (512, ), (1, ), 0), reinterpret_tensor(buf611, (2048, 512), (512, 1), 0), reinterpret_tensor(buf613, (2048, ), (1, ), 0), reinterpret_tensor(buf606, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf608, (512, ), (1, ), 0), reinterpret_tensor(buf596, (1536, 512), (512, 1), 0), reinterpret_tensor(buf598, (1536, ), (1, ), 0), reinterpret_tensor(buf584, (512, 512), (512, 1), 0), reinterpret_tensor(buf586, (512, ), (1, ), 0), reinterpret_tensor(buf574, (2048, 512), (512, 1), 0), reinterpret_tensor(buf576, (2048, ), (1, ), 0), reinterpret_tensor(buf569, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf571, (512, ), (1, ), 0), reinterpret_tensor(buf559, (1536, 512), (512, 1), 0), reinterpret_tensor(buf561, (1536, ), (1, ), 0), reinterpret_tensor(buf547, (512, 512), (512, 1), 0), reinterpret_tensor(buf549, (512, ), (1, ), 0), reinterpret_tensor(buf537, (2048, 512), (512, 1), 0), reinterpret_tensor(buf539, (2048, ), (1, ), 0), reinterpret_tensor(buf532, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf534, (512, ), (1, ), 0), reinterpret_tensor(buf522, (1536, 512), (512, 1), 0), reinterpret_tensor(buf524, (1536, ), (1, ), 0), reinterpret_tensor(buf510, (512, 512), (512, 1), 0), reinterpret_tensor(buf512, (512, ), (1, ), 0), reinterpret_tensor(buf500, (2048, 512), (512, 1), 0), reinterpret_tensor(buf502, (2048, ), (1, ), 0), reinterpret_tensor(buf495, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf497, (512, ), (1, ), 0), reinterpret_tensor(buf485, (1536, 512), (512, 1), 0), reinterpret_tensor(buf487, (1536, ), (1, ), 0), reinterpret_tensor(buf473, (512, 512), (512, 1), 0), reinterpret_tensor(buf475, (512, ), (1, ), 0), reinterpret_tensor(buf463, (2048, 512), (512, 1), 0), reinterpret_tensor(buf465, (2048, ), (1, ), 0), reinterpret_tensor(buf458, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf460, (512, ), (1, ), 0), reinterpret_tensor(buf448, (1536, 512), (512, 1), 0), reinterpret_tensor(buf450, (1536, ), (1, ), 0), reinterpret_tensor(buf436, (512, 512), (512, 1), 0), reinterpret_tensor(buf438, (512, ), (1, ), 0), reinterpret_tensor(buf426, (2048, 512), (512, 1), 0), reinterpret_tensor(buf428, (2048, ), (1, ), 0), reinterpret_tensor(buf421, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf423, (512, ), (1, ), 0), reinterpret_tensor(buf411, (1536, 512), (512, 1), 0), reinterpret_tensor(buf413, (1536, ), (1, ), 0), reinterpret_tensor(buf399, (512, 512), (512, 1), 0), reinterpret_tensor(buf401, (512, ), (1, ), 0), reinterpret_tensor(buf389, (2048, 512), (512, 1), 0), reinterpret_tensor(buf391, (2048, ), (1, ), 0), reinterpret_tensor(buf384, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf386, (512, ), (1, ), 0), reinterpret_tensor(buf374, (1536, 512), (512, 1), 0), reinterpret_tensor(buf376, (1536, ), (1, ), 0), reinterpret_tensor(buf362, (512, 512), (512, 1), 0), reinterpret_tensor(buf364, (512, ), (1, ), 0), reinterpret_tensor(buf352, (2048, 512), (512, 1), 0), reinterpret_tensor(buf354, (2048, ), (1, ), 0), reinterpret_tensor(buf347, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf349, (512, ), (1, ), 0), reinterpret_tensor(buf337, (1536, 512), (512, 1), 0), reinterpret_tensor(buf339, (1536, ), (1, ), 0), reinterpret_tensor(buf325, (512, 512), (512, 1), 0), reinterpret_tensor(buf327, (512, ), (1, ), 0), reinterpret_tensor(buf315, (2048, 512), (512, 1), 0), reinterpret_tensor(buf317, (2048, ), (1, ), 0), reinterpret_tensor(buf310, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf312, (512, ), (1, ), 0), reinterpret_tensor(buf300, (1536, 512), (512, 1), 0), reinterpret_tensor(buf302, (1536, ), (1, ), 0), reinterpret_tensor(buf288, (512, 512), (512, 1), 0), reinterpret_tensor(buf290, (512, ), (1, ), 0), reinterpret_tensor(buf278, (2048, 512), (512, 1), 0), reinterpret_tensor(buf280, (2048, ), (1, ), 0), reinterpret_tensor(buf273, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf275, (512, ), (1, ), 0), reinterpret_tensor(buf263, (1536, 512), (512, 1), 0), reinterpret_tensor(buf265, (1536, ), (1, ), 0), reinterpret_tensor(buf251, (512, 512), (512, 1), 0), reinterpret_tensor(buf253, (512, ), (1, ), 0), reinterpret_tensor(buf241, (2048, 512), (512, 1), 0), reinterpret_tensor(buf243, (2048, ), (1, ), 0), reinterpret_tensor(buf236, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf238, (512, ), (1, ), 0), reinterpret_tensor(buf226, (1536, 512), (512, 1), 0), reinterpret_tensor(buf228, (1536, ), (1, ), 0), reinterpret_tensor(buf214, (512, 512), (512, 1), 0), reinterpret_tensor(buf216, (512, ), (1, ), 0), reinterpret_tensor(buf204, (2048, 512), (512, 1), 0), reinterpret_tensor(buf206, (2048, ), (1, ), 0), reinterpret_tensor(buf199, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf201, (512, ), (1, ), 0), reinterpret_tensor(buf189, (1536, 512), (512, 1), 0), reinterpret_tensor(buf191, (1536, ), (1, ), 0), reinterpret_tensor(buf177, (512, 512), (512, 1), 0), reinterpret_tensor(buf179, (512, ), (1, ), 0), reinterpret_tensor(buf167, (2048, 512), (512, 1), 0), reinterpret_tensor(buf169, (2048, ), (1, ), 0), reinterpret_tensor(buf162, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf164, (512, ), (1, ), 0), reinterpret_tensor(buf152, (1536, 512), (512, 1), 0), reinterpret_tensor(buf154, (1536, ), (1, ), 0), reinterpret_tensor(buf140, (512, 512), (512, 1), 0), reinterpret_tensor(buf142, (512, ), (1, ), 0), reinterpret_tensor(buf130, (2048, 512), (512, 1), 0), reinterpret_tensor(buf132, (2048, ), (1, ), 0), reinterpret_tensor(buf125, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf127, (512, ), (1, ), 0), reinterpret_tensor(buf115, (1536, 512), (512, 1), 0), reinterpret_tensor(buf117, (1536, ), (1, ), 0), reinterpret_tensor(buf103, (512, 512), (512, 1), 0), reinterpret_tensor(buf105, (512, ), (1, ), 0), reinterpret_tensor(buf93, (2048, 512), (512, 1), 0), reinterpret_tensor(buf95, (2048, ), (1, ), 0), reinterpret_tensor(buf88, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf90, (512, ), (1, ), 0), reinterpret_tensor(buf78, (1536, 512), (512, 1), 0), reinterpret_tensor(buf80, (1536, ), (1, ), 0), reinterpret_tensor(buf66, (512, 512), (512, 1), 0), reinterpret_tensor(buf68, (512, ), (1, ), 0), reinterpret_tensor(buf56, (2048, 512), (512, 1), 0), reinterpret_tensor(buf58, (2048, ), (1, ), 0), reinterpret_tensor(buf51, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf53, (512, ), (1, ), 0), reinterpret_tensor(buf41, (1536, 512), (512, 1), 0), reinterpret_tensor(buf43, (1536, ), (1, ), 0), reinterpret_tensor(buf29, (512, 512), (512, 1), 0), reinterpret_tensor(buf31, (512, ), (1, ), 0), reinterpret_tensor(buf19, (2048, 512), (512, 1), 0), reinterpret_tensor(buf21, (2048, ), (1, ), 0), reinterpret_tensor(buf14, (512, 2048), (2048, 1), 0), reinterpret_tensor(buf16, (512, ), (1, ), 0), reinterpret_tensor(buf1, (1000, 512), (512, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_2 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((128, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    mul = rand_strided((8, 16, 196, 128), (401408, 25088, 128, 1), device='cuda:0', dtype=torch.float32)
    view_2 = rand_strided((25088, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_12 = rand_strided((25088, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mul_4 = rand_strided((8, 16, 196, 128), (401408, 25088, 128, 1), device='cuda:0', dtype=torch.float32)
    view_14 = rand_strided((25088, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_2 = rand_strided((25088, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_16 = rand_strided((25088, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mul_9 = rand_strided((8, 16, 196, 128), (401408, 25088, 128, 1), device='cuda:0', dtype=torch.float32)
    view_18 = rand_strided((25088, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    view_28 = rand_strided((25088, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    bernoulli = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_14 = rand_strided((8, 16, 196, 128), (401408, 25088, 128, 1), device='cuda:0', dtype=torch.float32)
    view_30 = rand_strided((25088, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    addmm_6 = rand_strided((25088, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_32 = rand_strided((25088, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_1 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_17 = rand_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    mul_20 = rand_strided((8, 56, 56, 256), (802816, 14336, 256, 1), device='cuda:0', dtype=torch.float32)
    constant_pad_nd = rand_strided((8, 256, 57, 57), (831744, 1, 14592, 256), device='cuda:0', dtype=torch.float32)
    getitem_17 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.int64)
    mul_22 = rand_strided((8, 4, 196, 256), (200704, 50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_38 = rand_strided((6272, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    view_48 = rand_strided((6272, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_2 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_27 = rand_strided((8, 4, 196, 256), (200704, 50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_50 = rand_strided((6272, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((6272, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_52 = rand_strided((6272, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_3 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_33 = rand_strided((8, 4, 196, 256), (200704, 50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_54 = rand_strided((6272, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    view_64 = rand_strided((6272, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_4 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_38 = rand_strided((8, 4, 196, 256), (200704, 50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_66 = rand_strided((6272, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    addmm_14 = rand_strided((6272, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    view_68 = rand_strided((6272, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_5 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_37 = rand_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    mul_44 = rand_strided((8, 28, 28, 512), (401408, 14336, 512, 1), device='cuda:0', dtype=torch.float32)
    constant_pad_nd_1 = rand_strided((8, 512, 29, 29), (430592, 1, 14848, 512), device='cuda:0', dtype=torch.float32)
    getitem_35 = rand_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.int64)
    mul_46 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_74 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_84 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_6 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_51 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_86 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_18 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_88 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_7 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_57 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_90 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_100 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_8 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_62 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_102 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_104 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_9 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_68 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_106 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_116 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_10 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_73 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_118 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_26 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_120 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_11 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_79 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_122 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_132 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_12 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_84 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_134 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_30 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_136 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_13 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_90 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_138 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_148 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_14 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_95 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_150 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_152 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_15 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_101 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_154 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_164 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_16 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_106 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_166 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_38 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_168 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_17 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_112 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_170 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_180 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_18 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_117 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_182 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_42 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_184 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_19 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_123 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_186 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_196 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_20 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_128 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_198 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_46 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_200 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_21 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_134 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_202 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_212 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_22 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_139 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_214 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_50 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_216 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_23 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_145 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_218 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_228 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_24 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_150 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_230 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_54 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_232 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_25 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_156 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_234 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_244 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_26 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_161 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_246 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_58 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_248 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_27 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_167 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_250 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_260 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_28 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_172 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_262 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_62 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_264 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_29 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_178 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_266 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_276 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_30 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_183 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_278 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_66 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_280 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_31 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_189 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_282 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_292 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_32 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_194 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_294 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_70 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_296 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_33 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_200 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_298 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_308 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_34 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_205 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_310 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_74 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_312 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_35 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_211 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_314 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_324 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_36 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_216 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_326 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_78 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_328 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_37 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_222 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_330 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_340 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_38 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_227 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_342 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_82 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_344 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_39 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_233 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_346 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_356 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_40 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_238 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_358 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_86 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_360 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_41 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_244 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_362 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_372 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_42 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_249 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_374 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_90 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_376 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_43 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_255 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_378 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    view_388 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_44 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_260 = rand_strided((8, 1, 196, 512), (100352, 100352, 512, 1), device='cuda:0', dtype=torch.float32)
    view_390 = rand_strided((1568, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    addmm_94 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    view_392 = rand_strided((1568, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    bernoulli_45 = rand_strided((8, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_266 = rand_strided((8, 14, 14, 512), (100352, 7168, 512, 1), device='cuda:0', dtype=torch.float32)
    clone_174 = rand_strided((8, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_187 = rand_strided((1000, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_71 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_195 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_199 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_72 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_203 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_208 = rand_strided((128, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_209 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_24 = rand_strided((8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_210 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_211 = rand_strided((128, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_214 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_73 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_218 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_222 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_74 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_226 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_231 = rand_strided((128, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_232 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_25 = rand_strided((8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_233 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_234 = rand_strided((128, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_237 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_75 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_241 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_245 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_76 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_249 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_254 = rand_strided((128, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_255 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_26 = rand_strided((8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_256 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_257 = rand_strided((128, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_260 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_77 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_264 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_268 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_78 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_272 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_277 = rand_strided((128, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_278 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_27 = rand_strided((8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_279 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_280 = rand_strided((128, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_283 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_79 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_287 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_291 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_80 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_295 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_300 = rand_strided((128, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_301 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_28 = rand_strided((8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_302 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_303 = rand_strided((128, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_306 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_81 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_310 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_314 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_82 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_318 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_323 = rand_strided((128, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_324 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_29 = rand_strided((8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_325 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_326 = rand_strided((128, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_329 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_83 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_333 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_337 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_84 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_341 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_346 = rand_strided((128, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_347 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_30 = rand_strided((8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_348 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_349 = rand_strided((128, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_352 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_85 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_356 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_360 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_86 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_364 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_369 = rand_strided((128, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_370 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_31 = rand_strided((8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_371 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_372 = rand_strided((128, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_375 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_87 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_379 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_383 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_88 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_387 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_392 = rand_strided((128, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_393 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_32 = rand_strided((8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_394 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_395 = rand_strided((128, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_398 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_89 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_402 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_406 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_90 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_410 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_415 = rand_strided((128, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_416 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_33 = rand_strided((8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_417 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_418 = rand_strided((128, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_421 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_91 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_425 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_429 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_92 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_433 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_438 = rand_strided((128, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_439 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_34 = rand_strided((8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_440 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_441 = rand_strided((128, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_444 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_93 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_448 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_452 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_94 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_456 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_461 = rand_strided((128, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_462 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_35 = rand_strided((8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_463 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_464 = rand_strided((128, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_467 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_95 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_471 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_475 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_96 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_479 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_484 = rand_strided((128, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_485 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_36 = rand_strided((8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_486 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_487 = rand_strided((128, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_490 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_97 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_494 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_498 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_98 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_502 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_507 = rand_strided((128, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_508 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_37 = rand_strided((8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_509 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_510 = rand_strided((128, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_513 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_99 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_517 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_521 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_100 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_525 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_530 = rand_strided((128, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_531 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_38 = rand_strided((8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_532 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_533 = rand_strided((128, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_536 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_101 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_540 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_544 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_102 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_548 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_553 = rand_strided((128, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_554 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_39 = rand_strided((8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_555 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_556 = rand_strided((128, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_559 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_103 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_563 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_567 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_104 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_571 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_576 = rand_strided((128, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_577 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_40 = rand_strided((8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_578 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_579 = rand_strided((128, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_582 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_105 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_586 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_590 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_106 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_594 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_599 = rand_strided((128, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_600 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_41 = rand_strided((8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_601 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_602 = rand_strided((128, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_605 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_107 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_609 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_613 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_108 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_617 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_622 = rand_strided((128, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_623 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_42 = rand_strided((8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_624 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_625 = rand_strided((128, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_628 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_109 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_632 = rand_strided((512, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    permute_636 = rand_strided((2048, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_110 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_640 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_645 = rand_strided((128, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_646 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_43 = rand_strided((8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_647 = rand_strided((128, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_648 = rand_strided((128, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_651 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    div_111 = rand_strided((8, 1, 196, 1), (196, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    div_112 = rand_strided((8, 28, 28, 1), (784, 28, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_661 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_665 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_113 = rand_strided((8, 4, 196, 1), (784, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_669 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_674 = rand_strided((256, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_675 = rand_strided((256, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_44 = rand_strided((8, 8, 4, 196, 196), (1229312, 153664, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_676 = rand_strided((256, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_677 = rand_strided((256, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_680 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_114 = rand_strided((8, 4, 196, 1), (784, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_684 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_688 = rand_strided((1024, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_115 = rand_strided((8, 4, 196, 1), (784, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_692 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_697 = rand_strided((256, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_698 = rand_strided((256, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_45 = rand_strided((8, 8, 4, 196, 196), (1229312, 153664, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_699 = rand_strided((256, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_700 = rand_strided((256, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_703 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    div_116 = rand_strided((8, 4, 196, 1), (784, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    div_117 = rand_strided((8, 56, 56, 1), (3136, 56, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_713 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_717 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_118 = rand_strided((8, 16, 196, 1), (3136, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_721 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_726 = rand_strided((512, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_727 = rand_strided((512, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_46 = rand_strided((8, 4, 16, 196, 196), (2458624, 614656, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_728 = rand_strided((512, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_729 = rand_strided((512, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_732 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_119 = rand_strided((8, 16, 196, 1), (3136, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_736 = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_740 = rand_strided((512, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_120 = rand_strided((8, 16, 196, 1), (3136, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_744 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_749 = rand_strided((512, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_750 = rand_strided((512, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_47 = rand_strided((8, 4, 16, 196, 196), (2458624, 614656, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_751 = rand_strided((512, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_752 = rand_strided((512, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_755 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    div_121 = rand_strided((8, 16, 196, 1), (3136, 196, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_2, primals_4, primals_6, primals_8, primals_10, primals_13, primals_15, primals_17, primals_19, primals_21, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_38, primals_40, primals_42, primals_44, primals_46, primals_48, primals_50, primals_52, primals_54, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_106, primals_124, primals_142, primals_306, mul, view_2, view_12, mul_4, view_14, addmm_2, view_16, mul_9, view_18, view_28, bernoulli, mul_14, view_30, addmm_6, view_32, bernoulli_1, permute_17, mul_20, constant_pad_nd, getitem_17, mul_22, view_38, view_48, bernoulli_2, mul_27, view_50, addmm_10, view_52, bernoulli_3, mul_33, view_54, view_64, bernoulli_4, mul_38, view_66, addmm_14, view_68, bernoulli_5, permute_37, mul_44, constant_pad_nd_1, getitem_35, mul_46, view_74, view_84, bernoulli_6, mul_51, view_86, addmm_18, view_88, bernoulli_7, mul_57, view_90, view_100, bernoulli_8, mul_62, view_102, addmm_22, view_104, bernoulli_9, mul_68, view_106, view_116, bernoulli_10, mul_73, view_118, addmm_26, view_120, bernoulli_11, mul_79, view_122, view_132, bernoulli_12, mul_84, view_134, addmm_30, view_136, bernoulli_13, mul_90, view_138, view_148, bernoulli_14, mul_95, view_150, addmm_34, view_152, bernoulli_15, mul_101, view_154, view_164, bernoulli_16, mul_106, view_166, addmm_38, view_168, bernoulli_17, mul_112, view_170, view_180, bernoulli_18, mul_117, view_182, addmm_42, view_184, bernoulli_19, mul_123, view_186, view_196, bernoulli_20, mul_128, view_198, addmm_46, view_200, bernoulli_21, mul_134, view_202, view_212, bernoulli_22, mul_139, view_214, addmm_50, view_216, bernoulli_23, mul_145, view_218, view_228, bernoulli_24, mul_150, view_230, addmm_54, view_232, bernoulli_25, mul_156, view_234, view_244, bernoulli_26, mul_161, view_246, addmm_58, view_248, bernoulli_27, mul_167, view_250, view_260, bernoulli_28, mul_172, view_262, addmm_62, view_264, bernoulli_29, mul_178, view_266, view_276, bernoulli_30, mul_183, view_278, addmm_66, view_280, bernoulli_31, mul_189, view_282, view_292, bernoulli_32, mul_194, view_294, addmm_70, view_296, bernoulli_33, mul_200, view_298, view_308, bernoulli_34, mul_205, view_310, addmm_74, view_312, bernoulli_35, mul_211, view_314, view_324, bernoulli_36, mul_216, view_326, addmm_78, view_328, bernoulli_37, mul_222, view_330, view_340, bernoulli_38, mul_227, view_342, addmm_82, view_344, bernoulli_39, mul_233, view_346, view_356, bernoulli_40, mul_238, view_358, addmm_86, view_360, bernoulli_41, mul_244, view_362, view_372, bernoulli_42, mul_249, view_374, addmm_90, view_376, bernoulli_43, mul_255, view_378, view_388, bernoulli_44, mul_260, view_390, addmm_94, view_392, bernoulli_45, mul_266, clone_174, permute_187, div_71, permute_195, permute_199, div_72, permute_203, permute_208, permute_209, alias_24, permute_210, permute_211, permute_214, div_73, permute_218, permute_222, div_74, permute_226, permute_231, permute_232, alias_25, permute_233, permute_234, permute_237, div_75, permute_241, permute_245, div_76, permute_249, permute_254, permute_255, alias_26, permute_256, permute_257, permute_260, div_77, permute_264, permute_268, div_78, permute_272, permute_277, permute_278, alias_27, permute_279, permute_280, permute_283, div_79, permute_287, permute_291, div_80, permute_295, permute_300, permute_301, alias_28, permute_302, permute_303, permute_306, div_81, permute_310, permute_314, div_82, permute_318, permute_323, permute_324, alias_29, permute_325, permute_326, permute_329, div_83, permute_333, permute_337, div_84, permute_341, permute_346, permute_347, alias_30, permute_348, permute_349, permute_352, div_85, permute_356, permute_360, div_86, permute_364, permute_369, permute_370, alias_31, permute_371, permute_372, permute_375, div_87, permute_379, permute_383, div_88, permute_387, permute_392, permute_393, alias_32, permute_394, permute_395, permute_398, div_89, permute_402, permute_406, div_90, permute_410, permute_415, permute_416, alias_33, permute_417, permute_418, permute_421, div_91, permute_425, permute_429, div_92, permute_433, permute_438, permute_439, alias_34, permute_440, permute_441, permute_444, div_93, permute_448, permute_452, div_94, permute_456, permute_461, permute_462, alias_35, permute_463, permute_464, permute_467, div_95, permute_471, permute_475, div_96, permute_479, permute_484, permute_485, alias_36, permute_486, permute_487, permute_490, div_97, permute_494, permute_498, div_98, permute_502, permute_507, permute_508, alias_37, permute_509, permute_510, permute_513, div_99, permute_517, permute_521, div_100, permute_525, permute_530, permute_531, alias_38, permute_532, permute_533, permute_536, div_101, permute_540, permute_544, div_102, permute_548, permute_553, permute_554, alias_39, permute_555, permute_556, permute_559, div_103, permute_563, permute_567, div_104, permute_571, permute_576, permute_577, alias_40, permute_578, permute_579, permute_582, div_105, permute_586, permute_590, div_106, permute_594, permute_599, permute_600, alias_41, permute_601, permute_602, permute_605, div_107, permute_609, permute_613, div_108, permute_617, permute_622, permute_623, alias_42, permute_624, permute_625, permute_628, div_109, permute_632, permute_636, div_110, permute_640, permute_645, permute_646, alias_43, permute_647, permute_648, permute_651, div_111, div_112, permute_661, permute_665, div_113, permute_669, permute_674, permute_675, alias_44, permute_676, permute_677, permute_680, div_114, permute_684, permute_688, div_115, permute_692, permute_697, permute_698, alias_45, permute_699, permute_700, permute_703, div_116, div_117, permute_713, permute_717, div_118, permute_721, permute_726, permute_727, alias_46, permute_728, permute_729, permute_732, div_119, permute_736, permute_740, div_120, permute_744, permute_749, permute_750, alias_47, permute_751, permute_752, permute_755, div_121, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('jx_nest_base', benchmark_compiled_module)
