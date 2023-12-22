
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


# kernel path: /tmp/torchinductor_youkaichao/4l/c4l6nym44ckwu5dfrcwzq7ufzxyawpnxuer2z37u5x4yfspcxwc2.py
# Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_native_dropout_backward_native_layer_norm_backward_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_dropout_backward_native_layer_norm_backward_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1024
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
    tmp20 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
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
    tmp21 = tmp20.to(tl.float32)
    tmp22 = 1.1111111111111112
    tmp23 = tmp21 * tmp22
    tmp24 = tmp19 * tmp23
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp19, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp24, rmask & xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/rj/crjnqilmxspe22xbr574ibkyuvv62olshvlvfkppny55ijrnl6k6.py
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
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp7 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sd/csdinqrmfvsvaxzukno3j4cbhsyg2de3mkdpkyg24a3to77yw6yp.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
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
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pc/cpcjw4rnkkm4qkqfuwnf6k7ufujcamdmo6mnipqxg4uouddnkqme.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_3', 'mutated_arg_names': []}
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
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k5/ck5z4mso6o7enss2xnmvvbsyqoh6tl6sh4i2qodfimxljk6kqdos.py
# Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu, aten.gelu_backward]
# intermediate_output_11 => add_128, erf_11, mul_92
triton_poi_fused_gelu_gelu_backward_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
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


# kernel path: /tmp/torchinductor_youkaichao/ua/cuatikuospdlskifpprzaus533qacimaaslviscan33zidkwhr5u.py
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (3072*r2) + (393216*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cv/ccvyszilu2lq7qfjbbxt3jfwya2iqxnyxks7eeab2m4eebq35lbe.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 8
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


# kernel path: /tmp/torchinductor_youkaichao/sl/cslo2oeqx5q5q7rbdpz2crsllrzhnee62rsrv5nw4orktwbbb3sb.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1024
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
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tmp4 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = 768.0
    tmp17 = tmp4 * tmp16
    tmp18 = tmp17 - tmp8
    tmp19 = tmp9 * tmp14
    tmp20 = tmp18 - tmp19
    tmp21 = tmp15 * tmp20
    tmp23 = tmp22.to(tl.float32)
    tmp24 = 1.1111111111111112
    tmp25 = tmp23 * tmp24
    tmp26 = tmp21 * tmp25
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp26, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cd/ccduhhdr4avfmnrsw62iu4bzc3omqygb726izr53gnnv4kpe77n7.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp9 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr1 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ln/clnepi3zhvotea6npw72ahrwtkvlc4zckdryyxqnm6il3l3lpyoe.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 1024
    x2 = (xindex // 65536)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (768*x1)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cx/ccxk547av4id44wnbp4ggkjhw5g7fckz6vqvwz5wziy7udgcmr6d.py
# Source Nodes: [], Original ATen: [aten.arange]

triton_poi_fused_arange_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/om/comeu7ltfq7maiaiuffktjclugjrixdh3437lda3wyvonaga7c4t.py
# Source Nodes: [], Original ATen: [aten.index_add, aten.new_zeros]

triton_poi_fused_index_add_new_zeros_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_add_new_zeros_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6l/c6llv2dgtpbgxlxozfix5fbfsrrrcom7kq5kesfkvb6pvi356jxn.py
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
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 49152
    x1 = (xindex // 49152) % 4
    x2 = (xindex // 196608)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16384*x1) + (98304*x2)), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yl/cylvblhzcyu3v7jzlwyjwj7xqojhnn3hxtyxhnp5y5foeas7ojzb.py
# Source Nodes: [tril], Original ATen: [aten._softmax_backward_data, aten.copy, aten.masked_fill, aten.native_dropout_backward, aten.tril]
# tril => full_default_1
triton_red_fused__softmax_backward_data_copy_masked_fill_native_dropout_backward_tril_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[16384, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_backward_data_copy_masked_fill_native_dropout_backward_tril_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 513
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 12)
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last').to(tl.int1)
    x0 = xindex % 12
    x3 = xindex
    _tmp31 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp22 = tl.load(in_ptr2 + (r2 + (513*x3)), rmask, eviction_policy='evict_last').to(tl.int1)
        tmp28 = tl.load(in_ptr3 + (r2 + (513*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = r2
        tmp2 = tl.full([1, 1], 770, tl.int64)
        tmp3 = tmp1 < tmp2
        tmp4 = r2 + (770*(x1 % 256))
        tmp5 = tl.full([1, 1], 196864, tl.int64)
        tmp6 = tmp4 < tmp5
        tmp7 = tmp6 & tmp3
        tmp8 = (r2 + (770*(x1 % 256))) % 769
        tmp9 = tl.full([1, 1], 768, tl.int64)
        tmp10 = tmp8 < tmp9
        tmp11 = tmp10 & tmp7
        tmp12 = tl.load(in_ptr1 + ((768*(((r2 + (770*(x1 % 256))) // 769) % 256)) + (196608*(x1 // 256)) + (786432*x0) + ((r2 + (770*(x1 % 256))) % 769)), rmask & tmp11, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
        tmp14 = tl.where(tmp11, tmp12, tmp13)
        tmp15 = 0.0
        tmp16 = tl.where(tmp10, tmp14, tmp15)
        tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
        tmp18 = tl.where(tmp7, tmp16, tmp17)
        tmp19 = tl.where(tmp6, tmp18, tmp15)
        tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
        tmp21 = tl.where(tmp3, tmp19, tmp20)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = 1.1111111111111112
        tmp25 = tmp23 * tmp24
        tmp26 = tmp21 * tmp25
        tmp27 = tl.where(tmp0, tmp15, tmp26)
        tmp29 = tmp27 * tmp28
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp32 = _tmp31 + tmp30
        _tmp31 = tl.where(rmask, tmp32, _tmp31)
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp54 = tl.load(in_ptr2 + (r2 + (513*x3)), rmask, eviction_policy='evict_first').to(tl.int1)
        tmp60 = tl.load(in_ptr3 + (r2 + (513*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp33 = r2
        tmp34 = tl.full([1, 1], 770, tl.int64)
        tmp35 = tmp33 < tmp34
        tmp36 = r2 + (770*(x1 % 256))
        tmp37 = tl.full([1, 1], 196864, tl.int64)
        tmp38 = tmp36 < tmp37
        tmp39 = tmp38 & tmp35
        tmp40 = (r2 + (770*(x1 % 256))) % 769
        tmp41 = tl.full([1, 1], 768, tl.int64)
        tmp42 = tmp40 < tmp41
        tmp43 = tmp42 & tmp39
        tmp44 = tl.load(in_ptr1 + ((768*(((r2 + (770*(x1 % 256))) // 769) % 256)) + (196608*(x1 // 256)) + (786432*x0) + ((r2 + (770*(x1 % 256))) % 769)), rmask & tmp43, eviction_policy='evict_last', other=0.0)
        tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
        tmp46 = tl.where(tmp43, tmp44, tmp45)
        tmp47 = 0.0
        tmp48 = tl.where(tmp42, tmp46, tmp47)
        tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
        tmp50 = tl.where(tmp39, tmp48, tmp49)
        tmp51 = tl.where(tmp38, tmp50, tmp47)
        tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
        tmp53 = tl.where(tmp35, tmp51, tmp52)
        tmp55 = tmp54.to(tl.float32)
        tmp56 = 1.1111111111111112
        tmp57 = tmp55 * tmp56
        tmp58 = tmp53 * tmp57
        tmp59 = tl.where(tmp0, tmp47, tmp58)
        tmp61 = tmp59 * tmp60
        tmp62 = tmp60 * tmp31
        tmp63 = tmp61 - tmp62
        tl.store(out_ptr1 + (r2 + (513*x1) + (525312*x0)), tmp63, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yb/cybarytkqleigh2ovktxcfzqpr5qldn6zlqxvr6ywp72sikiufzx.py
# Source Nodes: [], Original ATen: [aten.as_strided_scatter]

triton_poi_fused_as_strided_scatter_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_as_strided_scatter_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6303744
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zo/czo5y7rewqcwcrqc5fy7qcimntthy5l5brus7d7est5qljx7acex.py
# Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.clone, aten.copy]

triton_poi_fused_as_strided_scatter_clone_copy_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_as_strided_scatter_clone_copy_15', 'mutated_arg_names': ['out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6303744
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
    tl.store(out_ptr2 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ou/couda4acz5yxtfp2moworydgo4efd6rou3f4poehurwpt6hmw5xs.py
# Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]

triton_poi_fused_as_strided_scatter_copy_zeros_like_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_as_strided_scatter_copy_zeros_like_16', 'mutated_arg_names': ['out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6303744
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x2 = (xindex // 513) % 1024
    x1 = xindex % 513
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = x2
    tmp2 = tl.full([1], 768, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = x1
    tmp5 = tl.full([1], 256, tl.int64)
    tmp6 = tmp4 >= tmp5
    tmp7 = tmp6 & tmp3
    tmp8 = tl.load(in_ptr1 + ((-197632) + x1 + (257*x2)), tmp7, eviction_policy='evict_last', other=0.0)
    tmp9 = (tmp8 != 0)
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp10, tmp0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp7, tmp11, tmp12)
    tmp14 = tl.where(tmp6, tmp13, tmp10)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp3, tmp14, tmp15)
    tmp17 = tl.where(tmp3, tmp16, tmp10)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
    tl.store(out_ptr2 + (x0), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wf/cwfbfrlzhd5vearuj5rzzq64z4ombwuersofsaz34oenembd4tyl.py
# Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]

triton_poi_fused_as_strided_scatter_copy_zeros_like_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_as_strided_scatter_copy_zeros_like_17', 'mutated_arg_names': ['out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 789504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 257
    x1 = (xindex // 257) % 256
    x2 = (xindex // 65792)
    tmp0 = 0.0
    tl.store(out_ptr0 + (394240 + x0 + (513*x1) + (525312*x2)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jy/cjyh225detfv7opqttu2qdlk46cvnr4f5nnnv7ocdsdfxmdxijsm.py
# Source Nodes: [], Original ATen: [aten.as_strided_scatter]

triton_poi_fused_as_strided_scatter_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_as_strided_scatter_18', 'mutated_arg_names': ['out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6303744
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 513
    x1 = (xindex // 513) % 12
    x2 = (xindex // 6156)
    tmp0 = tl.load(in_ptr0 + (x0 + (513*x2) + (525312*x1)), None)
    tl.store(out_ptr0 + (x0 + (513*x2) + (525312*x1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3m/c3mfypg5ig6xsjjx6mpwcz24xsiva3alxabpx7u7pvil4vuhcl5e.py
# Source Nodes: [], Original ATen: [aten.add, aten.as_strided_scatter, aten.copy, aten.zeros_like]

triton_poi_fused_add_as_strided_scatter_copy_zeros_like_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_as_strided_scatter_copy_zeros_like_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6303744
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
    tl.store(out_ptr1 + (x0), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/s3/cs3wgolol3cvcfcnjuzh3q3g4oomuasqriiksdarqfgj5l6bbva6.py
# Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]

triton_poi_fused_as_strided_scatter_copy_zeros_like_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_as_strided_scatter_copy_zeros_like_20', 'mutated_arg_names': ['out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 789504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 257
    x1 = (xindex // 257) % 256
    x2 = (xindex // 65792)
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + (513*x1) + (525312*x2)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r7/cr7johbkibiegn35mhs5auf6p5t2zjrd3we5lpho4ymbvdgnzynu.py
# Source Nodes: [], Original ATen: [aten.add, aten.as_strided_scatter, aten.copy, aten.zeros_like]

triton_poi_fused_add_as_strided_scatter_copy_zeros_like_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_as_strided_scatter_copy_zeros_like_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6303744
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 513) % 1024
    x0 = xindex % 513
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = x1
    tmp2 = tl.full([1], 256, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = x0
    tmp5 = tl.full([1], 257, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tmp6 & tmp3
    tmp8 = tl.load(in_ptr1 + (x0 + (257*x1)), tmp7, eviction_policy='evict_last', other=0.0)
    tmp9 = (tmp8 != 0)
    tmp10 = tl.load(in_out_ptr0 + (x3), tmp7, other=0.0)
    tmp11 = 0.0
    tmp12 = tl.where(tmp9, tmp11, tmp10)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp7, tmp12, tmp13)
    tmp15 = tl.where(tmp6, tmp14, tmp11)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp3, tmp15, tmp16)
    tmp18 = tl.where(tmp3, tmp17, tmp11)
    tmp19 = tmp0 + tmp18
    tl.store(in_out_ptr0 + (x3), tmp19, None)
    tl.store(out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nt/cntdzzjnlzttmzijvoykvmegcrbthexpelk6lbzpnt7tzkh3qomt.py
# Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]

triton_poi_fused_as_strided_scatter_copy_zeros_like_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0,), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_as_strided_scatter_copy_zeros_like_22', 'mutated_arg_names': ['out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 780300
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 255
    x1 = (xindex // 255) % 255
    x2 = (xindex // 65025)
    tmp0 = 0.0
    tl.store(out_ptr0 + (514 + x0 + (513*x1) + (525312*x2)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/33/c33fgf5cbs3w7q42mfty26t73geke7u6ic7hb4dvnzvmhvq4w3qj.py
# Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]

triton_poi_fused_as_strided_scatter_copy_zeros_like_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_as_strided_scatter_copy_zeros_like_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6303744
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/av/cavfc6zhzikdruawuag3abrfewj25g23obaqt3bkftbvzdton5jr.py
# Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]

triton_poi_fused_as_strided_scatter_copy_zeros_like_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_as_strided_scatter_copy_zeros_like_24', 'mutated_arg_names': ['out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256) % 768
    x2 = (xindex // 196608)
    tmp0 = 0.0
    tl.store(out_ptr0 + (131328 + x0 + (513*x1) + (525312*x2)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/or/cor3dkztavce3qea6kjgwcioathql4uctfv3b7l7pmoly6ewucsa.py
# Source Nodes: [], Original ATen: [aten.add, aten.clone, aten.select_backward, aten.slice_backward]

triton_poi_fused_add_clone_select_backward_slice_backward_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_select_backward_slice_backward_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9455616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 262656) % 3
    x1 = (xindex // 513) % 512
    x0 = xindex % 513
    x3 = (xindex // 787968)
    x4 = xindex % 262656
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x1
    tmp4 = tl.full([1], 255, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = x0
    tmp7 = tl.full([1], 258, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tmp8 & tmp5
    tmp10 = tl.load(in_ptr0 + (256 + x4 + (525312*x3)), tmp9, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = 0.0
    tmp14 = tl.where(tmp8, tmp12, tmp13)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp5, tmp14, tmp15)
    tmp17 = tl.where(tmp5, tmp16, tmp13)
    tmp18 = tl.where(tmp2, tmp17, tmp13)
    tmp19 = tmp3 >= tmp4
    tmp20 = tl.full([1], 511, tl.int64)
    tmp21 = tmp3 < tmp20
    tmp22 = tmp19 & tmp21
    tmp23 = tl.full([1], 257, tl.int64)
    tmp24 = tmp6 >= tmp23
    tmp25 = tmp24 & tmp22
    tmp26 = tl.load(in_ptr1 + (256 + x4 + (131328*x2) + (525312*x3)), tmp25, other=0.0)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp25, tmp26, tmp27)
    tmp29 = tl.where(tmp24, tmp28, tmp13)
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp22, tmp29, tmp30)
    tmp32 = tl.where(tmp22, tmp31, tmp13)
    tmp33 = tmp18 + tmp32
    tmp34 = tl.full([1], 2, tl.int32)
    tmp35 = tmp0 == tmp34
    tmp36 = tl.full([1], 256, tl.int64)
    tmp37 = tmp3 >= tmp36
    tmp38 = tmp6 < tmp23
    tmp39 = tmp38 & tmp37
    tmp40 = tl.load(in_ptr2 + (262912 + x4 + (525312*x3)), tmp39, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp39, tmp40, tmp41)
    tmp43 = tl.where(tmp38, tmp42, tmp13)
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp37, tmp43, tmp44)
    tmp46 = tl.where(tmp37, tmp45, tmp13)
    tmp47 = tl.where(tmp35, tmp46, tmp13)
    tmp48 = tmp33 + tmp47
    tl.store(out_ptr0 + (x5), tmp48, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tr/ctr4f4us64v7jabd6dr6p2jckdh4xejemjgkdzws7vpkszeinuz7.py
# Source Nodes: [], Original ATen: [aten.constant_pad_nd]

triton_poi_fused_constant_pad_nd_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9437184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 512) % 512
    x2 = (xindex // 262144) % 3
    x3 = (xindex // 786432)
    x4 = xindex % 262144
    x6 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 513, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x4 + (262656*x2) + (787968*x3) + (787968*((x4 + (262656*x2)) // 787968))), tmp2, other=0.0)
    tmp4 = (x4 // 513)
    tmp5 = tl.full([1], 256, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tmp6 & tmp2
    tmp8 = x4 % 513
    tmp9 = tl.full([1], 257, tl.int64)
    tmp10 = tmp8 < tmp9
    tmp11 = tmp10 & tmp7
    tmp12 = tl.load(in_ptr1 + (256 + x4 + (131328*x2) + (525312*x3) + (525312*((x4 + (262656*x2)) // 787968))), tmp11, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = 0.0
    tmp16 = tl.where(tmp10, tmp14, tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp7, tmp16, tmp17)
    tmp19 = tl.where(tmp6, tmp18, tmp15)
    tmp20 = tmp3 + tmp19
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp2, tmp20, tmp21)
    tl.store(out_ptr0 + (x6), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tq/ctqr5kb5ntc2cpjfa3suxn2mfr6mak4ppfg3qnhu7rtakf3dy2qd.py
# Source Nodes: [], Original ATen: [aten.arange]

triton_poi_fused_arange_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sh/cshyglc5n7tndga5zfos7q2ripe5lg4frxk76lvvbrbhomzrbn6v.py
# Source Nodes: [], Original ATen: [aten.index_add]

triton_poi_fused_index_add_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_add_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/of/cofni25k6pa4t3i6vlp7vch5we24gb4fuuts5gukn7c3vw3kr7oz.py
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
    size_hints=[32768, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 18432
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (512*x2) + (32768*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cf/ccfy4iyhdv2crz5s3gqd7pp3hqtg4z5g3s3rzt4rs7daok3jfjxz.py
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
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768) % 3
    x3 = (xindex // 98304)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x3) + (768*x1) + (196608*x2)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qz/cqzzlr7qklth3eibvcwxushxgqx2q66fabztgsss2liqubtxyx4l.py
# Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.div]

triton_poi_fused_as_strided_scatter_div_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_as_strided_scatter_div_31', 'mutated_arg_names': ['out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 8.0
    tmp2 = tmp0 / tmp1
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp2, None)
    tl.store(out_ptr2 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dh/cdhe3iaoun2dlbl6doiplboangns3gaqsps6w5qvoy63z4mdzamv.py
# Source Nodes: [], Original ATen: [aten.as_strided_scatter]

triton_poi_fused_as_strided_scatter_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_as_strided_scatter_32', 'mutated_arg_names': ['out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xx/cxxityv3hj3azhhmebbklj6uzupl67afe5emgf2udhzuisiosyvc.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 768)
    x0 = xindex % 768
    x2 = xindex
    tmp0 = 256 + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1536, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (16384 + (64*x1) + (98304*(x0 // 64)) + (x0 % 64)), tmp5, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tl.store(out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/65/c65cjveu75swfzcpeuhsxyjbv3zjwozi4ke2axhemzawl32d7ife.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 1024
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
    tmp2 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr7 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = tmp8 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = 768.0
    tmp21 = tmp8 * tmp20
    tmp22 = tmp21 - tmp12
    tmp23 = tmp13 * tmp18
    tmp24 = tmp22 - tmp23
    tmp25 = tmp19 * tmp24
    tmp27 = tmp26.to(tl.float32)
    tmp28 = 1.1111111111111112
    tmp29 = tmp27 * tmp28
    tmp30 = tmp25 * tmp29
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp25, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp30, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zx/czxkiqxodjcv2ssaeu4yxj4awo3rdln4dguqerjnxmxnxw2xso2r.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr3 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr4 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 + tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tl.store(out_ptr0 + (x0), tmp12, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rn/crnzky7c266adf47xjsir4t55mvozg7ycxqw2buoggdsehqqzwjs.py
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
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_36', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tl.load(in_ptr1 + (x0), None)
    tmp4 = tl.load(in_ptr2 + (x0), None)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 + tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_9, primals_15, primals_25, primals_31, primals_41, primals_47, primals_57, primals_63, primals_73, primals_79, primals_89, primals_95, primals_105, primals_111, primals_121, primals_127, primals_137, primals_143, primals_153, primals_159, primals_169, primals_175, primals_185, primals_191, view, slice_64, rev_1, unsqueeze_16, getitem_1, view_69, getitem_3, mul_1, view_71, addmm_4, view_73, getitem_7, mul_6, view_75, getitem_11, view_144, getitem_13, mul_9, view_146, addmm_10, view_148, getitem_17, mul_14, view_150, getitem_21, view_219, getitem_23, mul_17, view_221, addmm_16, view_223, getitem_27, mul_22, view_225, getitem_31, view_294, getitem_33, mul_25, view_296, addmm_22, view_298, getitem_37, mul_30, view_300, getitem_41, view_369, getitem_43, mul_33, view_371, addmm_28, view_373, getitem_47, mul_38, view_375, getitem_51, view_444, getitem_53, mul_41, view_446, addmm_34, view_448, getitem_57, mul_46, view_450, getitem_61, view_519, getitem_63, mul_49, view_521, addmm_40, view_523, getitem_67, mul_54, view_525, getitem_71, view_594, getitem_73, mul_57, view_596, addmm_46, view_598, getitem_77, mul_62, view_600, getitem_81, view_669, getitem_83, mul_65, view_671, addmm_52, view_673, getitem_87, mul_70, view_675, getitem_91, view_744, getitem_93, mul_73, view_746, addmm_58, view_748, getitem_97, mul_78, view_750, getitem_101, view_819, getitem_103, mul_81, view_821, addmm_64, view_823, getitem_107, mul_86, view_825, getitem_111, view_894, getitem_113, mul_89, view_896, addmm_70, view_898, getitem_117, mul_94, div_120, permute_756, permute_760, div_121, permute_764, permute_772, permute_773, alias_12, permute_783, permute_784, permute_795, permute_799, permute_808, div_123, permute_814, permute_818, div_124, permute_822, permute_830, permute_831, alias_13, permute_841, permute_842, permute_853, permute_857, permute_866, div_126, permute_872, permute_876, div_127, permute_880, permute_888, permute_889, alias_14, permute_899, permute_900, permute_911, permute_915, permute_924, div_129, permute_930, permute_934, div_130, permute_938, permute_946, permute_947, alias_15, permute_957, permute_958, permute_969, permute_973, permute_982, div_132, permute_988, permute_992, div_133, permute_996, permute_1004, permute_1005, alias_16, permute_1015, permute_1016, permute_1027, permute_1031, permute_1040, div_135, permute_1046, permute_1050, div_136, permute_1054, permute_1062, permute_1063, alias_17, permute_1073, permute_1074, permute_1085, permute_1089, permute_1098, div_138, permute_1104, permute_1108, div_139, permute_1112, permute_1120, permute_1121, alias_18, permute_1131, permute_1132, permute_1143, permute_1147, permute_1156, div_141, permute_1162, permute_1166, div_142, permute_1170, permute_1178, permute_1179, alias_19, permute_1189, permute_1190, permute_1201, permute_1205, permute_1214, div_144, permute_1220, permute_1224, div_145, permute_1228, permute_1236, permute_1237, alias_20, permute_1247, permute_1248, permute_1259, permute_1263, permute_1272, div_147, permute_1278, permute_1282, div_148, permute_1286, permute_1294, permute_1295, alias_21, permute_1305, permute_1306, permute_1317, permute_1321, permute_1330, div_150, permute_1336, permute_1340, div_151, permute_1344, permute_1352, permute_1353, alias_22, permute_1363, permute_1364, permute_1375, permute_1379, permute_1388, div_153, permute_1394, permute_1398, div_154, permute_1402, permute_1410, permute_1411, alias_23, permute_1421, permute_1422, permute_1433, permute_1437, permute_1446, tangents_1 = args
    args.clear()
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_25, (768, ), (1, ))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_57, (768, ), (1, ))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_73, (768, ), (1, ))
    assert_size_stride(primals_79, (768, ), (1, ))
    assert_size_stride(primals_89, (768, ), (1, ))
    assert_size_stride(primals_95, (768, ), (1, ))
    assert_size_stride(primals_105, (768, ), (1, ))
    assert_size_stride(primals_111, (768, ), (1, ))
    assert_size_stride(primals_121, (768, ), (1, ))
    assert_size_stride(primals_127, (768, ), (1, ))
    assert_size_stride(primals_137, (768, ), (1, ))
    assert_size_stride(primals_143, (768, ), (1, ))
    assert_size_stride(primals_153, (768, ), (1, ))
    assert_size_stride(primals_159, (768, ), (1, ))
    assert_size_stride(primals_169, (768, ), (1, ))
    assert_size_stride(primals_175, (768, ), (1, ))
    assert_size_stride(primals_185, (768, ), (1, ))
    assert_size_stride(primals_191, (768, ), (1, ))
    assert_size_stride(view, (1024, 768), (768, 1))
    assert_size_stride(slice_64, (1, 256, 1, 257), (65792, 257, 257, 1))
    assert_size_stride(rev_1, (1, 256, 1, 257), (65792, 257, 257, 1))
    assert_size_stride(unsqueeze_16, (1, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(getitem_1, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(view_69, (1024, 768), (768, 1))
    assert_size_stride(getitem_3, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_1, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_71, (1024, 768), (768, 1))
    assert_size_stride(addmm_4, (1024, 3072), (3072, 1))
    assert_size_stride(view_73, (1024, 3072), (3072, 1))
    assert_size_stride(getitem_7, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_6, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_75, (1024, 768), (768, 1))
    assert_size_stride(getitem_11, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(view_144, (1024, 768), (768, 1))
    assert_size_stride(getitem_13, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_9, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_146, (1024, 768), (768, 1))
    assert_size_stride(addmm_10, (1024, 3072), (3072, 1))
    assert_size_stride(view_148, (1024, 3072), (3072, 1))
    assert_size_stride(getitem_17, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_14, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_150, (1024, 768), (768, 1))
    assert_size_stride(getitem_21, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(view_219, (1024, 768), (768, 1))
    assert_size_stride(getitem_23, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_17, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_221, (1024, 768), (768, 1))
    assert_size_stride(addmm_16, (1024, 3072), (3072, 1))
    assert_size_stride(view_223, (1024, 3072), (3072, 1))
    assert_size_stride(getitem_27, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_22, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_225, (1024, 768), (768, 1))
    assert_size_stride(getitem_31, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(view_294, (1024, 768), (768, 1))
    assert_size_stride(getitem_33, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_25, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_296, (1024, 768), (768, 1))
    assert_size_stride(addmm_22, (1024, 3072), (3072, 1))
    assert_size_stride(view_298, (1024, 3072), (3072, 1))
    assert_size_stride(getitem_37, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_30, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_300, (1024, 768), (768, 1))
    assert_size_stride(getitem_41, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(view_369, (1024, 768), (768, 1))
    assert_size_stride(getitem_43, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_33, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_371, (1024, 768), (768, 1))
    assert_size_stride(addmm_28, (1024, 3072), (3072, 1))
    assert_size_stride(view_373, (1024, 3072), (3072, 1))
    assert_size_stride(getitem_47, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_38, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_375, (1024, 768), (768, 1))
    assert_size_stride(getitem_51, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(view_444, (1024, 768), (768, 1))
    assert_size_stride(getitem_53, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_41, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_446, (1024, 768), (768, 1))
    assert_size_stride(addmm_34, (1024, 3072), (3072, 1))
    assert_size_stride(view_448, (1024, 3072), (3072, 1))
    assert_size_stride(getitem_57, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_46, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_450, (1024, 768), (768, 1))
    assert_size_stride(getitem_61, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(view_519, (1024, 768), (768, 1))
    assert_size_stride(getitem_63, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_49, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_521, (1024, 768), (768, 1))
    assert_size_stride(addmm_40, (1024, 3072), (3072, 1))
    assert_size_stride(view_523, (1024, 3072), (3072, 1))
    assert_size_stride(getitem_67, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_54, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_525, (1024, 768), (768, 1))
    assert_size_stride(getitem_71, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(view_594, (1024, 768), (768, 1))
    assert_size_stride(getitem_73, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_57, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_596, (1024, 768), (768, 1))
    assert_size_stride(addmm_46, (1024, 3072), (3072, 1))
    assert_size_stride(view_598, (1024, 3072), (3072, 1))
    assert_size_stride(getitem_77, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_62, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_600, (1024, 768), (768, 1))
    assert_size_stride(getitem_81, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(view_669, (1024, 768), (768, 1))
    assert_size_stride(getitem_83, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_65, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_671, (1024, 768), (768, 1))
    assert_size_stride(addmm_52, (1024, 3072), (3072, 1))
    assert_size_stride(view_673, (1024, 3072), (3072, 1))
    assert_size_stride(getitem_87, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_70, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_675, (1024, 768), (768, 1))
    assert_size_stride(getitem_91, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(view_744, (1024, 768), (768, 1))
    assert_size_stride(getitem_93, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_73, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_746, (1024, 768), (768, 1))
    assert_size_stride(addmm_58, (1024, 3072), (3072, 1))
    assert_size_stride(view_748, (1024, 3072), (3072, 1))
    assert_size_stride(getitem_97, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_78, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_750, (1024, 768), (768, 1))
    assert_size_stride(getitem_101, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(view_819, (1024, 768), (768, 1))
    assert_size_stride(getitem_103, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_81, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_821, (1024, 768), (768, 1))
    assert_size_stride(addmm_64, (1024, 3072), (3072, 1))
    assert_size_stride(view_823, (1024, 3072), (3072, 1))
    assert_size_stride(getitem_107, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_86, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_825, (1024, 768), (768, 1))
    assert_size_stride(getitem_111, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(view_894, (1024, 768), (768, 1))
    assert_size_stride(getitem_113, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_89, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_896, (1024, 768), (768, 1))
    assert_size_stride(addmm_70, (1024, 3072), (3072, 1))
    assert_size_stride(view_898, (1024, 3072), (3072, 1))
    assert_size_stride(getitem_117, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_94, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(div_120, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_756, (768, 3072), (3072, 1))
    assert_size_stride(permute_760, (3072, 768), (768, 1))
    assert_size_stride(div_121, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_764, (768, 768), (768, 1))
    assert_size_stride(permute_772, (48, 768, 256), (197120, 1, 769))
    assert_size_stride(permute_773, (48, 64, 768), (49152, 1, 64))
    assert_size_stride(alias_12, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(permute_783, (36, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_784, (36, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_795, (768, 768), (768, 1))
    assert_size_stride(permute_799, (768, 768), (768, 1))
    assert_size_stride(permute_808, (768, 768), (768, 1))
    assert_size_stride(div_123, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_814, (768, 3072), (3072, 1))
    assert_size_stride(permute_818, (3072, 768), (768, 1))
    assert_size_stride(div_124, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_822, (768, 768), (768, 1))
    assert_size_stride(permute_830, (48, 768, 256), (197120, 1, 769))
    assert_size_stride(permute_831, (48, 64, 768), (49152, 1, 64))
    assert_size_stride(alias_13, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(permute_841, (36, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_842, (36, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_853, (768, 768), (768, 1))
    assert_size_stride(permute_857, (768, 768), (768, 1))
    assert_size_stride(permute_866, (768, 768), (768, 1))
    assert_size_stride(div_126, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_872, (768, 3072), (3072, 1))
    assert_size_stride(permute_876, (3072, 768), (768, 1))
    assert_size_stride(div_127, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_880, (768, 768), (768, 1))
    assert_size_stride(permute_888, (48, 768, 256), (197120, 1, 769))
    assert_size_stride(permute_889, (48, 64, 768), (49152, 1, 64))
    assert_size_stride(alias_14, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(permute_899, (36, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_900, (36, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_911, (768, 768), (768, 1))
    assert_size_stride(permute_915, (768, 768), (768, 1))
    assert_size_stride(permute_924, (768, 768), (768, 1))
    assert_size_stride(div_129, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_930, (768, 3072), (3072, 1))
    assert_size_stride(permute_934, (3072, 768), (768, 1))
    assert_size_stride(div_130, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_938, (768, 768), (768, 1))
    assert_size_stride(permute_946, (48, 768, 256), (197120, 1, 769))
    assert_size_stride(permute_947, (48, 64, 768), (49152, 1, 64))
    assert_size_stride(alias_15, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(permute_957, (36, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_958, (36, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_969, (768, 768), (768, 1))
    assert_size_stride(permute_973, (768, 768), (768, 1))
    assert_size_stride(permute_982, (768, 768), (768, 1))
    assert_size_stride(div_132, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_988, (768, 3072), (3072, 1))
    assert_size_stride(permute_992, (3072, 768), (768, 1))
    assert_size_stride(div_133, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_996, (768, 768), (768, 1))
    assert_size_stride(permute_1004, (48, 768, 256), (197120, 1, 769))
    assert_size_stride(permute_1005, (48, 64, 768), (49152, 1, 64))
    assert_size_stride(alias_16, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(permute_1015, (36, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_1016, (36, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_1027, (768, 768), (768, 1))
    assert_size_stride(permute_1031, (768, 768), (768, 1))
    assert_size_stride(permute_1040, (768, 768), (768, 1))
    assert_size_stride(div_135, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_1046, (768, 3072), (3072, 1))
    assert_size_stride(permute_1050, (3072, 768), (768, 1))
    assert_size_stride(div_136, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_1054, (768, 768), (768, 1))
    assert_size_stride(permute_1062, (48, 768, 256), (197120, 1, 769))
    assert_size_stride(permute_1063, (48, 64, 768), (49152, 1, 64))
    assert_size_stride(alias_17, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(permute_1073, (36, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_1074, (36, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_1085, (768, 768), (768, 1))
    assert_size_stride(permute_1089, (768, 768), (768, 1))
    assert_size_stride(permute_1098, (768, 768), (768, 1))
    assert_size_stride(div_138, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_1104, (768, 3072), (3072, 1))
    assert_size_stride(permute_1108, (3072, 768), (768, 1))
    assert_size_stride(div_139, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_1112, (768, 768), (768, 1))
    assert_size_stride(permute_1120, (48, 768, 256), (197120, 1, 769))
    assert_size_stride(permute_1121, (48, 64, 768), (49152, 1, 64))
    assert_size_stride(alias_18, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(permute_1131, (36, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_1132, (36, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_1143, (768, 768), (768, 1))
    assert_size_stride(permute_1147, (768, 768), (768, 1))
    assert_size_stride(permute_1156, (768, 768), (768, 1))
    assert_size_stride(div_141, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_1162, (768, 3072), (3072, 1))
    assert_size_stride(permute_1166, (3072, 768), (768, 1))
    assert_size_stride(div_142, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_1170, (768, 768), (768, 1))
    assert_size_stride(permute_1178, (48, 768, 256), (197120, 1, 769))
    assert_size_stride(permute_1179, (48, 64, 768), (49152, 1, 64))
    assert_size_stride(alias_19, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(permute_1189, (36, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_1190, (36, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_1201, (768, 768), (768, 1))
    assert_size_stride(permute_1205, (768, 768), (768, 1))
    assert_size_stride(permute_1214, (768, 768), (768, 1))
    assert_size_stride(div_144, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_1220, (768, 3072), (3072, 1))
    assert_size_stride(permute_1224, (3072, 768), (768, 1))
    assert_size_stride(div_145, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_1228, (768, 768), (768, 1))
    assert_size_stride(permute_1236, (48, 768, 256), (197120, 1, 769))
    assert_size_stride(permute_1237, (48, 64, 768), (49152, 1, 64))
    assert_size_stride(alias_20, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(permute_1247, (36, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_1248, (36, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_1259, (768, 768), (768, 1))
    assert_size_stride(permute_1263, (768, 768), (768, 1))
    assert_size_stride(permute_1272, (768, 768), (768, 1))
    assert_size_stride(div_147, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_1278, (768, 3072), (3072, 1))
    assert_size_stride(permute_1282, (3072, 768), (768, 1))
    assert_size_stride(div_148, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_1286, (768, 768), (768, 1))
    assert_size_stride(permute_1294, (48, 768, 256), (197120, 1, 769))
    assert_size_stride(permute_1295, (48, 64, 768), (49152, 1, 64))
    assert_size_stride(alias_21, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(permute_1305, (36, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_1306, (36, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_1317, (768, 768), (768, 1))
    assert_size_stride(permute_1321, (768, 768), (768, 1))
    assert_size_stride(permute_1330, (768, 768), (768, 1))
    assert_size_stride(div_150, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_1336, (768, 3072), (3072, 1))
    assert_size_stride(permute_1340, (3072, 768), (768, 1))
    assert_size_stride(div_151, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_1344, (768, 768), (768, 1))
    assert_size_stride(permute_1352, (48, 768, 256), (197120, 1, 769))
    assert_size_stride(permute_1353, (48, 64, 768), (49152, 1, 64))
    assert_size_stride(alias_22, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(permute_1363, (36, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_1364, (36, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_1375, (768, 768), (768, 1))
    assert_size_stride(permute_1379, (768, 768), (768, 1))
    assert_size_stride(permute_1388, (768, 768), (768, 1))
    assert_size_stride(div_153, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_1394, (768, 3072), (3072, 1))
    assert_size_stride(permute_1398, (3072, 768), (768, 1))
    assert_size_stride(div_154, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_1402, (768, 768), (768, 1))
    assert_size_stride(permute_1410, (48, 768, 256), (197120, 1, 769))
    assert_size_stride(permute_1411, (48, 64, 768), (49152, 1, 64))
    assert_size_stride(alias_23, (1, 1024, 12, 513), (6303744, 6156, 513, 1))
    assert_size_stride(permute_1421, (36, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_1422, (36, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_1433, (768, 768), (768, 1))
    assert_size_stride(permute_1437, (768, 768), (768, 1))
    assert_size_stride(permute_1446, (768, 768), (768, 1))
    assert_size_stride(tangents_1, (1, 1024, 768), (786432, 768, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf2 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf5 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]
        stream0 = get_cuda_stream(0)
        triton_per_fused_native_dropout_backward_native_layer_norm_backward_0.run(tangents_1, primals_191, mul_94, div_120, getitem_117, buf2, buf5, 1024, 768, grid=grid(1024), stream=stream0)
        del div_120
        del getitem_117
        del primals_191
        buf3 = empty((768, ), device='cuda', dtype=torch.float32)
        buf4 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_1.run(tangents_1, mul_94, buf3, buf4, 768, 1024, grid=grid(768), stream=stream0)
        del mul_94
        del tangents_1
        buf6 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (1024, 768), (768, 1), 0), permute_756, out=buf6)
        del permute_756
        buf7 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (768, 1024), (1, 768), 0), view_898, out=buf7)
        del view_898
        buf8 = empty_strided((1, 768, 8), (6144, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf5, buf8, 6144, 128, grid=grid(6144), stream=stream0)
        buf9 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf8, buf9, 768, 8, grid=grid(768), stream=stream0)
        buf10 = reinterpret_tensor(buf6, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf6  # reuse
        # Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_4.run(buf10, addmm_70, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_70
        buf11 = reinterpret_tensor(buf5, (1024, 768), (768, 1), 0); del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (1024, 3072), (3072, 1), 0), permute_760, out=buf11)
        del permute_760
        buf12 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (3072, 1024), (1, 3072), 0), view_896, out=buf12)
        del view_896
        buf13 = empty_strided((1, 3072, 8), (24576, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf10, buf13, 24576, 128, grid=grid(24576), stream=stream0)
        buf14 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf13, buf14, 3072, 8, grid=grid(3072), stream=stream0)
        buf17 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf20 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7.run(buf2, buf11, primals_185, mul_89, div_121, getitem_113, buf17, buf20, 1024, 768, grid=grid(1024), stream=stream0)
        del div_121
        del getitem_113
        del primals_185
        buf18 = empty((768, ), device='cuda', dtype=torch.float32)
        buf19 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf2, buf11, mul_89, buf18, buf19, 768, 1024, grid=grid(768), stream=stream0)
        del mul_89
        buf21 = reinterpret_tensor(buf2, (1024, 768), (768, 1), 0); del buf2  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (1024, 768), (768, 1), 0), permute_764, out=buf21)
        del permute_764
        buf22 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (768, 1024), (1, 768), 0), view_894, out=buf22)
        del view_894
        buf23 = buf8; del buf8  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf20, buf23, 6144, 128, grid=grid(6144), stream=stream0)
        buf24 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf23, buf24, 768, 8, grid=grid(768), stream=stream0)
        buf25 = reinterpret_tensor(buf20, (12, 4, 256, 1, 64), (65536, 16384, 64, 64, 1), 0); del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf21, buf25, 786432, grid=grid(786432), stream=stream0)
        buf26 = empty((48, 768, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_772, reinterpret_tensor(buf25, (48, 256, 64), (16384, 64, 1), 0), out=buf26)
        del permute_772
        buf27 = empty((48, 256, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf25, (48, 256, 64), (16384, 64, 1), 0), permute_773, out=buf27)
        del permute_773
        buf28 = empty((1179648, ), device='cuda', dtype=torch.int64)
        # Source Nodes: [], Original ATen: [aten.arange]
        triton_poi_fused_arange_10.run(buf28, 1179648, grid=grid(1179648), stream=stream0)
        buf29 = empty((1179648, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_add, aten.new_zeros]
        triton_poi_fused_index_add_new_zeros_11.run(buf29, 1179648, grid=grid(1179648), stream=stream0)
        buf30 = empty((12, 4, 768, 64), device='cuda', dtype=torch.int64)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_12.run(buf28, buf30, 2359296, grid=grid(2359296), stream=stream0)
        aten.index_put_(buf29, [reinterpret_tensor(buf30, (2359296, ), (1, ), 0)], reinterpret_tensor(buf26, (2359296, ), (1, ), 0), True)
        buf35 = empty_strided((1024, 12, 513), (513, 525312, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [tril], Original ATen: [aten._softmax_backward_data, aten.copy, aten.masked_fill, aten.native_dropout_backward, aten.tril]
        triton_red_fused__softmax_backward_data_copy_masked_fill_native_dropout_backward_tril_13.run(unsqueeze_16, buf27, getitem_111, alias_12, buf35, 12288, 513, grid=grid(12288), stream=stream0)
        del alias_12
        del getitem_111
        buf36 = empty((6303744, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_14.run(buf36, 6303744, grid=grid(6303744), stream=stream0)
        buf39 = empty_strided((1, 1024, 12, 513), (6303744, 513, 525312, 1), device='cuda', dtype=torch.float32)
        buf40 = empty((12, 4, 256, 513), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.clone, aten.copy]
        triton_poi_fused_as_strided_scatter_clone_copy_15.run(buf35, buf36, buf39, buf40, 6303744, grid=grid(6303744), stream=stream0)
        buf43 = reinterpret_tensor(buf36, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf36  # reuse
        buf45 = buf35; del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_16.run(buf39, rev_1, buf40, buf43, buf45, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_17.run(buf43, 789504, grid=grid(789504), stream=stream0)
        buf46 = reinterpret_tensor(buf40, (6303744, ), (1, ), 0); del buf40  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_14.run(buf46, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_18.run(buf45, buf46, 6303744, grid=grid(6303744), stream=stream0)
        buf48 = reinterpret_tensor(buf45, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf45  # reuse
        buf50 = reinterpret_tensor(buf39, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_add_as_strided_scatter_copy_zeros_like_19.run(buf43, buf46, buf48, buf50, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_20.run(buf50, 789504, grid=grid(789504), stream=stream0)
        buf52 = buf48; del buf48  # reuse
        buf54 = reinterpret_tensor(buf46, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_add_as_strided_scatter_copy_zeros_like_21.run(buf52, buf50, slice_64, buf54, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_22.run(buf54, 780300, grid=grid(780300), stream=stream0)
        buf57 = buf50; del buf50  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_23.run(buf54, buf57, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_24.run(buf57, 2359296, grid=grid(2359296), stream=stream0)
        buf60 = buf43; del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_23.run(buf57, buf60, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_17.run(buf60, 789504, grid=grid(789504), stream=stream0)
        buf62 = empty((12, 3, 512, 513), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.clone, aten.select_backward, aten.slice_backward]
        triton_poi_fused_add_clone_select_backward_slice_backward_25.run(buf52, buf54, buf57, buf62, 9455616, grid=grid(9455616), stream=stream0)
        buf64 = reinterpret_tensor(buf27, (12, 3, 512, 512), (786432, 262144, 512, 1), 0); del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_26.run(buf62, buf60, buf64, 9437184, grid=grid(9437184), stream=stream0)
        buf65 = empty((36, 64, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_783, reinterpret_tensor(buf64, (36, 512, 512), (262144, 512, 1), 0), out=buf65)
        del permute_783
        buf66 = empty((36, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf64, (36, 512, 512), (262144, 512, 1), 0), permute_784, out=buf66)
        del permute_784
        buf68 = empty((786432, ), device='cuda', dtype=torch.int64)
        # Source Nodes: [], Original ATen: [aten.arange]
        triton_poi_fused_arange_27.run(buf68, 786432, grid=grid(786432), stream=stream0)
        buf69 = reinterpret_tensor(buf25, (786432, ), (1, ), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add]
        triton_poi_fused_index_add_28.run(buf69, 786432, grid=grid(786432), stream=stream0)
        buf70 = empty((12, 3, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf65, buf70, 18432, 64, grid=grid(18432, 64), stream=stream0)
        buf71 = reinterpret_tensor(buf28, (12, 3, 512, 64), (98304, 32768, 64, 1), 0); del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_30.run(buf68, buf71, 1179648, grid=grid(1179648), stream=stream0)
        del buf68
        aten.index_put_(buf69, [reinterpret_tensor(buf71, (1179648, ), (1, ), 0)], reinterpret_tensor(buf70, (1179648, ), (1, ), 0), True)
        buf74 = reinterpret_tensor(buf21, (786432, ), (1, ), 0); del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add]
        triton_poi_fused_index_add_28.run(buf74, 786432, grid=grid(786432), stream=stream0)
        aten.index_put_(buf74, [reinterpret_tensor(buf71, (1179648, ), (1, ), 0)], reinterpret_tensor(buf66, (1179648, ), (1, ), 0), True)
        buf77 = reinterpret_tensor(buf11, (786432, ), (1, ), 0); del buf11  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_index_add_28.run(buf77, 786432, grid=grid(786432), stream=stream0)
        buf80 = empty((1024, 1, 768), device='cuda', dtype=torch.float32)
        buf81 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.div]
        triton_poi_fused_as_strided_scatter_div_31.run(buf74, buf77, buf80, buf81, 786432, grid=grid(786432), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_32.run(buf80, buf81, 786432, grid=grid(786432), stream=stream0)
        buf83 = reinterpret_tensor(buf80, (1024, 768), (768, 1), 0); del buf80  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_33.run(buf29, buf83, 786432, grid=grid(786432), stream=stream0)
        buf84 = reinterpret_tensor(buf77, (1024, 768), (768, 1), 0); del buf77  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf83, permute_795, out=buf84)
        del permute_795
        buf85 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf83, (768, 1024), (1, 768), 0), view_825, out=buf85)
        buf86 = buf23; del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf83, buf86, 6144, 128, grid=grid(6144), stream=stream0)
        buf87 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf86, buf87, 768, 8, grid=grid(768), stream=stream0)
        buf88 = buf83; del buf83  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (1024, 768), (768, 1), 0), permute_799, out=buf88)
        del permute_799
        buf89 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (768, 1024), (1, 768), 0), view_825, out=buf89)
        buf90 = buf86; del buf86  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf69, buf90, 6144, 128, grid=grid(6144), stream=stream0)
        buf91 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf90, buf91, 768, 8, grid=grid(768), stream=stream0)
        buf92 = reinterpret_tensor(buf69, (1024, 768), (768, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf81, permute_808, out=buf92)
        del permute_808
        buf93 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf81, (768, 1024), (1, 768), 0), view_825, out=buf93)
        del view_825
        buf94 = buf90; del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf81, buf94, 6144, 128, grid=grid(6144), stream=stream0)
        buf95 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf94, buf95, 768, 8, grid=grid(768), stream=stream0)
        buf99 = reinterpret_tensor(buf81, (1, 1024, 768), (786432, 768, 1), 0); del buf81  # reuse
        buf102 = reinterpret_tensor(buf74, (1, 1024, 768), (786432, 768, 1), 0); del buf74  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_34.run(buf17, buf84, buf88, buf92, primals_175, mul_86, div_123, getitem_107, buf99, buf102, 1024, 768, grid=grid(1024), stream=stream0)
        del div_123
        del getitem_107
        del primals_175
        buf100 = empty((768, ), device='cuda', dtype=torch.float32)
        buf101 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_35.run(buf17, buf84, buf88, buf92, mul_86, buf100, buf101, 768, 1024, grid=grid(768), stream=stream0)
        del mul_86
        buf103 = reinterpret_tensor(buf10, (1024, 3072), (3072, 1), 0); del buf10  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf102, (1024, 768), (768, 1), 0), permute_814, out=buf103)
        del permute_814
        buf104 = reinterpret_tensor(buf26, (768, 3072), (3072, 1), 0); del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf102, (768, 1024), (1, 768), 0), view_823, out=buf104)
        del view_823
        buf105 = buf94; del buf94  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf102, buf105, 6144, 128, grid=grid(6144), stream=stream0)
        buf106 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf105, buf106, 768, 8, grid=grid(768), stream=stream0)
        buf107 = reinterpret_tensor(buf103, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf103  # reuse
        # Source Nodes: [intermediate_output_10], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_4.run(buf107, addmm_64, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_64
        buf108 = reinterpret_tensor(buf102, (1024, 768), (768, 1), 0); del buf102  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf107, (1024, 3072), (3072, 1), 0), permute_818, out=buf108)
        del permute_818
        buf109 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf107, (3072, 1024), (1, 3072), 0), view_821, out=buf109)
        del view_821
        buf110 = buf13; del buf13  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf107, buf110, 24576, 128, grid=grid(24576), stream=stream0)
        buf111 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf110, buf111, 3072, 8, grid=grid(3072), stream=stream0)
        buf114 = reinterpret_tensor(buf92, (1, 1024, 768), (786432, 768, 1), 0); del buf92  # reuse
        buf117 = reinterpret_tensor(buf88, (1, 1024, 768), (786432, 768, 1), 0); del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7.run(buf99, buf108, primals_169, mul_81, div_124, getitem_103, buf114, buf117, 1024, 768, grid=grid(1024), stream=stream0)
        del div_124
        del getitem_103
        del primals_169
        buf115 = empty((768, ), device='cuda', dtype=torch.float32)
        buf116 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf99, buf108, mul_81, buf115, buf116, 768, 1024, grid=grid(768), stream=stream0)
        del mul_81
        buf118 = reinterpret_tensor(buf99, (1024, 768), (768, 1), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf117, (1024, 768), (768, 1), 0), permute_822, out=buf118)
        del permute_822
        buf119 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf117, (768, 1024), (1, 768), 0), view_819, out=buf119)
        del view_819
        buf120 = buf105; del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf117, buf120, 6144, 128, grid=grid(6144), stream=stream0)
        buf121 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf120, buf121, 768, 8, grid=grid(768), stream=stream0)
        buf122 = reinterpret_tensor(buf117, (12, 4, 256, 1, 64), (65536, 16384, 64, 64, 1), 0); del buf117  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf118, buf122, 786432, grid=grid(786432), stream=stream0)
        buf123 = empty((48, 768, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_830, reinterpret_tensor(buf122, (48, 256, 64), (16384, 64, 1), 0), out=buf123)
        del permute_830
        buf124 = reinterpret_tensor(buf64, (48, 256, 768), (196608, 768, 1), 0); del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf122, (48, 256, 64), (16384, 64, 1), 0), permute_831, out=buf124)
        del permute_831
        buf125 = buf29; del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add, aten.new_zeros]
        triton_poi_fused_index_add_new_zeros_11.run(buf125, 1179648, grid=grid(1179648), stream=stream0)
        aten.index_put_(buf125, [reinterpret_tensor(buf30, (2359296, ), (1, ), 0)], reinterpret_tensor(buf123, (2359296, ), (1, ), 0), True)
        buf129 = reinterpret_tensor(buf60, (1024, 12, 513), (513, 525312, 1), 0); del buf60  # reuse
        # Source Nodes: [tril], Original ATen: [aten._softmax_backward_data, aten.copy, aten.masked_fill, aten.native_dropout_backward, aten.tril]
        triton_red_fused__softmax_backward_data_copy_masked_fill_native_dropout_backward_tril_13.run(unsqueeze_16, buf124, getitem_101, alias_13, buf129, 12288, 513, grid=grid(12288), stream=stream0)
        del alias_13
        del getitem_101
        buf130 = reinterpret_tensor(buf57, (6303744, ), (1, ), 0); del buf57  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_14.run(buf130, 6303744, grid=grid(6303744), stream=stream0)
        buf133 = reinterpret_tensor(buf54, (1, 1024, 12, 513), (6303744, 513, 525312, 1), 0); del buf54  # reuse
        buf134 = buf52; del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.clone, aten.copy]
        triton_poi_fused_as_strided_scatter_clone_copy_15.run(buf129, buf130, buf133, buf134, 6303744, grid=grid(6303744), stream=stream0)
        buf137 = reinterpret_tensor(buf130, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf130  # reuse
        buf139 = buf129; del buf129  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_16.run(buf133, rev_1, buf134, buf137, buf139, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_17.run(buf137, 789504, grid=grid(789504), stream=stream0)
        buf140 = reinterpret_tensor(buf134, (6303744, ), (1, ), 0); del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_14.run(buf140, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_18.run(buf139, buf140, 6303744, grid=grid(6303744), stream=stream0)
        buf142 = reinterpret_tensor(buf139, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf139  # reuse
        buf144 = reinterpret_tensor(buf133, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_add_as_strided_scatter_copy_zeros_like_19.run(buf137, buf140, buf142, buf144, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_20.run(buf144, 789504, grid=grid(789504), stream=stream0)
        buf146 = buf142; del buf142  # reuse
        buf148 = reinterpret_tensor(buf140, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf140  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_add_as_strided_scatter_copy_zeros_like_21.run(buf146, buf144, slice_64, buf148, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_22.run(buf148, 780300, grid=grid(780300), stream=stream0)
        buf151 = buf144; del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_23.run(buf148, buf151, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_24.run(buf151, 2359296, grid=grid(2359296), stream=stream0)
        buf154 = buf137; del buf137  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_23.run(buf151, buf154, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_17.run(buf154, 789504, grid=grid(789504), stream=stream0)
        buf156 = buf62; del buf62  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.clone, aten.select_backward, aten.slice_backward]
        triton_poi_fused_add_clone_select_backward_slice_backward_25.run(buf146, buf148, buf151, buf156, 9455616, grid=grid(9455616), stream=stream0)
        buf158 = reinterpret_tensor(buf124, (12, 3, 512, 512), (786432, 262144, 512, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_26.run(buf156, buf154, buf158, 9437184, grid=grid(9437184), stream=stream0)
        buf159 = reinterpret_tensor(buf66, (36, 64, 512), (32768, 512, 1), 0); del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_841, reinterpret_tensor(buf158, (36, 512, 512), (262144, 512, 1), 0), out=buf159)
        del permute_841
        buf160 = reinterpret_tensor(buf70, (36, 512, 64), (32768, 64, 1), 0); del buf70  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf158, (36, 512, 512), (262144, 512, 1), 0), permute_842, out=buf160)
        del permute_842
        buf161 = reinterpret_tensor(buf122, (786432, ), (1, ), 0); del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add]
        triton_poi_fused_index_add_28.run(buf161, 786432, grid=grid(786432), stream=stream0)
        buf162 = reinterpret_tensor(buf65, (12, 3, 512, 64), (98304, 32768, 64, 1), 0); del buf65  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf159, buf162, 18432, 64, grid=grid(18432, 64), stream=stream0)
        aten.index_put_(buf161, [reinterpret_tensor(buf71, (1179648, ), (1, ), 0)], reinterpret_tensor(buf162, (1179648, ), (1, ), 0), True)
        buf165 = reinterpret_tensor(buf118, (786432, ), (1, ), 0); del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add]
        triton_poi_fused_index_add_28.run(buf165, 786432, grid=grid(786432), stream=stream0)
        aten.index_put_(buf165, [reinterpret_tensor(buf71, (1179648, ), (1, ), 0)], reinterpret_tensor(buf160, (1179648, ), (1, ), 0), True)
        buf168 = reinterpret_tensor(buf108, (786432, ), (1, ), 0); del buf108  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_index_add_28.run(buf168, 786432, grid=grid(786432), stream=stream0)
        buf171 = reinterpret_tensor(buf84, (1024, 1, 768), (768, 768, 1), 0); del buf84  # reuse
        buf172 = reinterpret_tensor(buf17, (1024, 768), (768, 1), 0); del buf17  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.div]
        triton_poi_fused_as_strided_scatter_div_31.run(buf165, buf168, buf171, buf172, 786432, grid=grid(786432), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_32.run(buf171, buf172, 786432, grid=grid(786432), stream=stream0)
        buf174 = reinterpret_tensor(buf171, (1024, 768), (768, 1), 0); del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_33.run(buf125, buf174, 786432, grid=grid(786432), stream=stream0)
        buf175 = reinterpret_tensor(buf168, (1024, 768), (768, 1), 0); del buf168  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf174, permute_853, out=buf175)
        del permute_853
        buf176 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf174, (768, 1024), (1, 768), 0), view_750, out=buf176)
        buf177 = buf120; del buf120  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf174, buf177, 6144, 128, grid=grid(6144), stream=stream0)
        buf178 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf177, buf178, 768, 8, grid=grid(768), stream=stream0)
        buf179 = buf174; del buf174  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf161, (1024, 768), (768, 1), 0), permute_857, out=buf179)
        del permute_857
        buf180 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf161, (768, 1024), (1, 768), 0), view_750, out=buf180)
        buf181 = buf177; del buf177  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf161, buf181, 6144, 128, grid=grid(6144), stream=stream0)
        buf182 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf181, buf182, 768, 8, grid=grid(768), stream=stream0)
        buf183 = reinterpret_tensor(buf161, (1024, 768), (768, 1), 0); del buf161  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf172, permute_866, out=buf183)
        del permute_866
        buf184 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf172, (768, 1024), (1, 768), 0), view_750, out=buf184)
        del view_750
        buf185 = buf181; del buf181  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf172, buf185, 6144, 128, grid=grid(6144), stream=stream0)
        buf186 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf185, buf186, 768, 8, grid=grid(768), stream=stream0)
        buf190 = reinterpret_tensor(buf172, (1, 1024, 768), (786432, 768, 1), 0); del buf172  # reuse
        buf193 = reinterpret_tensor(buf165, (1, 1024, 768), (786432, 768, 1), 0); del buf165  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_34.run(buf114, buf175, buf179, buf183, primals_159, mul_78, div_126, getitem_97, buf190, buf193, 1024, 768, grid=grid(1024), stream=stream0)
        del div_126
        del getitem_97
        del primals_159
        buf191 = empty((768, ), device='cuda', dtype=torch.float32)
        buf192 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_35.run(buf114, buf175, buf179, buf183, mul_78, buf191, buf192, 768, 1024, grid=grid(768), stream=stream0)
        del mul_78
        buf194 = reinterpret_tensor(buf107, (1024, 3072), (3072, 1), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf193, (1024, 768), (768, 1), 0), permute_872, out=buf194)
        del permute_872
        buf195 = reinterpret_tensor(buf123, (768, 3072), (3072, 1), 0); del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf193, (768, 1024), (1, 768), 0), view_748, out=buf195)
        del view_748
        buf196 = buf185; del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf193, buf196, 6144, 128, grid=grid(6144), stream=stream0)
        buf197 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf196, buf197, 768, 8, grid=grid(768), stream=stream0)
        buf198 = reinterpret_tensor(buf194, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf194  # reuse
        # Source Nodes: [intermediate_output_9], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_4.run(buf198, addmm_58, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_58
        buf199 = reinterpret_tensor(buf193, (1024, 768), (768, 1), 0); del buf193  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf198, (1024, 3072), (3072, 1), 0), permute_876, out=buf199)
        del permute_876
        buf200 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf198, (3072, 1024), (1, 3072), 0), view_746, out=buf200)
        del view_746
        buf201 = buf110; del buf110  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf198, buf201, 24576, 128, grid=grid(24576), stream=stream0)
        buf202 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf201, buf202, 3072, 8, grid=grid(3072), stream=stream0)
        buf205 = reinterpret_tensor(buf183, (1, 1024, 768), (786432, 768, 1), 0); del buf183  # reuse
        buf208 = reinterpret_tensor(buf179, (1, 1024, 768), (786432, 768, 1), 0); del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7.run(buf190, buf199, primals_153, mul_73, div_127, getitem_93, buf205, buf208, 1024, 768, grid=grid(1024), stream=stream0)
        del div_127
        del getitem_93
        del primals_153
        buf206 = empty((768, ), device='cuda', dtype=torch.float32)
        buf207 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf190, buf199, mul_73, buf206, buf207, 768, 1024, grid=grid(768), stream=stream0)
        del mul_73
        buf209 = buf199; del buf199  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (1024, 768), (768, 1), 0), permute_880, out=buf209)
        del permute_880
        buf210 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (768, 1024), (1, 768), 0), view_744, out=buf210)
        del view_744
        buf211 = buf196; del buf196  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf208, buf211, 6144, 128, grid=grid(6144), stream=stream0)
        buf212 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf211, buf212, 768, 8, grid=grid(768), stream=stream0)
        buf213 = reinterpret_tensor(buf208, (12, 4, 256, 1, 64), (65536, 16384, 64, 64, 1), 0); del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf209, buf213, 786432, grid=grid(786432), stream=stream0)
        buf214 = empty((48, 768, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_888, reinterpret_tensor(buf213, (48, 256, 64), (16384, 64, 1), 0), out=buf214)
        del permute_888
        buf215 = reinterpret_tensor(buf158, (48, 256, 768), (196608, 768, 1), 0); del buf158  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf213, (48, 256, 64), (16384, 64, 1), 0), permute_889, out=buf215)
        del permute_889
        buf216 = buf125; del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add, aten.new_zeros]
        triton_poi_fused_index_add_new_zeros_11.run(buf216, 1179648, grid=grid(1179648), stream=stream0)
        aten.index_put_(buf216, [reinterpret_tensor(buf30, (2359296, ), (1, ), 0)], reinterpret_tensor(buf214, (2359296, ), (1, ), 0), True)
        buf220 = reinterpret_tensor(buf154, (1024, 12, 513), (513, 525312, 1), 0); del buf154  # reuse
        # Source Nodes: [tril], Original ATen: [aten._softmax_backward_data, aten.copy, aten.masked_fill, aten.native_dropout_backward, aten.tril]
        triton_red_fused__softmax_backward_data_copy_masked_fill_native_dropout_backward_tril_13.run(unsqueeze_16, buf215, getitem_91, alias_14, buf220, 12288, 513, grid=grid(12288), stream=stream0)
        del alias_14
        del getitem_91
        buf221 = reinterpret_tensor(buf151, (6303744, ), (1, ), 0); del buf151  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_14.run(buf221, 6303744, grid=grid(6303744), stream=stream0)
        buf224 = reinterpret_tensor(buf148, (1, 1024, 12, 513), (6303744, 513, 525312, 1), 0); del buf148  # reuse
        buf225 = buf146; del buf146  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.clone, aten.copy]
        triton_poi_fused_as_strided_scatter_clone_copy_15.run(buf220, buf221, buf224, buf225, 6303744, grid=grid(6303744), stream=stream0)
        buf228 = reinterpret_tensor(buf221, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf221  # reuse
        buf230 = buf220; del buf220  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_16.run(buf224, rev_1, buf225, buf228, buf230, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_17.run(buf228, 789504, grid=grid(789504), stream=stream0)
        buf231 = reinterpret_tensor(buf225, (6303744, ), (1, ), 0); del buf225  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_14.run(buf231, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_18.run(buf230, buf231, 6303744, grid=grid(6303744), stream=stream0)
        buf233 = reinterpret_tensor(buf230, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf230  # reuse
        buf235 = reinterpret_tensor(buf224, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_add_as_strided_scatter_copy_zeros_like_19.run(buf228, buf231, buf233, buf235, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_20.run(buf235, 789504, grid=grid(789504), stream=stream0)
        buf237 = buf233; del buf233  # reuse
        buf239 = reinterpret_tensor(buf231, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf231  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_add_as_strided_scatter_copy_zeros_like_21.run(buf237, buf235, slice_64, buf239, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_22.run(buf239, 780300, grid=grid(780300), stream=stream0)
        buf242 = buf235; del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_23.run(buf239, buf242, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_24.run(buf242, 2359296, grid=grid(2359296), stream=stream0)
        buf245 = buf228; del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_23.run(buf242, buf245, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_17.run(buf245, 789504, grid=grid(789504), stream=stream0)
        buf247 = buf156; del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.clone, aten.select_backward, aten.slice_backward]
        triton_poi_fused_add_clone_select_backward_slice_backward_25.run(buf237, buf239, buf242, buf247, 9455616, grid=grid(9455616), stream=stream0)
        buf249 = reinterpret_tensor(buf215, (12, 3, 512, 512), (786432, 262144, 512, 1), 0); del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_26.run(buf247, buf245, buf249, 9437184, grid=grid(9437184), stream=stream0)
        buf250 = reinterpret_tensor(buf160, (36, 64, 512), (32768, 512, 1), 0); del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_899, reinterpret_tensor(buf249, (36, 512, 512), (262144, 512, 1), 0), out=buf250)
        del permute_899
        buf251 = reinterpret_tensor(buf162, (36, 512, 64), (32768, 64, 1), 0); del buf162  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf249, (36, 512, 512), (262144, 512, 1), 0), permute_900, out=buf251)
        del permute_900
        buf252 = reinterpret_tensor(buf213, (786432, ), (1, ), 0); del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add]
        triton_poi_fused_index_add_28.run(buf252, 786432, grid=grid(786432), stream=stream0)
        buf253 = reinterpret_tensor(buf159, (12, 3, 512, 64), (98304, 32768, 64, 1), 0); del buf159  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf250, buf253, 18432, 64, grid=grid(18432, 64), stream=stream0)
        aten.index_put_(buf252, [reinterpret_tensor(buf71, (1179648, ), (1, ), 0)], reinterpret_tensor(buf253, (1179648, ), (1, ), 0), True)
        buf256 = reinterpret_tensor(buf209, (786432, ), (1, ), 0); del buf209  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add]
        triton_poi_fused_index_add_28.run(buf256, 786432, grid=grid(786432), stream=stream0)
        aten.index_put_(buf256, [reinterpret_tensor(buf71, (1179648, ), (1, ), 0)], reinterpret_tensor(buf251, (1179648, ), (1, ), 0), True)
        buf259 = reinterpret_tensor(buf190, (786432, ), (1, ), 0); del buf190  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_index_add_28.run(buf259, 786432, grid=grid(786432), stream=stream0)
        buf262 = reinterpret_tensor(buf175, (1024, 1, 768), (768, 768, 1), 0); del buf175  # reuse
        buf263 = reinterpret_tensor(buf114, (1024, 768), (768, 1), 0); del buf114  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.div]
        triton_poi_fused_as_strided_scatter_div_31.run(buf256, buf259, buf262, buf263, 786432, grid=grid(786432), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_32.run(buf262, buf263, 786432, grid=grid(786432), stream=stream0)
        buf265 = reinterpret_tensor(buf262, (1024, 768), (768, 1), 0); del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_33.run(buf216, buf265, 786432, grid=grid(786432), stream=stream0)
        buf266 = reinterpret_tensor(buf259, (1024, 768), (768, 1), 0); del buf259  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf265, permute_911, out=buf266)
        del permute_911
        buf267 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf265, (768, 1024), (1, 768), 0), view_675, out=buf267)
        buf268 = buf211; del buf211  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf265, buf268, 6144, 128, grid=grid(6144), stream=stream0)
        buf269 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf268, buf269, 768, 8, grid=grid(768), stream=stream0)
        buf270 = buf265; del buf265  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf252, (1024, 768), (768, 1), 0), permute_915, out=buf270)
        del permute_915
        buf271 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf252, (768, 1024), (1, 768), 0), view_675, out=buf271)
        buf272 = buf268; del buf268  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf252, buf272, 6144, 128, grid=grid(6144), stream=stream0)
        buf273 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf272, buf273, 768, 8, grid=grid(768), stream=stream0)
        buf274 = reinterpret_tensor(buf252, (1024, 768), (768, 1), 0); del buf252  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf263, permute_924, out=buf274)
        del permute_924
        buf275 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf263, (768, 1024), (1, 768), 0), view_675, out=buf275)
        del view_675
        buf276 = buf272; del buf272  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf263, buf276, 6144, 128, grid=grid(6144), stream=stream0)
        buf277 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf276, buf277, 768, 8, grid=grid(768), stream=stream0)
        buf281 = reinterpret_tensor(buf263, (1, 1024, 768), (786432, 768, 1), 0); del buf263  # reuse
        buf284 = reinterpret_tensor(buf256, (1, 1024, 768), (786432, 768, 1), 0); del buf256  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_34.run(buf205, buf266, buf270, buf274, primals_143, mul_70, div_129, getitem_87, buf281, buf284, 1024, 768, grid=grid(1024), stream=stream0)
        del div_129
        del getitem_87
        del primals_143
        buf282 = empty((768, ), device='cuda', dtype=torch.float32)
        buf283 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_35.run(buf205, buf266, buf270, buf274, mul_70, buf282, buf283, 768, 1024, grid=grid(768), stream=stream0)
        del mul_70
        buf285 = reinterpret_tensor(buf198, (1024, 3072), (3072, 1), 0); del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf284, (1024, 768), (768, 1), 0), permute_930, out=buf285)
        del permute_930
        buf286 = reinterpret_tensor(buf214, (768, 3072), (3072, 1), 0); del buf214  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf284, (768, 1024), (1, 768), 0), view_673, out=buf286)
        del view_673
        buf287 = buf276; del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf284, buf287, 6144, 128, grid=grid(6144), stream=stream0)
        buf288 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf287, buf288, 768, 8, grid=grid(768), stream=stream0)
        buf289 = reinterpret_tensor(buf285, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf285  # reuse
        # Source Nodes: [intermediate_output_8], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_4.run(buf289, addmm_52, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_52
        buf290 = reinterpret_tensor(buf284, (1024, 768), (768, 1), 0); del buf284  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (1024, 3072), (3072, 1), 0), permute_934, out=buf290)
        del permute_934
        buf291 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (3072, 1024), (1, 3072), 0), view_671, out=buf291)
        del view_671
        buf292 = buf201; del buf201  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf289, buf292, 24576, 128, grid=grid(24576), stream=stream0)
        buf293 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf292, buf293, 3072, 8, grid=grid(3072), stream=stream0)
        buf296 = reinterpret_tensor(buf274, (1, 1024, 768), (786432, 768, 1), 0); del buf274  # reuse
        buf299 = reinterpret_tensor(buf270, (1, 1024, 768), (786432, 768, 1), 0); del buf270  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7.run(buf281, buf290, primals_137, mul_65, div_130, getitem_83, buf296, buf299, 1024, 768, grid=grid(1024), stream=stream0)
        del div_130
        del getitem_83
        del primals_137
        buf297 = empty((768, ), device='cuda', dtype=torch.float32)
        buf298 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf281, buf290, mul_65, buf297, buf298, 768, 1024, grid=grid(768), stream=stream0)
        del mul_65
        buf300 = buf290; del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf299, (1024, 768), (768, 1), 0), permute_938, out=buf300)
        del permute_938
        buf301 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf299, (768, 1024), (1, 768), 0), view_669, out=buf301)
        del view_669
        buf302 = buf287; del buf287  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf299, buf302, 6144, 128, grid=grid(6144), stream=stream0)
        buf303 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf302, buf303, 768, 8, grid=grid(768), stream=stream0)
        buf304 = reinterpret_tensor(buf299, (12, 4, 256, 1, 64), (65536, 16384, 64, 64, 1), 0); del buf299  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf300, buf304, 786432, grid=grid(786432), stream=stream0)
        buf305 = empty((48, 768, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_946, reinterpret_tensor(buf304, (48, 256, 64), (16384, 64, 1), 0), out=buf305)
        del permute_946
        buf306 = reinterpret_tensor(buf249, (48, 256, 768), (196608, 768, 1), 0); del buf249  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf304, (48, 256, 64), (16384, 64, 1), 0), permute_947, out=buf306)
        del permute_947
        buf307 = buf216; del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add, aten.new_zeros]
        triton_poi_fused_index_add_new_zeros_11.run(buf307, 1179648, grid=grid(1179648), stream=stream0)
        aten.index_put_(buf307, [reinterpret_tensor(buf30, (2359296, ), (1, ), 0)], reinterpret_tensor(buf305, (2359296, ), (1, ), 0), True)
        buf311 = reinterpret_tensor(buf245, (1024, 12, 513), (513, 525312, 1), 0); del buf245  # reuse
        # Source Nodes: [tril], Original ATen: [aten._softmax_backward_data, aten.copy, aten.masked_fill, aten.native_dropout_backward, aten.tril]
        triton_red_fused__softmax_backward_data_copy_masked_fill_native_dropout_backward_tril_13.run(unsqueeze_16, buf306, getitem_81, alias_15, buf311, 12288, 513, grid=grid(12288), stream=stream0)
        del alias_15
        del getitem_81
        buf312 = reinterpret_tensor(buf242, (6303744, ), (1, ), 0); del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_14.run(buf312, 6303744, grid=grid(6303744), stream=stream0)
        buf315 = reinterpret_tensor(buf239, (1, 1024, 12, 513), (6303744, 513, 525312, 1), 0); del buf239  # reuse
        buf316 = buf237; del buf237  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.clone, aten.copy]
        triton_poi_fused_as_strided_scatter_clone_copy_15.run(buf311, buf312, buf315, buf316, 6303744, grid=grid(6303744), stream=stream0)
        buf319 = reinterpret_tensor(buf312, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf312  # reuse
        buf321 = buf311; del buf311  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_16.run(buf315, rev_1, buf316, buf319, buf321, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_17.run(buf319, 789504, grid=grid(789504), stream=stream0)
        buf322 = reinterpret_tensor(buf316, (6303744, ), (1, ), 0); del buf316  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_14.run(buf322, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_18.run(buf321, buf322, 6303744, grid=grid(6303744), stream=stream0)
        buf324 = reinterpret_tensor(buf321, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf321  # reuse
        buf326 = reinterpret_tensor(buf315, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf315  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_add_as_strided_scatter_copy_zeros_like_19.run(buf319, buf322, buf324, buf326, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_20.run(buf326, 789504, grid=grid(789504), stream=stream0)
        buf328 = buf324; del buf324  # reuse
        buf330 = reinterpret_tensor(buf322, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf322  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_add_as_strided_scatter_copy_zeros_like_21.run(buf328, buf326, slice_64, buf330, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_22.run(buf330, 780300, grid=grid(780300), stream=stream0)
        buf333 = buf326; del buf326  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_23.run(buf330, buf333, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_24.run(buf333, 2359296, grid=grid(2359296), stream=stream0)
        buf336 = buf319; del buf319  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_23.run(buf333, buf336, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_17.run(buf336, 789504, grid=grid(789504), stream=stream0)
        buf338 = buf247; del buf247  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.clone, aten.select_backward, aten.slice_backward]
        triton_poi_fused_add_clone_select_backward_slice_backward_25.run(buf328, buf330, buf333, buf338, 9455616, grid=grid(9455616), stream=stream0)
        buf340 = reinterpret_tensor(buf306, (12, 3, 512, 512), (786432, 262144, 512, 1), 0); del buf306  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_26.run(buf338, buf336, buf340, 9437184, grid=grid(9437184), stream=stream0)
        buf341 = reinterpret_tensor(buf251, (36, 64, 512), (32768, 512, 1), 0); del buf251  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_957, reinterpret_tensor(buf340, (36, 512, 512), (262144, 512, 1), 0), out=buf341)
        del permute_957
        buf342 = reinterpret_tensor(buf253, (36, 512, 64), (32768, 64, 1), 0); del buf253  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf340, (36, 512, 512), (262144, 512, 1), 0), permute_958, out=buf342)
        del permute_958
        buf343 = reinterpret_tensor(buf304, (786432, ), (1, ), 0); del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add]
        triton_poi_fused_index_add_28.run(buf343, 786432, grid=grid(786432), stream=stream0)
        buf344 = reinterpret_tensor(buf250, (12, 3, 512, 64), (98304, 32768, 64, 1), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf341, buf344, 18432, 64, grid=grid(18432, 64), stream=stream0)
        aten.index_put_(buf343, [reinterpret_tensor(buf71, (1179648, ), (1, ), 0)], reinterpret_tensor(buf344, (1179648, ), (1, ), 0), True)
        buf347 = reinterpret_tensor(buf300, (786432, ), (1, ), 0); del buf300  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add]
        triton_poi_fused_index_add_28.run(buf347, 786432, grid=grid(786432), stream=stream0)
        aten.index_put_(buf347, [reinterpret_tensor(buf71, (1179648, ), (1, ), 0)], reinterpret_tensor(buf342, (1179648, ), (1, ), 0), True)
        buf350 = reinterpret_tensor(buf281, (786432, ), (1, ), 0); del buf281  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_index_add_28.run(buf350, 786432, grid=grid(786432), stream=stream0)
        buf353 = reinterpret_tensor(buf266, (1024, 1, 768), (768, 768, 1), 0); del buf266  # reuse
        buf354 = reinterpret_tensor(buf205, (1024, 768), (768, 1), 0); del buf205  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.div]
        triton_poi_fused_as_strided_scatter_div_31.run(buf347, buf350, buf353, buf354, 786432, grid=grid(786432), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_32.run(buf353, buf354, 786432, grid=grid(786432), stream=stream0)
        buf356 = reinterpret_tensor(buf353, (1024, 768), (768, 1), 0); del buf353  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_33.run(buf307, buf356, 786432, grid=grid(786432), stream=stream0)
        buf357 = reinterpret_tensor(buf350, (1024, 768), (768, 1), 0); del buf350  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf356, permute_969, out=buf357)
        del permute_969
        buf358 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf356, (768, 1024), (1, 768), 0), view_600, out=buf358)
        buf359 = buf302; del buf302  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf356, buf359, 6144, 128, grid=grid(6144), stream=stream0)
        buf360 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf359, buf360, 768, 8, grid=grid(768), stream=stream0)
        buf361 = buf356; del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf343, (1024, 768), (768, 1), 0), permute_973, out=buf361)
        del permute_973
        buf362 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf343, (768, 1024), (1, 768), 0), view_600, out=buf362)
        buf363 = buf359; del buf359  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf343, buf363, 6144, 128, grid=grid(6144), stream=stream0)
        buf364 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf363, buf364, 768, 8, grid=grid(768), stream=stream0)
        buf365 = reinterpret_tensor(buf343, (1024, 768), (768, 1), 0); del buf343  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf354, permute_982, out=buf365)
        del permute_982
        buf366 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf354, (768, 1024), (1, 768), 0), view_600, out=buf366)
        del view_600
        buf367 = buf363; del buf363  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf354, buf367, 6144, 128, grid=grid(6144), stream=stream0)
        buf368 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf367, buf368, 768, 8, grid=grid(768), stream=stream0)
        buf372 = reinterpret_tensor(buf354, (1, 1024, 768), (786432, 768, 1), 0); del buf354  # reuse
        buf375 = reinterpret_tensor(buf347, (1, 1024, 768), (786432, 768, 1), 0); del buf347  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_34.run(buf296, buf357, buf361, buf365, primals_127, mul_62, div_132, getitem_77, buf372, buf375, 1024, 768, grid=grid(1024), stream=stream0)
        del div_132
        del getitem_77
        del primals_127
        buf373 = empty((768, ), device='cuda', dtype=torch.float32)
        buf374 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_35.run(buf296, buf357, buf361, buf365, mul_62, buf373, buf374, 768, 1024, grid=grid(768), stream=stream0)
        del mul_62
        buf376 = reinterpret_tensor(buf289, (1024, 3072), (3072, 1), 0); del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf375, (1024, 768), (768, 1), 0), permute_988, out=buf376)
        del permute_988
        buf377 = reinterpret_tensor(buf305, (768, 3072), (3072, 1), 0); del buf305  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf375, (768, 1024), (1, 768), 0), view_598, out=buf377)
        del view_598
        buf378 = buf367; del buf367  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf375, buf378, 6144, 128, grid=grid(6144), stream=stream0)
        buf379 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf378, buf379, 768, 8, grid=grid(768), stream=stream0)
        buf380 = reinterpret_tensor(buf376, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf376  # reuse
        # Source Nodes: [intermediate_output_7], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_4.run(buf380, addmm_46, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_46
        buf381 = reinterpret_tensor(buf375, (1024, 768), (768, 1), 0); del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf380, (1024, 3072), (3072, 1), 0), permute_992, out=buf381)
        del permute_992
        buf382 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf380, (3072, 1024), (1, 3072), 0), view_596, out=buf382)
        del view_596
        buf383 = buf292; del buf292  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf380, buf383, 24576, 128, grid=grid(24576), stream=stream0)
        buf384 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf383, buf384, 3072, 8, grid=grid(3072), stream=stream0)
        buf387 = reinterpret_tensor(buf365, (1, 1024, 768), (786432, 768, 1), 0); del buf365  # reuse
        buf390 = reinterpret_tensor(buf361, (1, 1024, 768), (786432, 768, 1), 0); del buf361  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7.run(buf372, buf381, primals_121, mul_57, div_133, getitem_73, buf387, buf390, 1024, 768, grid=grid(1024), stream=stream0)
        del div_133
        del getitem_73
        del primals_121
        buf388 = empty((768, ), device='cuda', dtype=torch.float32)
        buf389 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf372, buf381, mul_57, buf388, buf389, 768, 1024, grid=grid(768), stream=stream0)
        del mul_57
        buf391 = buf381; del buf381  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf390, (1024, 768), (768, 1), 0), permute_996, out=buf391)
        del permute_996
        buf392 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf390, (768, 1024), (1, 768), 0), view_594, out=buf392)
        del view_594
        buf393 = buf378; del buf378  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf390, buf393, 6144, 128, grid=grid(6144), stream=stream0)
        buf394 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf393, buf394, 768, 8, grid=grid(768), stream=stream0)
        buf395 = reinterpret_tensor(buf390, (12, 4, 256, 1, 64), (65536, 16384, 64, 64, 1), 0); del buf390  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf391, buf395, 786432, grid=grid(786432), stream=stream0)
        buf396 = empty((48, 768, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1004, reinterpret_tensor(buf395, (48, 256, 64), (16384, 64, 1), 0), out=buf396)
        del permute_1004
        buf397 = reinterpret_tensor(buf340, (48, 256, 768), (196608, 768, 1), 0); del buf340  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf395, (48, 256, 64), (16384, 64, 1), 0), permute_1005, out=buf397)
        del permute_1005
        buf398 = buf307; del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add, aten.new_zeros]
        triton_poi_fused_index_add_new_zeros_11.run(buf398, 1179648, grid=grid(1179648), stream=stream0)
        aten.index_put_(buf398, [reinterpret_tensor(buf30, (2359296, ), (1, ), 0)], reinterpret_tensor(buf396, (2359296, ), (1, ), 0), True)
        buf402 = reinterpret_tensor(buf336, (1024, 12, 513), (513, 525312, 1), 0); del buf336  # reuse
        # Source Nodes: [tril], Original ATen: [aten._softmax_backward_data, aten.copy, aten.masked_fill, aten.native_dropout_backward, aten.tril]
        triton_red_fused__softmax_backward_data_copy_masked_fill_native_dropout_backward_tril_13.run(unsqueeze_16, buf397, getitem_71, alias_16, buf402, 12288, 513, grid=grid(12288), stream=stream0)
        del alias_16
        del getitem_71
        buf403 = reinterpret_tensor(buf333, (6303744, ), (1, ), 0); del buf333  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_14.run(buf403, 6303744, grid=grid(6303744), stream=stream0)
        buf406 = reinterpret_tensor(buf330, (1, 1024, 12, 513), (6303744, 513, 525312, 1), 0); del buf330  # reuse
        buf407 = buf328; del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.clone, aten.copy]
        triton_poi_fused_as_strided_scatter_clone_copy_15.run(buf402, buf403, buf406, buf407, 6303744, grid=grid(6303744), stream=stream0)
        buf410 = reinterpret_tensor(buf403, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf403  # reuse
        buf412 = buf402; del buf402  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_16.run(buf406, rev_1, buf407, buf410, buf412, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_17.run(buf410, 789504, grid=grid(789504), stream=stream0)
        buf413 = reinterpret_tensor(buf407, (6303744, ), (1, ), 0); del buf407  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_14.run(buf413, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_18.run(buf412, buf413, 6303744, grid=grid(6303744), stream=stream0)
        buf415 = reinterpret_tensor(buf412, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf412  # reuse
        buf417 = reinterpret_tensor(buf406, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf406  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_add_as_strided_scatter_copy_zeros_like_19.run(buf410, buf413, buf415, buf417, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_20.run(buf417, 789504, grid=grid(789504), stream=stream0)
        buf419 = buf415; del buf415  # reuse
        buf421 = reinterpret_tensor(buf413, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf413  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_add_as_strided_scatter_copy_zeros_like_21.run(buf419, buf417, slice_64, buf421, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_22.run(buf421, 780300, grid=grid(780300), stream=stream0)
        buf424 = buf417; del buf417  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_23.run(buf421, buf424, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_24.run(buf424, 2359296, grid=grid(2359296), stream=stream0)
        buf427 = buf410; del buf410  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_23.run(buf424, buf427, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_17.run(buf427, 789504, grid=grid(789504), stream=stream0)
        buf429 = buf338; del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.clone, aten.select_backward, aten.slice_backward]
        triton_poi_fused_add_clone_select_backward_slice_backward_25.run(buf419, buf421, buf424, buf429, 9455616, grid=grid(9455616), stream=stream0)
        buf431 = reinterpret_tensor(buf397, (12, 3, 512, 512), (786432, 262144, 512, 1), 0); del buf397  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_26.run(buf429, buf427, buf431, 9437184, grid=grid(9437184), stream=stream0)
        buf432 = reinterpret_tensor(buf342, (36, 64, 512), (32768, 512, 1), 0); del buf342  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1015, reinterpret_tensor(buf431, (36, 512, 512), (262144, 512, 1), 0), out=buf432)
        del permute_1015
        buf433 = reinterpret_tensor(buf344, (36, 512, 64), (32768, 64, 1), 0); del buf344  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf431, (36, 512, 512), (262144, 512, 1), 0), permute_1016, out=buf433)
        del permute_1016
        buf434 = reinterpret_tensor(buf395, (786432, ), (1, ), 0); del buf395  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add]
        triton_poi_fused_index_add_28.run(buf434, 786432, grid=grid(786432), stream=stream0)
        buf435 = reinterpret_tensor(buf341, (12, 3, 512, 64), (98304, 32768, 64, 1), 0); del buf341  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf432, buf435, 18432, 64, grid=grid(18432, 64), stream=stream0)
        aten.index_put_(buf434, [reinterpret_tensor(buf71, (1179648, ), (1, ), 0)], reinterpret_tensor(buf435, (1179648, ), (1, ), 0), True)
        buf438 = reinterpret_tensor(buf391, (786432, ), (1, ), 0); del buf391  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add]
        triton_poi_fused_index_add_28.run(buf438, 786432, grid=grid(786432), stream=stream0)
        aten.index_put_(buf438, [reinterpret_tensor(buf71, (1179648, ), (1, ), 0)], reinterpret_tensor(buf433, (1179648, ), (1, ), 0), True)
        buf441 = reinterpret_tensor(buf372, (786432, ), (1, ), 0); del buf372  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_index_add_28.run(buf441, 786432, grid=grid(786432), stream=stream0)
        buf444 = reinterpret_tensor(buf357, (1024, 1, 768), (768, 768, 1), 0); del buf357  # reuse
        buf445 = reinterpret_tensor(buf296, (1024, 768), (768, 1), 0); del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.div]
        triton_poi_fused_as_strided_scatter_div_31.run(buf438, buf441, buf444, buf445, 786432, grid=grid(786432), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_32.run(buf444, buf445, 786432, grid=grid(786432), stream=stream0)
        buf447 = reinterpret_tensor(buf444, (1024, 768), (768, 1), 0); del buf444  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_33.run(buf398, buf447, 786432, grid=grid(786432), stream=stream0)
        buf448 = reinterpret_tensor(buf441, (1024, 768), (768, 1), 0); del buf441  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf447, permute_1027, out=buf448)
        del permute_1027
        buf449 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf447, (768, 1024), (1, 768), 0), view_525, out=buf449)
        buf450 = buf393; del buf393  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf447, buf450, 6144, 128, grid=grid(6144), stream=stream0)
        buf451 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf450, buf451, 768, 8, grid=grid(768), stream=stream0)
        buf452 = buf447; del buf447  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf434, (1024, 768), (768, 1), 0), permute_1031, out=buf452)
        del permute_1031
        buf453 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf434, (768, 1024), (1, 768), 0), view_525, out=buf453)
        buf454 = buf450; del buf450  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf434, buf454, 6144, 128, grid=grid(6144), stream=stream0)
        buf455 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf454, buf455, 768, 8, grid=grid(768), stream=stream0)
        buf456 = reinterpret_tensor(buf434, (1024, 768), (768, 1), 0); del buf434  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf445, permute_1040, out=buf456)
        del permute_1040
        buf457 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf445, (768, 1024), (1, 768), 0), view_525, out=buf457)
        del view_525
        buf458 = buf454; del buf454  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf445, buf458, 6144, 128, grid=grid(6144), stream=stream0)
        buf459 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf458, buf459, 768, 8, grid=grid(768), stream=stream0)
        buf463 = reinterpret_tensor(buf445, (1, 1024, 768), (786432, 768, 1), 0); del buf445  # reuse
        buf466 = reinterpret_tensor(buf438, (1, 1024, 768), (786432, 768, 1), 0); del buf438  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_34.run(buf387, buf448, buf452, buf456, primals_111, mul_54, div_135, getitem_67, buf463, buf466, 1024, 768, grid=grid(1024), stream=stream0)
        del div_135
        del getitem_67
        del primals_111
        buf464 = empty((768, ), device='cuda', dtype=torch.float32)
        buf465 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_35.run(buf387, buf448, buf452, buf456, mul_54, buf464, buf465, 768, 1024, grid=grid(768), stream=stream0)
        del mul_54
        buf467 = reinterpret_tensor(buf380, (1024, 3072), (3072, 1), 0); del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf466, (1024, 768), (768, 1), 0), permute_1046, out=buf467)
        del permute_1046
        buf468 = reinterpret_tensor(buf396, (768, 3072), (3072, 1), 0); del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf466, (768, 1024), (1, 768), 0), view_523, out=buf468)
        del view_523
        buf469 = buf458; del buf458  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf466, buf469, 6144, 128, grid=grid(6144), stream=stream0)
        buf470 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf469, buf470, 768, 8, grid=grid(768), stream=stream0)
        buf471 = reinterpret_tensor(buf467, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf467  # reuse
        # Source Nodes: [intermediate_output_6], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_4.run(buf471, addmm_40, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_40
        buf472 = reinterpret_tensor(buf466, (1024, 768), (768, 1), 0); del buf466  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf471, (1024, 3072), (3072, 1), 0), permute_1050, out=buf472)
        del permute_1050
        buf473 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf471, (3072, 1024), (1, 3072), 0), view_521, out=buf473)
        del view_521
        buf474 = buf383; del buf383  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf471, buf474, 24576, 128, grid=grid(24576), stream=stream0)
        buf475 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf474, buf475, 3072, 8, grid=grid(3072), stream=stream0)
        buf478 = reinterpret_tensor(buf456, (1, 1024, 768), (786432, 768, 1), 0); del buf456  # reuse
        buf481 = reinterpret_tensor(buf452, (1, 1024, 768), (786432, 768, 1), 0); del buf452  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7.run(buf463, buf472, primals_105, mul_49, div_136, getitem_63, buf478, buf481, 1024, 768, grid=grid(1024), stream=stream0)
        del div_136
        del getitem_63
        del primals_105
        buf479 = empty((768, ), device='cuda', dtype=torch.float32)
        buf480 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf463, buf472, mul_49, buf479, buf480, 768, 1024, grid=grid(768), stream=stream0)
        del mul_49
        buf482 = buf472; del buf472  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf481, (1024, 768), (768, 1), 0), permute_1054, out=buf482)
        del permute_1054
        buf483 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf481, (768, 1024), (1, 768), 0), view_519, out=buf483)
        del view_519
        buf484 = buf469; del buf469  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf481, buf484, 6144, 128, grid=grid(6144), stream=stream0)
        buf485 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf484, buf485, 768, 8, grid=grid(768), stream=stream0)
        buf486 = reinterpret_tensor(buf481, (12, 4, 256, 1, 64), (65536, 16384, 64, 64, 1), 0); del buf481  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf482, buf486, 786432, grid=grid(786432), stream=stream0)
        buf487 = empty((48, 768, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1062, reinterpret_tensor(buf486, (48, 256, 64), (16384, 64, 1), 0), out=buf487)
        del permute_1062
        buf488 = reinterpret_tensor(buf431, (48, 256, 768), (196608, 768, 1), 0); del buf431  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf486, (48, 256, 64), (16384, 64, 1), 0), permute_1063, out=buf488)
        del permute_1063
        buf489 = buf398; del buf398  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add, aten.new_zeros]
        triton_poi_fused_index_add_new_zeros_11.run(buf489, 1179648, grid=grid(1179648), stream=stream0)
        aten.index_put_(buf489, [reinterpret_tensor(buf30, (2359296, ), (1, ), 0)], reinterpret_tensor(buf487, (2359296, ), (1, ), 0), True)
        buf493 = reinterpret_tensor(buf427, (1024, 12, 513), (513, 525312, 1), 0); del buf427  # reuse
        # Source Nodes: [tril], Original ATen: [aten._softmax_backward_data, aten.copy, aten.masked_fill, aten.native_dropout_backward, aten.tril]
        triton_red_fused__softmax_backward_data_copy_masked_fill_native_dropout_backward_tril_13.run(unsqueeze_16, buf488, getitem_61, alias_17, buf493, 12288, 513, grid=grid(12288), stream=stream0)
        del alias_17
        del getitem_61
        buf494 = reinterpret_tensor(buf424, (6303744, ), (1, ), 0); del buf424  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_14.run(buf494, 6303744, grid=grid(6303744), stream=stream0)
        buf497 = reinterpret_tensor(buf421, (1, 1024, 12, 513), (6303744, 513, 525312, 1), 0); del buf421  # reuse
        buf498 = buf419; del buf419  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.clone, aten.copy]
        triton_poi_fused_as_strided_scatter_clone_copy_15.run(buf493, buf494, buf497, buf498, 6303744, grid=grid(6303744), stream=stream0)
        buf501 = reinterpret_tensor(buf494, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf494  # reuse
        buf503 = buf493; del buf493  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_16.run(buf497, rev_1, buf498, buf501, buf503, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_17.run(buf501, 789504, grid=grid(789504), stream=stream0)
        buf504 = reinterpret_tensor(buf498, (6303744, ), (1, ), 0); del buf498  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_14.run(buf504, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_18.run(buf503, buf504, 6303744, grid=grid(6303744), stream=stream0)
        buf506 = reinterpret_tensor(buf503, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf503  # reuse
        buf508 = reinterpret_tensor(buf497, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf497  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_add_as_strided_scatter_copy_zeros_like_19.run(buf501, buf504, buf506, buf508, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_20.run(buf508, 789504, grid=grid(789504), stream=stream0)
        buf510 = buf506; del buf506  # reuse
        buf512 = reinterpret_tensor(buf504, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf504  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_add_as_strided_scatter_copy_zeros_like_21.run(buf510, buf508, slice_64, buf512, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_22.run(buf512, 780300, grid=grid(780300), stream=stream0)
        buf515 = buf508; del buf508  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_23.run(buf512, buf515, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_24.run(buf515, 2359296, grid=grid(2359296), stream=stream0)
        buf518 = buf501; del buf501  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_23.run(buf515, buf518, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_17.run(buf518, 789504, grid=grid(789504), stream=stream0)
        buf520 = buf429; del buf429  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.clone, aten.select_backward, aten.slice_backward]
        triton_poi_fused_add_clone_select_backward_slice_backward_25.run(buf510, buf512, buf515, buf520, 9455616, grid=grid(9455616), stream=stream0)
        buf522 = reinterpret_tensor(buf488, (12, 3, 512, 512), (786432, 262144, 512, 1), 0); del buf488  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_26.run(buf520, buf518, buf522, 9437184, grid=grid(9437184), stream=stream0)
        buf523 = reinterpret_tensor(buf433, (36, 64, 512), (32768, 512, 1), 0); del buf433  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1073, reinterpret_tensor(buf522, (36, 512, 512), (262144, 512, 1), 0), out=buf523)
        del permute_1073
        buf524 = reinterpret_tensor(buf435, (36, 512, 64), (32768, 64, 1), 0); del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf522, (36, 512, 512), (262144, 512, 1), 0), permute_1074, out=buf524)
        del permute_1074
        buf525 = reinterpret_tensor(buf486, (786432, ), (1, ), 0); del buf486  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add]
        triton_poi_fused_index_add_28.run(buf525, 786432, grid=grid(786432), stream=stream0)
        buf526 = reinterpret_tensor(buf432, (12, 3, 512, 64), (98304, 32768, 64, 1), 0); del buf432  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf523, buf526, 18432, 64, grid=grid(18432, 64), stream=stream0)
        aten.index_put_(buf525, [reinterpret_tensor(buf71, (1179648, ), (1, ), 0)], reinterpret_tensor(buf526, (1179648, ), (1, ), 0), True)
        buf529 = reinterpret_tensor(buf482, (786432, ), (1, ), 0); del buf482  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add]
        triton_poi_fused_index_add_28.run(buf529, 786432, grid=grid(786432), stream=stream0)
        aten.index_put_(buf529, [reinterpret_tensor(buf71, (1179648, ), (1, ), 0)], reinterpret_tensor(buf524, (1179648, ), (1, ), 0), True)
        buf532 = reinterpret_tensor(buf463, (786432, ), (1, ), 0); del buf463  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_index_add_28.run(buf532, 786432, grid=grid(786432), stream=stream0)
        buf535 = reinterpret_tensor(buf448, (1024, 1, 768), (768, 768, 1), 0); del buf448  # reuse
        buf536 = reinterpret_tensor(buf387, (1024, 768), (768, 1), 0); del buf387  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.div]
        triton_poi_fused_as_strided_scatter_div_31.run(buf529, buf532, buf535, buf536, 786432, grid=grid(786432), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_32.run(buf535, buf536, 786432, grid=grid(786432), stream=stream0)
        buf538 = reinterpret_tensor(buf535, (1024, 768), (768, 1), 0); del buf535  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_33.run(buf489, buf538, 786432, grid=grid(786432), stream=stream0)
        buf539 = reinterpret_tensor(buf532, (1024, 768), (768, 1), 0); del buf532  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf538, permute_1085, out=buf539)
        del permute_1085
        buf540 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf538, (768, 1024), (1, 768), 0), view_450, out=buf540)
        buf541 = buf484; del buf484  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf538, buf541, 6144, 128, grid=grid(6144), stream=stream0)
        buf542 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf541, buf542, 768, 8, grid=grid(768), stream=stream0)
        buf543 = buf538; del buf538  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf525, (1024, 768), (768, 1), 0), permute_1089, out=buf543)
        del permute_1089
        buf544 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf525, (768, 1024), (1, 768), 0), view_450, out=buf544)
        buf545 = buf541; del buf541  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf525, buf545, 6144, 128, grid=grid(6144), stream=stream0)
        buf546 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf545, buf546, 768, 8, grid=grid(768), stream=stream0)
        buf547 = reinterpret_tensor(buf525, (1024, 768), (768, 1), 0); del buf525  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf536, permute_1098, out=buf547)
        del permute_1098
        buf548 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf536, (768, 1024), (1, 768), 0), view_450, out=buf548)
        del view_450
        buf549 = buf545; del buf545  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf536, buf549, 6144, 128, grid=grid(6144), stream=stream0)
        buf550 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf549, buf550, 768, 8, grid=grid(768), stream=stream0)
        buf554 = reinterpret_tensor(buf536, (1, 1024, 768), (786432, 768, 1), 0); del buf536  # reuse
        buf557 = reinterpret_tensor(buf529, (1, 1024, 768), (786432, 768, 1), 0); del buf529  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_34.run(buf478, buf539, buf543, buf547, primals_95, mul_46, div_138, getitem_57, buf554, buf557, 1024, 768, grid=grid(1024), stream=stream0)
        del div_138
        del getitem_57
        del primals_95
        buf555 = empty((768, ), device='cuda', dtype=torch.float32)
        buf556 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_35.run(buf478, buf539, buf543, buf547, mul_46, buf555, buf556, 768, 1024, grid=grid(768), stream=stream0)
        del mul_46
        buf558 = reinterpret_tensor(buf471, (1024, 3072), (3072, 1), 0); del buf471  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf557, (1024, 768), (768, 1), 0), permute_1104, out=buf558)
        del permute_1104
        buf559 = reinterpret_tensor(buf487, (768, 3072), (3072, 1), 0); del buf487  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf557, (768, 1024), (1, 768), 0), view_448, out=buf559)
        del view_448
        buf560 = buf549; del buf549  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf557, buf560, 6144, 128, grid=grid(6144), stream=stream0)
        buf561 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf560, buf561, 768, 8, grid=grid(768), stream=stream0)
        buf562 = reinterpret_tensor(buf558, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf558  # reuse
        # Source Nodes: [intermediate_output_5], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_4.run(buf562, addmm_34, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_34
        buf563 = reinterpret_tensor(buf557, (1024, 768), (768, 1), 0); del buf557  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf562, (1024, 3072), (3072, 1), 0), permute_1108, out=buf563)
        del permute_1108
        buf564 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf562, (3072, 1024), (1, 3072), 0), view_446, out=buf564)
        del view_446
        buf565 = buf474; del buf474  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf562, buf565, 24576, 128, grid=grid(24576), stream=stream0)
        buf566 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf565, buf566, 3072, 8, grid=grid(3072), stream=stream0)
        buf569 = reinterpret_tensor(buf547, (1, 1024, 768), (786432, 768, 1), 0); del buf547  # reuse
        buf572 = reinterpret_tensor(buf543, (1, 1024, 768), (786432, 768, 1), 0); del buf543  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7.run(buf554, buf563, primals_89, mul_41, div_139, getitem_53, buf569, buf572, 1024, 768, grid=grid(1024), stream=stream0)
        del div_139
        del getitem_53
        del primals_89
        buf570 = empty((768, ), device='cuda', dtype=torch.float32)
        buf571 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf554, buf563, mul_41, buf570, buf571, 768, 1024, grid=grid(768), stream=stream0)
        del mul_41
        buf573 = buf563; del buf563  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf572, (1024, 768), (768, 1), 0), permute_1112, out=buf573)
        del permute_1112
        buf574 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf572, (768, 1024), (1, 768), 0), view_444, out=buf574)
        del view_444
        buf575 = buf560; del buf560  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf572, buf575, 6144, 128, grid=grid(6144), stream=stream0)
        buf576 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf575, buf576, 768, 8, grid=grid(768), stream=stream0)
        buf577 = reinterpret_tensor(buf572, (12, 4, 256, 1, 64), (65536, 16384, 64, 64, 1), 0); del buf572  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf573, buf577, 786432, grid=grid(786432), stream=stream0)
        buf578 = empty((48, 768, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1120, reinterpret_tensor(buf577, (48, 256, 64), (16384, 64, 1), 0), out=buf578)
        del permute_1120
        buf579 = reinterpret_tensor(buf522, (48, 256, 768), (196608, 768, 1), 0); del buf522  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf577, (48, 256, 64), (16384, 64, 1), 0), permute_1121, out=buf579)
        del permute_1121
        buf580 = buf489; del buf489  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add, aten.new_zeros]
        triton_poi_fused_index_add_new_zeros_11.run(buf580, 1179648, grid=grid(1179648), stream=stream0)
        aten.index_put_(buf580, [reinterpret_tensor(buf30, (2359296, ), (1, ), 0)], reinterpret_tensor(buf578, (2359296, ), (1, ), 0), True)
        buf584 = reinterpret_tensor(buf518, (1024, 12, 513), (513, 525312, 1), 0); del buf518  # reuse
        # Source Nodes: [tril], Original ATen: [aten._softmax_backward_data, aten.copy, aten.masked_fill, aten.native_dropout_backward, aten.tril]
        triton_red_fused__softmax_backward_data_copy_masked_fill_native_dropout_backward_tril_13.run(unsqueeze_16, buf579, getitem_51, alias_18, buf584, 12288, 513, grid=grid(12288), stream=stream0)
        del alias_18
        del getitem_51
        buf585 = reinterpret_tensor(buf515, (6303744, ), (1, ), 0); del buf515  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_14.run(buf585, 6303744, grid=grid(6303744), stream=stream0)
        buf588 = reinterpret_tensor(buf512, (1, 1024, 12, 513), (6303744, 513, 525312, 1), 0); del buf512  # reuse
        buf589 = buf510; del buf510  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.clone, aten.copy]
        triton_poi_fused_as_strided_scatter_clone_copy_15.run(buf584, buf585, buf588, buf589, 6303744, grid=grid(6303744), stream=stream0)
        buf592 = reinterpret_tensor(buf585, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf585  # reuse
        buf594 = buf584; del buf584  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_16.run(buf588, rev_1, buf589, buf592, buf594, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_17.run(buf592, 789504, grid=grid(789504), stream=stream0)
        buf595 = reinterpret_tensor(buf589, (6303744, ), (1, ), 0); del buf589  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_14.run(buf595, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_18.run(buf594, buf595, 6303744, grid=grid(6303744), stream=stream0)
        buf597 = reinterpret_tensor(buf594, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf594  # reuse
        buf599 = reinterpret_tensor(buf588, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf588  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_add_as_strided_scatter_copy_zeros_like_19.run(buf592, buf595, buf597, buf599, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_20.run(buf599, 789504, grid=grid(789504), stream=stream0)
        buf601 = buf597; del buf597  # reuse
        buf603 = reinterpret_tensor(buf595, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf595  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_add_as_strided_scatter_copy_zeros_like_21.run(buf601, buf599, slice_64, buf603, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_22.run(buf603, 780300, grid=grid(780300), stream=stream0)
        buf606 = buf599; del buf599  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_23.run(buf603, buf606, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_24.run(buf606, 2359296, grid=grid(2359296), stream=stream0)
        buf609 = buf592; del buf592  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_23.run(buf606, buf609, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_17.run(buf609, 789504, grid=grid(789504), stream=stream0)
        buf611 = buf520; del buf520  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.clone, aten.select_backward, aten.slice_backward]
        triton_poi_fused_add_clone_select_backward_slice_backward_25.run(buf601, buf603, buf606, buf611, 9455616, grid=grid(9455616), stream=stream0)
        buf613 = reinterpret_tensor(buf579, (12, 3, 512, 512), (786432, 262144, 512, 1), 0); del buf579  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_26.run(buf611, buf609, buf613, 9437184, grid=grid(9437184), stream=stream0)
        buf614 = reinterpret_tensor(buf524, (36, 64, 512), (32768, 512, 1), 0); del buf524  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1131, reinterpret_tensor(buf613, (36, 512, 512), (262144, 512, 1), 0), out=buf614)
        del permute_1131
        buf615 = reinterpret_tensor(buf526, (36, 512, 64), (32768, 64, 1), 0); del buf526  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf613, (36, 512, 512), (262144, 512, 1), 0), permute_1132, out=buf615)
        del permute_1132
        buf616 = reinterpret_tensor(buf577, (786432, ), (1, ), 0); del buf577  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add]
        triton_poi_fused_index_add_28.run(buf616, 786432, grid=grid(786432), stream=stream0)
        buf617 = reinterpret_tensor(buf523, (12, 3, 512, 64), (98304, 32768, 64, 1), 0); del buf523  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf614, buf617, 18432, 64, grid=grid(18432, 64), stream=stream0)
        aten.index_put_(buf616, [reinterpret_tensor(buf71, (1179648, ), (1, ), 0)], reinterpret_tensor(buf617, (1179648, ), (1, ), 0), True)
        buf620 = reinterpret_tensor(buf573, (786432, ), (1, ), 0); del buf573  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add]
        triton_poi_fused_index_add_28.run(buf620, 786432, grid=grid(786432), stream=stream0)
        aten.index_put_(buf620, [reinterpret_tensor(buf71, (1179648, ), (1, ), 0)], reinterpret_tensor(buf615, (1179648, ), (1, ), 0), True)
        buf623 = reinterpret_tensor(buf554, (786432, ), (1, ), 0); del buf554  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_index_add_28.run(buf623, 786432, grid=grid(786432), stream=stream0)
        buf626 = reinterpret_tensor(buf539, (1024, 1, 768), (768, 768, 1), 0); del buf539  # reuse
        buf627 = reinterpret_tensor(buf478, (1024, 768), (768, 1), 0); del buf478  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.div]
        triton_poi_fused_as_strided_scatter_div_31.run(buf620, buf623, buf626, buf627, 786432, grid=grid(786432), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_32.run(buf626, buf627, 786432, grid=grid(786432), stream=stream0)
        buf629 = reinterpret_tensor(buf626, (1024, 768), (768, 1), 0); del buf626  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_33.run(buf580, buf629, 786432, grid=grid(786432), stream=stream0)
        buf630 = reinterpret_tensor(buf623, (1024, 768), (768, 1), 0); del buf623  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf629, permute_1143, out=buf630)
        del permute_1143
        buf631 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf629, (768, 1024), (1, 768), 0), view_375, out=buf631)
        buf632 = buf575; del buf575  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf629, buf632, 6144, 128, grid=grid(6144), stream=stream0)
        buf633 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf632, buf633, 768, 8, grid=grid(768), stream=stream0)
        buf634 = buf629; del buf629  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf616, (1024, 768), (768, 1), 0), permute_1147, out=buf634)
        del permute_1147
        buf635 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf616, (768, 1024), (1, 768), 0), view_375, out=buf635)
        buf636 = buf632; del buf632  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf616, buf636, 6144, 128, grid=grid(6144), stream=stream0)
        buf637 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf636, buf637, 768, 8, grid=grid(768), stream=stream0)
        buf638 = reinterpret_tensor(buf616, (1024, 768), (768, 1), 0); del buf616  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf627, permute_1156, out=buf638)
        del permute_1156
        buf639 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf627, (768, 1024), (1, 768), 0), view_375, out=buf639)
        del view_375
        buf640 = buf636; del buf636  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf627, buf640, 6144, 128, grid=grid(6144), stream=stream0)
        buf641 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf640, buf641, 768, 8, grid=grid(768), stream=stream0)
        buf645 = reinterpret_tensor(buf627, (1, 1024, 768), (786432, 768, 1), 0); del buf627  # reuse
        buf648 = reinterpret_tensor(buf620, (1, 1024, 768), (786432, 768, 1), 0); del buf620  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_34.run(buf569, buf630, buf634, buf638, primals_79, mul_38, div_141, getitem_47, buf645, buf648, 1024, 768, grid=grid(1024), stream=stream0)
        del div_141
        del getitem_47
        del primals_79
        buf646 = empty((768, ), device='cuda', dtype=torch.float32)
        buf647 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_35.run(buf569, buf630, buf634, buf638, mul_38, buf646, buf647, 768, 1024, grid=grid(768), stream=stream0)
        del mul_38
        buf649 = reinterpret_tensor(buf562, (1024, 3072), (3072, 1), 0); del buf562  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf648, (1024, 768), (768, 1), 0), permute_1162, out=buf649)
        del permute_1162
        buf650 = reinterpret_tensor(buf578, (768, 3072), (3072, 1), 0); del buf578  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf648, (768, 1024), (1, 768), 0), view_373, out=buf650)
        del view_373
        buf651 = buf640; del buf640  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf648, buf651, 6144, 128, grid=grid(6144), stream=stream0)
        buf652 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf651, buf652, 768, 8, grid=grid(768), stream=stream0)
        buf653 = reinterpret_tensor(buf649, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf649  # reuse
        # Source Nodes: [intermediate_output_4], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_4.run(buf653, addmm_28, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_28
        buf654 = reinterpret_tensor(buf648, (1024, 768), (768, 1), 0); del buf648  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf653, (1024, 3072), (3072, 1), 0), permute_1166, out=buf654)
        del permute_1166
        buf655 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf653, (3072, 1024), (1, 3072), 0), view_371, out=buf655)
        del view_371
        buf656 = buf565; del buf565  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf653, buf656, 24576, 128, grid=grid(24576), stream=stream0)
        buf657 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf656, buf657, 3072, 8, grid=grid(3072), stream=stream0)
        buf660 = reinterpret_tensor(buf638, (1, 1024, 768), (786432, 768, 1), 0); del buf638  # reuse
        buf663 = reinterpret_tensor(buf634, (1, 1024, 768), (786432, 768, 1), 0); del buf634  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7.run(buf645, buf654, primals_73, mul_33, div_142, getitem_43, buf660, buf663, 1024, 768, grid=grid(1024), stream=stream0)
        del div_142
        del getitem_43
        del primals_73
        buf661 = empty((768, ), device='cuda', dtype=torch.float32)
        buf662 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf645, buf654, mul_33, buf661, buf662, 768, 1024, grid=grid(768), stream=stream0)
        del mul_33
        buf664 = buf654; del buf654  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf663, (1024, 768), (768, 1), 0), permute_1170, out=buf664)
        del permute_1170
        buf665 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf663, (768, 1024), (1, 768), 0), view_369, out=buf665)
        del view_369
        buf666 = buf651; del buf651  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf663, buf666, 6144, 128, grid=grid(6144), stream=stream0)
        buf667 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf666, buf667, 768, 8, grid=grid(768), stream=stream0)
        buf668 = reinterpret_tensor(buf663, (12, 4, 256, 1, 64), (65536, 16384, 64, 64, 1), 0); del buf663  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf664, buf668, 786432, grid=grid(786432), stream=stream0)
        buf669 = empty((48, 768, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1178, reinterpret_tensor(buf668, (48, 256, 64), (16384, 64, 1), 0), out=buf669)
        del permute_1178
        buf670 = reinterpret_tensor(buf613, (48, 256, 768), (196608, 768, 1), 0); del buf613  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf668, (48, 256, 64), (16384, 64, 1), 0), permute_1179, out=buf670)
        del permute_1179
        buf671 = buf580; del buf580  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add, aten.new_zeros]
        triton_poi_fused_index_add_new_zeros_11.run(buf671, 1179648, grid=grid(1179648), stream=stream0)
        aten.index_put_(buf671, [reinterpret_tensor(buf30, (2359296, ), (1, ), 0)], reinterpret_tensor(buf669, (2359296, ), (1, ), 0), True)
        buf675 = reinterpret_tensor(buf609, (1024, 12, 513), (513, 525312, 1), 0); del buf609  # reuse
        # Source Nodes: [tril], Original ATen: [aten._softmax_backward_data, aten.copy, aten.masked_fill, aten.native_dropout_backward, aten.tril]
        triton_red_fused__softmax_backward_data_copy_masked_fill_native_dropout_backward_tril_13.run(unsqueeze_16, buf670, getitem_41, alias_19, buf675, 12288, 513, grid=grid(12288), stream=stream0)
        del alias_19
        del getitem_41
        buf676 = reinterpret_tensor(buf606, (6303744, ), (1, ), 0); del buf606  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_14.run(buf676, 6303744, grid=grid(6303744), stream=stream0)
        buf679 = reinterpret_tensor(buf603, (1, 1024, 12, 513), (6303744, 513, 525312, 1), 0); del buf603  # reuse
        buf680 = buf601; del buf601  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.clone, aten.copy]
        triton_poi_fused_as_strided_scatter_clone_copy_15.run(buf675, buf676, buf679, buf680, 6303744, grid=grid(6303744), stream=stream0)
        buf683 = reinterpret_tensor(buf676, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf676  # reuse
        buf685 = buf675; del buf675  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_16.run(buf679, rev_1, buf680, buf683, buf685, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_17.run(buf683, 789504, grid=grid(789504), stream=stream0)
        buf686 = reinterpret_tensor(buf680, (6303744, ), (1, ), 0); del buf680  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_14.run(buf686, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_18.run(buf685, buf686, 6303744, grid=grid(6303744), stream=stream0)
        buf688 = reinterpret_tensor(buf685, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf685  # reuse
        buf690 = reinterpret_tensor(buf679, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf679  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_add_as_strided_scatter_copy_zeros_like_19.run(buf683, buf686, buf688, buf690, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_20.run(buf690, 789504, grid=grid(789504), stream=stream0)
        buf692 = buf688; del buf688  # reuse
        buf694 = reinterpret_tensor(buf686, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf686  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_add_as_strided_scatter_copy_zeros_like_21.run(buf692, buf690, slice_64, buf694, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_22.run(buf694, 780300, grid=grid(780300), stream=stream0)
        buf697 = buf690; del buf690  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_23.run(buf694, buf697, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_24.run(buf697, 2359296, grid=grid(2359296), stream=stream0)
        buf700 = buf683; del buf683  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_23.run(buf697, buf700, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_17.run(buf700, 789504, grid=grid(789504), stream=stream0)
        buf702 = buf611; del buf611  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.clone, aten.select_backward, aten.slice_backward]
        triton_poi_fused_add_clone_select_backward_slice_backward_25.run(buf692, buf694, buf697, buf702, 9455616, grid=grid(9455616), stream=stream0)
        buf704 = reinterpret_tensor(buf670, (12, 3, 512, 512), (786432, 262144, 512, 1), 0); del buf670  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_26.run(buf702, buf700, buf704, 9437184, grid=grid(9437184), stream=stream0)
        buf705 = reinterpret_tensor(buf615, (36, 64, 512), (32768, 512, 1), 0); del buf615  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1189, reinterpret_tensor(buf704, (36, 512, 512), (262144, 512, 1), 0), out=buf705)
        del permute_1189
        buf706 = reinterpret_tensor(buf617, (36, 512, 64), (32768, 64, 1), 0); del buf617  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf704, (36, 512, 512), (262144, 512, 1), 0), permute_1190, out=buf706)
        del permute_1190
        buf707 = reinterpret_tensor(buf668, (786432, ), (1, ), 0); del buf668  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add]
        triton_poi_fused_index_add_28.run(buf707, 786432, grid=grid(786432), stream=stream0)
        buf708 = reinterpret_tensor(buf614, (12, 3, 512, 64), (98304, 32768, 64, 1), 0); del buf614  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf705, buf708, 18432, 64, grid=grid(18432, 64), stream=stream0)
        aten.index_put_(buf707, [reinterpret_tensor(buf71, (1179648, ), (1, ), 0)], reinterpret_tensor(buf708, (1179648, ), (1, ), 0), True)
        buf711 = reinterpret_tensor(buf664, (786432, ), (1, ), 0); del buf664  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add]
        triton_poi_fused_index_add_28.run(buf711, 786432, grid=grid(786432), stream=stream0)
        aten.index_put_(buf711, [reinterpret_tensor(buf71, (1179648, ), (1, ), 0)], reinterpret_tensor(buf706, (1179648, ), (1, ), 0), True)
        buf714 = reinterpret_tensor(buf645, (786432, ), (1, ), 0); del buf645  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_index_add_28.run(buf714, 786432, grid=grid(786432), stream=stream0)
        buf717 = reinterpret_tensor(buf630, (1024, 1, 768), (768, 768, 1), 0); del buf630  # reuse
        buf718 = reinterpret_tensor(buf569, (1024, 768), (768, 1), 0); del buf569  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.div]
        triton_poi_fused_as_strided_scatter_div_31.run(buf711, buf714, buf717, buf718, 786432, grid=grid(786432), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_32.run(buf717, buf718, 786432, grid=grid(786432), stream=stream0)
        buf720 = reinterpret_tensor(buf717, (1024, 768), (768, 1), 0); del buf717  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_33.run(buf671, buf720, 786432, grid=grid(786432), stream=stream0)
        buf721 = reinterpret_tensor(buf714, (1024, 768), (768, 1), 0); del buf714  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf720, permute_1201, out=buf721)
        del permute_1201
        buf722 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf720, (768, 1024), (1, 768), 0), view_300, out=buf722)
        buf723 = buf666; del buf666  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf720, buf723, 6144, 128, grid=grid(6144), stream=stream0)
        buf724 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf723, buf724, 768, 8, grid=grid(768), stream=stream0)
        buf725 = buf720; del buf720  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf707, (1024, 768), (768, 1), 0), permute_1205, out=buf725)
        del permute_1205
        buf726 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf707, (768, 1024), (1, 768), 0), view_300, out=buf726)
        buf727 = buf723; del buf723  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf707, buf727, 6144, 128, grid=grid(6144), stream=stream0)
        buf728 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf727, buf728, 768, 8, grid=grid(768), stream=stream0)
        buf729 = reinterpret_tensor(buf707, (1024, 768), (768, 1), 0); del buf707  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf718, permute_1214, out=buf729)
        del permute_1214
        buf730 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf718, (768, 1024), (1, 768), 0), view_300, out=buf730)
        del view_300
        buf731 = buf727; del buf727  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf718, buf731, 6144, 128, grid=grid(6144), stream=stream0)
        buf732 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf731, buf732, 768, 8, grid=grid(768), stream=stream0)
        buf736 = reinterpret_tensor(buf718, (1, 1024, 768), (786432, 768, 1), 0); del buf718  # reuse
        buf739 = reinterpret_tensor(buf711, (1, 1024, 768), (786432, 768, 1), 0); del buf711  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_34.run(buf660, buf721, buf725, buf729, primals_63, mul_30, div_144, getitem_37, buf736, buf739, 1024, 768, grid=grid(1024), stream=stream0)
        del div_144
        del getitem_37
        del primals_63
        buf737 = empty((768, ), device='cuda', dtype=torch.float32)
        buf738 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_35.run(buf660, buf721, buf725, buf729, mul_30, buf737, buf738, 768, 1024, grid=grid(768), stream=stream0)
        del mul_30
        buf740 = reinterpret_tensor(buf653, (1024, 3072), (3072, 1), 0); del buf653  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf739, (1024, 768), (768, 1), 0), permute_1220, out=buf740)
        del permute_1220
        buf741 = reinterpret_tensor(buf669, (768, 3072), (3072, 1), 0); del buf669  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf739, (768, 1024), (1, 768), 0), view_298, out=buf741)
        del view_298
        buf742 = buf731; del buf731  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf739, buf742, 6144, 128, grid=grid(6144), stream=stream0)
        buf743 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf742, buf743, 768, 8, grid=grid(768), stream=stream0)
        buf744 = reinterpret_tensor(buf740, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf740  # reuse
        # Source Nodes: [intermediate_output_3], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_4.run(buf744, addmm_22, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_22
        buf745 = reinterpret_tensor(buf739, (1024, 768), (768, 1), 0); del buf739  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf744, (1024, 3072), (3072, 1), 0), permute_1224, out=buf745)
        del permute_1224
        buf746 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf744, (3072, 1024), (1, 3072), 0), view_296, out=buf746)
        del view_296
        buf747 = buf656; del buf656  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf744, buf747, 24576, 128, grid=grid(24576), stream=stream0)
        buf748 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf747, buf748, 3072, 8, grid=grid(3072), stream=stream0)
        buf751 = reinterpret_tensor(buf729, (1, 1024, 768), (786432, 768, 1), 0); del buf729  # reuse
        buf754 = reinterpret_tensor(buf725, (1, 1024, 768), (786432, 768, 1), 0); del buf725  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7.run(buf736, buf745, primals_57, mul_25, div_145, getitem_33, buf751, buf754, 1024, 768, grid=grid(1024), stream=stream0)
        del div_145
        del getitem_33
        del primals_57
        buf752 = empty((768, ), device='cuda', dtype=torch.float32)
        buf753 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf736, buf745, mul_25, buf752, buf753, 768, 1024, grid=grid(768), stream=stream0)
        del mul_25
        buf755 = buf745; del buf745  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf754, (1024, 768), (768, 1), 0), permute_1228, out=buf755)
        del permute_1228
        buf756 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf754, (768, 1024), (1, 768), 0), view_294, out=buf756)
        del view_294
        buf757 = buf742; del buf742  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf754, buf757, 6144, 128, grid=grid(6144), stream=stream0)
        buf758 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf757, buf758, 768, 8, grid=grid(768), stream=stream0)
        buf759 = reinterpret_tensor(buf754, (12, 4, 256, 1, 64), (65536, 16384, 64, 64, 1), 0); del buf754  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf755, buf759, 786432, grid=grid(786432), stream=stream0)
        buf760 = empty((48, 768, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1236, reinterpret_tensor(buf759, (48, 256, 64), (16384, 64, 1), 0), out=buf760)
        del permute_1236
        buf761 = reinterpret_tensor(buf704, (48, 256, 768), (196608, 768, 1), 0); del buf704  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf759, (48, 256, 64), (16384, 64, 1), 0), permute_1237, out=buf761)
        del permute_1237
        buf762 = buf671; del buf671  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add, aten.new_zeros]
        triton_poi_fused_index_add_new_zeros_11.run(buf762, 1179648, grid=grid(1179648), stream=stream0)
        aten.index_put_(buf762, [reinterpret_tensor(buf30, (2359296, ), (1, ), 0)], reinterpret_tensor(buf760, (2359296, ), (1, ), 0), True)
        buf766 = reinterpret_tensor(buf700, (1024, 12, 513), (513, 525312, 1), 0); del buf700  # reuse
        # Source Nodes: [tril], Original ATen: [aten._softmax_backward_data, aten.copy, aten.masked_fill, aten.native_dropout_backward, aten.tril]
        triton_red_fused__softmax_backward_data_copy_masked_fill_native_dropout_backward_tril_13.run(unsqueeze_16, buf761, getitem_31, alias_20, buf766, 12288, 513, grid=grid(12288), stream=stream0)
        del alias_20
        del getitem_31
        buf767 = reinterpret_tensor(buf697, (6303744, ), (1, ), 0); del buf697  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_14.run(buf767, 6303744, grid=grid(6303744), stream=stream0)
        buf770 = reinterpret_tensor(buf694, (1, 1024, 12, 513), (6303744, 513, 525312, 1), 0); del buf694  # reuse
        buf771 = buf692; del buf692  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.clone, aten.copy]
        triton_poi_fused_as_strided_scatter_clone_copy_15.run(buf766, buf767, buf770, buf771, 6303744, grid=grid(6303744), stream=stream0)
        buf774 = reinterpret_tensor(buf767, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf767  # reuse
        buf776 = buf766; del buf766  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_16.run(buf770, rev_1, buf771, buf774, buf776, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_17.run(buf774, 789504, grid=grid(789504), stream=stream0)
        buf777 = reinterpret_tensor(buf771, (6303744, ), (1, ), 0); del buf771  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_14.run(buf777, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_18.run(buf776, buf777, 6303744, grid=grid(6303744), stream=stream0)
        buf779 = reinterpret_tensor(buf776, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf776  # reuse
        buf781 = reinterpret_tensor(buf770, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf770  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_add_as_strided_scatter_copy_zeros_like_19.run(buf774, buf777, buf779, buf781, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_20.run(buf781, 789504, grid=grid(789504), stream=stream0)
        buf783 = buf779; del buf779  # reuse
        buf785 = reinterpret_tensor(buf777, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf777  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_add_as_strided_scatter_copy_zeros_like_21.run(buf783, buf781, slice_64, buf785, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_22.run(buf785, 780300, grid=grid(780300), stream=stream0)
        buf788 = buf781; del buf781  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_23.run(buf785, buf788, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_24.run(buf788, 2359296, grid=grid(2359296), stream=stream0)
        buf791 = buf774; del buf774  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_23.run(buf788, buf791, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_17.run(buf791, 789504, grid=grid(789504), stream=stream0)
        buf793 = buf702; del buf702  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.clone, aten.select_backward, aten.slice_backward]
        triton_poi_fused_add_clone_select_backward_slice_backward_25.run(buf783, buf785, buf788, buf793, 9455616, grid=grid(9455616), stream=stream0)
        buf795 = reinterpret_tensor(buf761, (12, 3, 512, 512), (786432, 262144, 512, 1), 0); del buf761  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_26.run(buf793, buf791, buf795, 9437184, grid=grid(9437184), stream=stream0)
        buf796 = reinterpret_tensor(buf706, (36, 64, 512), (32768, 512, 1), 0); del buf706  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1247, reinterpret_tensor(buf795, (36, 512, 512), (262144, 512, 1), 0), out=buf796)
        del permute_1247
        buf797 = reinterpret_tensor(buf708, (36, 512, 64), (32768, 64, 1), 0); del buf708  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf795, (36, 512, 512), (262144, 512, 1), 0), permute_1248, out=buf797)
        del permute_1248
        buf798 = reinterpret_tensor(buf759, (786432, ), (1, ), 0); del buf759  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add]
        triton_poi_fused_index_add_28.run(buf798, 786432, grid=grid(786432), stream=stream0)
        buf799 = reinterpret_tensor(buf705, (12, 3, 512, 64), (98304, 32768, 64, 1), 0); del buf705  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf796, buf799, 18432, 64, grid=grid(18432, 64), stream=stream0)
        aten.index_put_(buf798, [reinterpret_tensor(buf71, (1179648, ), (1, ), 0)], reinterpret_tensor(buf799, (1179648, ), (1, ), 0), True)
        buf802 = reinterpret_tensor(buf755, (786432, ), (1, ), 0); del buf755  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add]
        triton_poi_fused_index_add_28.run(buf802, 786432, grid=grid(786432), stream=stream0)
        aten.index_put_(buf802, [reinterpret_tensor(buf71, (1179648, ), (1, ), 0)], reinterpret_tensor(buf797, (1179648, ), (1, ), 0), True)
        buf805 = reinterpret_tensor(buf736, (786432, ), (1, ), 0); del buf736  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_index_add_28.run(buf805, 786432, grid=grid(786432), stream=stream0)
        buf808 = reinterpret_tensor(buf721, (1024, 1, 768), (768, 768, 1), 0); del buf721  # reuse
        buf809 = reinterpret_tensor(buf660, (1024, 768), (768, 1), 0); del buf660  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.div]
        triton_poi_fused_as_strided_scatter_div_31.run(buf802, buf805, buf808, buf809, 786432, grid=grid(786432), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_32.run(buf808, buf809, 786432, grid=grid(786432), stream=stream0)
        buf811 = reinterpret_tensor(buf808, (1024, 768), (768, 1), 0); del buf808  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_33.run(buf762, buf811, 786432, grid=grid(786432), stream=stream0)
        buf812 = reinterpret_tensor(buf805, (1024, 768), (768, 1), 0); del buf805  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf811, permute_1259, out=buf812)
        del permute_1259
        buf813 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf811, (768, 1024), (1, 768), 0), view_225, out=buf813)
        buf814 = buf757; del buf757  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf811, buf814, 6144, 128, grid=grid(6144), stream=stream0)
        buf815 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf814, buf815, 768, 8, grid=grid(768), stream=stream0)
        buf816 = buf811; del buf811  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf798, (1024, 768), (768, 1), 0), permute_1263, out=buf816)
        del permute_1263
        buf817 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf798, (768, 1024), (1, 768), 0), view_225, out=buf817)
        buf818 = buf814; del buf814  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf798, buf818, 6144, 128, grid=grid(6144), stream=stream0)
        buf819 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf818, buf819, 768, 8, grid=grid(768), stream=stream0)
        buf820 = reinterpret_tensor(buf798, (1024, 768), (768, 1), 0); del buf798  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf809, permute_1272, out=buf820)
        del permute_1272
        buf821 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf809, (768, 1024), (1, 768), 0), view_225, out=buf821)
        del view_225
        buf822 = buf818; del buf818  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf809, buf822, 6144, 128, grid=grid(6144), stream=stream0)
        buf823 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf822, buf823, 768, 8, grid=grid(768), stream=stream0)
        buf827 = reinterpret_tensor(buf809, (1, 1024, 768), (786432, 768, 1), 0); del buf809  # reuse
        buf830 = reinterpret_tensor(buf802, (1, 1024, 768), (786432, 768, 1), 0); del buf802  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_34.run(buf751, buf812, buf816, buf820, primals_47, mul_22, div_147, getitem_27, buf827, buf830, 1024, 768, grid=grid(1024), stream=stream0)
        del div_147
        del getitem_27
        del primals_47
        buf828 = empty((768, ), device='cuda', dtype=torch.float32)
        buf829 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_35.run(buf751, buf812, buf816, buf820, mul_22, buf828, buf829, 768, 1024, grid=grid(768), stream=stream0)
        del mul_22
        buf831 = reinterpret_tensor(buf744, (1024, 3072), (3072, 1), 0); del buf744  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf830, (1024, 768), (768, 1), 0), permute_1278, out=buf831)
        del permute_1278
        buf832 = reinterpret_tensor(buf760, (768, 3072), (3072, 1), 0); del buf760  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf830, (768, 1024), (1, 768), 0), view_223, out=buf832)
        del view_223
        buf833 = buf822; del buf822  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf830, buf833, 6144, 128, grid=grid(6144), stream=stream0)
        buf834 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf833, buf834, 768, 8, grid=grid(768), stream=stream0)
        buf835 = reinterpret_tensor(buf831, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf831  # reuse
        # Source Nodes: [intermediate_output_2], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_4.run(buf835, addmm_16, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_16
        buf836 = reinterpret_tensor(buf830, (1024, 768), (768, 1), 0); del buf830  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf835, (1024, 3072), (3072, 1), 0), permute_1282, out=buf836)
        del permute_1282
        buf837 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf835, (3072, 1024), (1, 3072), 0), view_221, out=buf837)
        del view_221
        buf838 = buf747; del buf747  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf835, buf838, 24576, 128, grid=grid(24576), stream=stream0)
        buf839 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf838, buf839, 3072, 8, grid=grid(3072), stream=stream0)
        buf842 = reinterpret_tensor(buf820, (1, 1024, 768), (786432, 768, 1), 0); del buf820  # reuse
        buf845 = reinterpret_tensor(buf816, (1, 1024, 768), (786432, 768, 1), 0); del buf816  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7.run(buf827, buf836, primals_41, mul_17, div_148, getitem_23, buf842, buf845, 1024, 768, grid=grid(1024), stream=stream0)
        del div_148
        del getitem_23
        del primals_41
        buf843 = empty((768, ), device='cuda', dtype=torch.float32)
        buf844 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf827, buf836, mul_17, buf843, buf844, 768, 1024, grid=grid(768), stream=stream0)
        del mul_17
        buf846 = buf836; del buf836  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf845, (1024, 768), (768, 1), 0), permute_1286, out=buf846)
        del permute_1286
        buf847 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf845, (768, 1024), (1, 768), 0), view_219, out=buf847)
        del view_219
        buf848 = buf833; del buf833  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf845, buf848, 6144, 128, grid=grid(6144), stream=stream0)
        buf849 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf848, buf849, 768, 8, grid=grid(768), stream=stream0)
        buf850 = reinterpret_tensor(buf845, (12, 4, 256, 1, 64), (65536, 16384, 64, 64, 1), 0); del buf845  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf846, buf850, 786432, grid=grid(786432), stream=stream0)
        buf851 = empty((48, 768, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1294, reinterpret_tensor(buf850, (48, 256, 64), (16384, 64, 1), 0), out=buf851)
        del permute_1294
        buf852 = reinterpret_tensor(buf795, (48, 256, 768), (196608, 768, 1), 0); del buf795  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf850, (48, 256, 64), (16384, 64, 1), 0), permute_1295, out=buf852)
        del permute_1295
        buf853 = buf762; del buf762  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add, aten.new_zeros]
        triton_poi_fused_index_add_new_zeros_11.run(buf853, 1179648, grid=grid(1179648), stream=stream0)
        aten.index_put_(buf853, [reinterpret_tensor(buf30, (2359296, ), (1, ), 0)], reinterpret_tensor(buf851, (2359296, ), (1, ), 0), True)
        buf857 = reinterpret_tensor(buf791, (1024, 12, 513), (513, 525312, 1), 0); del buf791  # reuse
        # Source Nodes: [tril], Original ATen: [aten._softmax_backward_data, aten.copy, aten.masked_fill, aten.native_dropout_backward, aten.tril]
        triton_red_fused__softmax_backward_data_copy_masked_fill_native_dropout_backward_tril_13.run(unsqueeze_16, buf852, getitem_21, alias_21, buf857, 12288, 513, grid=grid(12288), stream=stream0)
        del alias_21
        del getitem_21
        buf858 = reinterpret_tensor(buf788, (6303744, ), (1, ), 0); del buf788  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_14.run(buf858, 6303744, grid=grid(6303744), stream=stream0)
        buf861 = reinterpret_tensor(buf785, (1, 1024, 12, 513), (6303744, 513, 525312, 1), 0); del buf785  # reuse
        buf862 = buf783; del buf783  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.clone, aten.copy]
        triton_poi_fused_as_strided_scatter_clone_copy_15.run(buf857, buf858, buf861, buf862, 6303744, grid=grid(6303744), stream=stream0)
        buf865 = reinterpret_tensor(buf858, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf858  # reuse
        buf867 = buf857; del buf857  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_16.run(buf861, rev_1, buf862, buf865, buf867, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_17.run(buf865, 789504, grid=grid(789504), stream=stream0)
        buf868 = reinterpret_tensor(buf862, (6303744, ), (1, ), 0); del buf862  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_14.run(buf868, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_18.run(buf867, buf868, 6303744, grid=grid(6303744), stream=stream0)
        buf870 = reinterpret_tensor(buf867, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf867  # reuse
        buf872 = reinterpret_tensor(buf861, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf861  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_add_as_strided_scatter_copy_zeros_like_19.run(buf865, buf868, buf870, buf872, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_20.run(buf872, 789504, grid=grid(789504), stream=stream0)
        buf874 = buf870; del buf870  # reuse
        buf876 = reinterpret_tensor(buf868, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf868  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_add_as_strided_scatter_copy_zeros_like_21.run(buf874, buf872, slice_64, buf876, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_22.run(buf876, 780300, grid=grid(780300), stream=stream0)
        buf879 = buf872; del buf872  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_23.run(buf876, buf879, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_24.run(buf879, 2359296, grid=grid(2359296), stream=stream0)
        buf882 = buf865; del buf865  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_23.run(buf879, buf882, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_17.run(buf882, 789504, grid=grid(789504), stream=stream0)
        buf884 = buf793; del buf793  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.clone, aten.select_backward, aten.slice_backward]
        triton_poi_fused_add_clone_select_backward_slice_backward_25.run(buf874, buf876, buf879, buf884, 9455616, grid=grid(9455616), stream=stream0)
        buf886 = reinterpret_tensor(buf852, (12, 3, 512, 512), (786432, 262144, 512, 1), 0); del buf852  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_26.run(buf884, buf882, buf886, 9437184, grid=grid(9437184), stream=stream0)
        buf887 = reinterpret_tensor(buf797, (36, 64, 512), (32768, 512, 1), 0); del buf797  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1305, reinterpret_tensor(buf886, (36, 512, 512), (262144, 512, 1), 0), out=buf887)
        del permute_1305
        buf888 = reinterpret_tensor(buf799, (36, 512, 64), (32768, 64, 1), 0); del buf799  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf886, (36, 512, 512), (262144, 512, 1), 0), permute_1306, out=buf888)
        del permute_1306
        buf889 = reinterpret_tensor(buf850, (786432, ), (1, ), 0); del buf850  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add]
        triton_poi_fused_index_add_28.run(buf889, 786432, grid=grid(786432), stream=stream0)
        buf890 = reinterpret_tensor(buf796, (12, 3, 512, 64), (98304, 32768, 64, 1), 0); del buf796  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf887, buf890, 18432, 64, grid=grid(18432, 64), stream=stream0)
        aten.index_put_(buf889, [reinterpret_tensor(buf71, (1179648, ), (1, ), 0)], reinterpret_tensor(buf890, (1179648, ), (1, ), 0), True)
        buf893 = reinterpret_tensor(buf846, (786432, ), (1, ), 0); del buf846  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add]
        triton_poi_fused_index_add_28.run(buf893, 786432, grid=grid(786432), stream=stream0)
        aten.index_put_(buf893, [reinterpret_tensor(buf71, (1179648, ), (1, ), 0)], reinterpret_tensor(buf888, (1179648, ), (1, ), 0), True)
        buf896 = reinterpret_tensor(buf827, (786432, ), (1, ), 0); del buf827  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_index_add_28.run(buf896, 786432, grid=grid(786432), stream=stream0)
        buf899 = reinterpret_tensor(buf812, (1024, 1, 768), (768, 768, 1), 0); del buf812  # reuse
        buf900 = reinterpret_tensor(buf751, (1024, 768), (768, 1), 0); del buf751  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.div]
        triton_poi_fused_as_strided_scatter_div_31.run(buf893, buf896, buf899, buf900, 786432, grid=grid(786432), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_32.run(buf899, buf900, 786432, grid=grid(786432), stream=stream0)
        buf902 = reinterpret_tensor(buf899, (1024, 768), (768, 1), 0); del buf899  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_33.run(buf853, buf902, 786432, grid=grid(786432), stream=stream0)
        buf903 = reinterpret_tensor(buf896, (1024, 768), (768, 1), 0); del buf896  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf902, permute_1317, out=buf903)
        del permute_1317
        buf904 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf902, (768, 1024), (1, 768), 0), view_150, out=buf904)
        buf905 = buf848; del buf848  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf902, buf905, 6144, 128, grid=grid(6144), stream=stream0)
        buf906 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf905, buf906, 768, 8, grid=grid(768), stream=stream0)
        buf907 = buf902; del buf902  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf889, (1024, 768), (768, 1), 0), permute_1321, out=buf907)
        del permute_1321
        buf908 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf889, (768, 1024), (1, 768), 0), view_150, out=buf908)
        buf909 = buf905; del buf905  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf889, buf909, 6144, 128, grid=grid(6144), stream=stream0)
        buf910 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf909, buf910, 768, 8, grid=grid(768), stream=stream0)
        buf911 = reinterpret_tensor(buf889, (1024, 768), (768, 1), 0); del buf889  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf900, permute_1330, out=buf911)
        del permute_1330
        buf912 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf900, (768, 1024), (1, 768), 0), view_150, out=buf912)
        del view_150
        buf913 = buf909; del buf909  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf900, buf913, 6144, 128, grid=grid(6144), stream=stream0)
        buf914 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf913, buf914, 768, 8, grid=grid(768), stream=stream0)
        buf918 = reinterpret_tensor(buf900, (1, 1024, 768), (786432, 768, 1), 0); del buf900  # reuse
        buf921 = reinterpret_tensor(buf893, (1, 1024, 768), (786432, 768, 1), 0); del buf893  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_34.run(buf842, buf903, buf907, buf911, primals_31, mul_14, div_150, getitem_17, buf918, buf921, 1024, 768, grid=grid(1024), stream=stream0)
        del div_150
        del getitem_17
        del primals_31
        buf919 = empty((768, ), device='cuda', dtype=torch.float32)
        buf920 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_35.run(buf842, buf903, buf907, buf911, mul_14, buf919, buf920, 768, 1024, grid=grid(768), stream=stream0)
        del mul_14
        buf922 = reinterpret_tensor(buf835, (1024, 3072), (3072, 1), 0); del buf835  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf921, (1024, 768), (768, 1), 0), permute_1336, out=buf922)
        del permute_1336
        buf923 = reinterpret_tensor(buf851, (768, 3072), (3072, 1), 0); del buf851  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf921, (768, 1024), (1, 768), 0), view_148, out=buf923)
        del view_148
        buf924 = buf913; del buf913  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf921, buf924, 6144, 128, grid=grid(6144), stream=stream0)
        buf925 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf924, buf925, 768, 8, grid=grid(768), stream=stream0)
        buf926 = reinterpret_tensor(buf922, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf922  # reuse
        # Source Nodes: [intermediate_output_1], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_4.run(buf926, addmm_10, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_10
        buf927 = reinterpret_tensor(buf921, (1024, 768), (768, 1), 0); del buf921  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf926, (1024, 3072), (3072, 1), 0), permute_1340, out=buf927)
        del permute_1340
        buf928 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf926, (3072, 1024), (1, 3072), 0), view_146, out=buf928)
        del view_146
        buf929 = buf838; del buf838  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf926, buf929, 24576, 128, grid=grid(24576), stream=stream0)
        buf930 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf929, buf930, 3072, 8, grid=grid(3072), stream=stream0)
        buf933 = reinterpret_tensor(buf911, (1, 1024, 768), (786432, 768, 1), 0); del buf911  # reuse
        buf936 = reinterpret_tensor(buf907, (1, 1024, 768), (786432, 768, 1), 0); del buf907  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7.run(buf918, buf927, primals_25, mul_9, div_151, getitem_13, buf933, buf936, 1024, 768, grid=grid(1024), stream=stream0)
        del div_151
        del getitem_13
        del primals_25
        buf934 = empty((768, ), device='cuda', dtype=torch.float32)
        buf935 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf918, buf927, mul_9, buf934, buf935, 768, 1024, grid=grid(768), stream=stream0)
        del mul_9
        buf937 = buf927; del buf927  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf936, (1024, 768), (768, 1), 0), permute_1344, out=buf937)
        del permute_1344
        buf938 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf936, (768, 1024), (1, 768), 0), view_144, out=buf938)
        del view_144
        buf939 = buf924; del buf924  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf936, buf939, 6144, 128, grid=grid(6144), stream=stream0)
        buf940 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf939, buf940, 768, 8, grid=grid(768), stream=stream0)
        buf941 = reinterpret_tensor(buf936, (12, 4, 256, 1, 64), (65536, 16384, 64, 64, 1), 0); del buf936  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf937, buf941, 786432, grid=grid(786432), stream=stream0)
        buf942 = empty((48, 768, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1352, reinterpret_tensor(buf941, (48, 256, 64), (16384, 64, 1), 0), out=buf942)
        del permute_1352
        buf943 = reinterpret_tensor(buf886, (48, 256, 768), (196608, 768, 1), 0); del buf886  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf941, (48, 256, 64), (16384, 64, 1), 0), permute_1353, out=buf943)
        del permute_1353
        buf944 = buf853; del buf853  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add, aten.new_zeros]
        triton_poi_fused_index_add_new_zeros_11.run(buf944, 1179648, grid=grid(1179648), stream=stream0)
        aten.index_put_(buf944, [reinterpret_tensor(buf30, (2359296, ), (1, ), 0)], reinterpret_tensor(buf942, (2359296, ), (1, ), 0), True)
        buf948 = reinterpret_tensor(buf882, (1024, 12, 513), (513, 525312, 1), 0); del buf882  # reuse
        # Source Nodes: [tril], Original ATen: [aten._softmax_backward_data, aten.copy, aten.masked_fill, aten.native_dropout_backward, aten.tril]
        triton_red_fused__softmax_backward_data_copy_masked_fill_native_dropout_backward_tril_13.run(unsqueeze_16, buf943, getitem_11, alias_22, buf948, 12288, 513, grid=grid(12288), stream=stream0)
        del alias_22
        del getitem_11
        buf949 = reinterpret_tensor(buf879, (6303744, ), (1, ), 0); del buf879  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_14.run(buf949, 6303744, grid=grid(6303744), stream=stream0)
        buf952 = reinterpret_tensor(buf876, (1, 1024, 12, 513), (6303744, 513, 525312, 1), 0); del buf876  # reuse
        buf953 = buf874; del buf874  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.clone, aten.copy]
        triton_poi_fused_as_strided_scatter_clone_copy_15.run(buf948, buf949, buf952, buf953, 6303744, grid=grid(6303744), stream=stream0)
        buf956 = reinterpret_tensor(buf949, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf949  # reuse
        buf958 = buf948; del buf948  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_16.run(buf952, rev_1, buf953, buf956, buf958, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_17.run(buf956, 789504, grid=grid(789504), stream=stream0)
        buf959 = reinterpret_tensor(buf953, (6303744, ), (1, ), 0); del buf953  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_14.run(buf959, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_18.run(buf958, buf959, 6303744, grid=grid(6303744), stream=stream0)
        buf961 = reinterpret_tensor(buf958, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf958  # reuse
        buf963 = reinterpret_tensor(buf952, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf952  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_add_as_strided_scatter_copy_zeros_like_19.run(buf956, buf959, buf961, buf963, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_20.run(buf963, 789504, grid=grid(789504), stream=stream0)
        buf965 = buf961; del buf961  # reuse
        buf967 = reinterpret_tensor(buf959, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf959  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_add_as_strided_scatter_copy_zeros_like_21.run(buf965, buf963, slice_64, buf967, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_22.run(buf967, 780300, grid=grid(780300), stream=stream0)
        buf970 = buf963; del buf963  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_23.run(buf967, buf970, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_24.run(buf970, 2359296, grid=grid(2359296), stream=stream0)
        buf973 = buf956; del buf956  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_23.run(buf970, buf973, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_17.run(buf973, 789504, grid=grid(789504), stream=stream0)
        buf975 = buf884; del buf884  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.clone, aten.select_backward, aten.slice_backward]
        triton_poi_fused_add_clone_select_backward_slice_backward_25.run(buf965, buf967, buf970, buf975, 9455616, grid=grid(9455616), stream=stream0)
        buf977 = reinterpret_tensor(buf943, (12, 3, 512, 512), (786432, 262144, 512, 1), 0); del buf943  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_26.run(buf975, buf973, buf977, 9437184, grid=grid(9437184), stream=stream0)
        buf978 = reinterpret_tensor(buf888, (36, 64, 512), (32768, 512, 1), 0); del buf888  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1363, reinterpret_tensor(buf977, (36, 512, 512), (262144, 512, 1), 0), out=buf978)
        del permute_1363
        buf979 = reinterpret_tensor(buf890, (36, 512, 64), (32768, 64, 1), 0); del buf890  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf977, (36, 512, 512), (262144, 512, 1), 0), permute_1364, out=buf979)
        del permute_1364
        buf980 = reinterpret_tensor(buf941, (786432, ), (1, ), 0); del buf941  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add]
        triton_poi_fused_index_add_28.run(buf980, 786432, grid=grid(786432), stream=stream0)
        buf981 = reinterpret_tensor(buf887, (12, 3, 512, 64), (98304, 32768, 64, 1), 0); del buf887  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf978, buf981, 18432, 64, grid=grid(18432, 64), stream=stream0)
        aten.index_put_(buf980, [reinterpret_tensor(buf71, (1179648, ), (1, ), 0)], reinterpret_tensor(buf981, (1179648, ), (1, ), 0), True)
        buf984 = reinterpret_tensor(buf937, (786432, ), (1, ), 0); del buf937  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add]
        triton_poi_fused_index_add_28.run(buf984, 786432, grid=grid(786432), stream=stream0)
        aten.index_put_(buf984, [reinterpret_tensor(buf71, (1179648, ), (1, ), 0)], reinterpret_tensor(buf979, (1179648, ), (1, ), 0), True)
        buf987 = reinterpret_tensor(buf918, (786432, ), (1, ), 0); del buf918  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_index_add_28.run(buf987, 786432, grid=grid(786432), stream=stream0)
        buf990 = reinterpret_tensor(buf903, (1024, 1, 768), (768, 768, 1), 0); del buf903  # reuse
        buf991 = reinterpret_tensor(buf842, (1024, 768), (768, 1), 0); del buf842  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.div]
        triton_poi_fused_as_strided_scatter_div_31.run(buf984, buf987, buf990, buf991, 786432, grid=grid(786432), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_32.run(buf990, buf991, 786432, grid=grid(786432), stream=stream0)
        buf993 = reinterpret_tensor(buf990, (1024, 768), (768, 1), 0); del buf990  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_33.run(buf944, buf993, 786432, grid=grid(786432), stream=stream0)
        buf994 = reinterpret_tensor(buf987, (1024, 768), (768, 1), 0); del buf987  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf993, permute_1375, out=buf994)
        del permute_1375
        buf995 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf993, (768, 1024), (1, 768), 0), view_75, out=buf995)
        buf996 = buf939; del buf939  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf993, buf996, 6144, 128, grid=grid(6144), stream=stream0)
        buf997 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf996, buf997, 768, 8, grid=grid(768), stream=stream0)
        buf998 = buf993; del buf993  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf980, (1024, 768), (768, 1), 0), permute_1379, out=buf998)
        del permute_1379
        buf999 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf980, (768, 1024), (1, 768), 0), view_75, out=buf999)
        buf1000 = buf996; del buf996  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf980, buf1000, 6144, 128, grid=grid(6144), stream=stream0)
        buf1001 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf1000, buf1001, 768, 8, grid=grid(768), stream=stream0)
        buf1002 = reinterpret_tensor(buf980, (1024, 768), (768, 1), 0); del buf980  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf991, permute_1388, out=buf1002)
        del permute_1388
        buf1003 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf991, (768, 1024), (1, 768), 0), view_75, out=buf1003)
        del view_75
        buf1004 = buf1000; del buf1000  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf991, buf1004, 6144, 128, grid=grid(6144), stream=stream0)
        buf1005 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf1004, buf1005, 768, 8, grid=grid(768), stream=stream0)
        buf1009 = reinterpret_tensor(buf991, (1, 1024, 768), (786432, 768, 1), 0); del buf991  # reuse
        buf1012 = reinterpret_tensor(buf984, (1, 1024, 768), (786432, 768, 1), 0); del buf984  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_34.run(buf933, buf994, buf998, buf1002, primals_15, mul_6, div_153, getitem_7, buf1009, buf1012, 1024, 768, grid=grid(1024), stream=stream0)
        del div_153
        del getitem_7
        del primals_15
        buf1010 = empty((768, ), device='cuda', dtype=torch.float32)
        buf1011 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_35.run(buf933, buf994, buf998, buf1002, mul_6, buf1010, buf1011, 768, 1024, grid=grid(768), stream=stream0)
        del mul_6
        buf1013 = reinterpret_tensor(buf926, (1024, 3072), (3072, 1), 0); del buf926  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1012, (1024, 768), (768, 1), 0), permute_1394, out=buf1013)
        del permute_1394
        buf1014 = reinterpret_tensor(buf942, (768, 3072), (3072, 1), 0); del buf942  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1012, (768, 1024), (1, 768), 0), view_73, out=buf1014)
        del view_73
        buf1015 = buf1004; del buf1004  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf1012, buf1015, 6144, 128, grid=grid(6144), stream=stream0)
        buf1016 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf1015, buf1016, 768, 8, grid=grid(768), stream=stream0)
        buf1017 = reinterpret_tensor(buf1013, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf1013  # reuse
        # Source Nodes: [intermediate_output], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_4.run(buf1017, addmm_4, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_4
        buf1018 = reinterpret_tensor(buf1012, (1024, 768), (768, 1), 0); del buf1012  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1017, (1024, 3072), (3072, 1), 0), permute_1398, out=buf1018)
        del permute_1398
        buf1019 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1017, (3072, 1024), (1, 3072), 0), view_71, out=buf1019)
        del view_71
        buf1020 = buf929; del buf929  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf1017, buf1020, 24576, 128, grid=grid(24576), stream=stream0)
        del buf1017
        buf1021 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf1020, buf1021, 3072, 8, grid=grid(3072), stream=stream0)
        del buf1020
        buf1024 = reinterpret_tensor(buf998, (1, 1024, 768), (786432, 768, 1), 0); del buf998  # reuse
        buf1027 = reinterpret_tensor(buf994, (1, 1024, 768), (786432, 768, 1), 0); del buf994  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7.run(buf1009, buf1018, primals_9, mul_1, div_154, getitem_3, buf1024, buf1027, 1024, 768, grid=grid(1024), stream=stream0)
        del div_154
        del getitem_3
        del primals_9
        buf1025 = empty((768, ), device='cuda', dtype=torch.float32)
        buf1026 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf1009, buf1018, mul_1, buf1025, buf1026, 768, 1024, grid=grid(768), stream=stream0)
        del mul_1
        buf1028 = buf1018; del buf1018  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1027, (1024, 768), (768, 1), 0), permute_1402, out=buf1028)
        del permute_1402
        buf1029 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1027, (768, 1024), (1, 768), 0), view_69, out=buf1029)
        del view_69
        buf1030 = buf1015; del buf1015  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf1027, buf1030, 6144, 128, grid=grid(6144), stream=stream0)
        buf1031 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf1030, buf1031, 768, 8, grid=grid(768), stream=stream0)
        buf1032 = reinterpret_tensor(buf1027, (12, 4, 256, 1, 64), (65536, 16384, 64, 64, 1), 0); del buf1027  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf1028, buf1032, 786432, grid=grid(786432), stream=stream0)
        buf1033 = empty((48, 768, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1410, reinterpret_tensor(buf1032, (48, 256, 64), (16384, 64, 1), 0), out=buf1033)
        del permute_1410
        buf1034 = reinterpret_tensor(buf977, (48, 256, 768), (196608, 768, 1), 0); del buf977  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1032, (48, 256, 64), (16384, 64, 1), 0), permute_1411, out=buf1034)
        del permute_1411
        buf1035 = buf944; del buf944  # reuse
        # Source Nodes: [], Original ATen: [aten.new_zeros]
        triton_poi_fused_index_add_new_zeros_11.run(buf1035, 1179648, grid=grid(1179648), stream=stream0)
        aten.index_put_(buf1035, [reinterpret_tensor(buf30, (2359296, ), (1, ), 0)], reinterpret_tensor(buf1033, (2359296, ), (1, ), 0), True)
        del buf1033
        del buf30
        buf1039 = reinterpret_tensor(buf973, (1024, 12, 513), (513, 525312, 1), 0); del buf973  # reuse
        # Source Nodes: [tril], Original ATen: [aten._softmax_backward_data, aten.copy, aten.masked_fill, aten.native_dropout_backward, aten.tril]
        triton_red_fused__softmax_backward_data_copy_masked_fill_native_dropout_backward_tril_13.run(unsqueeze_16, buf1034, getitem_1, alias_23, buf1039, 12288, 513, grid=grid(12288), stream=stream0)
        del alias_23
        del getitem_1
        del unsqueeze_16
        buf1040 = reinterpret_tensor(buf970, (6303744, ), (1, ), 0); del buf970  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_14.run(buf1040, 6303744, grid=grid(6303744), stream=stream0)
        buf1043 = reinterpret_tensor(buf967, (1, 1024, 12, 513), (6303744, 513, 525312, 1), 0); del buf967  # reuse
        buf1044 = buf965; del buf965  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.clone, aten.copy]
        triton_poi_fused_as_strided_scatter_clone_copy_15.run(buf1039, buf1040, buf1043, buf1044, 6303744, grid=grid(6303744), stream=stream0)
        buf1047 = reinterpret_tensor(buf1040, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf1040  # reuse
        buf1049 = buf1039; del buf1039  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_16.run(buf1043, rev_1, buf1044, buf1047, buf1049, 6303744, grid=grid(6303744), stream=stream0)
        del rev_1
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_17.run(buf1047, 789504, grid=grid(789504), stream=stream0)
        buf1050 = reinterpret_tensor(buf1044, (6303744, ), (1, ), 0); del buf1044  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_14.run(buf1050, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_18.run(buf1049, buf1050, 6303744, grid=grid(6303744), stream=stream0)
        buf1052 = reinterpret_tensor(buf1049, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf1049  # reuse
        buf1054 = reinterpret_tensor(buf1043, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf1043  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_add_as_strided_scatter_copy_zeros_like_19.run(buf1047, buf1050, buf1052, buf1054, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_20.run(buf1054, 789504, grid=grid(789504), stream=stream0)
        buf1056 = buf1052; del buf1052  # reuse
        buf1058 = reinterpret_tensor(buf1050, (12, 4, 256, 513), (525312, 131328, 513, 1), 0); del buf1050  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_add_as_strided_scatter_copy_zeros_like_21.run(buf1056, buf1054, slice_64, buf1058, 6303744, grid=grid(6303744), stream=stream0)
        del slice_64
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_22.run(buf1058, 780300, grid=grid(780300), stream=stream0)
        buf1061 = buf1054; del buf1054  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_23.run(buf1058, buf1061, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_24.run(buf1061, 2359296, grid=grid(2359296), stream=stream0)
        buf1064 = buf1047; del buf1047  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_23.run(buf1061, buf1064, 6303744, grid=grid(6303744), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.copy, aten.zeros_like]
        triton_poi_fused_as_strided_scatter_copy_zeros_like_17.run(buf1064, 789504, grid=grid(789504), stream=stream0)
        buf1066 = buf975; del buf975  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.clone, aten.select_backward, aten.slice_backward]
        triton_poi_fused_add_clone_select_backward_slice_backward_25.run(buf1056, buf1058, buf1061, buf1066, 9455616, grid=grid(9455616), stream=stream0)
        del buf1056
        del buf1058
        del buf1061
        buf1068 = reinterpret_tensor(buf1034, (12, 3, 512, 512), (786432, 262144, 512, 1), 0); del buf1034  # reuse
        # Source Nodes: [], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_26.run(buf1066, buf1064, buf1068, 9437184, grid=grid(9437184), stream=stream0)
        del buf1064
        del buf1066
        buf1069 = reinterpret_tensor(buf979, (36, 64, 512), (32768, 512, 1), 0); del buf979  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1421, reinterpret_tensor(buf1068, (36, 512, 512), (262144, 512, 1), 0), out=buf1069)
        del permute_1421
        buf1070 = reinterpret_tensor(buf981, (36, 512, 64), (32768, 64, 1), 0); del buf981  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1068, (36, 512, 512), (262144, 512, 1), 0), permute_1422, out=buf1070)
        del buf1068
        del permute_1422
        buf1071 = reinterpret_tensor(buf1032, (786432, ), (1, ), 0); del buf1032  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add]
        triton_poi_fused_index_add_28.run(buf1071, 786432, grid=grid(786432), stream=stream0)
        buf1072 = reinterpret_tensor(buf978, (12, 3, 512, 64), (98304, 32768, 64, 1), 0); del buf978  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf1069, buf1072, 18432, 64, grid=grid(18432, 64), stream=stream0)
        del buf1069
        aten.index_put_(buf1071, [reinterpret_tensor(buf71, (1179648, ), (1, ), 0)], reinterpret_tensor(buf1072, (1179648, ), (1, ), 0), True)
        del buf1072
        buf1075 = reinterpret_tensor(buf1028, (786432, ), (1, ), 0); del buf1028  # reuse
        # Source Nodes: [], Original ATen: [aten.index_add]
        triton_poi_fused_index_add_28.run(buf1075, 786432, grid=grid(786432), stream=stream0)
        aten.index_put_(buf1075, [reinterpret_tensor(buf71, (1179648, ), (1, ), 0)], reinterpret_tensor(buf1070, (1179648, ), (1, ), 0), True)
        del buf1070
        del buf71
        buf1078 = reinterpret_tensor(buf1009, (786432, ), (1, ), 0); del buf1009  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_index_add_28.run(buf1078, 786432, grid=grid(786432), stream=stream0)
        buf1081 = reinterpret_tensor(buf933, (1024, 1, 768), (768, 768, 1), 0); del buf933  # reuse
        buf1082 = buf1002; del buf1002  # reuse
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter, aten.div]
        triton_poi_fused_as_strided_scatter_div_31.run(buf1075, buf1078, buf1081, buf1082, 786432, grid=grid(786432), stream=stream0)
        del buf1075
        # Source Nodes: [], Original ATen: [aten.as_strided_scatter]
        triton_poi_fused_as_strided_scatter_32.run(buf1081, buf1082, 786432, grid=grid(786432), stream=stream0)
        buf1084 = reinterpret_tensor(buf1081, (1024, 768), (768, 1), 0); del buf1081  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_33.run(buf1035, buf1084, 786432, grid=grid(786432), stream=stream0)
        del buf1035
        buf1085 = reinterpret_tensor(buf1078, (1024, 768), (768, 1), 0); del buf1078  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1084, permute_1433, out=buf1085)
        del permute_1433
        buf1086 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1084, (768, 1024), (1, 768), 0), view, out=buf1086)
        buf1087 = buf1030; del buf1030  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf1084, buf1087, 6144, 128, grid=grid(6144), stream=stream0)
        buf1088 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf1087, buf1088, 768, 8, grid=grid(768), stream=stream0)
        buf1089 = buf1084; del buf1084  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1071, (1024, 768), (768, 1), 0), permute_1437, out=buf1089)
        del permute_1437
        buf1090 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1071, (768, 1024), (1, 768), 0), view, out=buf1090)
        buf1091 = buf1087; del buf1087  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf1071, buf1091, 6144, 128, grid=grid(6144), stream=stream0)
        buf1092 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf1091, buf1092, 768, 8, grid=grid(768), stream=stream0)
        buf1093 = reinterpret_tensor(buf1071, (1024, 768), (768, 1), 0); del buf1071  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf1082, permute_1446, out=buf1093)
        del permute_1446
        buf1094 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1082, (768, 1024), (1, 768), 0), view, out=buf1094)
        del view
        buf1095 = buf1091; del buf1091  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf1082, buf1095, 6144, 128, grid=grid(6144), stream=stream0)
        del buf1082
        buf1096 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf1095, buf1096, 768, 8, grid=grid(768), stream=stream0)
        del buf1095
        buf1097 = buf1024; del buf1024  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_36.run(buf1097, buf1085, buf1089, buf1093, 786432, grid=grid(786432), stream=stream0)
        return (reinterpret_tensor(buf1094, (768, 768), (768, 1), 0), reinterpret_tensor(buf1096, (768, ), (1, ), 0), reinterpret_tensor(buf1090, (768, 768), (768, 1), 0), reinterpret_tensor(buf1092, (768, ), (1, ), 0), reinterpret_tensor(buf1086, (768, 768), (768, 1), 0), reinterpret_tensor(buf1088, (768, ), (1, ), 0), reinterpret_tensor(buf1029, (768, 768), (768, 1), 0), reinterpret_tensor(buf1031, (768, ), (1, ), 0), buf1025, buf1026, reinterpret_tensor(buf1019, (3072, 768), (768, 1), 0), reinterpret_tensor(buf1021, (3072, ), (1, ), 0), reinterpret_tensor(buf1014, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf1016, (768, ), (1, ), 0), buf1010, buf1011, reinterpret_tensor(buf1003, (768, 768), (768, 1), 0), reinterpret_tensor(buf1005, (768, ), (1, ), 0), reinterpret_tensor(buf999, (768, 768), (768, 1), 0), reinterpret_tensor(buf1001, (768, ), (1, ), 0), reinterpret_tensor(buf995, (768, 768), (768, 1), 0), reinterpret_tensor(buf997, (768, ), (1, ), 0), reinterpret_tensor(buf938, (768, 768), (768, 1), 0), reinterpret_tensor(buf940, (768, ), (1, ), 0), buf934, buf935, reinterpret_tensor(buf928, (3072, 768), (768, 1), 0), reinterpret_tensor(buf930, (3072, ), (1, ), 0), reinterpret_tensor(buf923, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf925, (768, ), (1, ), 0), buf919, buf920, reinterpret_tensor(buf912, (768, 768), (768, 1), 0), reinterpret_tensor(buf914, (768, ), (1, ), 0), reinterpret_tensor(buf908, (768, 768), (768, 1), 0), reinterpret_tensor(buf910, (768, ), (1, ), 0), reinterpret_tensor(buf904, (768, 768), (768, 1), 0), reinterpret_tensor(buf906, (768, ), (1, ), 0), reinterpret_tensor(buf847, (768, 768), (768, 1), 0), reinterpret_tensor(buf849, (768, ), (1, ), 0), buf843, buf844, reinterpret_tensor(buf837, (3072, 768), (768, 1), 0), reinterpret_tensor(buf839, (3072, ), (1, ), 0), reinterpret_tensor(buf832, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf834, (768, ), (1, ), 0), buf828, buf829, reinterpret_tensor(buf821, (768, 768), (768, 1), 0), reinterpret_tensor(buf823, (768, ), (1, ), 0), reinterpret_tensor(buf817, (768, 768), (768, 1), 0), reinterpret_tensor(buf819, (768, ), (1, ), 0), reinterpret_tensor(buf813, (768, 768), (768, 1), 0), reinterpret_tensor(buf815, (768, ), (1, ), 0), reinterpret_tensor(buf756, (768, 768), (768, 1), 0), reinterpret_tensor(buf758, (768, ), (1, ), 0), buf752, buf753, reinterpret_tensor(buf746, (3072, 768), (768, 1), 0), reinterpret_tensor(buf748, (3072, ), (1, ), 0), reinterpret_tensor(buf741, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf743, (768, ), (1, ), 0), buf737, buf738, reinterpret_tensor(buf730, (768, 768), (768, 1), 0), reinterpret_tensor(buf732, (768, ), (1, ), 0), reinterpret_tensor(buf726, (768, 768), (768, 1), 0), reinterpret_tensor(buf728, (768, ), (1, ), 0), reinterpret_tensor(buf722, (768, 768), (768, 1), 0), reinterpret_tensor(buf724, (768, ), (1, ), 0), reinterpret_tensor(buf665, (768, 768), (768, 1), 0), reinterpret_tensor(buf667, (768, ), (1, ), 0), buf661, buf662, reinterpret_tensor(buf655, (3072, 768), (768, 1), 0), reinterpret_tensor(buf657, (3072, ), (1, ), 0), reinterpret_tensor(buf650, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf652, (768, ), (1, ), 0), buf646, buf647, reinterpret_tensor(buf639, (768, 768), (768, 1), 0), reinterpret_tensor(buf641, (768, ), (1, ), 0), reinterpret_tensor(buf635, (768, 768), (768, 1), 0), reinterpret_tensor(buf637, (768, ), (1, ), 0), reinterpret_tensor(buf631, (768, 768), (768, 1), 0), reinterpret_tensor(buf633, (768, ), (1, ), 0), reinterpret_tensor(buf574, (768, 768), (768, 1), 0), reinterpret_tensor(buf576, (768, ), (1, ), 0), buf570, buf571, reinterpret_tensor(buf564, (3072, 768), (768, 1), 0), reinterpret_tensor(buf566, (3072, ), (1, ), 0), reinterpret_tensor(buf559, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf561, (768, ), (1, ), 0), buf555, buf556, reinterpret_tensor(buf548, (768, 768), (768, 1), 0), reinterpret_tensor(buf550, (768, ), (1, ), 0), reinterpret_tensor(buf544, (768, 768), (768, 1), 0), reinterpret_tensor(buf546, (768, ), (1, ), 0), reinterpret_tensor(buf540, (768, 768), (768, 1), 0), reinterpret_tensor(buf542, (768, ), (1, ), 0), reinterpret_tensor(buf483, (768, 768), (768, 1), 0), reinterpret_tensor(buf485, (768, ), (1, ), 0), buf479, buf480, reinterpret_tensor(buf473, (3072, 768), (768, 1), 0), reinterpret_tensor(buf475, (3072, ), (1, ), 0), reinterpret_tensor(buf468, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf470, (768, ), (1, ), 0), buf464, buf465, reinterpret_tensor(buf457, (768, 768), (768, 1), 0), reinterpret_tensor(buf459, (768, ), (1, ), 0), reinterpret_tensor(buf453, (768, 768), (768, 1), 0), reinterpret_tensor(buf455, (768, ), (1, ), 0), reinterpret_tensor(buf449, (768, 768), (768, 1), 0), reinterpret_tensor(buf451, (768, ), (1, ), 0), reinterpret_tensor(buf392, (768, 768), (768, 1), 0), reinterpret_tensor(buf394, (768, ), (1, ), 0), buf388, buf389, reinterpret_tensor(buf382, (3072, 768), (768, 1), 0), reinterpret_tensor(buf384, (3072, ), (1, ), 0), reinterpret_tensor(buf377, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf379, (768, ), (1, ), 0), buf373, buf374, reinterpret_tensor(buf366, (768, 768), (768, 1), 0), reinterpret_tensor(buf368, (768, ), (1, ), 0), reinterpret_tensor(buf362, (768, 768), (768, 1), 0), reinterpret_tensor(buf364, (768, ), (1, ), 0), reinterpret_tensor(buf358, (768, 768), (768, 1), 0), reinterpret_tensor(buf360, (768, ), (1, ), 0), reinterpret_tensor(buf301, (768, 768), (768, 1), 0), reinterpret_tensor(buf303, (768, ), (1, ), 0), buf297, buf298, reinterpret_tensor(buf291, (3072, 768), (768, 1), 0), reinterpret_tensor(buf293, (3072, ), (1, ), 0), reinterpret_tensor(buf286, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf288, (768, ), (1, ), 0), buf282, buf283, reinterpret_tensor(buf275, (768, 768), (768, 1), 0), reinterpret_tensor(buf277, (768, ), (1, ), 0), reinterpret_tensor(buf271, (768, 768), (768, 1), 0), reinterpret_tensor(buf273, (768, ), (1, ), 0), reinterpret_tensor(buf267, (768, 768), (768, 1), 0), reinterpret_tensor(buf269, (768, ), (1, ), 0), reinterpret_tensor(buf210, (768, 768), (768, 1), 0), reinterpret_tensor(buf212, (768, ), (1, ), 0), buf206, buf207, reinterpret_tensor(buf200, (3072, 768), (768, 1), 0), reinterpret_tensor(buf202, (3072, ), (1, ), 0), reinterpret_tensor(buf195, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf197, (768, ), (1, ), 0), buf191, buf192, reinterpret_tensor(buf184, (768, 768), (768, 1), 0), reinterpret_tensor(buf186, (768, ), (1, ), 0), reinterpret_tensor(buf180, (768, 768), (768, 1), 0), reinterpret_tensor(buf182, (768, ), (1, ), 0), reinterpret_tensor(buf176, (768, 768), (768, 1), 0), reinterpret_tensor(buf178, (768, ), (1, ), 0), reinterpret_tensor(buf119, (768, 768), (768, 1), 0), reinterpret_tensor(buf121, (768, ), (1, ), 0), buf115, buf116, reinterpret_tensor(buf109, (3072, 768), (768, 1), 0), reinterpret_tensor(buf111, (3072, ), (1, ), 0), reinterpret_tensor(buf104, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf106, (768, ), (1, ), 0), buf100, buf101, reinterpret_tensor(buf93, (768, 768), (768, 1), 0), reinterpret_tensor(buf95, (768, ), (1, ), 0), reinterpret_tensor(buf89, (768, 768), (768, 1), 0), reinterpret_tensor(buf91, (768, ), (1, ), 0), reinterpret_tensor(buf85, (768, 768), (768, 1), 0), reinterpret_tensor(buf87, (768, ), (1, ), 0), reinterpret_tensor(buf22, (768, 768), (768, 1), 0), reinterpret_tensor(buf24, (768, ), (1, ), 0), buf18, buf19, reinterpret_tensor(buf12, (3072, 768), (768, 1), 0), reinterpret_tensor(buf14, (3072, ), (1, ), 0), reinterpret_tensor(buf7, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf9, (768, ), (1, ), 0), buf3, buf4, buf1097, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_9 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    slice_64 = rand_strided((1, 256, 1, 257), (65792, 257, 257, 1), device='cuda:0', dtype=torch.float32)
    rev_1 = rand_strided((1, 256, 1, 257), (65792, 257, 257, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_16 = rand_strided((1, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.bool)
    getitem_1 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cuda:0', dtype=torch.bool)
    view_69 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_1 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    view_71 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_4 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_73 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_6 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    view_75 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_11 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cuda:0', dtype=torch.bool)
    view_144 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_13 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_9 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    view_146 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_148 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_17 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_14 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    view_150 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_21 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cuda:0', dtype=torch.bool)
    view_219 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_23 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_17 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    view_221 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_16 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_223 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_27 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_22 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    view_225 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_31 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cuda:0', dtype=torch.bool)
    view_294 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_33 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_25 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    view_296 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_298 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_37 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_30 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    view_300 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_41 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cuda:0', dtype=torch.bool)
    view_369 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_43 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_33 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    view_371 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_28 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_373 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_47 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_38 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    view_375 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_51 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cuda:0', dtype=torch.bool)
    view_444 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_53 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_41 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    view_446 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_448 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_57 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_46 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    view_450 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_61 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cuda:0', dtype=torch.bool)
    view_519 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_63 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_49 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    view_521 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_40 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_523 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_67 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_54 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    view_525 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_71 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cuda:0', dtype=torch.bool)
    view_594 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_73 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_57 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    view_596 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_46 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_598 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_77 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_62 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    view_600 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_81 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cuda:0', dtype=torch.bool)
    view_669 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_83 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_65 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    view_671 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_52 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_673 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_87 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_70 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    view_675 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_91 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cuda:0', dtype=torch.bool)
    view_744 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_93 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_73 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    view_746 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_58 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_748 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_97 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_78 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    view_750 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_101 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cuda:0', dtype=torch.bool)
    view_819 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_103 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_81 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    view_821 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_64 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_823 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_107 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_86 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    view_825 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_111 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cuda:0', dtype=torch.bool)
    view_894 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_113 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_89 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    view_896 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_70 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_898 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_117 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_94 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    div_120 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_756 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_760 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_121 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_764 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_772 = rand_strided((48, 768, 256), (197120, 1, 769), device='cuda:0', dtype=torch.float32)
    permute_773 = rand_strided((48, 64, 768), (49152, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_12 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cuda:0', dtype=torch.float32)
    permute_783 = rand_strided((36, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_784 = rand_strided((36, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_795 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_799 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_808 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_123 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_814 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_818 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_124 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_822 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_830 = rand_strided((48, 768, 256), (197120, 1, 769), device='cuda:0', dtype=torch.float32)
    permute_831 = rand_strided((48, 64, 768), (49152, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_13 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cuda:0', dtype=torch.float32)
    permute_841 = rand_strided((36, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_842 = rand_strided((36, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_853 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_857 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_866 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_126 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_872 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_876 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_127 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_880 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_888 = rand_strided((48, 768, 256), (197120, 1, 769), device='cuda:0', dtype=torch.float32)
    permute_889 = rand_strided((48, 64, 768), (49152, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_14 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cuda:0', dtype=torch.float32)
    permute_899 = rand_strided((36, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_900 = rand_strided((36, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_911 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_915 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_924 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_129 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_930 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_934 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_130 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_938 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_946 = rand_strided((48, 768, 256), (197120, 1, 769), device='cuda:0', dtype=torch.float32)
    permute_947 = rand_strided((48, 64, 768), (49152, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_15 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cuda:0', dtype=torch.float32)
    permute_957 = rand_strided((36, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_958 = rand_strided((36, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_969 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_973 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_982 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_132 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_988 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_992 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_133 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_996 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_1004 = rand_strided((48, 768, 256), (197120, 1, 769), device='cuda:0', dtype=torch.float32)
    permute_1005 = rand_strided((48, 64, 768), (49152, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_16 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cuda:0', dtype=torch.float32)
    permute_1015 = rand_strided((36, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1016 = rand_strided((36, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_1027 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_1031 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_1040 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_135 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1046 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_1050 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_136 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1054 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_1062 = rand_strided((48, 768, 256), (197120, 1, 769), device='cuda:0', dtype=torch.float32)
    permute_1063 = rand_strided((48, 64, 768), (49152, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_17 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cuda:0', dtype=torch.float32)
    permute_1073 = rand_strided((36, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1074 = rand_strided((36, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_1085 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_1089 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_1098 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_138 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1104 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_1108 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_139 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1112 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_1120 = rand_strided((48, 768, 256), (197120, 1, 769), device='cuda:0', dtype=torch.float32)
    permute_1121 = rand_strided((48, 64, 768), (49152, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_18 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cuda:0', dtype=torch.float32)
    permute_1131 = rand_strided((36, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1132 = rand_strided((36, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_1143 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_1147 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_1156 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_141 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1162 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_1166 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_142 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1170 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_1178 = rand_strided((48, 768, 256), (197120, 1, 769), device='cuda:0', dtype=torch.float32)
    permute_1179 = rand_strided((48, 64, 768), (49152, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_19 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cuda:0', dtype=torch.float32)
    permute_1189 = rand_strided((36, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1190 = rand_strided((36, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_1201 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_1205 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_1214 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_144 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1220 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_1224 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_145 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1228 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_1236 = rand_strided((48, 768, 256), (197120, 1, 769), device='cuda:0', dtype=torch.float32)
    permute_1237 = rand_strided((48, 64, 768), (49152, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_20 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cuda:0', dtype=torch.float32)
    permute_1247 = rand_strided((36, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1248 = rand_strided((36, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_1259 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_1263 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_1272 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_147 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1278 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_1282 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_148 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1286 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_1294 = rand_strided((48, 768, 256), (197120, 1, 769), device='cuda:0', dtype=torch.float32)
    permute_1295 = rand_strided((48, 64, 768), (49152, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_21 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cuda:0', dtype=torch.float32)
    permute_1305 = rand_strided((36, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1306 = rand_strided((36, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_1317 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_1321 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_1330 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_150 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1336 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_1340 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_151 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1344 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_1352 = rand_strided((48, 768, 256), (197120, 1, 769), device='cuda:0', dtype=torch.float32)
    permute_1353 = rand_strided((48, 64, 768), (49152, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_22 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cuda:0', dtype=torch.float32)
    permute_1363 = rand_strided((36, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1364 = rand_strided((36, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_1375 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_1379 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_1388 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_153 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1394 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_1398 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_154 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_1402 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_1410 = rand_strided((48, 768, 256), (197120, 1, 769), device='cuda:0', dtype=torch.float32)
    permute_1411 = rand_strided((48, 64, 768), (49152, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_23 = rand_strided((1, 1024, 12, 513), (6303744, 6156, 513, 1), device='cuda:0', dtype=torch.float32)
    permute_1421 = rand_strided((36, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_1422 = rand_strided((36, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_1433 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_1437 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_1446 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_9, primals_15, primals_25, primals_31, primals_41, primals_47, primals_57, primals_63, primals_73, primals_79, primals_89, primals_95, primals_105, primals_111, primals_121, primals_127, primals_137, primals_143, primals_153, primals_159, primals_169, primals_175, primals_185, primals_191, view, slice_64, rev_1, unsqueeze_16, getitem_1, view_69, getitem_3, mul_1, view_71, addmm_4, view_73, getitem_7, mul_6, view_75, getitem_11, view_144, getitem_13, mul_9, view_146, addmm_10, view_148, getitem_17, mul_14, view_150, getitem_21, view_219, getitem_23, mul_17, view_221, addmm_16, view_223, getitem_27, mul_22, view_225, getitem_31, view_294, getitem_33, mul_25, view_296, addmm_22, view_298, getitem_37, mul_30, view_300, getitem_41, view_369, getitem_43, mul_33, view_371, addmm_28, view_373, getitem_47, mul_38, view_375, getitem_51, view_444, getitem_53, mul_41, view_446, addmm_34, view_448, getitem_57, mul_46, view_450, getitem_61, view_519, getitem_63, mul_49, view_521, addmm_40, view_523, getitem_67, mul_54, view_525, getitem_71, view_594, getitem_73, mul_57, view_596, addmm_46, view_598, getitem_77, mul_62, view_600, getitem_81, view_669, getitem_83, mul_65, view_671, addmm_52, view_673, getitem_87, mul_70, view_675, getitem_91, view_744, getitem_93, mul_73, view_746, addmm_58, view_748, getitem_97, mul_78, view_750, getitem_101, view_819, getitem_103, mul_81, view_821, addmm_64, view_823, getitem_107, mul_86, view_825, getitem_111, view_894, getitem_113, mul_89, view_896, addmm_70, view_898, getitem_117, mul_94, div_120, permute_756, permute_760, div_121, permute_764, permute_772, permute_773, alias_12, permute_783, permute_784, permute_795, permute_799, permute_808, div_123, permute_814, permute_818, div_124, permute_822, permute_830, permute_831, alias_13, permute_841, permute_842, permute_853, permute_857, permute_866, div_126, permute_872, permute_876, div_127, permute_880, permute_888, permute_889, alias_14, permute_899, permute_900, permute_911, permute_915, permute_924, div_129, permute_930, permute_934, div_130, permute_938, permute_946, permute_947, alias_15, permute_957, permute_958, permute_969, permute_973, permute_982, div_132, permute_988, permute_992, div_133, permute_996, permute_1004, permute_1005, alias_16, permute_1015, permute_1016, permute_1027, permute_1031, permute_1040, div_135, permute_1046, permute_1050, div_136, permute_1054, permute_1062, permute_1063, alias_17, permute_1073, permute_1074, permute_1085, permute_1089, permute_1098, div_138, permute_1104, permute_1108, div_139, permute_1112, permute_1120, permute_1121, alias_18, permute_1131, permute_1132, permute_1143, permute_1147, permute_1156, div_141, permute_1162, permute_1166, div_142, permute_1170, permute_1178, permute_1179, alias_19, permute_1189, permute_1190, permute_1201, permute_1205, permute_1214, div_144, permute_1220, permute_1224, div_145, permute_1228, permute_1236, permute_1237, alias_20, permute_1247, permute_1248, permute_1259, permute_1263, permute_1272, div_147, permute_1278, permute_1282, div_148, permute_1286, permute_1294, permute_1295, alias_21, permute_1305, permute_1306, permute_1317, permute_1321, permute_1330, div_150, permute_1336, permute_1340, div_151, permute_1344, permute_1352, permute_1353, alias_22, permute_1363, permute_1364, permute_1375, permute_1379, permute_1388, div_153, permute_1394, permute_1398, div_154, permute_1402, permute_1410, permute_1411, alias_23, permute_1421, permute_1422, permute_1433, permute_1437, permute_1446, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('AllenaiLongformerBase', benchmark_compiled_module)
