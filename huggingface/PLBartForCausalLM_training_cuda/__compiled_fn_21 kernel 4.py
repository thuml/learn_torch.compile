
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
# Source Nodes: [hidden_states_4], Original ATen: [aten.gelu, aten.gelu_backward]
# hidden_states_4 => add_4, erf, mul_4
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


# kernel path: /tmp/torchinductor_youkaichao/zx/czxnveg6dql4vdey2yygka3ln3mq3bxgvrwo2bpkb53s4b337tsm.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.native_dropout_backward]

triton_per_fused__softmax_backward_data_native_dropout_backward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_native_dropout_backward_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel):
    xnumel = 12288
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
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask).to(tl.int1)
    tmp6 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tmp6 * tmp11
    tmp13 = tmp7 - tmp12
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp13, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oa/coaercbserojokowar7wuozbzzlxqd7sfegdobg7mc2xmvqaxbp3.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64) % 1024
    x2 = (xindex // 65536)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0 + (64*x2) + (768*x1)), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5q/c5qwz3uqbbhh57bny75jpgunjrsxnyvrj73vkll5xcnaphrcjjzq.py
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
    size_hints=[16384, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12288
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (1024*x2) + (65536*y1)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (64*y1) + (768*y0)), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/da/cdake7wdy3hgzqmxojyjejhqbsuicg7qes7lnhcgy55fgsnttnhd.py
# Source Nodes: [], Original ATen: [aten.mul, aten.view]

triton_poi_fused_mul_view_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_view_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 768
    x1 = (xindex // 768)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (65536*(x0 // 64)) + (x0 % 64)), None)
    tmp1 = 0.125
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/uh/cuh5bu3vzsizadulv63mnmut23yjgvtuvp3ga4ogynwny5m6p4lm.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_13', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_9, primals_15, view, getitem_1, view_16, getitem_3, mul_1, view_18, addmm_4, view_20, getitem_7, mul_6, div_1, permute_11, permute_15, div_2, permute_19, permute_24, permute_25, alias_1, permute_26, permute_27, permute_31, permute_36, permute_40, tangents_1, tangents_2, tangents_3 = args
    args.clear()
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(view, (1024, 768), (768, 1))
    assert_size_stride(getitem_1, (12, 1024, 1024), (1048576, 1024, 1))
    assert_size_stride(view_16, (1024, 768), (768, 1))
    assert_size_stride(getitem_3, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_1, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_18, (1024, 768), (768, 1))
    assert_size_stride(addmm_4, (1024, 3072), (3072, 1))
    assert_size_stride(view_20, (1024, 3072), (3072, 1))
    assert_size_stride(getitem_7, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_6, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(div_1, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_11, (768, 3072), (3072, 1))
    assert_size_stride(permute_15, (3072, 768), (768, 1))
    assert_size_stride(div_2, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_19, (768, 768), (768, 1))
    assert_size_stride(permute_24, (12, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_25, (12, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_1, (12, 1024, 1024), (1048576, 1024, 1))
    assert_size_stride(permute_26, (12, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_27, (12, 1024, 64), (65536, 64, 1))
    assert_size_stride(permute_31, (768, 768), (768, 1))
    assert_size_stride(permute_36, (768, 768), (768, 1))
    assert_size_stride(permute_40, (768, 768), (768, 1))
    assert_size_stride(tangents_1, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(tangents_2, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_3, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf2 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf5 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]
        stream0 = get_cuda_stream(0)
        triton_per_fused_native_dropout_backward_native_layer_norm_backward_0.run(tangents_1, primals_15, mul_6, div_1, getitem_7, buf2, buf5, 1024, 768, grid=grid(1024), stream=stream0)
        del div_1
        del getitem_7
        del primals_15
        buf3 = empty((768, ), device='cuda', dtype=torch.float32)
        buf4 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_1.run(tangents_1, mul_6, buf3, buf4, 768, 1024, grid=grid(768), stream=stream0)
        del mul_6
        del tangents_1
        buf6 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (1024, 768), (768, 1), 0), permute_11, out=buf6)
        del permute_11
        buf7 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (768, 1024), (1, 768), 0), view_20, out=buf7)
        del view_20
        buf8 = empty_strided((1, 768, 8), (6144, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf5, buf8, 6144, 128, grid=grid(6144), stream=stream0)
        buf9 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf8, buf9, 768, 8, grid=grid(768), stream=stream0)
        buf10 = reinterpret_tensor(buf6, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf6  # reuse
        # Source Nodes: [hidden_states_4], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_4.run(buf10, addmm_4, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_4
        buf11 = reinterpret_tensor(buf5, (1024, 768), (768, 1), 0); del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (1024, 3072), (3072, 1), 0), permute_15, out=buf11)
        del permute_15
        buf12 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (3072, 1024), (1, 3072), 0), view_18, out=buf12)
        del view_18
        buf13 = empty_strided((1, 3072, 8), (24576, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf10, buf13, 24576, 128, grid=grid(24576), stream=stream0)
        del buf10
        buf14 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf13, buf14, 3072, 8, grid=grid(3072), stream=stream0)
        del buf13
        buf17 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf20 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7.run(buf2, buf11, primals_9, mul_1, div_2, getitem_3, buf17, buf20, 1024, 768, grid=grid(1024), stream=stream0)
        del div_2
        del getitem_3
        del primals_9
        buf18 = empty((768, ), device='cuda', dtype=torch.float32)
        buf19 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf2, buf11, mul_1, buf18, buf19, 768, 1024, grid=grid(768), stream=stream0)
        del mul_1
        buf21 = reinterpret_tensor(buf2, (1024, 768), (768, 1), 0); del buf2  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (1024, 768), (768, 1), 0), permute_19, out=buf21)
        del permute_19
        buf22 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (768, 1024), (1, 768), 0), view_16, out=buf22)
        del view_16
        buf23 = buf8; del buf8  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf20, buf23, 6144, 128, grid=grid(6144), stream=stream0)
        buf24 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf23, buf24, 768, 8, grid=grid(768), stream=stream0)
        buf25 = reinterpret_tensor(buf20, (12, 1024, 64), (65536, 64, 1), 0); del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_24, reinterpret_tensor(buf21, (12, 1024, 64), (64, 768, 1), 0), out=buf25)
        del permute_24
        buf26 = empty((12, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf21, (12, 1024, 64), (64, 768, 1), 0), permute_25, out=buf26)
        del permute_25
        buf28 = empty((12, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.native_dropout_backward]
        triton_per_fused__softmax_backward_data_native_dropout_backward_9.run(buf26, getitem_1, alias_1, buf28, 12288, 1024, grid=grid(12288), stream=stream0)
        del alias_1
        del buf26
        del getitem_1
        buf29 = reinterpret_tensor(buf21, (12, 64, 1024), (65536, 1024, 1), 0); del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_26, reinterpret_tensor(buf28, (12, 1024, 1024), (1048576, 1024, 1), 0), out=buf29)
        del permute_26
        buf30 = reinterpret_tensor(buf11, (12, 1024, 64), (65536, 64, 1), 0); del buf11  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf28, (12, 1024, 1024), (1048576, 1024, 1), 0), permute_27, out=buf30)
        del buf28
        del permute_27
        buf31 = empty((1, 1024, 12, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(tangents_3, buf25, buf31, 786432, grid=grid(786432), stream=stream0)
        del tangents_3
        buf32 = reinterpret_tensor(buf25, (1024, 768), (768, 1), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf31, (1024, 768), (768, 1), 0), permute_31, out=buf32)
        del permute_31
        buf33 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf31, (768, 1024), (1, 768), 0), view, out=buf33)
        buf34 = buf23; del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf31, buf34, 6144, 128, grid=grid(6144), stream=stream0)
        buf35 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf34, buf35, 768, 8, grid=grid(768), stream=stream0)
        buf36 = buf31; del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(tangents_2, buf29, buf36, 12288, 64, grid=grid(12288, 64), stream=stream0)
        del tangents_2
        buf37 = reinterpret_tensor(buf29, (1024, 768), (768, 1), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (1024, 768), (768, 1), 0), permute_36, out=buf37)
        del permute_36
        buf38 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (768, 1024), (1, 768), 0), view, out=buf38)
        buf39 = buf34; del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf36, buf39, 6144, 128, grid=grid(6144), stream=stream0)
        buf40 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf39, buf40, 768, 8, grid=grid(768), stream=stream0)
        buf41 = reinterpret_tensor(buf36, (1024, 768), (768, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_12.run(buf30, buf41, 786432, grid=grid(786432), stream=stream0)
        buf42 = reinterpret_tensor(buf30, (1024, 768), (768, 1), 0); del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf41, permute_40, out=buf42)
        del permute_40
        buf43 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf41, (768, 1024), (1, 768), 0), view, out=buf43)
        del view
        buf44 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf41, buf44, 6144, 128, grid=grid(6144), stream=stream0)
        del buf41
        buf45 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf44, buf45, 768, 8, grid=grid(768), stream=stream0)
        del buf44
        buf46 = buf17; del buf17  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_13.run(buf46, buf32, buf37, buf42, 786432, grid=grid(786432), stream=stream0)
        return (reinterpret_tensor(buf43, (768, 768), (768, 1), 0), reinterpret_tensor(buf45, (768, ), (1, ), 0), reinterpret_tensor(buf38, (768, 768), (768, 1), 0), reinterpret_tensor(buf40, (768, ), (1, ), 0), reinterpret_tensor(buf33, (768, 768), (768, 1), 0), reinterpret_tensor(buf35, (768, ), (1, ), 0), reinterpret_tensor(buf22, (768, 768), (768, 1), 0), reinterpret_tensor(buf24, (768, ), (1, ), 0), buf18, buf19, reinterpret_tensor(buf12, (3072, 768), (768, 1), 0), reinterpret_tensor(buf14, (3072, ), (1, ), 0), reinterpret_tensor(buf7, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf9, (768, ), (1, ), 0), buf3, buf4, buf46, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_9 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((12, 1024, 1024), (1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_16 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_1 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    view_18 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_4 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_20 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_6 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    div_1 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_11 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_15 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_2 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_19 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_24 = rand_strided((12, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_25 = rand_strided((12, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_1 = rand_strided((12, 1024, 1024), (1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_26 = rand_strided((12, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_27 = rand_strided((12, 1024, 64), (65536, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_31 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_36 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_40 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_3 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_9, primals_15, view, getitem_1, view_16, getitem_3, mul_1, view_18, addmm_4, view_20, getitem_7, mul_6, div_1, permute_11, permute_15, div_2, permute_19, permute_24, permute_25, alias_1, permute_26, permute_27, permute_31, permute_36, permute_40, tangents_1, tangents_2, tangents_3]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('PLBartForCausalLM', benchmark_compiled_module)
