
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
# Source Nodes: [hidden_states_8], Original ATen: [aten.gelu, aten.gelu_backward]
# hidden_states_8 => add_7, erf, mul_7
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


# kernel path: /tmp/torchinductor_youkaichao/p4/cp4jukwt4kh2ae7tks3xxylvdfwx4yhpeuyk4qahlifrec5czizv.py
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 12
    x2 = (xindex // 768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (65536*x1)), None)
    tmp1 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hk/chk7ol4ke24r2l6gxb2yb3fozzpaepieorbflfnmrhy5y5cr6oxe.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tu/ctutjcjtc4dq5g4wjdt7itaqudjutlb25cxwgju4gblo5koy32s5.py
# Source Nodes: [], Original ATen: [aten.mul, aten.view]

triton_poi_fused_mul_view_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_view_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = 0.125
    tmp2 = tmp0 * tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7p/c7pj6c77blhwgag4ylp25lpz7fs5f3qmck2eefhw3l2nwl5uqx2d.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.native_dropout_backward]

triton_per_fused__softmax_backward_data_native_dropout_backward_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_native_dropout_backward_12', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/im/cimrxofrtn2ut5dpjayreue3qafwmxqn4ba6akiuxbxci5wxpxka.py
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
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_13', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ra/cra62i4nkcy2ovgyljslgeywibxe7jcm4osvsglserpyrt5tngtp.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_14', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/pt/cptgunjni67lmxdrpkswc4vpvjslx2ivztuma55aa3k6rboczc3h.py
# Source Nodes: [], Original ATen: [aten.mul, aten.view]

triton_poi_fused_mul_view_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_view_15', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ua/cuasclwfhp3ba6s2xxj4jdihhjixwsbuva5k5raandzvxeziwfwm.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_16', 'mutated_arg_names': ['in_out_ptr0']},
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
    primals_9, primals_19, primals_25, view, getitem_1, view_16, getitem_3, mul_1, view_18, view_20, unsqueeze_default, unsqueeze_default_1, unsqueeze_default_2, getitem_17, getitem_18, getitem_19, alias_default_1, view_32, getitem_9, mul_4, view_34, addmm_8, view_36, getitem_13, mul_9, div_2, permute_20, permute_24, div_3, permute_28, permute_40, permute_45, permute_49, div_4, permute_53, permute_58, permute_59, alias_3, permute_60, permute_61, permute_65, permute_70, permute_74, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5 = args
    args.clear()
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_25, (768, ), (1, ))
    assert_size_stride(view, (1024, 768), (768, 1))
    assert_size_stride(getitem_1, (12, 1024, 1024), (1048576, 1024, 1))
    assert_size_stride(view_16, (1024, 768), (768, 1))
    assert_size_stride(getitem_3, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_1, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_18, (1024, 768), (768, 1))
    assert_size_stride(view_20, (1024, 768), (768, 1))
    assert_size_stride(unsqueeze_default, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(unsqueeze_default_1, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(unsqueeze_default_2, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(getitem_17, (1, 12, 1024), (12288, 1024, 1))
    assert_size_stride(getitem_18, (), ())
    assert_size_stride(getitem_19, (), ())
    assert_size_stride(alias_default_1, (1, 12, 1024, 64), (786432, 64, 768, 1))
    assert_size_stride(view_32, (1024, 768), (768, 1))
    assert_size_stride(getitem_9, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_4, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_34, (1024, 768), (768, 1))
    assert_size_stride(addmm_8, (1024, 3072), (3072, 1))
    assert_size_stride(view_36, (1024, 3072), (3072, 1))
    assert_size_stride(getitem_13, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_9, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(div_2, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_20, (768, 3072), (3072, 1))
    assert_size_stride(permute_24, (3072, 768), (768, 1))
    assert_size_stride(div_3, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_28, (768, 768), (768, 1))
    assert_size_stride(permute_40, (768, 768), (768, 1))
    assert_size_stride(permute_45, (768, 768), (768, 1))
    assert_size_stride(permute_49, (768, 768), (768, 1))
    assert_size_stride(div_4, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_53, (768, 768), (768, 1))
    assert_size_stride(permute_58, (12, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_59, (12, 64, 1024), (65536, 1, 64))
    assert_size_stride(alias_3, (12, 1024, 1024), (1048576, 1024, 1))
    assert_size_stride(permute_60, (12, 64, 1024), (65536, 1, 64))
    assert_size_stride(permute_61, (12, 1024, 64), (65536, 64, 1))
    assert_size_stride(permute_65, (768, 768), (768, 1))
    assert_size_stride(permute_70, (768, 768), (768, 1))
    assert_size_stride(permute_74, (768, 768), (768, 1))
    assert_size_stride(tangents_1, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(tangents_2, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_3, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_4, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_5, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf2 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf5 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]
        stream0 = get_cuda_stream(0)
        triton_per_fused_native_dropout_backward_native_layer_norm_backward_0.run(tangents_1, primals_25, mul_9, div_2, getitem_13, buf2, buf5, 1024, 768, grid=grid(1024), stream=stream0)
        del div_2
        del getitem_13
        del primals_25
        buf3 = empty((768, ), device='cuda', dtype=torch.float32)
        buf4 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_1.run(tangents_1, mul_9, buf3, buf4, 768, 1024, grid=grid(768), stream=stream0)
        del mul_9
        del tangents_1
        buf6 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (1024, 768), (768, 1), 0), permute_20, out=buf6)
        del permute_20
        buf7 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (768, 1024), (1, 768), 0), view_36, out=buf7)
        del view_36
        buf8 = empty_strided((1, 768, 8), (6144, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf5, buf8, 6144, 128, grid=grid(6144), stream=stream0)
        buf9 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf8, buf9, 768, 8, grid=grid(768), stream=stream0)
        buf10 = reinterpret_tensor(buf6, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf6  # reuse
        # Source Nodes: [hidden_states_8], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_4.run(buf10, addmm_8, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_8
        buf11 = reinterpret_tensor(buf5, (1024, 768), (768, 1), 0); del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (1024, 3072), (3072, 1), 0), permute_24, out=buf11)
        del permute_24
        buf12 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (3072, 1024), (1, 3072), 0), view_34, out=buf12)
        del view_34
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
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7.run(buf2, buf11, primals_19, mul_4, div_3, getitem_9, buf17, buf20, 1024, 768, grid=grid(1024), stream=stream0)
        del div_3
        del getitem_9
        del primals_19
        buf18 = empty((768, ), device='cuda', dtype=torch.float32)
        buf19 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf2, buf11, mul_4, buf18, buf19, 768, 1024, grid=grid(768), stream=stream0)
        del buf11
        del mul_4
        buf21 = reinterpret_tensor(buf2, (1024, 768), (768, 1), 0); del buf2  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (1024, 768), (768, 1), 0), permute_28, out=buf21)
        del permute_28
        buf22 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (768, 1024), (1, 768), 0), view_32, out=buf22)
        del view_32
        buf23 = buf8; del buf8  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf20, buf23, 6144, 128, grid=grid(6144), stream=stream0)
        buf24 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf23, buf24, 768, 8, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf25 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf21, (1, 12, 1024, 64), (0, 64, 768, 1), 0), unsqueeze_default, unsqueeze_default_1, unsqueeze_default_2, None, alias_default_1, getitem_17, getitem_18, getitem_19, 0.1, [True, True, True, False], scale=1.0)
        del alias_default_1
        del getitem_17
        del getitem_18
        del getitem_19
        del unsqueeze_default
        del unsqueeze_default_1
        del unsqueeze_default_2
        buf26 = buf25[0]
        buf27 = buf25[1]
        buf28 = buf25[2]
        del buf25
        buf29 = reinterpret_tensor(buf28, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf29, tangents_5, 786432, grid=grid(786432), stream=stream0)
        del tangents_5
        buf30 = buf21; del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf29, (1024, 768), (768, 1), 0), permute_40, out=buf30)
        del permute_40
        buf31 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf29, (768, 1024), (1, 768), 0), view_20, out=buf31)
        buf32 = buf23; del buf23  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf29, buf32, 6144, 128, grid=grid(6144), stream=stream0)
        buf33 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf32, buf33, 768, 8, grid=grid(768), stream=stream0)
        buf34 = reinterpret_tensor(buf27, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf34, tangents_4, 786432, grid=grid(786432), stream=stream0)
        del tangents_4
        buf35 = reinterpret_tensor(buf29, (1024, 768), (768, 1), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf34, (1024, 768), (768, 1), 0), permute_45, out=buf35)
        del permute_45
        buf36 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf34, (768, 1024), (1, 768), 0), view_20, out=buf36)
        del view_20
        buf37 = buf32; del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf34, buf37, 6144, 128, grid=grid(6144), stream=stream0)
        buf38 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf37, buf38, 768, 8, grid=grid(768), stream=stream0)
        buf39 = reinterpret_tensor(buf30, (1, 1024, 768), (786432, 768, 1), 0); del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_10.run(buf39, buf35, 786432, grid=grid(786432), stream=stream0)
        buf40 = reinterpret_tensor(buf26, (1024, 768), (768, 1), 0); del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_11.run(buf40, 786432, grid=grid(786432), stream=stream0)
        buf41 = buf35; del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf40, permute_49, out=buf41)
        del permute_49
        buf42 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf40, (768, 1024), (1, 768), 0), view_18, out=buf42)
        del view_18
        buf43 = buf37; del buf37  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf40, buf43, 6144, 128, grid=grid(6144), stream=stream0)
        buf44 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf43, buf44, 768, 8, grid=grid(768), stream=stream0)
        buf47 = reinterpret_tensor(buf40, (1, 1024, 768), (786432, 768, 1), 0); del buf40  # reuse
        buf50 = reinterpret_tensor(buf34, (1, 1024, 768), (786432, 768, 1), 0); del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_7.run(buf17, buf41, primals_9, mul_1, div_4, getitem_3, buf47, buf50, 1024, 768, grid=grid(1024), stream=stream0)
        del div_4
        del getitem_3
        del primals_9
        buf48 = empty((768, ), device='cuda', dtype=torch.float32)
        buf49 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_8.run(buf17, buf41, mul_1, buf48, buf49, 768, 1024, grid=grid(768), stream=stream0)
        del mul_1
        buf51 = buf41; del buf41  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf50, (1024, 768), (768, 1), 0), permute_53, out=buf51)
        del permute_53
        buf52 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf50, (768, 1024), (1, 768), 0), view_16, out=buf52)
        del view_16
        buf53 = buf43; del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf50, buf53, 6144, 128, grid=grid(6144), stream=stream0)
        buf54 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf53, buf54, 768, 8, grid=grid(768), stream=stream0)
        buf55 = reinterpret_tensor(buf50, (12, 1024, 64), (65536, 64, 1), 0); del buf50  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_58, reinterpret_tensor(buf51, (12, 1024, 64), (64, 768, 1), 0), out=buf55)
        del permute_58
        buf56 = empty((12, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf51, (12, 1024, 64), (64, 768, 1), 0), permute_59, out=buf56)
        del permute_59
        buf58 = empty((12, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.native_dropout_backward]
        triton_per_fused__softmax_backward_data_native_dropout_backward_12.run(buf56, getitem_1, alias_3, buf58, 12288, 1024, grid=grid(12288), stream=stream0)
        del alias_3
        del buf56
        del getitem_1
        buf59 = reinterpret_tensor(buf51, (12, 64, 1024), (65536, 1024, 1), 0); del buf51  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_60, reinterpret_tensor(buf58, (12, 1024, 1024), (1048576, 1024, 1), 0), out=buf59)
        del permute_60
        buf60 = reinterpret_tensor(buf17, (12, 1024, 64), (65536, 64, 1), 0); del buf17  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf58, (12, 1024, 1024), (1048576, 1024, 1), 0), permute_61, out=buf60)
        del buf58
        del permute_61
        buf61 = reinterpret_tensor(buf20, (1, 1024, 12, 64), (786432, 768, 64, 1), 0); del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(tangents_3, buf55, buf61, 786432, grid=grid(786432), stream=stream0)
        del tangents_3
        buf62 = reinterpret_tensor(buf55, (1024, 768), (768, 1), 0); del buf55  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf61, (1024, 768), (768, 1), 0), permute_65, out=buf62)
        del permute_65
        buf63 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf61, (768, 1024), (1, 768), 0), view, out=buf63)
        buf64 = buf53; del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf61, buf64, 6144, 128, grid=grid(6144), stream=stream0)
        buf65 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf64, buf65, 768, 8, grid=grid(768), stream=stream0)
        buf66 = buf61; del buf61  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(tangents_2, buf59, buf66, 12288, 64, grid=grid(12288, 64), stream=stream0)
        del tangents_2
        buf67 = reinterpret_tensor(buf59, (1024, 768), (768, 1), 0); del buf59  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf66, (1024, 768), (768, 1), 0), permute_70, out=buf67)
        del permute_70
        buf68 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf66, (768, 1024), (1, 768), 0), view, out=buf68)
        buf69 = buf64; del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf66, buf69, 6144, 128, grid=grid(6144), stream=stream0)
        buf70 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf69, buf70, 768, 8, grid=grid(768), stream=stream0)
        buf71 = reinterpret_tensor(buf66, (1024, 768), (768, 1), 0); del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.mul, aten.view]
        triton_poi_fused_mul_view_15.run(buf60, buf71, 786432, grid=grid(786432), stream=stream0)
        buf72 = reinterpret_tensor(buf60, (1024, 768), (768, 1), 0); del buf60  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf71, permute_74, out=buf72)
        del permute_74
        buf73 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf71, (768, 1024), (1, 768), 0), view, out=buf73)
        del view
        buf74 = buf69; del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf71, buf74, 6144, 128, grid=grid(6144), stream=stream0)
        del buf71
        buf75 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf74, buf75, 768, 8, grid=grid(768), stream=stream0)
        del buf74
        buf76 = buf47; del buf47  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_16.run(buf76, buf62, buf67, buf72, 786432, grid=grid(786432), stream=stream0)
        return (reinterpret_tensor(buf73, (768, 768), (768, 1), 0), reinterpret_tensor(buf75, (768, ), (1, ), 0), reinterpret_tensor(buf68, (768, 768), (768, 1), 0), reinterpret_tensor(buf70, (768, ), (1, ), 0), reinterpret_tensor(buf63, (768, 768), (768, 1), 0), reinterpret_tensor(buf65, (768, ), (1, ), 0), reinterpret_tensor(buf52, (768, 768), (768, 1), 0), reinterpret_tensor(buf54, (768, ), (1, ), 0), buf48, buf49, reinterpret_tensor(buf42, (768, 768), (768, 1), 0), reinterpret_tensor(buf44, (768, ), (1, ), 0), reinterpret_tensor(buf36, (768, 768), (768, 1), 0), reinterpret_tensor(buf38, (768, ), (1, ), 0), reinterpret_tensor(buf31, (768, 768), (768, 1), 0), reinterpret_tensor(buf33, (768, ), (1, ), 0), reinterpret_tensor(buf22, (768, 768), (768, 1), 0), reinterpret_tensor(buf24, (768, ), (1, ), 0), buf18, buf19, reinterpret_tensor(buf12, (3072, 768), (768, 1), 0), reinterpret_tensor(buf14, (3072, ), (1, ), 0), reinterpret_tensor(buf7, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf9, (768, ), (1, ), 0), buf3, buf4, buf76, None, buf39, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_9 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((12, 1024, 1024), (1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    view_16 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_1 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    view_18 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    view_20 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_default = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_default_1 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_default_2 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_17 = rand_strided((1, 12, 1024), (12288, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_18 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_19 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_1 = rand_strided((1, 12, 1024, 64), (786432, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    view_32 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_9 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_4 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    view_34 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_8 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_36 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_13 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_9 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    div_2 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_20 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_24 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_3 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_28 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_40 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_45 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_49 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_4 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_53 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_58 = rand_strided((12, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_59 = rand_strided((12, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_3 = rand_strided((12, 1024, 1024), (1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_60 = rand_strided((12, 64, 1024), (65536, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_61 = rand_strided((12, 1024, 64), (65536, 64, 1), device='cuda:0', dtype=torch.float32)
    permute_65 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_70 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_74 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_3 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_4 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_5 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_9, primals_19, primals_25, view, getitem_1, view_16, getitem_3, mul_1, view_18, view_20, unsqueeze_default, unsqueeze_default_1, unsqueeze_default_2, getitem_17, getitem_18, getitem_19, alias_default_1, view_32, getitem_9, mul_4, view_34, addmm_8, view_36, getitem_13, mul_9, div_2, permute_20, permute_24, div_3, permute_28, permute_40, permute_45, permute_49, div_4, permute_53, permute_58, permute_59, alias_3, permute_60, permute_61, permute_65, permute_70, permute_74, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('PLBartForConditionalGeneration', benchmark_compiled_module)
