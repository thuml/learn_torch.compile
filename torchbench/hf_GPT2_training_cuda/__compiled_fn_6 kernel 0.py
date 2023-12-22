
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


# kernel path: /tmp/torchinductor_youkaichao/rd/crdone53gwsw5heohklsfcifurcyqdulxnf746wgjxncbp243qy6.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
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

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/cj/ccjdbzawblxot4ipjkcdurq4dtiyogavjtswpn3v7dr32xsh5hv7.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/6q/c6qgsy3ishko2vbctwnensr5uddujv7iikskwogbnexeqhzwgde5.py
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
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_2', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/xm/cxm6vofly47ggcb3a5pcu3qx2qufdodq4juum5ulijesv675ezkc.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_3', 'mutated_arg_names': []}
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
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lz/clzy2iy6o26dyzbfzyzdwkpwn5t52ymjucmoeuizmrdgg5iomezp.py
# Source Nodes: [add_47, mul_44], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
# add_47 => add_95
# mul_44 => mul_92
triton_poi_fused_add_mul_pow_tanh_backward_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_tanh_backward_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp5 = tl.load(in_ptr1 + (x0), None)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 * tmp3
    tmp6 = tmp5 * tmp5
    tmp7 = 1.0
    tmp8 = tmp7 - tmp6
    tmp9 = tmp4 * tmp8
    tmp10 = 0.7978845608028654
    tmp11 = tmp9 * tmp10
    tmp12 = 0.044715
    tmp13 = tmp11 * tmp12
    tmp14 = tmp1 * tmp1
    tmp15 = 3.0
    tmp16 = tmp14 * tmp15
    tmp17 = tmp13 * tmp16
    tmp18 = tmp11 + tmp17
    tmp19 = tmp5 + tmp7
    tmp20 = tmp0 * tmp19
    tmp21 = tmp20 * tmp2
    tmp22 = tmp18 + tmp21
    tl.store(in_out_ptr0 + (x0), tmp22, None)
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


# kernel path: /tmp/torchinductor_youkaichao/lb/clbolohqzhuajxuvmlphkxwl6fzjtqutqxzprey77nmq362qpoob.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_7', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
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


# kernel path: /tmp/torchinductor_youkaichao/gh/cgh26ckbjs2feb2i46dgizqagxd6p5e2ssrofp5p74yntn42d4yq.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 512
    x2 = (xindex // 32768) % 12
    x3 = (xindex // 393216)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (768*x1) + (393216*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/j2/cj2ullwygcw4twvx77eidczzx2zeop7kbwl2t7xeyihwasvkfdjc.py
# Source Nodes: [full], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.scalar_tensor, aten.where]
# full => full_default
triton_per_fused__softmax_backward_data_div_full_scalar_tensor_where_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_div_full_scalar_tensor_where_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel):
    xnumel = 12288
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
    x2 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (1024*x2)), rmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp1 * tmp6
    tmp9 = tmp2 - tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp7, tmp9, tmp10)
    tmp12 = 8.0
    tmp13 = tmp11 / tmp12
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp13, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fa/cfagzmkct2nemnttyogg54hpbv5kbd5akwwrldxxhgcx3xtbjl3f.py
# Source Nodes: [], Original ATen: [aten.cat]

triton_poi_fused_cat_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 2304
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
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 768, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((64*y0) + (32768*((x2 // 64) % 12)) + (393216*y1) + (x2 % 64)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 1536, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((64*y0) + (32768*((x2 // 64) % 12)) + (393216*y1) + (x2 % 64)), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr2 + (y0 + (512*(x2 % 768)) + (393216*y1)), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp11, tmp14, tmp15)
    tmp17 = tmp0 >= tmp9
    tmp18 = tl.full([1, 1], 2304, tl.int64)
    tmp19 = tmp0 < tmp18
    tmp20 = tl.load(in_ptr3 + ((64*y0) + (32768*((x2 // 64) % 12)) + (393216*y1) + (x2 % 64)), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr4 + ((64*y0) + (32768*((x2 // 64) % 12)) + (393216*y1) + (x2 % 64)), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp17, tmp22, tmp23)
    tmp25 = tl.where(tmp11, tmp16, tmp24)
    tmp26 = tl.where(tmp4, tmp7, tmp25)
    tl.store(out_ptr0 + (x2 + (2304*y3)), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/li/cliborb7dkdn7oqba6fju4g2qtxfznpdynmzdkbap66t4sw7wcfy.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18432
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2304
    x1 = (xindex // 2304)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (2304*r2) + (294912*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kd/ckdxvsztspk27bjbkafo7cvigwnpwtqogboyn2bddqnbfo2ofr4z.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2304
    rnumel = 8
    RBLOCK: tl.constexpr = 8
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


# kernel path: /tmp/torchinductor_youkaichao/sl/cslh5fglruqubvnw2bqjmjotgnazpketsjdvxe6iuszcjeluu2xe.py
# Source Nodes: [], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_layer_norm_backward, aten.scalar_tensor]

triton_per_fused_add_embedding_dense_backward_native_layer_norm_backward_scalar_tensor_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i64', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_dense_backward_native_layer_norm_backward_scalar_tensor_13', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
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
    tmp13 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp23 = tl.full([1], -1, tl.int64)
    tmp24 = tmp22 == tmp23
    tmp25 = 0.0
    tmp26 = tl.where(tmp24, tmp25, tmp21)
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp26, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/os/cosgcel5izhyd7ow4fdg7ikknavkwmtp3kvujmshrphepa5jyipy.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_14', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/xn/cxnxarntpxfe2vz7ry7i3jdikjknfosdl3mpqqojunsnasqw6ovy.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward, aten.scalar_tensor, aten.sum]

triton_poi_fused_embedding_dense_backward_scalar_tensor_sum_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_scalar_tensor_sum_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (393216 + x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], False, tl.int1)
    tmp4 = 0.0
    tmp5 = tl.where(tmp3, tmp4, tmp2)
    tl.store(out_ptr0 + (x0), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/el/cel4ex6nptcxxtkx4bxbeahrdvehtmyvaagiqtg3e44223kwabbn.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 38597376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, view, view_1, mul, slice_4, mul_2, addmm_2, tanh, mul_8, slice_8, mul_10, addmm_6, tanh_1, mul_16, slice_12, mul_18, addmm_10, tanh_2, mul_24, slice_16, mul_26, addmm_14, tanh_3, mul_32, slice_20, mul_34, addmm_18, tanh_4, mul_40, slice_24, mul_42, addmm_22, tanh_5, mul_48, slice_28, mul_50, addmm_26, tanh_6, mul_56, slice_32, mul_58, addmm_30, tanh_7, mul_64, slice_36, mul_66, addmm_34, tanh_8, mul_72, slice_40, mul_74, addmm_38, tanh_9, mul_80, slice_44, mul_82, addmm_42, tanh_10, mul_88, slice_48, mul_90, addmm_46, tanh_11, mul_96, view_219, permute_63, div_24, permute_65, permute_66, permute_67, permute_68, div_25, permute_69, permute_70, permute_72, permute_73, alias_25, permute_74, permute_75, permute_80, permute_81, div_27, permute_82, permute_83, permute_84, permute_85, div_28, permute_86, permute_87, permute_89, permute_90, alias_27, permute_91, permute_92, permute_97, permute_98, div_30, permute_99, permute_100, permute_101, permute_102, div_31, permute_103, permute_104, permute_106, permute_107, alias_29, permute_108, permute_109, permute_114, permute_115, div_33, permute_116, permute_117, permute_118, permute_119, div_34, permute_120, permute_121, permute_123, permute_124, alias_31, permute_125, permute_126, permute_131, permute_132, div_36, permute_133, permute_134, permute_135, permute_136, div_37, permute_137, permute_138, permute_140, permute_141, alias_33, permute_142, permute_143, permute_148, permute_149, div_39, permute_150, permute_151, permute_152, permute_153, div_40, permute_154, permute_155, permute_157, permute_158, alias_35, permute_159, permute_160, permute_165, permute_166, div_42, permute_167, permute_168, permute_169, permute_170, div_43, permute_171, permute_172, permute_174, permute_175, alias_37, permute_176, permute_177, permute_182, permute_183, div_45, permute_184, permute_185, permute_186, permute_187, div_46, permute_188, permute_189, permute_191, permute_192, alias_39, permute_193, permute_194, permute_199, permute_200, div_48, permute_201, permute_202, permute_203, permute_204, div_49, permute_205, permute_206, permute_208, permute_209, alias_41, permute_210, permute_211, permute_216, permute_217, div_51, permute_218, permute_219, permute_220, permute_221, div_52, permute_222, permute_223, permute_225, permute_226, alias_43, permute_227, permute_228, permute_233, permute_234, div_54, permute_235, permute_236, permute_237, permute_238, div_55, permute_239, permute_240, permute_242, permute_243, alias_45, permute_244, permute_245, permute_250, permute_251, div_57, permute_252, permute_253, permute_254, permute_255, div_58, permute_256, permute_257, permute_259, permute_260, alias_47, permute_261, permute_262, permute_267, permute_268, div_60, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25 = args
    args.clear()
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_101, (768, ), (1, ))
    assert_size_stride(primals_103, (768, ), (1, ))
    assert_size_stride(primals_105, (768, ), (1, ))
    assert_size_stride(primals_107, (768, ), (1, ))
    assert_size_stride(primals_109, (768, ), (1, ))
    assert_size_stride(primals_111, (768, ), (1, ))
    assert_size_stride(primals_113, (768, ), (1, ))
    assert_size_stride(primals_115, (768, ), (1, ))
    assert_size_stride(primals_117, (768, ), (1, ))
    assert_size_stride(primals_119, (768, ), (1, ))
    assert_size_stride(primals_121, (768, ), (1, ))
    assert_size_stride(primals_123, (768, ), (1, ))
    assert_size_stride(primals_125, (768, ), (1, ))
    assert_size_stride(primals_127, (768, ), (1, ))
    assert_size_stride(primals_129, (768, ), (1, ))
    assert_size_stride(primals_131, (768, ), (1, ))
    assert_size_stride(primals_133, (768, ), (1, ))
    assert_size_stride(primals_135, (768, ), (1, ))
    assert_size_stride(primals_137, (768, ), (1, ))
    assert_size_stride(primals_139, (768, ), (1, ))
    assert_size_stride(primals_141, (768, ), (1, ))
    assert_size_stride(primals_143, (768, ), (1, ))
    assert_size_stride(primals_145, (768, ), (1, ))
    assert_size_stride(primals_147, (768, ), (1, ))
    assert_size_stride(view, (2, 512), (512, 1))
    assert_size_stride(view_1, (1, 512), (512, 1))
    assert_size_stride(mul, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_4, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(mul_2, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_2, (1024, 3072), (3072, 1))
    assert_size_stride(tanh, (2, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(mul_8, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_8, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(mul_10, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_6, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_1, (2, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(mul_16, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_12, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(mul_18, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_10, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_2, (2, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(mul_24, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_16, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(mul_26, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_14, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_3, (2, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(mul_32, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_20, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(mul_34, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_18, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_4, (2, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(mul_40, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_24, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(mul_42, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_22, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_5, (2, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(mul_48, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_28, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(mul_50, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_26, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_6, (2, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(mul_56, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_32, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(mul_58, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_30, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_7, (2, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(mul_64, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_36, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(mul_66, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_34, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_8, (2, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(mul_72, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_40, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(mul_74, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_38, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_9, (2, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(mul_80, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_44, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(mul_82, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_42, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_10, (2, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(mul_88, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(slice_48, (1, 1, 512, 512), (1048576, 1048576, 1024, 1))
    assert_size_stride(mul_90, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(addmm_46, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_11, (2, 512, 3072), (1572864, 3072, 1))
    assert_size_stride(mul_96, (2, 512, 768), (393216, 768, 1))
    assert_size_stride(view_219, (1024, 768), (768, 1))
    assert_size_stride(permute_63, (50257, 768), (768, 1))
    assert_size_stride(div_24, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_65, (768, 3072), (1, 768))
    assert_size_stride(permute_66, (3072, 1024), (1, 3072))
    assert_size_stride(permute_67, (3072, 768), (1, 3072))
    assert_size_stride(permute_68, (768, 1024), (1, 768))
    assert_size_stride(div_25, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_69, (768, 768), (1, 768))
    assert_size_stride(permute_70, (768, 1024), (1, 768))
    assert_size_stride(permute_72, (24, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_73, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_25, (2, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_74, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_75, (24, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_80, (2304, 768), (1, 2304))
    assert_size_stride(permute_81, (768, 1024), (1, 768))
    assert_size_stride(div_27, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_82, (768, 3072), (1, 768))
    assert_size_stride(permute_83, (3072, 1024), (1, 3072))
    assert_size_stride(permute_84, (3072, 768), (1, 3072))
    assert_size_stride(permute_85, (768, 1024), (1, 768))
    assert_size_stride(div_28, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_86, (768, 768), (1, 768))
    assert_size_stride(permute_87, (768, 1024), (1, 768))
    assert_size_stride(permute_89, (24, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_90, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_27, (2, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_91, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_92, (24, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_97, (2304, 768), (1, 2304))
    assert_size_stride(permute_98, (768, 1024), (1, 768))
    assert_size_stride(div_30, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_99, (768, 3072), (1, 768))
    assert_size_stride(permute_100, (3072, 1024), (1, 3072))
    assert_size_stride(permute_101, (3072, 768), (1, 3072))
    assert_size_stride(permute_102, (768, 1024), (1, 768))
    assert_size_stride(div_31, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_103, (768, 768), (1, 768))
    assert_size_stride(permute_104, (768, 1024), (1, 768))
    assert_size_stride(permute_106, (24, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_107, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_29, (2, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_108, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_109, (24, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_114, (2304, 768), (1, 2304))
    assert_size_stride(permute_115, (768, 1024), (1, 768))
    assert_size_stride(div_33, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_116, (768, 3072), (1, 768))
    assert_size_stride(permute_117, (3072, 1024), (1, 3072))
    assert_size_stride(permute_118, (3072, 768), (1, 3072))
    assert_size_stride(permute_119, (768, 1024), (1, 768))
    assert_size_stride(div_34, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_120, (768, 768), (1, 768))
    assert_size_stride(permute_121, (768, 1024), (1, 768))
    assert_size_stride(permute_123, (24, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_124, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_31, (2, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_125, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_126, (24, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_131, (2304, 768), (1, 2304))
    assert_size_stride(permute_132, (768, 1024), (1, 768))
    assert_size_stride(div_36, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_133, (768, 3072), (1, 768))
    assert_size_stride(permute_134, (3072, 1024), (1, 3072))
    assert_size_stride(permute_135, (3072, 768), (1, 3072))
    assert_size_stride(permute_136, (768, 1024), (1, 768))
    assert_size_stride(div_37, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_137, (768, 768), (1, 768))
    assert_size_stride(permute_138, (768, 1024), (1, 768))
    assert_size_stride(permute_140, (24, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_141, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_33, (2, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_142, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_143, (24, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_148, (2304, 768), (1, 2304))
    assert_size_stride(permute_149, (768, 1024), (1, 768))
    assert_size_stride(div_39, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_150, (768, 3072), (1, 768))
    assert_size_stride(permute_151, (3072, 1024), (1, 3072))
    assert_size_stride(permute_152, (3072, 768), (1, 3072))
    assert_size_stride(permute_153, (768, 1024), (1, 768))
    assert_size_stride(div_40, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_154, (768, 768), (1, 768))
    assert_size_stride(permute_155, (768, 1024), (1, 768))
    assert_size_stride(permute_157, (24, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_158, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_35, (2, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_159, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_160, (24, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_165, (2304, 768), (1, 2304))
    assert_size_stride(permute_166, (768, 1024), (1, 768))
    assert_size_stride(div_42, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_167, (768, 3072), (1, 768))
    assert_size_stride(permute_168, (3072, 1024), (1, 3072))
    assert_size_stride(permute_169, (3072, 768), (1, 3072))
    assert_size_stride(permute_170, (768, 1024), (1, 768))
    assert_size_stride(div_43, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_171, (768, 768), (1, 768))
    assert_size_stride(permute_172, (768, 1024), (1, 768))
    assert_size_stride(permute_174, (24, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_175, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_37, (2, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_176, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_177, (24, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_182, (2304, 768), (1, 2304))
    assert_size_stride(permute_183, (768, 1024), (1, 768))
    assert_size_stride(div_45, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_184, (768, 3072), (1, 768))
    assert_size_stride(permute_185, (3072, 1024), (1, 3072))
    assert_size_stride(permute_186, (3072, 768), (1, 3072))
    assert_size_stride(permute_187, (768, 1024), (1, 768))
    assert_size_stride(div_46, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_188, (768, 768), (1, 768))
    assert_size_stride(permute_189, (768, 1024), (1, 768))
    assert_size_stride(permute_191, (24, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_192, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_39, (2, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_193, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_194, (24, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_199, (2304, 768), (1, 2304))
    assert_size_stride(permute_200, (768, 1024), (1, 768))
    assert_size_stride(div_48, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_201, (768, 3072), (1, 768))
    assert_size_stride(permute_202, (3072, 1024), (1, 3072))
    assert_size_stride(permute_203, (3072, 768), (1, 3072))
    assert_size_stride(permute_204, (768, 1024), (1, 768))
    assert_size_stride(div_49, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_205, (768, 768), (1, 768))
    assert_size_stride(permute_206, (768, 1024), (1, 768))
    assert_size_stride(permute_208, (24, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_209, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_41, (2, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_210, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_211, (24, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_216, (2304, 768), (1, 2304))
    assert_size_stride(permute_217, (768, 1024), (1, 768))
    assert_size_stride(div_51, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_218, (768, 3072), (1, 768))
    assert_size_stride(permute_219, (3072, 1024), (1, 3072))
    assert_size_stride(permute_220, (3072, 768), (1, 3072))
    assert_size_stride(permute_221, (768, 1024), (1, 768))
    assert_size_stride(div_52, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_222, (768, 768), (1, 768))
    assert_size_stride(permute_223, (768, 1024), (1, 768))
    assert_size_stride(permute_225, (24, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_226, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_43, (2, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_227, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_228, (24, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_233, (2304, 768), (1, 2304))
    assert_size_stride(permute_234, (768, 1024), (1, 768))
    assert_size_stride(div_54, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_235, (768, 3072), (1, 768))
    assert_size_stride(permute_236, (3072, 1024), (1, 3072))
    assert_size_stride(permute_237, (3072, 768), (1, 3072))
    assert_size_stride(permute_238, (768, 1024), (1, 768))
    assert_size_stride(div_55, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_239, (768, 768), (1, 768))
    assert_size_stride(permute_240, (768, 1024), (1, 768))
    assert_size_stride(permute_242, (24, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_243, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_45, (2, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_244, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_245, (24, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_250, (2304, 768), (1, 2304))
    assert_size_stride(permute_251, (768, 1024), (1, 768))
    assert_size_stride(div_57, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_252, (768, 3072), (1, 768))
    assert_size_stride(permute_253, (3072, 1024), (1, 3072))
    assert_size_stride(permute_254, (3072, 768), (1, 3072))
    assert_size_stride(permute_255, (768, 1024), (1, 768))
    assert_size_stride(div_58, (2, 512, 1), (512, 1, 1))
    assert_size_stride(permute_256, (768, 768), (1, 768))
    assert_size_stride(permute_257, (768, 1024), (1, 768))
    assert_size_stride(permute_259, (24, 512, 512), (262144, 1, 512))
    assert_size_stride(permute_260, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(alias_47, (2, 12, 512, 512), (3145728, 262144, 512, 1))
    assert_size_stride(permute_261, (24, 64, 512), (32768, 1, 64))
    assert_size_stride(permute_262, (24, 512, 64), (32768, 1, 512))
    assert_size_stride(permute_267, (2304, 768), (1, 2304))
    assert_size_stride(permute_268, (768, 1024), (1, 768))
    assert_size_stride(div_60, (2, 512, 1), (512, 1, 1))
    assert_size_stride(tangents_1, (2, 512, 50257), (25731584, 50257, 1))
    assert_size_stride(tangents_2, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_3, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_4, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_5, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_6, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_7, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_8, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_9, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_10, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_11, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_12, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_13, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_14, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_15, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_16, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_17, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_18, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_19, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_20, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_21, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_22, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_23, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_24, (2, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(tangents_25, (2, 12, 512, 64), (393216, 32768, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((50257, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (50257, 1024), (1, 50257), 0), view_219, out=buf0)
        del view_219
        buf1 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1024, 50257), (50257, 1), 0), permute_63, out=buf1)
        del permute_63
        del tangents_1
        buf4 = empty((2, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        stream0 = get_cuda_stream(0)
        triton_per_fused_native_layer_norm_backward_0.run(buf1, primals_147, mul_96, div_24, buf4, 1024, 768, grid=grid(1024), stream=stream0)
        del div_24
        del primals_147
        buf5 = empty_strided((768, 8), (1, 768), device='cuda', dtype=torch.float32)
        buf7 = empty_strided((768, 8), (1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_1.run(buf1, mul_96, buf5, buf7, 6144, 128, grid=grid(6144), stream=stream0)
        del mul_96
        buf6 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf5, buf6, 768, 8, grid=grid(768), stream=stream0)
        buf8 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf7, buf8, 768, 8, grid=grid(768), stream=stream0)
        buf9 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (1024, 768), (768, 1), 0), permute_65, out=buf9)
        del permute_65
        buf10 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_66, reinterpret_tensor(buf4, (1024, 768), (768, 1), 0), out=buf10)
        del permute_66
        buf11 = reinterpret_tensor(buf7, (1, 768, 8), (6144, 1, 768), 0); del buf7  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf4, buf11, 6144, 128, grid=grid(6144), stream=stream0)
        buf12 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf11, buf12, 768, 8, grid=grid(768), stream=stream0)
        buf13 = reinterpret_tensor(buf9, (2, 512, 3072), (1572864, 3072, 1), 0); del buf9  # reuse
        # Source Nodes: [add_47, mul_44], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_4.run(buf13, addmm_46, tanh_11, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_46
        del tanh_11
        buf14 = buf1; del buf1  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (1024, 3072), (3072, 1), 0), permute_67, out=buf14)
        del permute_67
        buf15 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_68, reinterpret_tensor(buf13, (1024, 3072), (3072, 1), 0), out=buf15)
        del permute_68
        buf16 = empty_strided((1, 3072, 8), (24576, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf13, buf16, 24576, 128, grid=grid(24576), stream=stream0)
        buf17 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf16, buf17, 3072, 8, grid=grid(3072), stream=stream0)
        buf24 = buf4; del buf4  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf24, buf14, primals_145, mul_90, div_25, 1024, 768, grid=grid(1024), stream=stream0)
        del div_25
        del primals_145
        buf20 = reinterpret_tensor(buf11, (768, 8), (1, 768), 0); del buf11  # reuse
        buf22 = buf5; del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_1.run(buf14, mul_90, buf20, buf22, 6144, 128, grid=grid(6144), stream=stream0)
        del mul_90
        buf21 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf20, buf21, 768, 8, grid=grid(768), stream=stream0)
        buf23 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf22, buf23, 768, 8, grid=grid(768), stream=stream0)
        buf25 = buf14; del buf14  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf24, (1024, 768), (768, 1), 0), permute_69, out=buf25)
        del permute_69
        buf26 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_70, reinterpret_tensor(buf24, (1024, 768), (768, 1), 0), out=buf26)
        del permute_70
        buf27 = reinterpret_tensor(buf22, (1, 768, 8), (6144, 1, 768), 0); del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf24, buf27, 6144, 128, grid=grid(6144), stream=stream0)
        buf28 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf27, buf28, 768, 8, grid=grid(768), stream=stream0)
        buf29 = empty((2, 12, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf25, buf29, 786432, grid=grid(786432), stream=stream0)
        buf30 = reinterpret_tensor(buf25, (24, 512, 64), (32768, 64, 1), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_72, reinterpret_tensor(buf29, (24, 512, 64), (32768, 64, 1), 0), out=buf30)
        del permute_72
        buf31 = empty((24, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf29, (24, 512, 64), (32768, 64, 1), 0), permute_73, out=buf31)
        del permute_73
        buf33 = empty((2, 12, 512, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [full], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.scalar_tensor, aten.where]
        triton_per_fused__softmax_backward_data_div_full_scalar_tensor_where_9.run(buf31, alias_25, slice_48, buf33, 12288, 512, grid=grid(12288), stream=stream0)
        del alias_25
        del slice_48
        buf34 = reinterpret_tensor(buf29, (24, 64, 512), (32768, 512, 1), 0); del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_74, reinterpret_tensor(buf33, (24, 512, 512), (262144, 512, 1), 0), out=buf34)
        del permute_74
        buf35 = empty((24, 512, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf33, (24, 512, 512), (262144, 512, 1), 0), permute_75, out=buf35)
        del permute_75
        buf36 = empty((2, 512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_10.run(buf35, tangents_24, buf34, tangents_25, buf30, buf36, 1024, 2304, grid=grid(1024, 2304), stream=stream0)
        del tangents_24
        del tangents_25
        buf37 = reinterpret_tensor(buf35, (1024, 768), (768, 1), 0); del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (1024, 2304), (2304, 1), 0), permute_80, out=buf37)
        del permute_80
        buf38 = empty((768, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_81, reinterpret_tensor(buf36, (1024, 2304), (2304, 1), 0), out=buf38)
        del permute_81
        buf39 = empty_strided((1, 2304, 8), (18432, 1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf36, buf39, 18432, 128, grid=grid(18432), stream=stream0)
        buf40 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf39, buf40, 2304, 8, grid=grid(2304), stream=stream0)
        buf47 = buf24; del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf47, buf37, primals_143, mul_88, div_27, 1024, 768, grid=grid(1024), stream=stream0)
        del div_27
        del primals_143
        buf43 = reinterpret_tensor(buf27, (768, 8), (1, 768), 0); del buf27  # reuse
        buf45 = buf20; del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_1.run(buf37, mul_88, buf43, buf45, 6144, 128, grid=grid(6144), stream=stream0)
        del mul_88
        buf44 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf43, buf44, 768, 8, grid=grid(768), stream=stream0)
        buf46 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf45, buf46, 768, 8, grid=grid(768), stream=stream0)
        buf48 = reinterpret_tensor(buf13, (1024, 3072), (3072, 1), 0); del buf13  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf47, (1024, 768), (768, 1), 0), permute_82, out=buf48)
        del permute_82
        buf49 = reinterpret_tensor(buf36, (3072, 768), (768, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_83, reinterpret_tensor(buf47, (1024, 768), (768, 1), 0), out=buf49)
        del permute_83
        buf50 = reinterpret_tensor(buf45, (1, 768, 8), (6144, 1, 768), 0); del buf45  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf47, buf50, 6144, 128, grid=grid(6144), stream=stream0)
        buf51 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf50, buf51, 768, 8, grid=grid(768), stream=stream0)
        buf52 = reinterpret_tensor(buf48, (2, 512, 3072), (1572864, 3072, 1), 0); del buf48  # reuse
        # Source Nodes: [add_43, mul_40], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_4.run(buf52, addmm_42, tanh_10, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_42
        del tanh_10
        buf53 = buf37; del buf37  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf52, (1024, 3072), (3072, 1), 0), permute_84, out=buf53)
        del permute_84
        buf54 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_85, reinterpret_tensor(buf52, (1024, 3072), (3072, 1), 0), out=buf54)
        del permute_85
        buf55 = buf16; del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf52, buf55, 24576, 128, grid=grid(24576), stream=stream0)
        buf56 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf55, buf56, 3072, 8, grid=grid(3072), stream=stream0)
        buf63 = buf47; del buf47  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf63, buf53, primals_141, mul_82, div_28, 1024, 768, grid=grid(1024), stream=stream0)
        del div_28
        del primals_141
        buf59 = reinterpret_tensor(buf50, (768, 8), (1, 768), 0); del buf50  # reuse
        buf61 = buf43; del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_1.run(buf53, mul_82, buf59, buf61, 6144, 128, grid=grid(6144), stream=stream0)
        del mul_82
        buf60 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf59, buf60, 768, 8, grid=grid(768), stream=stream0)
        buf62 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf61, buf62, 768, 8, grid=grid(768), stream=stream0)
        buf64 = buf53; del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf63, (1024, 768), (768, 1), 0), permute_86, out=buf64)
        del permute_86
        buf65 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_87, reinterpret_tensor(buf63, (1024, 768), (768, 1), 0), out=buf65)
        del permute_87
        buf66 = reinterpret_tensor(buf61, (1, 768, 8), (6144, 1, 768), 0); del buf61  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf63, buf66, 6144, 128, grid=grid(6144), stream=stream0)
        buf67 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf66, buf67, 768, 8, grid=grid(768), stream=stream0)
        buf68 = reinterpret_tensor(buf34, (2, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf64, buf68, 786432, grid=grid(786432), stream=stream0)
        buf69 = reinterpret_tensor(buf64, (24, 512, 64), (32768, 64, 1), 0); del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_89, reinterpret_tensor(buf68, (24, 512, 64), (32768, 64, 1), 0), out=buf69)
        del permute_89
        buf70 = reinterpret_tensor(buf33, (24, 512, 512), (262144, 512, 1), 0); del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf68, (24, 512, 64), (32768, 64, 1), 0), permute_90, out=buf70)
        del permute_90
        buf72 = reinterpret_tensor(buf31, (2, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf31  # reuse
        # Source Nodes: [full], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.scalar_tensor, aten.where]
        triton_per_fused__softmax_backward_data_div_full_scalar_tensor_where_9.run(buf70, alias_27, slice_44, buf72, 12288, 512, grid=grid(12288), stream=stream0)
        del alias_27
        del slice_44
        buf73 = reinterpret_tensor(buf68, (24, 64, 512), (32768, 512, 1), 0); del buf68  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_91, reinterpret_tensor(buf72, (24, 512, 512), (262144, 512, 1), 0), out=buf73)
        del permute_91
        buf74 = buf30; del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf72, (24, 512, 512), (262144, 512, 1), 0), permute_92, out=buf74)
        del permute_92
        buf75 = empty((2, 512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_10.run(buf74, tangents_22, buf73, tangents_23, buf69, buf75, 1024, 2304, grid=grid(1024, 2304), stream=stream0)
        del tangents_22
        del tangents_23
        buf76 = reinterpret_tensor(buf74, (1024, 768), (768, 1), 0); del buf74  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf75, (1024, 2304), (2304, 1), 0), permute_97, out=buf76)
        del permute_97
        buf77 = empty((768, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_98, reinterpret_tensor(buf75, (1024, 2304), (2304, 1), 0), out=buf77)
        del permute_98
        buf78 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf75, buf78, 18432, 128, grid=grid(18432), stream=stream0)
        buf79 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf78, buf79, 2304, 8, grid=grid(2304), stream=stream0)
        buf86 = buf63; del buf63  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf86, buf76, primals_139, mul_80, div_30, 1024, 768, grid=grid(1024), stream=stream0)
        del div_30
        del primals_139
        buf82 = reinterpret_tensor(buf66, (768, 8), (1, 768), 0); del buf66  # reuse
        buf84 = buf59; del buf59  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_1.run(buf76, mul_80, buf82, buf84, 6144, 128, grid=grid(6144), stream=stream0)
        del mul_80
        buf83 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf82, buf83, 768, 8, grid=grid(768), stream=stream0)
        buf85 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf84, buf85, 768, 8, grid=grid(768), stream=stream0)
        buf87 = reinterpret_tensor(buf52, (1024, 3072), (3072, 1), 0); del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf86, (1024, 768), (768, 1), 0), permute_99, out=buf87)
        del permute_99
        buf88 = reinterpret_tensor(buf75, (3072, 768), (768, 1), 0); del buf75  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_100, reinterpret_tensor(buf86, (1024, 768), (768, 1), 0), out=buf88)
        del permute_100
        buf89 = reinterpret_tensor(buf84, (1, 768, 8), (6144, 1, 768), 0); del buf84  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf86, buf89, 6144, 128, grid=grid(6144), stream=stream0)
        buf90 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf89, buf90, 768, 8, grid=grid(768), stream=stream0)
        buf91 = reinterpret_tensor(buf87, (2, 512, 3072), (1572864, 3072, 1), 0); del buf87  # reuse
        # Source Nodes: [add_39, mul_36], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_4.run(buf91, addmm_38, tanh_9, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_38
        del tanh_9
        buf92 = buf76; del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf91, (1024, 3072), (3072, 1), 0), permute_101, out=buf92)
        del permute_101
        buf93 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_102, reinterpret_tensor(buf91, (1024, 3072), (3072, 1), 0), out=buf93)
        del permute_102
        buf94 = buf55; del buf55  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf91, buf94, 24576, 128, grid=grid(24576), stream=stream0)
        buf95 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf94, buf95, 3072, 8, grid=grid(3072), stream=stream0)
        buf102 = buf86; del buf86  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf102, buf92, primals_137, mul_74, div_31, 1024, 768, grid=grid(1024), stream=stream0)
        del div_31
        del primals_137
        buf98 = reinterpret_tensor(buf89, (768, 8), (1, 768), 0); del buf89  # reuse
        buf100 = buf82; del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_1.run(buf92, mul_74, buf98, buf100, 6144, 128, grid=grid(6144), stream=stream0)
        del mul_74
        buf99 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf98, buf99, 768, 8, grid=grid(768), stream=stream0)
        buf101 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf100, buf101, 768, 8, grid=grid(768), stream=stream0)
        buf103 = buf92; del buf92  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf102, (1024, 768), (768, 1), 0), permute_103, out=buf103)
        del permute_103
        buf104 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_104, reinterpret_tensor(buf102, (1024, 768), (768, 1), 0), out=buf104)
        del permute_104
        buf105 = reinterpret_tensor(buf100, (1, 768, 8), (6144, 1, 768), 0); del buf100  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf102, buf105, 6144, 128, grid=grid(6144), stream=stream0)
        buf106 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf105, buf106, 768, 8, grid=grid(768), stream=stream0)
        buf107 = reinterpret_tensor(buf73, (2, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf73  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf103, buf107, 786432, grid=grid(786432), stream=stream0)
        buf108 = reinterpret_tensor(buf103, (24, 512, 64), (32768, 64, 1), 0); del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_106, reinterpret_tensor(buf107, (24, 512, 64), (32768, 64, 1), 0), out=buf108)
        del permute_106
        buf109 = reinterpret_tensor(buf72, (24, 512, 512), (262144, 512, 1), 0); del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf107, (24, 512, 64), (32768, 64, 1), 0), permute_107, out=buf109)
        del permute_107
        buf111 = reinterpret_tensor(buf70, (2, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf70  # reuse
        # Source Nodes: [full], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.scalar_tensor, aten.where]
        triton_per_fused__softmax_backward_data_div_full_scalar_tensor_where_9.run(buf109, alias_29, slice_40, buf111, 12288, 512, grid=grid(12288), stream=stream0)
        del alias_29
        del slice_40
        buf112 = reinterpret_tensor(buf107, (24, 64, 512), (32768, 512, 1), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_108, reinterpret_tensor(buf111, (24, 512, 512), (262144, 512, 1), 0), out=buf112)
        del permute_108
        buf113 = buf69; del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf111, (24, 512, 512), (262144, 512, 1), 0), permute_109, out=buf113)
        del permute_109
        buf114 = empty((2, 512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_10.run(buf113, tangents_20, buf112, tangents_21, buf108, buf114, 1024, 2304, grid=grid(1024, 2304), stream=stream0)
        del tangents_20
        del tangents_21
        buf115 = reinterpret_tensor(buf113, (1024, 768), (768, 1), 0); del buf113  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf114, (1024, 2304), (2304, 1), 0), permute_114, out=buf115)
        del permute_114
        buf116 = empty((768, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_115, reinterpret_tensor(buf114, (1024, 2304), (2304, 1), 0), out=buf116)
        del permute_115
        buf117 = buf78; del buf78  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf114, buf117, 18432, 128, grid=grid(18432), stream=stream0)
        buf118 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf117, buf118, 2304, 8, grid=grid(2304), stream=stream0)
        buf125 = buf102; del buf102  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf125, buf115, primals_135, mul_72, div_33, 1024, 768, grid=grid(1024), stream=stream0)
        del div_33
        del primals_135
        buf121 = reinterpret_tensor(buf105, (768, 8), (1, 768), 0); del buf105  # reuse
        buf123 = buf98; del buf98  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_1.run(buf115, mul_72, buf121, buf123, 6144, 128, grid=grid(6144), stream=stream0)
        del mul_72
        buf122 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf121, buf122, 768, 8, grid=grid(768), stream=stream0)
        buf124 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf123, buf124, 768, 8, grid=grid(768), stream=stream0)
        buf126 = reinterpret_tensor(buf91, (1024, 3072), (3072, 1), 0); del buf91  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (1024, 768), (768, 1), 0), permute_116, out=buf126)
        del permute_116
        buf127 = reinterpret_tensor(buf114, (3072, 768), (768, 1), 0); del buf114  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_117, reinterpret_tensor(buf125, (1024, 768), (768, 1), 0), out=buf127)
        del permute_117
        buf128 = reinterpret_tensor(buf123, (1, 768, 8), (6144, 1, 768), 0); del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf125, buf128, 6144, 128, grid=grid(6144), stream=stream0)
        buf129 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf128, buf129, 768, 8, grid=grid(768), stream=stream0)
        buf130 = reinterpret_tensor(buf126, (2, 512, 3072), (1572864, 3072, 1), 0); del buf126  # reuse
        # Source Nodes: [add_35, mul_32], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_4.run(buf130, addmm_34, tanh_8, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_34
        del tanh_8
        buf131 = buf115; del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf130, (1024, 3072), (3072, 1), 0), permute_118, out=buf131)
        del permute_118
        buf132 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_119, reinterpret_tensor(buf130, (1024, 3072), (3072, 1), 0), out=buf132)
        del permute_119
        buf133 = buf94; del buf94  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf130, buf133, 24576, 128, grid=grid(24576), stream=stream0)
        buf134 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf133, buf134, 3072, 8, grid=grid(3072), stream=stream0)
        buf141 = buf125; del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf141, buf131, primals_133, mul_66, div_34, 1024, 768, grid=grid(1024), stream=stream0)
        del div_34
        del primals_133
        buf137 = reinterpret_tensor(buf128, (768, 8), (1, 768), 0); del buf128  # reuse
        buf139 = buf121; del buf121  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_1.run(buf131, mul_66, buf137, buf139, 6144, 128, grid=grid(6144), stream=stream0)
        del mul_66
        buf138 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf137, buf138, 768, 8, grid=grid(768), stream=stream0)
        buf140 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf139, buf140, 768, 8, grid=grid(768), stream=stream0)
        buf142 = buf131; del buf131  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf141, (1024, 768), (768, 1), 0), permute_120, out=buf142)
        del permute_120
        buf143 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_121, reinterpret_tensor(buf141, (1024, 768), (768, 1), 0), out=buf143)
        del permute_121
        buf144 = reinterpret_tensor(buf139, (1, 768, 8), (6144, 1, 768), 0); del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf141, buf144, 6144, 128, grid=grid(6144), stream=stream0)
        buf145 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf144, buf145, 768, 8, grid=grid(768), stream=stream0)
        buf146 = reinterpret_tensor(buf112, (2, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf112  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf142, buf146, 786432, grid=grid(786432), stream=stream0)
        buf147 = reinterpret_tensor(buf142, (24, 512, 64), (32768, 64, 1), 0); del buf142  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_123, reinterpret_tensor(buf146, (24, 512, 64), (32768, 64, 1), 0), out=buf147)
        del permute_123
        buf148 = reinterpret_tensor(buf111, (24, 512, 512), (262144, 512, 1), 0); del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf146, (24, 512, 64), (32768, 64, 1), 0), permute_124, out=buf148)
        del permute_124
        buf150 = reinterpret_tensor(buf109, (2, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf109  # reuse
        # Source Nodes: [full], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.scalar_tensor, aten.where]
        triton_per_fused__softmax_backward_data_div_full_scalar_tensor_where_9.run(buf148, alias_31, slice_36, buf150, 12288, 512, grid=grid(12288), stream=stream0)
        del alias_31
        del slice_36
        buf151 = reinterpret_tensor(buf146, (24, 64, 512), (32768, 512, 1), 0); del buf146  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_125, reinterpret_tensor(buf150, (24, 512, 512), (262144, 512, 1), 0), out=buf151)
        del permute_125
        buf152 = buf108; del buf108  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf150, (24, 512, 512), (262144, 512, 1), 0), permute_126, out=buf152)
        del permute_126
        buf153 = empty((2, 512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_10.run(buf152, tangents_18, buf151, tangents_19, buf147, buf153, 1024, 2304, grid=grid(1024, 2304), stream=stream0)
        del tangents_18
        del tangents_19
        buf154 = reinterpret_tensor(buf152, (1024, 768), (768, 1), 0); del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf153, (1024, 2304), (2304, 1), 0), permute_131, out=buf154)
        del permute_131
        buf155 = empty((768, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_132, reinterpret_tensor(buf153, (1024, 2304), (2304, 1), 0), out=buf155)
        del permute_132
        buf156 = buf117; del buf117  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf153, buf156, 18432, 128, grid=grid(18432), stream=stream0)
        buf157 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf156, buf157, 2304, 8, grid=grid(2304), stream=stream0)
        buf164 = buf141; del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf164, buf154, primals_131, mul_64, div_36, 1024, 768, grid=grid(1024), stream=stream0)
        del div_36
        del primals_131
        buf160 = reinterpret_tensor(buf144, (768, 8), (1, 768), 0); del buf144  # reuse
        buf162 = buf137; del buf137  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_1.run(buf154, mul_64, buf160, buf162, 6144, 128, grid=grid(6144), stream=stream0)
        del mul_64
        buf161 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf160, buf161, 768, 8, grid=grid(768), stream=stream0)
        buf163 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf162, buf163, 768, 8, grid=grid(768), stream=stream0)
        buf165 = reinterpret_tensor(buf130, (1024, 3072), (3072, 1), 0); del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf164, (1024, 768), (768, 1), 0), permute_133, out=buf165)
        del permute_133
        buf166 = reinterpret_tensor(buf153, (3072, 768), (768, 1), 0); del buf153  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_134, reinterpret_tensor(buf164, (1024, 768), (768, 1), 0), out=buf166)
        del permute_134
        buf167 = reinterpret_tensor(buf162, (1, 768, 8), (6144, 1, 768), 0); del buf162  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf164, buf167, 6144, 128, grid=grid(6144), stream=stream0)
        buf168 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf167, buf168, 768, 8, grid=grid(768), stream=stream0)
        buf169 = reinterpret_tensor(buf165, (2, 512, 3072), (1572864, 3072, 1), 0); del buf165  # reuse
        # Source Nodes: [add_31, mul_28], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_4.run(buf169, addmm_30, tanh_7, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_30
        del tanh_7
        buf170 = buf154; del buf154  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf169, (1024, 3072), (3072, 1), 0), permute_135, out=buf170)
        del permute_135
        buf171 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_136, reinterpret_tensor(buf169, (1024, 3072), (3072, 1), 0), out=buf171)
        del permute_136
        buf172 = buf133; del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf169, buf172, 24576, 128, grid=grid(24576), stream=stream0)
        buf173 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf172, buf173, 3072, 8, grid=grid(3072), stream=stream0)
        buf180 = buf164; del buf164  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf180, buf170, primals_129, mul_58, div_37, 1024, 768, grid=grid(1024), stream=stream0)
        del div_37
        del primals_129
        buf176 = reinterpret_tensor(buf167, (768, 8), (1, 768), 0); del buf167  # reuse
        buf178 = buf160; del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_1.run(buf170, mul_58, buf176, buf178, 6144, 128, grid=grid(6144), stream=stream0)
        del mul_58
        buf177 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf176, buf177, 768, 8, grid=grid(768), stream=stream0)
        buf179 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf178, buf179, 768, 8, grid=grid(768), stream=stream0)
        buf181 = buf170; del buf170  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf180, (1024, 768), (768, 1), 0), permute_137, out=buf181)
        del permute_137
        buf182 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_138, reinterpret_tensor(buf180, (1024, 768), (768, 1), 0), out=buf182)
        del permute_138
        buf183 = reinterpret_tensor(buf178, (1, 768, 8), (6144, 1, 768), 0); del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf180, buf183, 6144, 128, grid=grid(6144), stream=stream0)
        buf184 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf183, buf184, 768, 8, grid=grid(768), stream=stream0)
        buf185 = reinterpret_tensor(buf151, (2, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf151  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf181, buf185, 786432, grid=grid(786432), stream=stream0)
        buf186 = reinterpret_tensor(buf181, (24, 512, 64), (32768, 64, 1), 0); del buf181  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_140, reinterpret_tensor(buf185, (24, 512, 64), (32768, 64, 1), 0), out=buf186)
        del permute_140
        buf187 = reinterpret_tensor(buf150, (24, 512, 512), (262144, 512, 1), 0); del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf185, (24, 512, 64), (32768, 64, 1), 0), permute_141, out=buf187)
        del permute_141
        buf189 = reinterpret_tensor(buf148, (2, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf148  # reuse
        # Source Nodes: [full], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.scalar_tensor, aten.where]
        triton_per_fused__softmax_backward_data_div_full_scalar_tensor_where_9.run(buf187, alias_33, slice_32, buf189, 12288, 512, grid=grid(12288), stream=stream0)
        del alias_33
        del slice_32
        buf190 = reinterpret_tensor(buf185, (24, 64, 512), (32768, 512, 1), 0); del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_142, reinterpret_tensor(buf189, (24, 512, 512), (262144, 512, 1), 0), out=buf190)
        del permute_142
        buf191 = buf147; del buf147  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf189, (24, 512, 512), (262144, 512, 1), 0), permute_143, out=buf191)
        del permute_143
        buf192 = empty((2, 512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_10.run(buf191, tangents_16, buf190, tangents_17, buf186, buf192, 1024, 2304, grid=grid(1024, 2304), stream=stream0)
        del tangents_16
        del tangents_17
        buf193 = reinterpret_tensor(buf191, (1024, 768), (768, 1), 0); del buf191  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf192, (1024, 2304), (2304, 1), 0), permute_148, out=buf193)
        del permute_148
        buf194 = empty((768, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_149, reinterpret_tensor(buf192, (1024, 2304), (2304, 1), 0), out=buf194)
        del permute_149
        buf195 = buf156; del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf192, buf195, 18432, 128, grid=grid(18432), stream=stream0)
        buf196 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf195, buf196, 2304, 8, grid=grid(2304), stream=stream0)
        buf203 = buf180; del buf180  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf203, buf193, primals_127, mul_56, div_39, 1024, 768, grid=grid(1024), stream=stream0)
        del div_39
        del primals_127
        buf199 = reinterpret_tensor(buf183, (768, 8), (1, 768), 0); del buf183  # reuse
        buf201 = buf176; del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_1.run(buf193, mul_56, buf199, buf201, 6144, 128, grid=grid(6144), stream=stream0)
        del mul_56
        buf200 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf199, buf200, 768, 8, grid=grid(768), stream=stream0)
        buf202 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf201, buf202, 768, 8, grid=grid(768), stream=stream0)
        buf204 = reinterpret_tensor(buf169, (1024, 3072), (3072, 1), 0); del buf169  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf203, (1024, 768), (768, 1), 0), permute_150, out=buf204)
        del permute_150
        buf205 = reinterpret_tensor(buf192, (3072, 768), (768, 1), 0); del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_151, reinterpret_tensor(buf203, (1024, 768), (768, 1), 0), out=buf205)
        del permute_151
        buf206 = reinterpret_tensor(buf201, (1, 768, 8), (6144, 1, 768), 0); del buf201  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf203, buf206, 6144, 128, grid=grid(6144), stream=stream0)
        buf207 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf206, buf207, 768, 8, grid=grid(768), stream=stream0)
        buf208 = reinterpret_tensor(buf204, (2, 512, 3072), (1572864, 3072, 1), 0); del buf204  # reuse
        # Source Nodes: [add_27, mul_24], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_4.run(buf208, addmm_26, tanh_6, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_26
        del tanh_6
        buf209 = buf193; del buf193  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (1024, 3072), (3072, 1), 0), permute_152, out=buf209)
        del permute_152
        buf210 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_153, reinterpret_tensor(buf208, (1024, 3072), (3072, 1), 0), out=buf210)
        del permute_153
        buf211 = buf172; del buf172  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf208, buf211, 24576, 128, grid=grid(24576), stream=stream0)
        buf212 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf211, buf212, 3072, 8, grid=grid(3072), stream=stream0)
        buf219 = buf203; del buf203  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf219, buf209, primals_125, mul_50, div_40, 1024, 768, grid=grid(1024), stream=stream0)
        del div_40
        del primals_125
        buf215 = reinterpret_tensor(buf206, (768, 8), (1, 768), 0); del buf206  # reuse
        buf217 = buf199; del buf199  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_1.run(buf209, mul_50, buf215, buf217, 6144, 128, grid=grid(6144), stream=stream0)
        del mul_50
        buf216 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf215, buf216, 768, 8, grid=grid(768), stream=stream0)
        buf218 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf217, buf218, 768, 8, grid=grid(768), stream=stream0)
        buf220 = buf209; del buf209  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf219, (1024, 768), (768, 1), 0), permute_154, out=buf220)
        del permute_154
        buf221 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_155, reinterpret_tensor(buf219, (1024, 768), (768, 1), 0), out=buf221)
        del permute_155
        buf222 = reinterpret_tensor(buf217, (1, 768, 8), (6144, 1, 768), 0); del buf217  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf219, buf222, 6144, 128, grid=grid(6144), stream=stream0)
        buf223 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf222, buf223, 768, 8, grid=grid(768), stream=stream0)
        buf224 = reinterpret_tensor(buf190, (2, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf190  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf220, buf224, 786432, grid=grid(786432), stream=stream0)
        buf225 = reinterpret_tensor(buf220, (24, 512, 64), (32768, 64, 1), 0); del buf220  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_157, reinterpret_tensor(buf224, (24, 512, 64), (32768, 64, 1), 0), out=buf225)
        del permute_157
        buf226 = reinterpret_tensor(buf189, (24, 512, 512), (262144, 512, 1), 0); del buf189  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf224, (24, 512, 64), (32768, 64, 1), 0), permute_158, out=buf226)
        del permute_158
        buf228 = reinterpret_tensor(buf187, (2, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf187  # reuse
        # Source Nodes: [full], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.scalar_tensor, aten.where]
        triton_per_fused__softmax_backward_data_div_full_scalar_tensor_where_9.run(buf226, alias_35, slice_28, buf228, 12288, 512, grid=grid(12288), stream=stream0)
        del alias_35
        del slice_28
        buf229 = reinterpret_tensor(buf224, (24, 64, 512), (32768, 512, 1), 0); del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_159, reinterpret_tensor(buf228, (24, 512, 512), (262144, 512, 1), 0), out=buf229)
        del permute_159
        buf230 = buf186; del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf228, (24, 512, 512), (262144, 512, 1), 0), permute_160, out=buf230)
        del permute_160
        buf231 = empty((2, 512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_10.run(buf230, tangents_14, buf229, tangents_15, buf225, buf231, 1024, 2304, grid=grid(1024, 2304), stream=stream0)
        del tangents_14
        del tangents_15
        buf232 = reinterpret_tensor(buf230, (1024, 768), (768, 1), 0); del buf230  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf231, (1024, 2304), (2304, 1), 0), permute_165, out=buf232)
        del permute_165
        buf233 = empty((768, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_166, reinterpret_tensor(buf231, (1024, 2304), (2304, 1), 0), out=buf233)
        del permute_166
        buf234 = buf195; del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf231, buf234, 18432, 128, grid=grid(18432), stream=stream0)
        buf235 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf234, buf235, 2304, 8, grid=grid(2304), stream=stream0)
        buf242 = buf219; del buf219  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf242, buf232, primals_123, mul_48, div_42, 1024, 768, grid=grid(1024), stream=stream0)
        del div_42
        del primals_123
        buf238 = reinterpret_tensor(buf222, (768, 8), (1, 768), 0); del buf222  # reuse
        buf240 = buf215; del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_1.run(buf232, mul_48, buf238, buf240, 6144, 128, grid=grid(6144), stream=stream0)
        del mul_48
        buf239 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf238, buf239, 768, 8, grid=grid(768), stream=stream0)
        buf241 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf240, buf241, 768, 8, grid=grid(768), stream=stream0)
        buf243 = reinterpret_tensor(buf208, (1024, 3072), (3072, 1), 0); del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf242, (1024, 768), (768, 1), 0), permute_167, out=buf243)
        del permute_167
        buf244 = reinterpret_tensor(buf231, (3072, 768), (768, 1), 0); del buf231  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_168, reinterpret_tensor(buf242, (1024, 768), (768, 1), 0), out=buf244)
        del permute_168
        buf245 = reinterpret_tensor(buf240, (1, 768, 8), (6144, 1, 768), 0); del buf240  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf242, buf245, 6144, 128, grid=grid(6144), stream=stream0)
        buf246 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf245, buf246, 768, 8, grid=grid(768), stream=stream0)
        buf247 = reinterpret_tensor(buf243, (2, 512, 3072), (1572864, 3072, 1), 0); del buf243  # reuse
        # Source Nodes: [add_23, mul_20], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_4.run(buf247, addmm_22, tanh_5, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_22
        del tanh_5
        buf248 = buf232; del buf232  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf247, (1024, 3072), (3072, 1), 0), permute_169, out=buf248)
        del permute_169
        buf249 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_170, reinterpret_tensor(buf247, (1024, 3072), (3072, 1), 0), out=buf249)
        del permute_170
        buf250 = buf211; del buf211  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf247, buf250, 24576, 128, grid=grid(24576), stream=stream0)
        buf251 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf250, buf251, 3072, 8, grid=grid(3072), stream=stream0)
        buf258 = buf242; del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf258, buf248, primals_121, mul_42, div_43, 1024, 768, grid=grid(1024), stream=stream0)
        del div_43
        del primals_121
        buf254 = reinterpret_tensor(buf245, (768, 8), (1, 768), 0); del buf245  # reuse
        buf256 = buf238; del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_1.run(buf248, mul_42, buf254, buf256, 6144, 128, grid=grid(6144), stream=stream0)
        del mul_42
        buf255 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf254, buf255, 768, 8, grid=grid(768), stream=stream0)
        buf257 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf256, buf257, 768, 8, grid=grid(768), stream=stream0)
        buf259 = buf248; del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf258, (1024, 768), (768, 1), 0), permute_171, out=buf259)
        del permute_171
        buf260 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_172, reinterpret_tensor(buf258, (1024, 768), (768, 1), 0), out=buf260)
        del permute_172
        buf261 = reinterpret_tensor(buf256, (1, 768, 8), (6144, 1, 768), 0); del buf256  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf258, buf261, 6144, 128, grid=grid(6144), stream=stream0)
        buf262 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf261, buf262, 768, 8, grid=grid(768), stream=stream0)
        buf263 = reinterpret_tensor(buf229, (2, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf229  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf259, buf263, 786432, grid=grid(786432), stream=stream0)
        buf264 = reinterpret_tensor(buf259, (24, 512, 64), (32768, 64, 1), 0); del buf259  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_174, reinterpret_tensor(buf263, (24, 512, 64), (32768, 64, 1), 0), out=buf264)
        del permute_174
        buf265 = reinterpret_tensor(buf228, (24, 512, 512), (262144, 512, 1), 0); del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf263, (24, 512, 64), (32768, 64, 1), 0), permute_175, out=buf265)
        del permute_175
        buf267 = reinterpret_tensor(buf226, (2, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf226  # reuse
        # Source Nodes: [full], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.scalar_tensor, aten.where]
        triton_per_fused__softmax_backward_data_div_full_scalar_tensor_where_9.run(buf265, alias_37, slice_24, buf267, 12288, 512, grid=grid(12288), stream=stream0)
        del alias_37
        del slice_24
        buf268 = reinterpret_tensor(buf263, (24, 64, 512), (32768, 512, 1), 0); del buf263  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_176, reinterpret_tensor(buf267, (24, 512, 512), (262144, 512, 1), 0), out=buf268)
        del permute_176
        buf269 = buf225; del buf225  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf267, (24, 512, 512), (262144, 512, 1), 0), permute_177, out=buf269)
        del permute_177
        buf270 = empty((2, 512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_10.run(buf269, tangents_12, buf268, tangents_13, buf264, buf270, 1024, 2304, grid=grid(1024, 2304), stream=stream0)
        del tangents_12
        del tangents_13
        buf271 = reinterpret_tensor(buf269, (1024, 768), (768, 1), 0); del buf269  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf270, (1024, 2304), (2304, 1), 0), permute_182, out=buf271)
        del permute_182
        buf272 = empty((768, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_183, reinterpret_tensor(buf270, (1024, 2304), (2304, 1), 0), out=buf272)
        del permute_183
        buf273 = buf234; del buf234  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf270, buf273, 18432, 128, grid=grid(18432), stream=stream0)
        buf274 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf273, buf274, 2304, 8, grid=grid(2304), stream=stream0)
        buf281 = buf258; del buf258  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf281, buf271, primals_119, mul_40, div_45, 1024, 768, grid=grid(1024), stream=stream0)
        del div_45
        del primals_119
        buf277 = reinterpret_tensor(buf261, (768, 8), (1, 768), 0); del buf261  # reuse
        buf279 = buf254; del buf254  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_1.run(buf271, mul_40, buf277, buf279, 6144, 128, grid=grid(6144), stream=stream0)
        del mul_40
        buf278 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf277, buf278, 768, 8, grid=grid(768), stream=stream0)
        buf280 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf279, buf280, 768, 8, grid=grid(768), stream=stream0)
        buf282 = reinterpret_tensor(buf247, (1024, 3072), (3072, 1), 0); del buf247  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf281, (1024, 768), (768, 1), 0), permute_184, out=buf282)
        del permute_184
        buf283 = reinterpret_tensor(buf270, (3072, 768), (768, 1), 0); del buf270  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_185, reinterpret_tensor(buf281, (1024, 768), (768, 1), 0), out=buf283)
        del permute_185
        buf284 = reinterpret_tensor(buf279, (1, 768, 8), (6144, 1, 768), 0); del buf279  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf281, buf284, 6144, 128, grid=grid(6144), stream=stream0)
        buf285 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf284, buf285, 768, 8, grid=grid(768), stream=stream0)
        buf286 = reinterpret_tensor(buf282, (2, 512, 3072), (1572864, 3072, 1), 0); del buf282  # reuse
        # Source Nodes: [add_19, mul_16], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_4.run(buf286, addmm_18, tanh_4, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_18
        del tanh_4
        buf287 = buf271; del buf271  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf286, (1024, 3072), (3072, 1), 0), permute_186, out=buf287)
        del permute_186
        buf288 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_187, reinterpret_tensor(buf286, (1024, 3072), (3072, 1), 0), out=buf288)
        del permute_187
        buf289 = buf250; del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf286, buf289, 24576, 128, grid=grid(24576), stream=stream0)
        buf290 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf289, buf290, 3072, 8, grid=grid(3072), stream=stream0)
        buf297 = buf281; del buf281  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf297, buf287, primals_117, mul_34, div_46, 1024, 768, grid=grid(1024), stream=stream0)
        del div_46
        del primals_117
        buf293 = reinterpret_tensor(buf284, (768, 8), (1, 768), 0); del buf284  # reuse
        buf295 = buf277; del buf277  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_1.run(buf287, mul_34, buf293, buf295, 6144, 128, grid=grid(6144), stream=stream0)
        del mul_34
        buf294 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf293, buf294, 768, 8, grid=grid(768), stream=stream0)
        buf296 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf295, buf296, 768, 8, grid=grid(768), stream=stream0)
        buf298 = buf287; del buf287  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf297, (1024, 768), (768, 1), 0), permute_188, out=buf298)
        del permute_188
        buf299 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_189, reinterpret_tensor(buf297, (1024, 768), (768, 1), 0), out=buf299)
        del permute_189
        buf300 = reinterpret_tensor(buf295, (1, 768, 8), (6144, 1, 768), 0); del buf295  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf297, buf300, 6144, 128, grid=grid(6144), stream=stream0)
        buf301 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf300, buf301, 768, 8, grid=grid(768), stream=stream0)
        buf302 = reinterpret_tensor(buf268, (2, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf268  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf298, buf302, 786432, grid=grid(786432), stream=stream0)
        buf303 = reinterpret_tensor(buf298, (24, 512, 64), (32768, 64, 1), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_191, reinterpret_tensor(buf302, (24, 512, 64), (32768, 64, 1), 0), out=buf303)
        del permute_191
        buf304 = reinterpret_tensor(buf267, (24, 512, 512), (262144, 512, 1), 0); del buf267  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf302, (24, 512, 64), (32768, 64, 1), 0), permute_192, out=buf304)
        del permute_192
        buf306 = reinterpret_tensor(buf265, (2, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf265  # reuse
        # Source Nodes: [full], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.scalar_tensor, aten.where]
        triton_per_fused__softmax_backward_data_div_full_scalar_tensor_where_9.run(buf304, alias_39, slice_20, buf306, 12288, 512, grid=grid(12288), stream=stream0)
        del alias_39
        del slice_20
        buf307 = reinterpret_tensor(buf302, (24, 64, 512), (32768, 512, 1), 0); del buf302  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_193, reinterpret_tensor(buf306, (24, 512, 512), (262144, 512, 1), 0), out=buf307)
        del permute_193
        buf308 = buf264; del buf264  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf306, (24, 512, 512), (262144, 512, 1), 0), permute_194, out=buf308)
        del permute_194
        buf309 = empty((2, 512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_10.run(buf308, tangents_10, buf307, tangents_11, buf303, buf309, 1024, 2304, grid=grid(1024, 2304), stream=stream0)
        del tangents_10
        del tangents_11
        buf310 = reinterpret_tensor(buf308, (1024, 768), (768, 1), 0); del buf308  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf309, (1024, 2304), (2304, 1), 0), permute_199, out=buf310)
        del permute_199
        buf311 = empty((768, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_200, reinterpret_tensor(buf309, (1024, 2304), (2304, 1), 0), out=buf311)
        del permute_200
        buf312 = buf273; del buf273  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf309, buf312, 18432, 128, grid=grid(18432), stream=stream0)
        buf313 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf312, buf313, 2304, 8, grid=grid(2304), stream=stream0)
        buf320 = buf297; del buf297  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf320, buf310, primals_115, mul_32, div_48, 1024, 768, grid=grid(1024), stream=stream0)
        del div_48
        del primals_115
        buf316 = reinterpret_tensor(buf300, (768, 8), (1, 768), 0); del buf300  # reuse
        buf318 = buf293; del buf293  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_1.run(buf310, mul_32, buf316, buf318, 6144, 128, grid=grid(6144), stream=stream0)
        del mul_32
        buf317 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf316, buf317, 768, 8, grid=grid(768), stream=stream0)
        buf319 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf318, buf319, 768, 8, grid=grid(768), stream=stream0)
        buf321 = reinterpret_tensor(buf286, (1024, 3072), (3072, 1), 0); del buf286  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf320, (1024, 768), (768, 1), 0), permute_201, out=buf321)
        del permute_201
        buf322 = reinterpret_tensor(buf309, (3072, 768), (768, 1), 0); del buf309  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_202, reinterpret_tensor(buf320, (1024, 768), (768, 1), 0), out=buf322)
        del permute_202
        buf323 = reinterpret_tensor(buf318, (1, 768, 8), (6144, 1, 768), 0); del buf318  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf320, buf323, 6144, 128, grid=grid(6144), stream=stream0)
        buf324 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf323, buf324, 768, 8, grid=grid(768), stream=stream0)
        buf325 = reinterpret_tensor(buf321, (2, 512, 3072), (1572864, 3072, 1), 0); del buf321  # reuse
        # Source Nodes: [add_15, mul_12], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_4.run(buf325, addmm_14, tanh_3, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_14
        del tanh_3
        buf326 = buf310; del buf310  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf325, (1024, 3072), (3072, 1), 0), permute_203, out=buf326)
        del permute_203
        buf327 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_204, reinterpret_tensor(buf325, (1024, 3072), (3072, 1), 0), out=buf327)
        del permute_204
        buf328 = buf289; del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf325, buf328, 24576, 128, grid=grid(24576), stream=stream0)
        buf329 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf328, buf329, 3072, 8, grid=grid(3072), stream=stream0)
        buf336 = buf320; del buf320  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf336, buf326, primals_113, mul_26, div_49, 1024, 768, grid=grid(1024), stream=stream0)
        del div_49
        del primals_113
        buf332 = reinterpret_tensor(buf323, (768, 8), (1, 768), 0); del buf323  # reuse
        buf334 = buf316; del buf316  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_1.run(buf326, mul_26, buf332, buf334, 6144, 128, grid=grid(6144), stream=stream0)
        del mul_26
        buf333 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf332, buf333, 768, 8, grid=grid(768), stream=stream0)
        buf335 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf334, buf335, 768, 8, grid=grid(768), stream=stream0)
        buf337 = buf326; del buf326  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf336, (1024, 768), (768, 1), 0), permute_205, out=buf337)
        del permute_205
        buf338 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_206, reinterpret_tensor(buf336, (1024, 768), (768, 1), 0), out=buf338)
        del permute_206
        buf339 = reinterpret_tensor(buf334, (1, 768, 8), (6144, 1, 768), 0); del buf334  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf336, buf339, 6144, 128, grid=grid(6144), stream=stream0)
        buf340 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf339, buf340, 768, 8, grid=grid(768), stream=stream0)
        buf341 = reinterpret_tensor(buf307, (2, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf337, buf341, 786432, grid=grid(786432), stream=stream0)
        buf342 = reinterpret_tensor(buf337, (24, 512, 64), (32768, 64, 1), 0); del buf337  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_208, reinterpret_tensor(buf341, (24, 512, 64), (32768, 64, 1), 0), out=buf342)
        del permute_208
        buf343 = reinterpret_tensor(buf306, (24, 512, 512), (262144, 512, 1), 0); del buf306  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf341, (24, 512, 64), (32768, 64, 1), 0), permute_209, out=buf343)
        del permute_209
        buf345 = reinterpret_tensor(buf304, (2, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf304  # reuse
        # Source Nodes: [full], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.scalar_tensor, aten.where]
        triton_per_fused__softmax_backward_data_div_full_scalar_tensor_where_9.run(buf343, alias_41, slice_16, buf345, 12288, 512, grid=grid(12288), stream=stream0)
        del alias_41
        del slice_16
        buf346 = reinterpret_tensor(buf341, (24, 64, 512), (32768, 512, 1), 0); del buf341  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_210, reinterpret_tensor(buf345, (24, 512, 512), (262144, 512, 1), 0), out=buf346)
        del permute_210
        buf347 = buf303; del buf303  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf345, (24, 512, 512), (262144, 512, 1), 0), permute_211, out=buf347)
        del permute_211
        buf348 = empty((2, 512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_10.run(buf347, tangents_8, buf346, tangents_9, buf342, buf348, 1024, 2304, grid=grid(1024, 2304), stream=stream0)
        del tangents_8
        del tangents_9
        buf349 = reinterpret_tensor(buf347, (1024, 768), (768, 1), 0); del buf347  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf348, (1024, 2304), (2304, 1), 0), permute_216, out=buf349)
        del permute_216
        buf350 = empty((768, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_217, reinterpret_tensor(buf348, (1024, 2304), (2304, 1), 0), out=buf350)
        del permute_217
        buf351 = buf312; del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf348, buf351, 18432, 128, grid=grid(18432), stream=stream0)
        buf352 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf351, buf352, 2304, 8, grid=grid(2304), stream=stream0)
        buf359 = buf336; del buf336  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf359, buf349, primals_111, mul_24, div_51, 1024, 768, grid=grid(1024), stream=stream0)
        del div_51
        del primals_111
        buf355 = reinterpret_tensor(buf339, (768, 8), (1, 768), 0); del buf339  # reuse
        buf357 = buf332; del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_1.run(buf349, mul_24, buf355, buf357, 6144, 128, grid=grid(6144), stream=stream0)
        del mul_24
        buf356 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf355, buf356, 768, 8, grid=grid(768), stream=stream0)
        buf358 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf357, buf358, 768, 8, grid=grid(768), stream=stream0)
        buf360 = reinterpret_tensor(buf325, (1024, 3072), (3072, 1), 0); del buf325  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf359, (1024, 768), (768, 1), 0), permute_218, out=buf360)
        del permute_218
        buf361 = reinterpret_tensor(buf348, (3072, 768), (768, 1), 0); del buf348  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_219, reinterpret_tensor(buf359, (1024, 768), (768, 1), 0), out=buf361)
        del permute_219
        buf362 = reinterpret_tensor(buf357, (1, 768, 8), (6144, 1, 768), 0); del buf357  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf359, buf362, 6144, 128, grid=grid(6144), stream=stream0)
        buf363 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf362, buf363, 768, 8, grid=grid(768), stream=stream0)
        buf364 = reinterpret_tensor(buf360, (2, 512, 3072), (1572864, 3072, 1), 0); del buf360  # reuse
        # Source Nodes: [add_11, mul_8], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_4.run(buf364, addmm_10, tanh_2, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_10
        del tanh_2
        buf365 = buf349; del buf349  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf364, (1024, 3072), (3072, 1), 0), permute_220, out=buf365)
        del permute_220
        buf366 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_221, reinterpret_tensor(buf364, (1024, 3072), (3072, 1), 0), out=buf366)
        del permute_221
        buf367 = buf328; del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf364, buf367, 24576, 128, grid=grid(24576), stream=stream0)
        buf368 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf367, buf368, 3072, 8, grid=grid(3072), stream=stream0)
        buf375 = buf359; del buf359  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf375, buf365, primals_109, mul_18, div_52, 1024, 768, grid=grid(1024), stream=stream0)
        del div_52
        del primals_109
        buf371 = reinterpret_tensor(buf362, (768, 8), (1, 768), 0); del buf362  # reuse
        buf373 = buf355; del buf355  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_1.run(buf365, mul_18, buf371, buf373, 6144, 128, grid=grid(6144), stream=stream0)
        del mul_18
        buf372 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf371, buf372, 768, 8, grid=grid(768), stream=stream0)
        buf374 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf373, buf374, 768, 8, grid=grid(768), stream=stream0)
        buf376 = buf365; del buf365  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf375, (1024, 768), (768, 1), 0), permute_222, out=buf376)
        del permute_222
        buf377 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_223, reinterpret_tensor(buf375, (1024, 768), (768, 1), 0), out=buf377)
        del permute_223
        buf378 = reinterpret_tensor(buf373, (1, 768, 8), (6144, 1, 768), 0); del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf375, buf378, 6144, 128, grid=grid(6144), stream=stream0)
        buf379 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf378, buf379, 768, 8, grid=grid(768), stream=stream0)
        buf380 = reinterpret_tensor(buf346, (2, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf346  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf376, buf380, 786432, grid=grid(786432), stream=stream0)
        buf381 = reinterpret_tensor(buf376, (24, 512, 64), (32768, 64, 1), 0); del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_225, reinterpret_tensor(buf380, (24, 512, 64), (32768, 64, 1), 0), out=buf381)
        del permute_225
        buf382 = reinterpret_tensor(buf345, (24, 512, 512), (262144, 512, 1), 0); del buf345  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf380, (24, 512, 64), (32768, 64, 1), 0), permute_226, out=buf382)
        del permute_226
        buf384 = reinterpret_tensor(buf343, (2, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf343  # reuse
        # Source Nodes: [full], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.scalar_tensor, aten.where]
        triton_per_fused__softmax_backward_data_div_full_scalar_tensor_where_9.run(buf382, alias_43, slice_12, buf384, 12288, 512, grid=grid(12288), stream=stream0)
        del alias_43
        del slice_12
        buf385 = reinterpret_tensor(buf380, (24, 64, 512), (32768, 512, 1), 0); del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_227, reinterpret_tensor(buf384, (24, 512, 512), (262144, 512, 1), 0), out=buf385)
        del permute_227
        buf386 = buf342; del buf342  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf384, (24, 512, 512), (262144, 512, 1), 0), permute_228, out=buf386)
        del permute_228
        buf387 = empty((2, 512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_10.run(buf386, tangents_6, buf385, tangents_7, buf381, buf387, 1024, 2304, grid=grid(1024, 2304), stream=stream0)
        del tangents_6
        del tangents_7
        buf388 = reinterpret_tensor(buf386, (1024, 768), (768, 1), 0); del buf386  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf387, (1024, 2304), (2304, 1), 0), permute_233, out=buf388)
        del permute_233
        buf389 = empty((768, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_234, reinterpret_tensor(buf387, (1024, 2304), (2304, 1), 0), out=buf389)
        del permute_234
        buf390 = buf351; del buf351  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf387, buf390, 18432, 128, grid=grid(18432), stream=stream0)
        buf391 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf390, buf391, 2304, 8, grid=grid(2304), stream=stream0)
        buf398 = buf375; del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf398, buf388, primals_107, mul_16, div_54, 1024, 768, grid=grid(1024), stream=stream0)
        del div_54
        del primals_107
        buf394 = reinterpret_tensor(buf378, (768, 8), (1, 768), 0); del buf378  # reuse
        buf396 = buf371; del buf371  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_1.run(buf388, mul_16, buf394, buf396, 6144, 128, grid=grid(6144), stream=stream0)
        del mul_16
        buf395 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf394, buf395, 768, 8, grid=grid(768), stream=stream0)
        buf397 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf396, buf397, 768, 8, grid=grid(768), stream=stream0)
        buf399 = reinterpret_tensor(buf364, (1024, 3072), (3072, 1), 0); del buf364  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf398, (1024, 768), (768, 1), 0), permute_235, out=buf399)
        del permute_235
        buf400 = reinterpret_tensor(buf387, (3072, 768), (768, 1), 0); del buf387  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_236, reinterpret_tensor(buf398, (1024, 768), (768, 1), 0), out=buf400)
        del permute_236
        buf401 = reinterpret_tensor(buf396, (1, 768, 8), (6144, 1, 768), 0); del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf398, buf401, 6144, 128, grid=grid(6144), stream=stream0)
        buf402 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf401, buf402, 768, 8, grid=grid(768), stream=stream0)
        buf403 = reinterpret_tensor(buf399, (2, 512, 3072), (1572864, 3072, 1), 0); del buf399  # reuse
        # Source Nodes: [add_7, mul_4], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_4.run(buf403, addmm_6, tanh_1, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_6
        del tanh_1
        buf404 = buf388; del buf388  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf403, (1024, 3072), (3072, 1), 0), permute_237, out=buf404)
        del permute_237
        buf405 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_238, reinterpret_tensor(buf403, (1024, 3072), (3072, 1), 0), out=buf405)
        del permute_238
        buf406 = buf367; del buf367  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf403, buf406, 24576, 128, grid=grid(24576), stream=stream0)
        buf407 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf406, buf407, 3072, 8, grid=grid(3072), stream=stream0)
        buf414 = buf398; del buf398  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf414, buf404, primals_105, mul_10, div_55, 1024, 768, grid=grid(1024), stream=stream0)
        del div_55
        del primals_105
        buf410 = reinterpret_tensor(buf401, (768, 8), (1, 768), 0); del buf401  # reuse
        buf412 = buf394; del buf394  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_1.run(buf404, mul_10, buf410, buf412, 6144, 128, grid=grid(6144), stream=stream0)
        del mul_10
        buf411 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf410, buf411, 768, 8, grid=grid(768), stream=stream0)
        buf413 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf412, buf413, 768, 8, grid=grid(768), stream=stream0)
        buf415 = buf404; del buf404  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf414, (1024, 768), (768, 1), 0), permute_239, out=buf415)
        del permute_239
        buf416 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_240, reinterpret_tensor(buf414, (1024, 768), (768, 1), 0), out=buf416)
        del permute_240
        buf417 = reinterpret_tensor(buf412, (1, 768, 8), (6144, 1, 768), 0); del buf412  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf414, buf417, 6144, 128, grid=grid(6144), stream=stream0)
        buf418 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf417, buf418, 768, 8, grid=grid(768), stream=stream0)
        buf419 = reinterpret_tensor(buf385, (2, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf385  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf415, buf419, 786432, grid=grid(786432), stream=stream0)
        buf420 = reinterpret_tensor(buf415, (24, 512, 64), (32768, 64, 1), 0); del buf415  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_242, reinterpret_tensor(buf419, (24, 512, 64), (32768, 64, 1), 0), out=buf420)
        del permute_242
        buf421 = reinterpret_tensor(buf384, (24, 512, 512), (262144, 512, 1), 0); del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf419, (24, 512, 64), (32768, 64, 1), 0), permute_243, out=buf421)
        del permute_243
        buf423 = reinterpret_tensor(buf382, (2, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf382  # reuse
        # Source Nodes: [full], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.scalar_tensor, aten.where]
        triton_per_fused__softmax_backward_data_div_full_scalar_tensor_where_9.run(buf421, alias_45, slice_8, buf423, 12288, 512, grid=grid(12288), stream=stream0)
        del alias_45
        del slice_8
        buf424 = reinterpret_tensor(buf419, (24, 64, 512), (32768, 512, 1), 0); del buf419  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_244, reinterpret_tensor(buf423, (24, 512, 512), (262144, 512, 1), 0), out=buf424)
        del permute_244
        buf425 = buf381; del buf381  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf423, (24, 512, 512), (262144, 512, 1), 0), permute_245, out=buf425)
        del permute_245
        buf426 = empty((2, 512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_10.run(buf425, tangents_4, buf424, tangents_5, buf420, buf426, 1024, 2304, grid=grid(1024, 2304), stream=stream0)
        del tangents_4
        del tangents_5
        buf427 = reinterpret_tensor(buf425, (1024, 768), (768, 1), 0); del buf425  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf426, (1024, 2304), (2304, 1), 0), permute_250, out=buf427)
        del permute_250
        buf428 = empty((768, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_251, reinterpret_tensor(buf426, (1024, 2304), (2304, 1), 0), out=buf428)
        del permute_251
        buf429 = buf390; del buf390  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf426, buf429, 18432, 128, grid=grid(18432), stream=stream0)
        buf430 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf429, buf430, 2304, 8, grid=grid(2304), stream=stream0)
        buf437 = buf414; del buf414  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf437, buf427, primals_103, mul_8, div_57, 1024, 768, grid=grid(1024), stream=stream0)
        del div_57
        del primals_103
        buf433 = reinterpret_tensor(buf417, (768, 8), (1, 768), 0); del buf417  # reuse
        buf435 = buf410; del buf410  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_1.run(buf427, mul_8, buf433, buf435, 6144, 128, grid=grid(6144), stream=stream0)
        del mul_8
        buf434 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf433, buf434, 768, 8, grid=grid(768), stream=stream0)
        buf436 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf435, buf436, 768, 8, grid=grid(768), stream=stream0)
        buf438 = reinterpret_tensor(buf403, (1024, 3072), (3072, 1), 0); del buf403  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf437, (1024, 768), (768, 1), 0), permute_252, out=buf438)
        del permute_252
        buf439 = reinterpret_tensor(buf426, (3072, 768), (768, 1), 0); del buf426  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_253, reinterpret_tensor(buf437, (1024, 768), (768, 1), 0), out=buf439)
        del permute_253
        buf440 = reinterpret_tensor(buf435, (1, 768, 8), (6144, 1, 768), 0); del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf437, buf440, 6144, 128, grid=grid(6144), stream=stream0)
        buf441 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf440, buf441, 768, 8, grid=grid(768), stream=stream0)
        buf442 = reinterpret_tensor(buf438, (2, 512, 3072), (1572864, 3072, 1), 0); del buf438  # reuse
        # Source Nodes: [add_3, mul], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_4.run(buf442, addmm_2, tanh, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_2
        del tanh
        buf443 = buf427; del buf427  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf442, (1024, 3072), (3072, 1), 0), permute_254, out=buf443)
        del permute_254
        buf444 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_255, reinterpret_tensor(buf442, (1024, 3072), (3072, 1), 0), out=buf444)
        del permute_255
        buf445 = buf406; del buf406  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_5.run(buf442, buf445, 24576, 128, grid=grid(24576), stream=stream0)
        del buf442
        buf446 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_6.run(buf445, buf446, 3072, 8, grid=grid(3072), stream=stream0)
        del buf445
        buf453 = buf437; del buf437  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_7.run(buf453, buf443, primals_101, mul_2, div_58, 1024, 768, grid=grid(1024), stream=stream0)
        del div_58
        del primals_101
        buf449 = reinterpret_tensor(buf440, (768, 8), (1, 768), 0); del buf440  # reuse
        buf451 = buf433; del buf433  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_1.run(buf443, mul_2, buf449, buf451, 6144, 128, grid=grid(6144), stream=stream0)
        del mul_2
        buf450 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf449, buf450, 768, 8, grid=grid(768), stream=stream0)
        buf452 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf451, buf452, 768, 8, grid=grid(768), stream=stream0)
        buf454 = buf443; del buf443  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf453, (1024, 768), (768, 1), 0), permute_256, out=buf454)
        del permute_256
        buf455 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_257, reinterpret_tensor(buf453, (1024, 768), (768, 1), 0), out=buf455)
        del permute_257
        buf456 = reinterpret_tensor(buf451, (1, 768, 8), (6144, 1, 768), 0); del buf451  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf453, buf456, 6144, 128, grid=grid(6144), stream=stream0)
        buf457 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_native_layer_norm_backward_2.run(buf456, buf457, 768, 8, grid=grid(768), stream=stream0)
        buf458 = reinterpret_tensor(buf424, (2, 12, 512, 64), (393216, 32768, 64, 1), 0); del buf424  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf454, buf458, 786432, grid=grid(786432), stream=stream0)
        buf459 = reinterpret_tensor(buf454, (24, 512, 64), (32768, 64, 1), 0); del buf454  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_259, reinterpret_tensor(buf458, (24, 512, 64), (32768, 64, 1), 0), out=buf459)
        del permute_259
        buf460 = reinterpret_tensor(buf423, (24, 512, 512), (262144, 512, 1), 0); del buf423  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf458, (24, 512, 64), (32768, 64, 1), 0), permute_260, out=buf460)
        del permute_260
        buf462 = reinterpret_tensor(buf421, (2, 12, 512, 512), (3145728, 262144, 512, 1), 0); del buf421  # reuse
        # Source Nodes: [full], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.scalar_tensor, aten.where]
        triton_per_fused__softmax_backward_data_div_full_scalar_tensor_where_9.run(buf460, alias_47, slice_4, buf462, 12288, 512, grid=grid(12288), stream=stream0)
        del alias_47
        del buf460
        del slice_4
        buf463 = reinterpret_tensor(buf458, (24, 64, 512), (32768, 512, 1), 0); del buf458  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_261, reinterpret_tensor(buf462, (24, 512, 512), (262144, 512, 1), 0), out=buf463)
        del permute_261
        buf464 = buf420; del buf420  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf462, (24, 512, 512), (262144, 512, 1), 0), permute_262, out=buf464)
        del buf462
        del permute_262
        buf465 = empty((2, 512, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_10.run(buf464, tangents_2, buf463, tangents_3, buf459, buf465, 1024, 2304, grid=grid(1024, 2304), stream=stream0)
        del buf459
        del tangents_2
        del tangents_3
        buf466 = reinterpret_tensor(buf464, (1024, 768), (768, 1), 0); del buf464  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf465, (1024, 2304), (2304, 1), 0), permute_267, out=buf466)
        del permute_267
        buf467 = empty((768, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_268, reinterpret_tensor(buf465, (1024, 2304), (2304, 1), 0), out=buf467)
        del permute_268
        buf468 = buf429; del buf429  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_11.run(buf465, buf468, 18432, 128, grid=grid(18432), stream=stream0)
        del buf465
        buf469 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_12.run(buf468, buf469, 2304, 8, grid=grid(2304), stream=stream0)
        del buf468
        buf476 = buf453; del buf453  # reuse
        buf482 = reinterpret_tensor(buf463, (2, 512, 768), (393216, 768, 1), 0); del buf463  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_layer_norm_backward, aten.scalar_tensor]
        triton_per_fused_add_embedding_dense_backward_native_layer_norm_backward_scalar_tensor_13.run(buf476, buf466, primals_99, mul, div_60, view, buf482, 1024, 768, grid=grid(1024), stream=stream0)
        del div_60
        del primals_99
        buf472 = reinterpret_tensor(buf456, (768, 8), (1, 768), 0); del buf456  # reuse
        buf474 = buf449; del buf449  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_1.run(buf466, mul, buf472, buf474, 6144, 128, grid=grid(6144), stream=stream0)
        del mul
        buf473 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf472, buf473, 768, 8, grid=grid(768), stream=stream0)
        del buf472
        buf475 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_2.run(buf474, buf475, 768, 8, grid=grid(768), stream=stream0)
        del buf474
        buf477 = buf466; del buf466  # reuse
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_14.run(buf477, 786432, grid=grid(786432), stream=stream0)
        buf478 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward, aten.scalar_tensor, aten.sum]
        triton_poi_fused_embedding_dense_backward_scalar_tensor_sum_15.run(buf476, buf478, 393216, grid=grid(393216), stream=stream0)
        del buf476
        aten.index_put_(buf477, [view_1], buf478, True)
        del buf478
        del view_1
        buf481 = empty((50257, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_16.run(buf481, 38597376, grid=grid(38597376), stream=stream0)
        aten.index_put_(buf481, [view], buf482, True)
        del buf482
        del view
        return (reinterpret_tensor(buf469, (2304, ), (1, ), 0), buf467, reinterpret_tensor(buf457, (768, ), (1, ), 0), buf455, reinterpret_tensor(buf446, (3072, ), (1, ), 0), buf444, reinterpret_tensor(buf441, (768, ), (1, ), 0), buf439, reinterpret_tensor(buf430, (2304, ), (1, ), 0), buf428, reinterpret_tensor(buf418, (768, ), (1, ), 0), buf416, reinterpret_tensor(buf407, (3072, ), (1, ), 0), buf405, reinterpret_tensor(buf402, (768, ), (1, ), 0), buf400, reinterpret_tensor(buf391, (2304, ), (1, ), 0), buf389, reinterpret_tensor(buf379, (768, ), (1, ), 0), buf377, reinterpret_tensor(buf368, (3072, ), (1, ), 0), buf366, reinterpret_tensor(buf363, (768, ), (1, ), 0), buf361, reinterpret_tensor(buf352, (2304, ), (1, ), 0), buf350, reinterpret_tensor(buf340, (768, ), (1, ), 0), buf338, reinterpret_tensor(buf329, (3072, ), (1, ), 0), buf327, reinterpret_tensor(buf324, (768, ), (1, ), 0), buf322, reinterpret_tensor(buf313, (2304, ), (1, ), 0), buf311, reinterpret_tensor(buf301, (768, ), (1, ), 0), buf299, reinterpret_tensor(buf290, (3072, ), (1, ), 0), buf288, reinterpret_tensor(buf285, (768, ), (1, ), 0), buf283, reinterpret_tensor(buf274, (2304, ), (1, ), 0), buf272, reinterpret_tensor(buf262, (768, ), (1, ), 0), buf260, reinterpret_tensor(buf251, (3072, ), (1, ), 0), buf249, reinterpret_tensor(buf246, (768, ), (1, ), 0), buf244, reinterpret_tensor(buf235, (2304, ), (1, ), 0), buf233, reinterpret_tensor(buf223, (768, ), (1, ), 0), buf221, reinterpret_tensor(buf212, (3072, ), (1, ), 0), buf210, reinterpret_tensor(buf207, (768, ), (1, ), 0), buf205, reinterpret_tensor(buf196, (2304, ), (1, ), 0), buf194, reinterpret_tensor(buf184, (768, ), (1, ), 0), buf182, reinterpret_tensor(buf173, (3072, ), (1, ), 0), buf171, reinterpret_tensor(buf168, (768, ), (1, ), 0), buf166, reinterpret_tensor(buf157, (2304, ), (1, ), 0), buf155, reinterpret_tensor(buf145, (768, ), (1, ), 0), buf143, reinterpret_tensor(buf134, (3072, ), (1, ), 0), buf132, reinterpret_tensor(buf129, (768, ), (1, ), 0), buf127, reinterpret_tensor(buf118, (2304, ), (1, ), 0), buf116, reinterpret_tensor(buf106, (768, ), (1, ), 0), buf104, reinterpret_tensor(buf95, (3072, ), (1, ), 0), buf93, reinterpret_tensor(buf90, (768, ), (1, ), 0), buf88, reinterpret_tensor(buf79, (2304, ), (1, ), 0), buf77, reinterpret_tensor(buf67, (768, ), (1, ), 0), buf65, reinterpret_tensor(buf56, (3072, ), (1, ), 0), buf54, reinterpret_tensor(buf51, (768, ), (1, ), 0), buf49, reinterpret_tensor(buf40, (2304, ), (1, ), 0), buf38, reinterpret_tensor(buf28, (768, ), (1, ), 0), buf26, reinterpret_tensor(buf17, (3072, ), (1, ), 0), buf15, reinterpret_tensor(buf12, (768, ), (1, ), 0), buf10, buf481, buf477, buf473, buf475, buf450, buf452, buf434, buf436, buf411, buf413, buf395, buf397, buf372, buf374, buf356, buf358, buf333, buf335, buf317, buf319, buf294, buf296, buf278, buf280, buf255, buf257, buf239, buf241, buf216, buf218, buf200, buf202, buf177, buf179, buf161, buf163, buf138, buf140, buf122, buf124, buf99, buf101, buf83, buf85, buf60, buf62, buf44, buf46, buf21, buf23, buf6, buf8, reinterpret_tensor(buf0, (50257, 768), (768, 1), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_99 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view = rand_strided((2, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    view_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    mul = rand_strided((2, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    slice_4 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_2 = rand_strided((2, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_2 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh = rand_strided((2, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    mul_8 = rand_strided((2, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    slice_8 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_10 = rand_strided((2, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_6 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_1 = rand_strided((2, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    mul_16 = rand_strided((2, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    slice_12 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_18 = rand_strided((2, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_2 = rand_strided((2, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    mul_24 = rand_strided((2, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    slice_16 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_26 = rand_strided((2, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_14 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_3 = rand_strided((2, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    mul_32 = rand_strided((2, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    slice_20 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_34 = rand_strided((2, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_18 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_4 = rand_strided((2, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    mul_40 = rand_strided((2, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    slice_24 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_42 = rand_strided((2, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_5 = rand_strided((2, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    mul_48 = rand_strided((2, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    slice_28 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_50 = rand_strided((2, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_26 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_6 = rand_strided((2, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    mul_56 = rand_strided((2, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    slice_32 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_58 = rand_strided((2, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_30 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_7 = rand_strided((2, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    mul_64 = rand_strided((2, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    slice_36 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_66 = rand_strided((2, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_8 = rand_strided((2, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    mul_72 = rand_strided((2, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    slice_40 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_74 = rand_strided((2, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_38 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_9 = rand_strided((2, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    mul_80 = rand_strided((2, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    slice_44 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_82 = rand_strided((2, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_42 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_10 = rand_strided((2, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    mul_88 = rand_strided((2, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    slice_48 = rand_strided((1, 1, 512, 512), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    mul_90 = rand_strided((2, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_46 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_11 = rand_strided((2, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    mul_96 = rand_strided((2, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_219 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_63 = rand_strided((50257, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_24 = rand_strided((2, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_65 = rand_strided((768, 3072), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_66 = rand_strided((3072, 1024), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_67 = rand_strided((3072, 768), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_68 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_25 = rand_strided((2, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_69 = rand_strided((768, 768), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_70 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_72 = rand_strided((24, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_73 = rand_strided((24, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_25 = rand_strided((2, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_74 = rand_strided((24, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_75 = rand_strided((24, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_80 = rand_strided((2304, 768), (1, 2304), device='cuda:0', dtype=torch.float32)
    permute_81 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_27 = rand_strided((2, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_82 = rand_strided((768, 3072), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_83 = rand_strided((3072, 1024), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_84 = rand_strided((3072, 768), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_85 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_28 = rand_strided((2, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_86 = rand_strided((768, 768), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_87 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_89 = rand_strided((24, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_90 = rand_strided((24, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_27 = rand_strided((2, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_91 = rand_strided((24, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_92 = rand_strided((24, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_97 = rand_strided((2304, 768), (1, 2304), device='cuda:0', dtype=torch.float32)
    permute_98 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_30 = rand_strided((2, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_99 = rand_strided((768, 3072), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_100 = rand_strided((3072, 1024), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_101 = rand_strided((3072, 768), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_102 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_31 = rand_strided((2, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_103 = rand_strided((768, 768), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_104 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_106 = rand_strided((24, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_107 = rand_strided((24, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_29 = rand_strided((2, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_108 = rand_strided((24, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_109 = rand_strided((24, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_114 = rand_strided((2304, 768), (1, 2304), device='cuda:0', dtype=torch.float32)
    permute_115 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_33 = rand_strided((2, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_116 = rand_strided((768, 3072), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_117 = rand_strided((3072, 1024), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_118 = rand_strided((3072, 768), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_119 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_34 = rand_strided((2, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_120 = rand_strided((768, 768), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_121 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_123 = rand_strided((24, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_124 = rand_strided((24, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_31 = rand_strided((2, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_125 = rand_strided((24, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_126 = rand_strided((24, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_131 = rand_strided((2304, 768), (1, 2304), device='cuda:0', dtype=torch.float32)
    permute_132 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_36 = rand_strided((2, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_133 = rand_strided((768, 3072), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_134 = rand_strided((3072, 1024), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_135 = rand_strided((3072, 768), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_136 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_37 = rand_strided((2, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_137 = rand_strided((768, 768), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_138 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_140 = rand_strided((24, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_141 = rand_strided((24, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_33 = rand_strided((2, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_142 = rand_strided((24, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_143 = rand_strided((24, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_148 = rand_strided((2304, 768), (1, 2304), device='cuda:0', dtype=torch.float32)
    permute_149 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_39 = rand_strided((2, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_150 = rand_strided((768, 3072), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_151 = rand_strided((3072, 1024), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_152 = rand_strided((3072, 768), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_153 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_40 = rand_strided((2, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_154 = rand_strided((768, 768), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_155 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_157 = rand_strided((24, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_158 = rand_strided((24, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_35 = rand_strided((2, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_159 = rand_strided((24, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_160 = rand_strided((24, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_165 = rand_strided((2304, 768), (1, 2304), device='cuda:0', dtype=torch.float32)
    permute_166 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_42 = rand_strided((2, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_167 = rand_strided((768, 3072), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_168 = rand_strided((3072, 1024), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_169 = rand_strided((3072, 768), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_170 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_43 = rand_strided((2, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_171 = rand_strided((768, 768), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_172 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_174 = rand_strided((24, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_175 = rand_strided((24, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_37 = rand_strided((2, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_176 = rand_strided((24, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_177 = rand_strided((24, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_182 = rand_strided((2304, 768), (1, 2304), device='cuda:0', dtype=torch.float32)
    permute_183 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_45 = rand_strided((2, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_184 = rand_strided((768, 3072), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_185 = rand_strided((3072, 1024), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_186 = rand_strided((3072, 768), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_187 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_46 = rand_strided((2, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_188 = rand_strided((768, 768), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_189 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_191 = rand_strided((24, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_192 = rand_strided((24, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_39 = rand_strided((2, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_193 = rand_strided((24, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_194 = rand_strided((24, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_199 = rand_strided((2304, 768), (1, 2304), device='cuda:0', dtype=torch.float32)
    permute_200 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_48 = rand_strided((2, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_201 = rand_strided((768, 3072), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_202 = rand_strided((3072, 1024), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_203 = rand_strided((3072, 768), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_204 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_49 = rand_strided((2, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_205 = rand_strided((768, 768), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_206 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_208 = rand_strided((24, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_209 = rand_strided((24, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_41 = rand_strided((2, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_210 = rand_strided((24, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_211 = rand_strided((24, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_216 = rand_strided((2304, 768), (1, 2304), device='cuda:0', dtype=torch.float32)
    permute_217 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_51 = rand_strided((2, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_218 = rand_strided((768, 3072), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_219 = rand_strided((3072, 1024), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_220 = rand_strided((3072, 768), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_221 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_52 = rand_strided((2, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_222 = rand_strided((768, 768), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_223 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_225 = rand_strided((24, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_226 = rand_strided((24, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_43 = rand_strided((2, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_227 = rand_strided((24, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_228 = rand_strided((24, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_233 = rand_strided((2304, 768), (1, 2304), device='cuda:0', dtype=torch.float32)
    permute_234 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_54 = rand_strided((2, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_235 = rand_strided((768, 3072), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_236 = rand_strided((3072, 1024), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_237 = rand_strided((3072, 768), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_238 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_55 = rand_strided((2, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_239 = rand_strided((768, 768), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_240 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_242 = rand_strided((24, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_243 = rand_strided((24, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_45 = rand_strided((2, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_244 = rand_strided((24, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_245 = rand_strided((24, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_250 = rand_strided((2304, 768), (1, 2304), device='cuda:0', dtype=torch.float32)
    permute_251 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_57 = rand_strided((2, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_252 = rand_strided((768, 3072), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_253 = rand_strided((3072, 1024), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_254 = rand_strided((3072, 768), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_255 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_58 = rand_strided((2, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_256 = rand_strided((768, 768), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_257 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_259 = rand_strided((24, 512, 512), (262144, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_260 = rand_strided((24, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_47 = rand_strided((2, 12, 512, 512), (3145728, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    permute_261 = rand_strided((24, 64, 512), (32768, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_262 = rand_strided((24, 512, 64), (32768, 1, 512), device='cuda:0', dtype=torch.float32)
    permute_267 = rand_strided((2304, 768), (1, 2304), device='cuda:0', dtype=torch.float32)
    permute_268 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_60 = rand_strided((2, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((2, 512, 50257), (25731584, 50257, 1), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_3 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_4 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_5 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_6 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_7 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_8 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_9 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_10 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_11 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_12 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_13 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_14 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_15 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_16 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_17 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_18 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_19 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_20 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_21 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_22 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_23 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_24 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_25 = rand_strided((2, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, view, view_1, mul, slice_4, mul_2, addmm_2, tanh, mul_8, slice_8, mul_10, addmm_6, tanh_1, mul_16, slice_12, mul_18, addmm_10, tanh_2, mul_24, slice_16, mul_26, addmm_14, tanh_3, mul_32, slice_20, mul_34, addmm_18, tanh_4, mul_40, slice_24, mul_42, addmm_22, tanh_5, mul_48, slice_28, mul_50, addmm_26, tanh_6, mul_56, slice_32, mul_58, addmm_30, tanh_7, mul_64, slice_36, mul_66, addmm_34, tanh_8, mul_72, slice_40, mul_74, addmm_38, tanh_9, mul_80, slice_44, mul_82, addmm_42, tanh_10, mul_88, slice_48, mul_90, addmm_46, tanh_11, mul_96, view_219, permute_63, div_24, permute_65, permute_66, permute_67, permute_68, div_25, permute_69, permute_70, permute_72, permute_73, alias_25, permute_74, permute_75, permute_80, permute_81, div_27, permute_82, permute_83, permute_84, permute_85, div_28, permute_86, permute_87, permute_89, permute_90, alias_27, permute_91, permute_92, permute_97, permute_98, div_30, permute_99, permute_100, permute_101, permute_102, div_31, permute_103, permute_104, permute_106, permute_107, alias_29, permute_108, permute_109, permute_114, permute_115, div_33, permute_116, permute_117, permute_118, permute_119, div_34, permute_120, permute_121, permute_123, permute_124, alias_31, permute_125, permute_126, permute_131, permute_132, div_36, permute_133, permute_134, permute_135, permute_136, div_37, permute_137, permute_138, permute_140, permute_141, alias_33, permute_142, permute_143, permute_148, permute_149, div_39, permute_150, permute_151, permute_152, permute_153, div_40, permute_154, permute_155, permute_157, permute_158, alias_35, permute_159, permute_160, permute_165, permute_166, div_42, permute_167, permute_168, permute_169, permute_170, div_43, permute_171, permute_172, permute_174, permute_175, alias_37, permute_176, permute_177, permute_182, permute_183, div_45, permute_184, permute_185, permute_186, permute_187, div_46, permute_188, permute_189, permute_191, permute_192, alias_39, permute_193, permute_194, permute_199, permute_200, div_48, permute_201, permute_202, permute_203, permute_204, div_49, permute_205, permute_206, permute_208, permute_209, alias_41, permute_210, permute_211, permute_216, permute_217, div_51, permute_218, permute_219, permute_220, permute_221, div_52, permute_222, permute_223, permute_225, permute_226, alias_43, permute_227, permute_228, permute_233, permute_234, div_54, permute_235, permute_236, permute_237, permute_238, div_55, permute_239, permute_240, permute_242, permute_243, alias_45, permute_244, permute_245, permute_250, permute_251, div_57, permute_252, permute_253, permute_254, permute_255, div_58, permute_256, permute_257, permute_259, permute_260, alias_47, permute_261, permute_262, permute_267, permute_268, div_60, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('hf_GPT2', benchmark_compiled_module)
