
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


# kernel path: /tmp/torchinductor_youkaichao/kv/ckvgysaqgp56tdde7kxdpedk53ktcbhuf62mk7bwxavujl22tbln.py
# Source Nodes: [], Original ATen: [aten.new_zeros]

triton_poi_fused_new_zeros_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_zeros_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/rh/crht5os7v6dgxkuzgfqjhdu7p2jco6o667bxxpcmttnai55arjqo.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_1', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/a4/ca4j44lqqvihc3iv2623atz3gnyoabubsv7jmqf5rnk6g2eqta5x.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_2', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/iq/ciqjn5sf7lunchqytjeijrppsef4ma7vt2erjnrbdf6olargwlhy.py
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
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/h3/ch3hh3q2ujuai7leeh3ifphksyypz4uze63hvbdowahgh54d236c.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_4', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/3b/c3bdeq6f5csjny4j5q7rsjfojgdl7opg46ujj6d5dfro3djketqh.py
# Source Nodes: [add_47, mul_44], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
# add_47 => add_95
# mul_44 => mul_92
triton_poi_fused_add_mul_pow_tanh_backward_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_tanh_backward_5', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/ov/cov4uf5iklgjeltzeqfkglocpprviqhhwjbssdrjup675k24dwpp.py
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_6', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/sd/csdt2z7yni3crzil36i52hmnoimmyngb2wf33aybt6cnmtrmd7ig.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_7', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/rs/crs2gqochwle3s7f6cd6jasifyup452h6lcebxtr35eacrdrpapw.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_8', 'mutated_arg_names': ['in_out_ptr0']}
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
    tmp22 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
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
    tmp23 = tmp22.to(tl.float32)
    tmp24 = 1.1111111111111112
    tmp25 = tmp23 * tmp24
    tmp26 = tmp21 * tmp25
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp26, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fl/cflommjsd5wrgt3nq76joatugxjmdk7joyzvqffff7buvyid7s6l.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_9', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/dp/cdpt5ixqddu67d3i42as7bqg6dqcfrqcquc5wvykxlhgzgprtw6l.py
# Source Nodes: [full], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.native_dropout_backward, aten.scalar_tensor, aten.where]
# full => full_default
triton_per_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel):
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
    x2 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask).to(tl.int1)
    tmp6 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask, other=0.0)
    tmp12 = tl.load(in_ptr3 + (r1 + (1024*x2)), rmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.1111111111111112
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp13 = tmp6 * tmp11
    tmp14 = tmp7 - tmp13
    tmp15 = 0.0
    tmp16 = tl.where(tmp12, tmp14, tmp15)
    tmp17 = 8.0
    tmp18 = tmp16 / tmp17
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp18, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nu/cnu3frcnayghvql2lpkrkabjbyo4foh4mmlbrmb47nxlvrry4unc.py
# Source Nodes: [], Original ATen: [aten.cat]

triton_poi_fused_cat_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_11', 'mutated_arg_names': []},
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
    x1 = xindex
    y0 = yindex
    tmp0 = x1
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 768, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((64*y0) + (65536*((x1 // 64) % 12)) + (x1 % 64)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 1536, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((64*y0) + (65536*((x1 // 64) % 12)) + (x1 % 64)), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr2 + (y0 + (1024*(x1 % 768))), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp11, tmp14, tmp15)
    tmp17 = tmp0 >= tmp9
    tmp18 = tl.full([1, 1], 2304, tl.int64)
    tmp19 = tmp0 < tmp18
    tmp20 = tl.load(in_ptr3 + ((64*y0) + (65536*((x1 // 64) % 12)) + (x1 % 64)), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr4 + ((64*y0) + (65536*((x1 // 64) % 12)) + (x1 % 64)), tmp17 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp17, tmp22, tmp23)
    tmp25 = tl.where(tmp11, tmp16, tmp24)
    tmp26 = tl.where(tmp4, tmp7, tmp25)
    tl.store(out_ptr0 + (x1 + (2304*y0)), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hm/chmmtel56ivyif7fcpybxrhryirbmpn7267hzcz7bhuxzrrmtwte.py
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_12', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/26/c26bb4s2rqmal5jkbwpjfxz6yj32jcmkeh7egpklnhddgwfqm3e7.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_13', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/bt/cbtgjdtl47kfysjaswxu65gqfuwt4orbkdrvf32l7daw2y3bkoli.py
# Source Nodes: [], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.scalar_tensor]

triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_scalar_tensor_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*i64', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_scalar_tensor_14', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
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
    tmp22 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp30 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
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
    tmp23 = tmp22.to(tl.float32)
    tmp24 = 1.1111111111111112
    tmp25 = tmp23 * tmp24
    tmp26 = tmp21 * tmp25
    tmp27 = tl.full([1], False, tl.int1)
    tmp28 = 0.0
    tmp29 = tl.where(tmp27, tmp28, tmp26)
    tmp31 = tl.full([1], -1, tl.int64)
    tmp32 = tmp30 == tmp31
    tmp33 = tl.where(tmp32, tmp28, tmp26)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp29, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp33, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/26/c26gc7zocuq7bnx7sfp3mlwnawpgoqlee32poqmbvtljfacjgha7.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_15', 'mutated_arg_names': []},
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
    primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, view, view_1, getitem_1, mul, getitem_8, getitem_10, mul_2, addmm_2, tanh, getitem_14, mul_8, getitem_21, getitem_23, mul_10, addmm_6, tanh_1, getitem_27, mul_16, getitem_34, getitem_36, mul_18, addmm_10, tanh_2, getitem_40, mul_24, getitem_47, getitem_49, mul_26, addmm_14, tanh_3, getitem_53, mul_32, getitem_60, getitem_62, mul_34, addmm_18, tanh_4, getitem_66, mul_40, getitem_73, getitem_75, mul_42, addmm_22, tanh_5, getitem_79, mul_48, getitem_86, getitem_88, mul_50, addmm_26, tanh_6, getitem_92, mul_56, getitem_99, getitem_101, mul_58, addmm_30, tanh_7, getitem_105, mul_64, getitem_112, getitem_114, mul_66, addmm_34, tanh_8, getitem_118, mul_72, getitem_125, getitem_127, mul_74, addmm_38, tanh_9, getitem_131, mul_80, getitem_138, getitem_140, mul_82, addmm_42, tanh_10, getitem_144, mul_88, getitem_151, getitem_153, mul_90, addmm_46, tanh_11, getitem_157, mul_96, view_219, sub_37, full_default_24, permute_63, div_24, permute_65, permute_66, permute_67, permute_68, div_25, permute_69, permute_70, permute_72, permute_73, alias_25, permute_74, permute_75, permute_80, permute_81, div_27, permute_82, permute_83, permute_84, permute_85, div_28, permute_86, permute_87, permute_89, permute_90, alias_27, permute_91, permute_92, permute_97, permute_98, div_30, permute_99, permute_100, permute_101, permute_102, div_31, permute_103, permute_104, permute_106, permute_107, alias_29, permute_108, permute_109, permute_114, permute_115, div_33, permute_116, permute_117, permute_118, permute_119, div_34, permute_120, permute_121, permute_123, permute_124, alias_31, permute_125, permute_126, permute_131, permute_132, div_36, permute_133, permute_134, permute_135, permute_136, div_37, permute_137, permute_138, permute_140, permute_141, alias_33, permute_142, permute_143, permute_148, permute_149, div_39, permute_150, permute_151, permute_152, permute_153, div_40, permute_154, permute_155, permute_157, permute_158, alias_35, permute_159, permute_160, permute_165, permute_166, div_42, permute_167, permute_168, permute_169, permute_170, div_43, permute_171, permute_172, permute_174, permute_175, alias_37, permute_176, permute_177, permute_182, permute_183, div_45, permute_184, permute_185, permute_186, permute_187, div_46, permute_188, permute_189, permute_191, permute_192, alias_39, permute_193, permute_194, permute_199, permute_200, div_48, permute_201, permute_202, permute_203, permute_204, div_49, permute_205, permute_206, permute_208, permute_209, alias_41, permute_210, permute_211, permute_216, permute_217, div_51, permute_218, permute_219, permute_220, permute_221, div_52, permute_222, permute_223, permute_225, permute_226, alias_43, permute_227, permute_228, permute_233, permute_234, div_54, permute_235, permute_236, permute_237, permute_238, div_55, permute_239, permute_240, permute_242, permute_243, alias_45, permute_244, permute_245, permute_250, permute_251, div_57, permute_252, permute_253, permute_254, permute_255, div_58, permute_256, permute_257, permute_259, permute_260, alias_47, permute_261, permute_262, permute_267, permute_268, div_60, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26 = args
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
    assert_size_stride(primals_150, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_151, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_152, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_153, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_154, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_155, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_156, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_157, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_158, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_159, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_160, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_161, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(view, (1, 1024), (1024, 1))
    assert_size_stride(view_1, (1, 1024), (1024, 1))
    assert_size_stride(getitem_1, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(getitem_8, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(getitem_10, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_2, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(addmm_2, (1024, 3072), (3072, 1))
    assert_size_stride(tanh, (1, 1024, 3072), (3145728, 3072, 1))
    assert_size_stride(getitem_14, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_8, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(getitem_21, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(getitem_23, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_10, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(addmm_6, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_1, (1, 1024, 3072), (3145728, 3072, 1))
    assert_size_stride(getitem_27, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_16, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(getitem_34, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(getitem_36, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_18, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(addmm_10, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_2, (1, 1024, 3072), (3145728, 3072, 1))
    assert_size_stride(getitem_40, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_24, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(getitem_47, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(getitem_49, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_26, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(addmm_14, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_3, (1, 1024, 3072), (3145728, 3072, 1))
    assert_size_stride(getitem_53, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_32, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(getitem_60, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(getitem_62, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_34, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(addmm_18, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_4, (1, 1024, 3072), (3145728, 3072, 1))
    assert_size_stride(getitem_66, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_40, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(getitem_73, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(getitem_75, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_42, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(addmm_22, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_5, (1, 1024, 3072), (3145728, 3072, 1))
    assert_size_stride(getitem_79, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_48, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(getitem_86, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(getitem_88, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_50, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(addmm_26, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_6, (1, 1024, 3072), (3145728, 3072, 1))
    assert_size_stride(getitem_92, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_56, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(getitem_99, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(getitem_101, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_58, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(addmm_30, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_7, (1, 1024, 3072), (3145728, 3072, 1))
    assert_size_stride(getitem_105, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_64, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(getitem_112, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(getitem_114, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_66, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(addmm_34, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_8, (1, 1024, 3072), (3145728, 3072, 1))
    assert_size_stride(getitem_118, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_72, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(getitem_125, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(getitem_127, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_74, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(addmm_38, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_9, (1, 1024, 3072), (3145728, 3072, 1))
    assert_size_stride(getitem_131, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_80, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(getitem_138, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(getitem_140, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_82, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(addmm_42, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_10, (1, 1024, 3072), (3145728, 3072, 1))
    assert_size_stride(getitem_144, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_88, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(getitem_151, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(getitem_153, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_90, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(addmm_46, (1024, 3072), (3072, 1))
    assert_size_stride(tanh_11, (1, 1024, 3072), (3145728, 3072, 1))
    assert_size_stride(getitem_157, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(mul_96, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(view_219, (1024, 768), (768, 1))
    assert_size_stride(sub_37, (1, ), (1, ))
    assert_size_stride(full_default_24, (1, ), (1, ))
    assert_size_stride(permute_63, (2, 768), (768, 1))
    assert_size_stride(div_24, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_65, (768, 3072), (1, 768))
    assert_size_stride(permute_66, (3072, 1024), (1, 3072))
    assert_size_stride(permute_67, (3072, 768), (1, 3072))
    assert_size_stride(permute_68, (768, 1024), (1, 768))
    assert_size_stride(div_25, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_69, (768, 768), (1, 768))
    assert_size_stride(permute_70, (768, 1024), (1, 768))
    assert_size_stride(permute_72, (12, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_73, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(alias_25, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(permute_74, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(permute_75, (12, 1024, 64), (64, 2304, 1))
    assert_size_stride(permute_80, (2304, 768), (1, 2304))
    assert_size_stride(permute_81, (768, 1024), (1, 768))
    assert_size_stride(div_27, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_82, (768, 3072), (1, 768))
    assert_size_stride(permute_83, (3072, 1024), (1, 3072))
    assert_size_stride(permute_84, (3072, 768), (1, 3072))
    assert_size_stride(permute_85, (768, 1024), (1, 768))
    assert_size_stride(div_28, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_86, (768, 768), (1, 768))
    assert_size_stride(permute_87, (768, 1024), (1, 768))
    assert_size_stride(permute_89, (12, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_90, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(alias_27, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(permute_91, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(permute_92, (12, 1024, 64), (64, 2304, 1))
    assert_size_stride(permute_97, (2304, 768), (1, 2304))
    assert_size_stride(permute_98, (768, 1024), (1, 768))
    assert_size_stride(div_30, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_99, (768, 3072), (1, 768))
    assert_size_stride(permute_100, (3072, 1024), (1, 3072))
    assert_size_stride(permute_101, (3072, 768), (1, 3072))
    assert_size_stride(permute_102, (768, 1024), (1, 768))
    assert_size_stride(div_31, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_103, (768, 768), (1, 768))
    assert_size_stride(permute_104, (768, 1024), (1, 768))
    assert_size_stride(permute_106, (12, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_107, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(alias_29, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(permute_108, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(permute_109, (12, 1024, 64), (64, 2304, 1))
    assert_size_stride(permute_114, (2304, 768), (1, 2304))
    assert_size_stride(permute_115, (768, 1024), (1, 768))
    assert_size_stride(div_33, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_116, (768, 3072), (1, 768))
    assert_size_stride(permute_117, (3072, 1024), (1, 3072))
    assert_size_stride(permute_118, (3072, 768), (1, 3072))
    assert_size_stride(permute_119, (768, 1024), (1, 768))
    assert_size_stride(div_34, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_120, (768, 768), (1, 768))
    assert_size_stride(permute_121, (768, 1024), (1, 768))
    assert_size_stride(permute_123, (12, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_124, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(alias_31, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(permute_125, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(permute_126, (12, 1024, 64), (64, 2304, 1))
    assert_size_stride(permute_131, (2304, 768), (1, 2304))
    assert_size_stride(permute_132, (768, 1024), (1, 768))
    assert_size_stride(div_36, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_133, (768, 3072), (1, 768))
    assert_size_stride(permute_134, (3072, 1024), (1, 3072))
    assert_size_stride(permute_135, (3072, 768), (1, 3072))
    assert_size_stride(permute_136, (768, 1024), (1, 768))
    assert_size_stride(div_37, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_137, (768, 768), (1, 768))
    assert_size_stride(permute_138, (768, 1024), (1, 768))
    assert_size_stride(permute_140, (12, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_141, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(alias_33, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(permute_142, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(permute_143, (12, 1024, 64), (64, 2304, 1))
    assert_size_stride(permute_148, (2304, 768), (1, 2304))
    assert_size_stride(permute_149, (768, 1024), (1, 768))
    assert_size_stride(div_39, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_150, (768, 3072), (1, 768))
    assert_size_stride(permute_151, (3072, 1024), (1, 3072))
    assert_size_stride(permute_152, (3072, 768), (1, 3072))
    assert_size_stride(permute_153, (768, 1024), (1, 768))
    assert_size_stride(div_40, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_154, (768, 768), (1, 768))
    assert_size_stride(permute_155, (768, 1024), (1, 768))
    assert_size_stride(permute_157, (12, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_158, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(alias_35, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(permute_159, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(permute_160, (12, 1024, 64), (64, 2304, 1))
    assert_size_stride(permute_165, (2304, 768), (1, 2304))
    assert_size_stride(permute_166, (768, 1024), (1, 768))
    assert_size_stride(div_42, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_167, (768, 3072), (1, 768))
    assert_size_stride(permute_168, (3072, 1024), (1, 3072))
    assert_size_stride(permute_169, (3072, 768), (1, 3072))
    assert_size_stride(permute_170, (768, 1024), (1, 768))
    assert_size_stride(div_43, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_171, (768, 768), (1, 768))
    assert_size_stride(permute_172, (768, 1024), (1, 768))
    assert_size_stride(permute_174, (12, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_175, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(alias_37, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(permute_176, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(permute_177, (12, 1024, 64), (64, 2304, 1))
    assert_size_stride(permute_182, (2304, 768), (1, 2304))
    assert_size_stride(permute_183, (768, 1024), (1, 768))
    assert_size_stride(div_45, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_184, (768, 3072), (1, 768))
    assert_size_stride(permute_185, (3072, 1024), (1, 3072))
    assert_size_stride(permute_186, (3072, 768), (1, 3072))
    assert_size_stride(permute_187, (768, 1024), (1, 768))
    assert_size_stride(div_46, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_188, (768, 768), (1, 768))
    assert_size_stride(permute_189, (768, 1024), (1, 768))
    assert_size_stride(permute_191, (12, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_192, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(alias_39, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(permute_193, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(permute_194, (12, 1024, 64), (64, 2304, 1))
    assert_size_stride(permute_199, (2304, 768), (1, 2304))
    assert_size_stride(permute_200, (768, 1024), (1, 768))
    assert_size_stride(div_48, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_201, (768, 3072), (1, 768))
    assert_size_stride(permute_202, (3072, 1024), (1, 3072))
    assert_size_stride(permute_203, (3072, 768), (1, 3072))
    assert_size_stride(permute_204, (768, 1024), (1, 768))
    assert_size_stride(div_49, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_205, (768, 768), (1, 768))
    assert_size_stride(permute_206, (768, 1024), (1, 768))
    assert_size_stride(permute_208, (12, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_209, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(alias_41, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(permute_210, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(permute_211, (12, 1024, 64), (64, 2304, 1))
    assert_size_stride(permute_216, (2304, 768), (1, 2304))
    assert_size_stride(permute_217, (768, 1024), (1, 768))
    assert_size_stride(div_51, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_218, (768, 3072), (1, 768))
    assert_size_stride(permute_219, (3072, 1024), (1, 3072))
    assert_size_stride(permute_220, (3072, 768), (1, 3072))
    assert_size_stride(permute_221, (768, 1024), (1, 768))
    assert_size_stride(div_52, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_222, (768, 768), (1, 768))
    assert_size_stride(permute_223, (768, 1024), (1, 768))
    assert_size_stride(permute_225, (12, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_226, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(alias_43, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(permute_227, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(permute_228, (12, 1024, 64), (64, 2304, 1))
    assert_size_stride(permute_233, (2304, 768), (1, 2304))
    assert_size_stride(permute_234, (768, 1024), (1, 768))
    assert_size_stride(div_54, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_235, (768, 3072), (1, 768))
    assert_size_stride(permute_236, (3072, 1024), (1, 3072))
    assert_size_stride(permute_237, (3072, 768), (1, 3072))
    assert_size_stride(permute_238, (768, 1024), (1, 768))
    assert_size_stride(div_55, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_239, (768, 768), (1, 768))
    assert_size_stride(permute_240, (768, 1024), (1, 768))
    assert_size_stride(permute_242, (12, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_243, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(alias_45, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(permute_244, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(permute_245, (12, 1024, 64), (64, 2304, 1))
    assert_size_stride(permute_250, (2304, 768), (1, 2304))
    assert_size_stride(permute_251, (768, 1024), (1, 768))
    assert_size_stride(div_57, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_252, (768, 3072), (1, 768))
    assert_size_stride(permute_253, (3072, 1024), (1, 3072))
    assert_size_stride(permute_254, (3072, 768), (1, 3072))
    assert_size_stride(permute_255, (768, 1024), (1, 768))
    assert_size_stride(div_58, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_256, (768, 768), (1, 768))
    assert_size_stride(permute_257, (768, 1024), (1, 768))
    assert_size_stride(permute_259, (12, 1024, 1024), (1048576, 1, 1024))
    assert_size_stride(permute_260, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(alias_47, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1))
    assert_size_stride(permute_261, (12, 64, 1024), (64, 1, 2304))
    assert_size_stride(permute_262, (12, 1024, 64), (64, 2304, 1))
    assert_size_stride(permute_267, (2304, 768), (1, 2304))
    assert_size_stride(permute_268, (768, 1024), (1, 768))
    assert_size_stride(div_60, (1, 1024, 1), (1024, 1, 1))
    assert_size_stride(tangents_1, (1, 1024, 768), (786432, 768, 1))
    assert_size_stride(tangents_2, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_3, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_4, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_5, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_6, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_7, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_8, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_9, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_10, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_11, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_12, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_13, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_14, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_15, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_16, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_17, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_18, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_19, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_20, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_21, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_22, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_23, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_24, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_25, (1, 12, 1024, 64), (786432, 65536, 64, 1))
    assert_size_stride(tangents_26, (1, 2), (2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 1024, 2), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.new_zeros]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_new_zeros_0.run(buf0, 2048, grid=grid(2048), stream=stream0)
        aten.index_put_(buf0, [full_default_24, sub_37], tangents_26, True)
        del full_default_24
        del sub_37
        del tangents_26
        buf3 = empty((2, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (2, 1024), (1, 2), 0), view_219, out=buf3)
        del view_219
        buf4 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (1024, 2), (2, 1), 0), permute_63, out=buf4)
        del buf0
        del permute_63
        buf7 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        buf10 = empty((1, 1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_1.run(tangents_1, buf4, primals_147, mul_96, div_24, getitem_157, buf7, buf10, 1024, 768, grid=grid(1024), stream=stream0)
        del div_24
        del getitem_157
        del primals_147
        buf8 = empty((768, ), device='cuda', dtype=torch.float32)
        buf9 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_2.run(tangents_1, buf4, mul_96, buf8, buf9, 768, 1024, grid=grid(768), stream=stream0)
        del mul_96
        del tangents_1
        buf11 = empty((1024, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (1024, 768), (768, 1), 0), permute_65, out=buf11)
        del permute_65
        buf12 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_66, reinterpret_tensor(buf10, (1024, 768), (768, 1), 0), out=buf12)
        del permute_66
        buf13 = empty_strided((1, 768, 8), (6144, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf10, buf13, 6144, 128, grid=grid(6144), stream=stream0)
        buf14 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf13, buf14, 768, 8, grid=grid(768), stream=stream0)
        buf15 = reinterpret_tensor(buf11, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf11  # reuse
        # Source Nodes: [add_47, mul_44], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_5.run(buf15, addmm_46, tanh_11, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_46
        del tanh_11
        buf16 = reinterpret_tensor(buf10, (1024, 768), (768, 1), 0); del buf10  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (1024, 3072), (3072, 1), 0), permute_67, out=buf16)
        del permute_67
        buf17 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_68, reinterpret_tensor(buf15, (1024, 3072), (3072, 1), 0), out=buf17)
        del permute_68
        buf18 = empty_strided((1, 3072, 8), (24576, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf15, buf18, 24576, 128, grid=grid(24576), stream=stream0)
        buf19 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf18, buf19, 3072, 8, grid=grid(3072), stream=stream0)
        buf24 = buf7; del buf7  # reuse
        buf25 = reinterpret_tensor(buf4, (1, 1024, 768), (786432, 768, 1), 0); del buf4  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_8.run(buf24, buf16, primals_145, mul_90, div_25, getitem_153, buf25, 1024, 768, grid=grid(1024), stream=stream0)
        del div_25
        del getitem_153
        del primals_145
        buf22 = empty((768, ), device='cuda', dtype=torch.float32)
        buf23 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf16, mul_90, buf22, buf23, 768, 1024, grid=grid(768), stream=stream0)
        del mul_90
        buf26 = buf16; del buf16  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf25, (1024, 768), (768, 1), 0), permute_69, out=buf26)
        del permute_69
        buf27 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_70, reinterpret_tensor(buf25, (1024, 768), (768, 1), 0), out=buf27)
        del permute_70
        buf28 = buf13; del buf13  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf25, buf28, 6144, 128, grid=grid(6144), stream=stream0)
        buf29 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf28, buf29, 768, 8, grid=grid(768), stream=stream0)
        buf30 = reinterpret_tensor(buf25, (12, 1024, 64), (65536, 64, 1), 0); del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_72, reinterpret_tensor(buf26, (12, 1024, 64), (64, 768, 1), 0), out=buf30)
        del permute_72
        buf31 = empty((12, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf26, (12, 1024, 64), (64, 768, 1), 0), permute_73, out=buf31)
        del permute_73
        buf33 = empty((1, 12, 1024, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [full], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.native_dropout_backward, aten.scalar_tensor, aten.where]
        triton_per_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_10.run(buf31, getitem_151, alias_25, primals_161, buf33, 12288, 1024, grid=grid(12288), stream=stream0)
        del alias_25
        del getitem_151
        del primals_161
        buf34 = reinterpret_tensor(buf26, (12, 64, 1024), (65536, 1024, 1), 0); del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_74, reinterpret_tensor(buf33, (12, 1024, 1024), (1048576, 1024, 1), 0), out=buf34)
        del permute_74
        buf35 = empty((12, 1024, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf33, (12, 1024, 1024), (1048576, 1024, 1), 0), permute_75, out=buf35)
        del permute_75
        buf36 = empty((1, 1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_11.run(buf35, tangents_24, buf34, tangents_25, buf30, buf36, 1024, 2304, grid=grid(1024, 2304), stream=stream0)
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
        triton_red_fused_sum_12.run(buf36, buf39, 18432, 128, grid=grid(18432), stream=stream0)
        buf40 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf39, buf40, 2304, 8, grid=grid(2304), stream=stream0)
        buf45 = buf24; del buf24  # reuse
        buf46 = reinterpret_tensor(buf34, (1, 1024, 768), (786432, 768, 1), 0); del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_8.run(buf45, buf37, primals_143, mul_88, div_27, getitem_144, buf46, 1024, 768, grid=grid(1024), stream=stream0)
        del div_27
        del getitem_144
        del primals_143
        buf43 = empty((768, ), device='cuda', dtype=torch.float32)
        buf44 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf37, mul_88, buf43, buf44, 768, 1024, grid=grid(768), stream=stream0)
        del mul_88
        buf47 = reinterpret_tensor(buf15, (1024, 3072), (3072, 1), 0); del buf15  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf46, (1024, 768), (768, 1), 0), permute_82, out=buf47)
        del permute_82
        buf48 = reinterpret_tensor(buf36, (3072, 768), (768, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_83, reinterpret_tensor(buf46, (1024, 768), (768, 1), 0), out=buf48)
        del permute_83
        buf49 = buf28; del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf46, buf49, 6144, 128, grid=grid(6144), stream=stream0)
        buf50 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf49, buf50, 768, 8, grid=grid(768), stream=stream0)
        buf51 = reinterpret_tensor(buf47, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf47  # reuse
        # Source Nodes: [add_43, mul_40], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_5.run(buf51, addmm_42, tanh_10, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_42
        del tanh_10
        buf52 = reinterpret_tensor(buf46, (1024, 768), (768, 1), 0); del buf46  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf51, (1024, 3072), (3072, 1), 0), permute_84, out=buf52)
        del permute_84
        buf53 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_85, reinterpret_tensor(buf51, (1024, 3072), (3072, 1), 0), out=buf53)
        del permute_85
        buf54 = buf18; del buf18  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf51, buf54, 24576, 128, grid=grid(24576), stream=stream0)
        buf55 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf54, buf55, 3072, 8, grid=grid(3072), stream=stream0)
        buf60 = buf45; del buf45  # reuse
        buf61 = reinterpret_tensor(buf37, (1, 1024, 768), (786432, 768, 1), 0); del buf37  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_8.run(buf60, buf52, primals_141, mul_82, div_28, getitem_140, buf61, 1024, 768, grid=grid(1024), stream=stream0)
        del div_28
        del getitem_140
        del primals_141
        buf58 = empty((768, ), device='cuda', dtype=torch.float32)
        buf59 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf52, mul_82, buf58, buf59, 768, 1024, grid=grid(768), stream=stream0)
        del mul_82
        buf62 = buf52; del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf61, (1024, 768), (768, 1), 0), permute_86, out=buf62)
        del permute_86
        buf63 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_87, reinterpret_tensor(buf61, (1024, 768), (768, 1), 0), out=buf63)
        del permute_87
        buf64 = buf49; del buf49  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf61, buf64, 6144, 128, grid=grid(6144), stream=stream0)
        buf65 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf64, buf65, 768, 8, grid=grid(768), stream=stream0)
        buf66 = reinterpret_tensor(buf61, (12, 1024, 64), (65536, 64, 1), 0); del buf61  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_89, reinterpret_tensor(buf62, (12, 1024, 64), (64, 768, 1), 0), out=buf66)
        del permute_89
        buf67 = reinterpret_tensor(buf33, (12, 1024, 1024), (1048576, 1024, 1), 0); del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf62, (12, 1024, 64), (64, 768, 1), 0), permute_90, out=buf67)
        del permute_90
        buf69 = reinterpret_tensor(buf31, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf31  # reuse
        # Source Nodes: [full], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.native_dropout_backward, aten.scalar_tensor, aten.where]
        triton_per_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_10.run(buf67, getitem_138, alias_27, primals_160, buf69, 12288, 1024, grid=grid(12288), stream=stream0)
        del alias_27
        del getitem_138
        del primals_160
        buf70 = reinterpret_tensor(buf62, (12, 64, 1024), (65536, 1024, 1), 0); del buf62  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_91, reinterpret_tensor(buf69, (12, 1024, 1024), (1048576, 1024, 1), 0), out=buf70)
        del permute_91
        buf71 = buf30; del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf69, (12, 1024, 1024), (1048576, 1024, 1), 0), permute_92, out=buf71)
        del permute_92
        buf72 = empty((1, 1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_11.run(buf71, tangents_22, buf70, tangents_23, buf66, buf72, 1024, 2304, grid=grid(1024, 2304), stream=stream0)
        del tangents_22
        del tangents_23
        buf73 = reinterpret_tensor(buf71, (1024, 768), (768, 1), 0); del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf72, (1024, 2304), (2304, 1), 0), permute_97, out=buf73)
        del permute_97
        buf74 = empty((768, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_98, reinterpret_tensor(buf72, (1024, 2304), (2304, 1), 0), out=buf74)
        del permute_98
        buf75 = buf39; del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf72, buf75, 18432, 128, grid=grid(18432), stream=stream0)
        buf76 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf75, buf76, 2304, 8, grid=grid(2304), stream=stream0)
        buf81 = buf60; del buf60  # reuse
        buf82 = reinterpret_tensor(buf70, (1, 1024, 768), (786432, 768, 1), 0); del buf70  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_8.run(buf81, buf73, primals_139, mul_80, div_30, getitem_131, buf82, 1024, 768, grid=grid(1024), stream=stream0)
        del div_30
        del getitem_131
        del primals_139
        buf79 = empty((768, ), device='cuda', dtype=torch.float32)
        buf80 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf73, mul_80, buf79, buf80, 768, 1024, grid=grid(768), stream=stream0)
        del mul_80
        buf83 = reinterpret_tensor(buf51, (1024, 3072), (3072, 1), 0); del buf51  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf82, (1024, 768), (768, 1), 0), permute_99, out=buf83)
        del permute_99
        buf84 = reinterpret_tensor(buf72, (3072, 768), (768, 1), 0); del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_100, reinterpret_tensor(buf82, (1024, 768), (768, 1), 0), out=buf84)
        del permute_100
        buf85 = buf64; del buf64  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf82, buf85, 6144, 128, grid=grid(6144), stream=stream0)
        buf86 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf85, buf86, 768, 8, grid=grid(768), stream=stream0)
        buf87 = reinterpret_tensor(buf83, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf83  # reuse
        # Source Nodes: [add_39, mul_36], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_5.run(buf87, addmm_38, tanh_9, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_38
        del tanh_9
        buf88 = reinterpret_tensor(buf82, (1024, 768), (768, 1), 0); del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf87, (1024, 3072), (3072, 1), 0), permute_101, out=buf88)
        del permute_101
        buf89 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_102, reinterpret_tensor(buf87, (1024, 3072), (3072, 1), 0), out=buf89)
        del permute_102
        buf90 = buf54; del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf87, buf90, 24576, 128, grid=grid(24576), stream=stream0)
        buf91 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf90, buf91, 3072, 8, grid=grid(3072), stream=stream0)
        buf96 = buf81; del buf81  # reuse
        buf97 = reinterpret_tensor(buf73, (1, 1024, 768), (786432, 768, 1), 0); del buf73  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_8.run(buf96, buf88, primals_137, mul_74, div_31, getitem_127, buf97, 1024, 768, grid=grid(1024), stream=stream0)
        del div_31
        del getitem_127
        del primals_137
        buf94 = empty((768, ), device='cuda', dtype=torch.float32)
        buf95 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf88, mul_74, buf94, buf95, 768, 1024, grid=grid(768), stream=stream0)
        del mul_74
        buf98 = buf88; del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf97, (1024, 768), (768, 1), 0), permute_103, out=buf98)
        del permute_103
        buf99 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_104, reinterpret_tensor(buf97, (1024, 768), (768, 1), 0), out=buf99)
        del permute_104
        buf100 = buf85; del buf85  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf97, buf100, 6144, 128, grid=grid(6144), stream=stream0)
        buf101 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf100, buf101, 768, 8, grid=grid(768), stream=stream0)
        buf102 = reinterpret_tensor(buf97, (12, 1024, 64), (65536, 64, 1), 0); del buf97  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_106, reinterpret_tensor(buf98, (12, 1024, 64), (64, 768, 1), 0), out=buf102)
        del permute_106
        buf103 = reinterpret_tensor(buf69, (12, 1024, 1024), (1048576, 1024, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf98, (12, 1024, 64), (64, 768, 1), 0), permute_107, out=buf103)
        del permute_107
        buf105 = reinterpret_tensor(buf67, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf67  # reuse
        # Source Nodes: [full], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.native_dropout_backward, aten.scalar_tensor, aten.where]
        triton_per_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_10.run(buf103, getitem_125, alias_29, primals_159, buf105, 12288, 1024, grid=grid(12288), stream=stream0)
        del alias_29
        del getitem_125
        del primals_159
        buf106 = reinterpret_tensor(buf98, (12, 64, 1024), (65536, 1024, 1), 0); del buf98  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_108, reinterpret_tensor(buf105, (12, 1024, 1024), (1048576, 1024, 1), 0), out=buf106)
        del permute_108
        buf107 = buf66; del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf105, (12, 1024, 1024), (1048576, 1024, 1), 0), permute_109, out=buf107)
        del permute_109
        buf108 = empty((1, 1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_11.run(buf107, tangents_20, buf106, tangents_21, buf102, buf108, 1024, 2304, grid=grid(1024, 2304), stream=stream0)
        del tangents_20
        del tangents_21
        buf109 = reinterpret_tensor(buf107, (1024, 768), (768, 1), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf108, (1024, 2304), (2304, 1), 0), permute_114, out=buf109)
        del permute_114
        buf110 = empty((768, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_115, reinterpret_tensor(buf108, (1024, 2304), (2304, 1), 0), out=buf110)
        del permute_115
        buf111 = buf75; del buf75  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf108, buf111, 18432, 128, grid=grid(18432), stream=stream0)
        buf112 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf111, buf112, 2304, 8, grid=grid(2304), stream=stream0)
        buf117 = buf96; del buf96  # reuse
        buf118 = reinterpret_tensor(buf106, (1, 1024, 768), (786432, 768, 1), 0); del buf106  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_8.run(buf117, buf109, primals_135, mul_72, div_33, getitem_118, buf118, 1024, 768, grid=grid(1024), stream=stream0)
        del div_33
        del getitem_118
        del primals_135
        buf115 = empty((768, ), device='cuda', dtype=torch.float32)
        buf116 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf109, mul_72, buf115, buf116, 768, 1024, grid=grid(768), stream=stream0)
        del mul_72
        buf119 = reinterpret_tensor(buf87, (1024, 3072), (3072, 1), 0); del buf87  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf118, (1024, 768), (768, 1), 0), permute_116, out=buf119)
        del permute_116
        buf120 = reinterpret_tensor(buf108, (3072, 768), (768, 1), 0); del buf108  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_117, reinterpret_tensor(buf118, (1024, 768), (768, 1), 0), out=buf120)
        del permute_117
        buf121 = buf100; del buf100  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf118, buf121, 6144, 128, grid=grid(6144), stream=stream0)
        buf122 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf121, buf122, 768, 8, grid=grid(768), stream=stream0)
        buf123 = reinterpret_tensor(buf119, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf119  # reuse
        # Source Nodes: [add_35, mul_32], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_5.run(buf123, addmm_34, tanh_8, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_34
        del tanh_8
        buf124 = reinterpret_tensor(buf118, (1024, 768), (768, 1), 0); del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf123, (1024, 3072), (3072, 1), 0), permute_118, out=buf124)
        del permute_118
        buf125 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_119, reinterpret_tensor(buf123, (1024, 3072), (3072, 1), 0), out=buf125)
        del permute_119
        buf126 = buf90; del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf123, buf126, 24576, 128, grid=grid(24576), stream=stream0)
        buf127 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf126, buf127, 3072, 8, grid=grid(3072), stream=stream0)
        buf132 = buf117; del buf117  # reuse
        buf133 = reinterpret_tensor(buf109, (1, 1024, 768), (786432, 768, 1), 0); del buf109  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_8.run(buf132, buf124, primals_133, mul_66, div_34, getitem_114, buf133, 1024, 768, grid=grid(1024), stream=stream0)
        del div_34
        del getitem_114
        del primals_133
        buf130 = empty((768, ), device='cuda', dtype=torch.float32)
        buf131 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf124, mul_66, buf130, buf131, 768, 1024, grid=grid(768), stream=stream0)
        del mul_66
        buf134 = buf124; del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf133, (1024, 768), (768, 1), 0), permute_120, out=buf134)
        del permute_120
        buf135 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_121, reinterpret_tensor(buf133, (1024, 768), (768, 1), 0), out=buf135)
        del permute_121
        buf136 = buf121; del buf121  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf133, buf136, 6144, 128, grid=grid(6144), stream=stream0)
        buf137 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf136, buf137, 768, 8, grid=grid(768), stream=stream0)
        buf138 = reinterpret_tensor(buf133, (12, 1024, 64), (65536, 64, 1), 0); del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_123, reinterpret_tensor(buf134, (12, 1024, 64), (64, 768, 1), 0), out=buf138)
        del permute_123
        buf139 = reinterpret_tensor(buf105, (12, 1024, 1024), (1048576, 1024, 1), 0); del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf134, (12, 1024, 64), (64, 768, 1), 0), permute_124, out=buf139)
        del permute_124
        buf141 = reinterpret_tensor(buf103, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf103  # reuse
        # Source Nodes: [full], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.native_dropout_backward, aten.scalar_tensor, aten.where]
        triton_per_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_10.run(buf139, getitem_112, alias_31, primals_158, buf141, 12288, 1024, grid=grid(12288), stream=stream0)
        del alias_31
        del getitem_112
        del primals_158
        buf142 = reinterpret_tensor(buf134, (12, 64, 1024), (65536, 1024, 1), 0); del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_125, reinterpret_tensor(buf141, (12, 1024, 1024), (1048576, 1024, 1), 0), out=buf142)
        del permute_125
        buf143 = buf102; del buf102  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf141, (12, 1024, 1024), (1048576, 1024, 1), 0), permute_126, out=buf143)
        del permute_126
        buf144 = empty((1, 1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_11.run(buf143, tangents_18, buf142, tangents_19, buf138, buf144, 1024, 2304, grid=grid(1024, 2304), stream=stream0)
        del tangents_18
        del tangents_19
        buf145 = reinterpret_tensor(buf143, (1024, 768), (768, 1), 0); del buf143  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf144, (1024, 2304), (2304, 1), 0), permute_131, out=buf145)
        del permute_131
        buf146 = empty((768, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_132, reinterpret_tensor(buf144, (1024, 2304), (2304, 1), 0), out=buf146)
        del permute_132
        buf147 = buf111; del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf144, buf147, 18432, 128, grid=grid(18432), stream=stream0)
        buf148 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf147, buf148, 2304, 8, grid=grid(2304), stream=stream0)
        buf153 = buf132; del buf132  # reuse
        buf154 = reinterpret_tensor(buf142, (1, 1024, 768), (786432, 768, 1), 0); del buf142  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_8.run(buf153, buf145, primals_131, mul_64, div_36, getitem_105, buf154, 1024, 768, grid=grid(1024), stream=stream0)
        del div_36
        del getitem_105
        del primals_131
        buf151 = empty((768, ), device='cuda', dtype=torch.float32)
        buf152 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf145, mul_64, buf151, buf152, 768, 1024, grid=grid(768), stream=stream0)
        del mul_64
        buf155 = reinterpret_tensor(buf123, (1024, 3072), (3072, 1), 0); del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (1024, 768), (768, 1), 0), permute_133, out=buf155)
        del permute_133
        buf156 = reinterpret_tensor(buf144, (3072, 768), (768, 1), 0); del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_134, reinterpret_tensor(buf154, (1024, 768), (768, 1), 0), out=buf156)
        del permute_134
        buf157 = buf136; del buf136  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf154, buf157, 6144, 128, grid=grid(6144), stream=stream0)
        buf158 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf157, buf158, 768, 8, grid=grid(768), stream=stream0)
        buf159 = reinterpret_tensor(buf155, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf155  # reuse
        # Source Nodes: [add_31, mul_28], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_5.run(buf159, addmm_30, tanh_7, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_30
        del tanh_7
        buf160 = reinterpret_tensor(buf154, (1024, 768), (768, 1), 0); del buf154  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf159, (1024, 3072), (3072, 1), 0), permute_135, out=buf160)
        del permute_135
        buf161 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_136, reinterpret_tensor(buf159, (1024, 3072), (3072, 1), 0), out=buf161)
        del permute_136
        buf162 = buf126; del buf126  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf159, buf162, 24576, 128, grid=grid(24576), stream=stream0)
        buf163 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf162, buf163, 3072, 8, grid=grid(3072), stream=stream0)
        buf168 = buf153; del buf153  # reuse
        buf169 = reinterpret_tensor(buf145, (1, 1024, 768), (786432, 768, 1), 0); del buf145  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_8.run(buf168, buf160, primals_129, mul_58, div_37, getitem_101, buf169, 1024, 768, grid=grid(1024), stream=stream0)
        del div_37
        del getitem_101
        del primals_129
        buf166 = empty((768, ), device='cuda', dtype=torch.float32)
        buf167 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf160, mul_58, buf166, buf167, 768, 1024, grid=grid(768), stream=stream0)
        del mul_58
        buf170 = buf160; del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf169, (1024, 768), (768, 1), 0), permute_137, out=buf170)
        del permute_137
        buf171 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_138, reinterpret_tensor(buf169, (1024, 768), (768, 1), 0), out=buf171)
        del permute_138
        buf172 = buf157; del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf169, buf172, 6144, 128, grid=grid(6144), stream=stream0)
        buf173 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf172, buf173, 768, 8, grid=grid(768), stream=stream0)
        buf174 = reinterpret_tensor(buf169, (12, 1024, 64), (65536, 64, 1), 0); del buf169  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_140, reinterpret_tensor(buf170, (12, 1024, 64), (64, 768, 1), 0), out=buf174)
        del permute_140
        buf175 = reinterpret_tensor(buf141, (12, 1024, 1024), (1048576, 1024, 1), 0); del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf170, (12, 1024, 64), (64, 768, 1), 0), permute_141, out=buf175)
        del permute_141
        buf177 = reinterpret_tensor(buf139, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf139  # reuse
        # Source Nodes: [full], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.native_dropout_backward, aten.scalar_tensor, aten.where]
        triton_per_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_10.run(buf175, getitem_99, alias_33, primals_157, buf177, 12288, 1024, grid=grid(12288), stream=stream0)
        del alias_33
        del getitem_99
        del primals_157
        buf178 = reinterpret_tensor(buf170, (12, 64, 1024), (65536, 1024, 1), 0); del buf170  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_142, reinterpret_tensor(buf177, (12, 1024, 1024), (1048576, 1024, 1), 0), out=buf178)
        del permute_142
        buf179 = buf138; del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf177, (12, 1024, 1024), (1048576, 1024, 1), 0), permute_143, out=buf179)
        del permute_143
        buf180 = empty((1, 1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_11.run(buf179, tangents_16, buf178, tangents_17, buf174, buf180, 1024, 2304, grid=grid(1024, 2304), stream=stream0)
        del tangents_16
        del tangents_17
        buf181 = reinterpret_tensor(buf179, (1024, 768), (768, 1), 0); del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf180, (1024, 2304), (2304, 1), 0), permute_148, out=buf181)
        del permute_148
        buf182 = empty((768, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_149, reinterpret_tensor(buf180, (1024, 2304), (2304, 1), 0), out=buf182)
        del permute_149
        buf183 = buf147; del buf147  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf180, buf183, 18432, 128, grid=grid(18432), stream=stream0)
        buf184 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf183, buf184, 2304, 8, grid=grid(2304), stream=stream0)
        buf189 = buf168; del buf168  # reuse
        buf190 = reinterpret_tensor(buf178, (1, 1024, 768), (786432, 768, 1), 0); del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_8.run(buf189, buf181, primals_127, mul_56, div_39, getitem_92, buf190, 1024, 768, grid=grid(1024), stream=stream0)
        del div_39
        del getitem_92
        del primals_127
        buf187 = empty((768, ), device='cuda', dtype=torch.float32)
        buf188 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf181, mul_56, buf187, buf188, 768, 1024, grid=grid(768), stream=stream0)
        del mul_56
        buf191 = reinterpret_tensor(buf159, (1024, 3072), (3072, 1), 0); del buf159  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf190, (1024, 768), (768, 1), 0), permute_150, out=buf191)
        del permute_150
        buf192 = reinterpret_tensor(buf180, (3072, 768), (768, 1), 0); del buf180  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_151, reinterpret_tensor(buf190, (1024, 768), (768, 1), 0), out=buf192)
        del permute_151
        buf193 = buf172; del buf172  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf190, buf193, 6144, 128, grid=grid(6144), stream=stream0)
        buf194 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf193, buf194, 768, 8, grid=grid(768), stream=stream0)
        buf195 = reinterpret_tensor(buf191, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf191  # reuse
        # Source Nodes: [add_27, mul_24], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_5.run(buf195, addmm_26, tanh_6, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_26
        del tanh_6
        buf196 = reinterpret_tensor(buf190, (1024, 768), (768, 1), 0); del buf190  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf195, (1024, 3072), (3072, 1), 0), permute_152, out=buf196)
        del permute_152
        buf197 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_153, reinterpret_tensor(buf195, (1024, 3072), (3072, 1), 0), out=buf197)
        del permute_153
        buf198 = buf162; del buf162  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf195, buf198, 24576, 128, grid=grid(24576), stream=stream0)
        buf199 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf198, buf199, 3072, 8, grid=grid(3072), stream=stream0)
        buf204 = buf189; del buf189  # reuse
        buf205 = reinterpret_tensor(buf181, (1, 1024, 768), (786432, 768, 1), 0); del buf181  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_8.run(buf204, buf196, primals_125, mul_50, div_40, getitem_88, buf205, 1024, 768, grid=grid(1024), stream=stream0)
        del div_40
        del getitem_88
        del primals_125
        buf202 = empty((768, ), device='cuda', dtype=torch.float32)
        buf203 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf196, mul_50, buf202, buf203, 768, 1024, grid=grid(768), stream=stream0)
        del mul_50
        buf206 = buf196; del buf196  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf205, (1024, 768), (768, 1), 0), permute_154, out=buf206)
        del permute_154
        buf207 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_155, reinterpret_tensor(buf205, (1024, 768), (768, 1), 0), out=buf207)
        del permute_155
        buf208 = buf193; del buf193  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf205, buf208, 6144, 128, grid=grid(6144), stream=stream0)
        buf209 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf208, buf209, 768, 8, grid=grid(768), stream=stream0)
        buf210 = reinterpret_tensor(buf205, (12, 1024, 64), (65536, 64, 1), 0); del buf205  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_157, reinterpret_tensor(buf206, (12, 1024, 64), (64, 768, 1), 0), out=buf210)
        del permute_157
        buf211 = reinterpret_tensor(buf177, (12, 1024, 1024), (1048576, 1024, 1), 0); del buf177  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf206, (12, 1024, 64), (64, 768, 1), 0), permute_158, out=buf211)
        del permute_158
        buf213 = reinterpret_tensor(buf175, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf175  # reuse
        # Source Nodes: [full], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.native_dropout_backward, aten.scalar_tensor, aten.where]
        triton_per_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_10.run(buf211, getitem_86, alias_35, primals_156, buf213, 12288, 1024, grid=grid(12288), stream=stream0)
        del alias_35
        del getitem_86
        del primals_156
        buf214 = reinterpret_tensor(buf206, (12, 64, 1024), (65536, 1024, 1), 0); del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_159, reinterpret_tensor(buf213, (12, 1024, 1024), (1048576, 1024, 1), 0), out=buf214)
        del permute_159
        buf215 = buf174; del buf174  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf213, (12, 1024, 1024), (1048576, 1024, 1), 0), permute_160, out=buf215)
        del permute_160
        buf216 = empty((1, 1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_11.run(buf215, tangents_14, buf214, tangents_15, buf210, buf216, 1024, 2304, grid=grid(1024, 2304), stream=stream0)
        del tangents_14
        del tangents_15
        buf217 = reinterpret_tensor(buf215, (1024, 768), (768, 1), 0); del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf216, (1024, 2304), (2304, 1), 0), permute_165, out=buf217)
        del permute_165
        buf218 = empty((768, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_166, reinterpret_tensor(buf216, (1024, 2304), (2304, 1), 0), out=buf218)
        del permute_166
        buf219 = buf183; del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf216, buf219, 18432, 128, grid=grid(18432), stream=stream0)
        buf220 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf219, buf220, 2304, 8, grid=grid(2304), stream=stream0)
        buf225 = buf204; del buf204  # reuse
        buf226 = reinterpret_tensor(buf214, (1, 1024, 768), (786432, 768, 1), 0); del buf214  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_8.run(buf225, buf217, primals_123, mul_48, div_42, getitem_79, buf226, 1024, 768, grid=grid(1024), stream=stream0)
        del div_42
        del getitem_79
        del primals_123
        buf223 = empty((768, ), device='cuda', dtype=torch.float32)
        buf224 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf217, mul_48, buf223, buf224, 768, 1024, grid=grid(768), stream=stream0)
        del mul_48
        buf227 = reinterpret_tensor(buf195, (1024, 3072), (3072, 1), 0); del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf226, (1024, 768), (768, 1), 0), permute_167, out=buf227)
        del permute_167
        buf228 = reinterpret_tensor(buf216, (3072, 768), (768, 1), 0); del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_168, reinterpret_tensor(buf226, (1024, 768), (768, 1), 0), out=buf228)
        del permute_168
        buf229 = buf208; del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf226, buf229, 6144, 128, grid=grid(6144), stream=stream0)
        buf230 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf229, buf230, 768, 8, grid=grid(768), stream=stream0)
        buf231 = reinterpret_tensor(buf227, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf227  # reuse
        # Source Nodes: [add_23, mul_20], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_5.run(buf231, addmm_22, tanh_5, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_22
        del tanh_5
        buf232 = reinterpret_tensor(buf226, (1024, 768), (768, 1), 0); del buf226  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf231, (1024, 3072), (3072, 1), 0), permute_169, out=buf232)
        del permute_169
        buf233 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_170, reinterpret_tensor(buf231, (1024, 3072), (3072, 1), 0), out=buf233)
        del permute_170
        buf234 = buf198; del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf231, buf234, 24576, 128, grid=grid(24576), stream=stream0)
        buf235 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf234, buf235, 3072, 8, grid=grid(3072), stream=stream0)
        buf240 = buf225; del buf225  # reuse
        buf241 = reinterpret_tensor(buf217, (1, 1024, 768), (786432, 768, 1), 0); del buf217  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_8.run(buf240, buf232, primals_121, mul_42, div_43, getitem_75, buf241, 1024, 768, grid=grid(1024), stream=stream0)
        del div_43
        del getitem_75
        del primals_121
        buf238 = empty((768, ), device='cuda', dtype=torch.float32)
        buf239 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf232, mul_42, buf238, buf239, 768, 1024, grid=grid(768), stream=stream0)
        del mul_42
        buf242 = buf232; del buf232  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf241, (1024, 768), (768, 1), 0), permute_171, out=buf242)
        del permute_171
        buf243 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_172, reinterpret_tensor(buf241, (1024, 768), (768, 1), 0), out=buf243)
        del permute_172
        buf244 = buf229; del buf229  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf241, buf244, 6144, 128, grid=grid(6144), stream=stream0)
        buf245 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf244, buf245, 768, 8, grid=grid(768), stream=stream0)
        buf246 = reinterpret_tensor(buf241, (12, 1024, 64), (65536, 64, 1), 0); del buf241  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_174, reinterpret_tensor(buf242, (12, 1024, 64), (64, 768, 1), 0), out=buf246)
        del permute_174
        buf247 = reinterpret_tensor(buf213, (12, 1024, 1024), (1048576, 1024, 1), 0); del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf242, (12, 1024, 64), (64, 768, 1), 0), permute_175, out=buf247)
        del permute_175
        buf249 = reinterpret_tensor(buf211, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf211  # reuse
        # Source Nodes: [full], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.native_dropout_backward, aten.scalar_tensor, aten.where]
        triton_per_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_10.run(buf247, getitem_73, alias_37, primals_155, buf249, 12288, 1024, grid=grid(12288), stream=stream0)
        del alias_37
        del getitem_73
        del primals_155
        buf250 = reinterpret_tensor(buf242, (12, 64, 1024), (65536, 1024, 1), 0); del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_176, reinterpret_tensor(buf249, (12, 1024, 1024), (1048576, 1024, 1), 0), out=buf250)
        del permute_176
        buf251 = buf210; del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf249, (12, 1024, 1024), (1048576, 1024, 1), 0), permute_177, out=buf251)
        del permute_177
        buf252 = empty((1, 1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_11.run(buf251, tangents_12, buf250, tangents_13, buf246, buf252, 1024, 2304, grid=grid(1024, 2304), stream=stream0)
        del tangents_12
        del tangents_13
        buf253 = reinterpret_tensor(buf251, (1024, 768), (768, 1), 0); del buf251  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf252, (1024, 2304), (2304, 1), 0), permute_182, out=buf253)
        del permute_182
        buf254 = empty((768, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_183, reinterpret_tensor(buf252, (1024, 2304), (2304, 1), 0), out=buf254)
        del permute_183
        buf255 = buf219; del buf219  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf252, buf255, 18432, 128, grid=grid(18432), stream=stream0)
        buf256 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf255, buf256, 2304, 8, grid=grid(2304), stream=stream0)
        buf261 = buf240; del buf240  # reuse
        buf262 = reinterpret_tensor(buf250, (1, 1024, 768), (786432, 768, 1), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_8.run(buf261, buf253, primals_119, mul_40, div_45, getitem_66, buf262, 1024, 768, grid=grid(1024), stream=stream0)
        del div_45
        del getitem_66
        del primals_119
        buf259 = empty((768, ), device='cuda', dtype=torch.float32)
        buf260 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf253, mul_40, buf259, buf260, 768, 1024, grid=grid(768), stream=stream0)
        del mul_40
        buf263 = reinterpret_tensor(buf231, (1024, 3072), (3072, 1), 0); del buf231  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf262, (1024, 768), (768, 1), 0), permute_184, out=buf263)
        del permute_184
        buf264 = reinterpret_tensor(buf252, (3072, 768), (768, 1), 0); del buf252  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_185, reinterpret_tensor(buf262, (1024, 768), (768, 1), 0), out=buf264)
        del permute_185
        buf265 = buf244; del buf244  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf262, buf265, 6144, 128, grid=grid(6144), stream=stream0)
        buf266 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf265, buf266, 768, 8, grid=grid(768), stream=stream0)
        buf267 = reinterpret_tensor(buf263, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf263  # reuse
        # Source Nodes: [add_19, mul_16], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_5.run(buf267, addmm_18, tanh_4, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_18
        del tanh_4
        buf268 = reinterpret_tensor(buf262, (1024, 768), (768, 1), 0); del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf267, (1024, 3072), (3072, 1), 0), permute_186, out=buf268)
        del permute_186
        buf269 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_187, reinterpret_tensor(buf267, (1024, 3072), (3072, 1), 0), out=buf269)
        del permute_187
        buf270 = buf234; del buf234  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf267, buf270, 24576, 128, grid=grid(24576), stream=stream0)
        buf271 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf270, buf271, 3072, 8, grid=grid(3072), stream=stream0)
        buf276 = buf261; del buf261  # reuse
        buf277 = reinterpret_tensor(buf253, (1, 1024, 768), (786432, 768, 1), 0); del buf253  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_8.run(buf276, buf268, primals_117, mul_34, div_46, getitem_62, buf277, 1024, 768, grid=grid(1024), stream=stream0)
        del div_46
        del getitem_62
        del primals_117
        buf274 = empty((768, ), device='cuda', dtype=torch.float32)
        buf275 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf268, mul_34, buf274, buf275, 768, 1024, grid=grid(768), stream=stream0)
        del mul_34
        buf278 = buf268; del buf268  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf277, (1024, 768), (768, 1), 0), permute_188, out=buf278)
        del permute_188
        buf279 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_189, reinterpret_tensor(buf277, (1024, 768), (768, 1), 0), out=buf279)
        del permute_189
        buf280 = buf265; del buf265  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf277, buf280, 6144, 128, grid=grid(6144), stream=stream0)
        buf281 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf280, buf281, 768, 8, grid=grid(768), stream=stream0)
        buf282 = reinterpret_tensor(buf277, (12, 1024, 64), (65536, 64, 1), 0); del buf277  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_191, reinterpret_tensor(buf278, (12, 1024, 64), (64, 768, 1), 0), out=buf282)
        del permute_191
        buf283 = reinterpret_tensor(buf249, (12, 1024, 1024), (1048576, 1024, 1), 0); del buf249  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf278, (12, 1024, 64), (64, 768, 1), 0), permute_192, out=buf283)
        del permute_192
        buf285 = reinterpret_tensor(buf247, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf247  # reuse
        # Source Nodes: [full], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.native_dropout_backward, aten.scalar_tensor, aten.where]
        triton_per_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_10.run(buf283, getitem_60, alias_39, primals_154, buf285, 12288, 1024, grid=grid(12288), stream=stream0)
        del alias_39
        del getitem_60
        del primals_154
        buf286 = reinterpret_tensor(buf278, (12, 64, 1024), (65536, 1024, 1), 0); del buf278  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_193, reinterpret_tensor(buf285, (12, 1024, 1024), (1048576, 1024, 1), 0), out=buf286)
        del permute_193
        buf287 = buf246; del buf246  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf285, (12, 1024, 1024), (1048576, 1024, 1), 0), permute_194, out=buf287)
        del permute_194
        buf288 = empty((1, 1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_11.run(buf287, tangents_10, buf286, tangents_11, buf282, buf288, 1024, 2304, grid=grid(1024, 2304), stream=stream0)
        del tangents_10
        del tangents_11
        buf289 = reinterpret_tensor(buf287, (1024, 768), (768, 1), 0); del buf287  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf288, (1024, 2304), (2304, 1), 0), permute_199, out=buf289)
        del permute_199
        buf290 = empty((768, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_200, reinterpret_tensor(buf288, (1024, 2304), (2304, 1), 0), out=buf290)
        del permute_200
        buf291 = buf255; del buf255  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf288, buf291, 18432, 128, grid=grid(18432), stream=stream0)
        buf292 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf291, buf292, 2304, 8, grid=grid(2304), stream=stream0)
        buf297 = buf276; del buf276  # reuse
        buf298 = reinterpret_tensor(buf286, (1, 1024, 768), (786432, 768, 1), 0); del buf286  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_8.run(buf297, buf289, primals_115, mul_32, div_48, getitem_53, buf298, 1024, 768, grid=grid(1024), stream=stream0)
        del div_48
        del getitem_53
        del primals_115
        buf295 = empty((768, ), device='cuda', dtype=torch.float32)
        buf296 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf289, mul_32, buf295, buf296, 768, 1024, grid=grid(768), stream=stream0)
        del mul_32
        buf299 = reinterpret_tensor(buf267, (1024, 3072), (3072, 1), 0); del buf267  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf298, (1024, 768), (768, 1), 0), permute_201, out=buf299)
        del permute_201
        buf300 = reinterpret_tensor(buf288, (3072, 768), (768, 1), 0); del buf288  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_202, reinterpret_tensor(buf298, (1024, 768), (768, 1), 0), out=buf300)
        del permute_202
        buf301 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf298, buf301, 6144, 128, grid=grid(6144), stream=stream0)
        buf302 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf301, buf302, 768, 8, grid=grid(768), stream=stream0)
        buf303 = reinterpret_tensor(buf299, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf299  # reuse
        # Source Nodes: [add_15, mul_12], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_5.run(buf303, addmm_14, tanh_3, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_14
        del tanh_3
        buf304 = reinterpret_tensor(buf298, (1024, 768), (768, 1), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf303, (1024, 3072), (3072, 1), 0), permute_203, out=buf304)
        del permute_203
        buf305 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_204, reinterpret_tensor(buf303, (1024, 3072), (3072, 1), 0), out=buf305)
        del permute_204
        buf306 = buf270; del buf270  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf303, buf306, 24576, 128, grid=grid(24576), stream=stream0)
        buf307 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf306, buf307, 3072, 8, grid=grid(3072), stream=stream0)
        buf312 = buf297; del buf297  # reuse
        buf313 = reinterpret_tensor(buf289, (1, 1024, 768), (786432, 768, 1), 0); del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_8.run(buf312, buf304, primals_113, mul_26, div_49, getitem_49, buf313, 1024, 768, grid=grid(1024), stream=stream0)
        del div_49
        del getitem_49
        del primals_113
        buf310 = empty((768, ), device='cuda', dtype=torch.float32)
        buf311 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf304, mul_26, buf310, buf311, 768, 1024, grid=grid(768), stream=stream0)
        del mul_26
        buf314 = buf304; del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf313, (1024, 768), (768, 1), 0), permute_205, out=buf314)
        del permute_205
        buf315 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_206, reinterpret_tensor(buf313, (1024, 768), (768, 1), 0), out=buf315)
        del permute_206
        buf316 = buf301; del buf301  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf313, buf316, 6144, 128, grid=grid(6144), stream=stream0)
        buf317 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf316, buf317, 768, 8, grid=grid(768), stream=stream0)
        buf318 = reinterpret_tensor(buf313, (12, 1024, 64), (65536, 64, 1), 0); del buf313  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_208, reinterpret_tensor(buf314, (12, 1024, 64), (64, 768, 1), 0), out=buf318)
        del permute_208
        buf319 = reinterpret_tensor(buf285, (12, 1024, 1024), (1048576, 1024, 1), 0); del buf285  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf314, (12, 1024, 64), (64, 768, 1), 0), permute_209, out=buf319)
        del permute_209
        buf321 = reinterpret_tensor(buf283, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf283  # reuse
        # Source Nodes: [full], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.native_dropout_backward, aten.scalar_tensor, aten.where]
        triton_per_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_10.run(buf319, getitem_47, alias_41, primals_153, buf321, 12288, 1024, grid=grid(12288), stream=stream0)
        del alias_41
        del getitem_47
        del primals_153
        buf322 = reinterpret_tensor(buf314, (12, 64, 1024), (65536, 1024, 1), 0); del buf314  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_210, reinterpret_tensor(buf321, (12, 1024, 1024), (1048576, 1024, 1), 0), out=buf322)
        del permute_210
        buf323 = buf282; del buf282  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf321, (12, 1024, 1024), (1048576, 1024, 1), 0), permute_211, out=buf323)
        del permute_211
        buf324 = empty((1, 1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_11.run(buf323, tangents_8, buf322, tangents_9, buf318, buf324, 1024, 2304, grid=grid(1024, 2304), stream=stream0)
        del tangents_8
        del tangents_9
        buf325 = reinterpret_tensor(buf323, (1024, 768), (768, 1), 0); del buf323  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf324, (1024, 2304), (2304, 1), 0), permute_216, out=buf325)
        del permute_216
        buf326 = empty((768, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_217, reinterpret_tensor(buf324, (1024, 2304), (2304, 1), 0), out=buf326)
        del permute_217
        buf327 = buf291; del buf291  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf324, buf327, 18432, 128, grid=grid(18432), stream=stream0)
        buf328 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf327, buf328, 2304, 8, grid=grid(2304), stream=stream0)
        buf333 = buf312; del buf312  # reuse
        buf334 = reinterpret_tensor(buf322, (1, 1024, 768), (786432, 768, 1), 0); del buf322  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_8.run(buf333, buf325, primals_111, mul_24, div_51, getitem_40, buf334, 1024, 768, grid=grid(1024), stream=stream0)
        del div_51
        del getitem_40
        del primals_111
        buf331 = empty((768, ), device='cuda', dtype=torch.float32)
        buf332 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf325, mul_24, buf331, buf332, 768, 1024, grid=grid(768), stream=stream0)
        del mul_24
        buf335 = reinterpret_tensor(buf303, (1024, 3072), (3072, 1), 0); del buf303  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf334, (1024, 768), (768, 1), 0), permute_218, out=buf335)
        del permute_218
        buf336 = reinterpret_tensor(buf324, (3072, 768), (768, 1), 0); del buf324  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_219, reinterpret_tensor(buf334, (1024, 768), (768, 1), 0), out=buf336)
        del permute_219
        buf337 = buf316; del buf316  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf334, buf337, 6144, 128, grid=grid(6144), stream=stream0)
        buf338 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf337, buf338, 768, 8, grid=grid(768), stream=stream0)
        buf339 = reinterpret_tensor(buf335, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf335  # reuse
        # Source Nodes: [add_11, mul_8], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_5.run(buf339, addmm_10, tanh_2, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_10
        del tanh_2
        buf340 = reinterpret_tensor(buf334, (1024, 768), (768, 1), 0); del buf334  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf339, (1024, 3072), (3072, 1), 0), permute_220, out=buf340)
        del permute_220
        buf341 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_221, reinterpret_tensor(buf339, (1024, 3072), (3072, 1), 0), out=buf341)
        del permute_221
        buf342 = buf306; del buf306  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf339, buf342, 24576, 128, grid=grid(24576), stream=stream0)
        buf343 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf342, buf343, 3072, 8, grid=grid(3072), stream=stream0)
        buf348 = buf333; del buf333  # reuse
        buf349 = reinterpret_tensor(buf325, (1, 1024, 768), (786432, 768, 1), 0); del buf325  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_8.run(buf348, buf340, primals_109, mul_18, div_52, getitem_36, buf349, 1024, 768, grid=grid(1024), stream=stream0)
        del div_52
        del getitem_36
        del primals_109
        buf346 = empty((768, ), device='cuda', dtype=torch.float32)
        buf347 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf340, mul_18, buf346, buf347, 768, 1024, grid=grid(768), stream=stream0)
        del mul_18
        buf350 = buf340; del buf340  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf349, (1024, 768), (768, 1), 0), permute_222, out=buf350)
        del permute_222
        buf351 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_223, reinterpret_tensor(buf349, (1024, 768), (768, 1), 0), out=buf351)
        del permute_223
        buf352 = buf337; del buf337  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf349, buf352, 6144, 128, grid=grid(6144), stream=stream0)
        buf353 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf352, buf353, 768, 8, grid=grid(768), stream=stream0)
        buf354 = reinterpret_tensor(buf349, (12, 1024, 64), (65536, 64, 1), 0); del buf349  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_225, reinterpret_tensor(buf350, (12, 1024, 64), (64, 768, 1), 0), out=buf354)
        del permute_225
        buf355 = reinterpret_tensor(buf321, (12, 1024, 1024), (1048576, 1024, 1), 0); del buf321  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf350, (12, 1024, 64), (64, 768, 1), 0), permute_226, out=buf355)
        del permute_226
        buf357 = reinterpret_tensor(buf319, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf319  # reuse
        # Source Nodes: [full], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.native_dropout_backward, aten.scalar_tensor, aten.where]
        triton_per_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_10.run(buf355, getitem_34, alias_43, primals_152, buf357, 12288, 1024, grid=grid(12288), stream=stream0)
        del alias_43
        del getitem_34
        del primals_152
        buf358 = reinterpret_tensor(buf350, (12, 64, 1024), (65536, 1024, 1), 0); del buf350  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_227, reinterpret_tensor(buf357, (12, 1024, 1024), (1048576, 1024, 1), 0), out=buf358)
        del permute_227
        buf359 = buf318; del buf318  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf357, (12, 1024, 1024), (1048576, 1024, 1), 0), permute_228, out=buf359)
        del permute_228
        buf360 = empty((1, 1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_11.run(buf359, tangents_6, buf358, tangents_7, buf354, buf360, 1024, 2304, grid=grid(1024, 2304), stream=stream0)
        del tangents_6
        del tangents_7
        buf361 = reinterpret_tensor(buf359, (1024, 768), (768, 1), 0); del buf359  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf360, (1024, 2304), (2304, 1), 0), permute_233, out=buf361)
        del permute_233
        buf362 = empty((768, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_234, reinterpret_tensor(buf360, (1024, 2304), (2304, 1), 0), out=buf362)
        del permute_234
        buf363 = buf327; del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf360, buf363, 18432, 128, grid=grid(18432), stream=stream0)
        buf364 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf363, buf364, 2304, 8, grid=grid(2304), stream=stream0)
        buf369 = buf348; del buf348  # reuse
        buf370 = reinterpret_tensor(buf358, (1, 1024, 768), (786432, 768, 1), 0); del buf358  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_8.run(buf369, buf361, primals_107, mul_16, div_54, getitem_27, buf370, 1024, 768, grid=grid(1024), stream=stream0)
        del div_54
        del getitem_27
        del primals_107
        buf367 = empty((768, ), device='cuda', dtype=torch.float32)
        buf368 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf361, mul_16, buf367, buf368, 768, 1024, grid=grid(768), stream=stream0)
        del mul_16
        buf371 = reinterpret_tensor(buf339, (1024, 3072), (3072, 1), 0); del buf339  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf370, (1024, 768), (768, 1), 0), permute_235, out=buf371)
        del permute_235
        buf372 = reinterpret_tensor(buf360, (3072, 768), (768, 1), 0); del buf360  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_236, reinterpret_tensor(buf370, (1024, 768), (768, 1), 0), out=buf372)
        del permute_236
        buf373 = buf352; del buf352  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf370, buf373, 6144, 128, grid=grid(6144), stream=stream0)
        buf374 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf373, buf374, 768, 8, grid=grid(768), stream=stream0)
        buf375 = reinterpret_tensor(buf371, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf371  # reuse
        # Source Nodes: [add_7, mul_4], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_5.run(buf375, addmm_6, tanh_1, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_6
        del tanh_1
        buf376 = reinterpret_tensor(buf370, (1024, 768), (768, 1), 0); del buf370  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf375, (1024, 3072), (3072, 1), 0), permute_237, out=buf376)
        del permute_237
        buf377 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_238, reinterpret_tensor(buf375, (1024, 3072), (3072, 1), 0), out=buf377)
        del permute_238
        buf378 = buf342; del buf342  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf375, buf378, 24576, 128, grid=grid(24576), stream=stream0)
        buf379 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf378, buf379, 3072, 8, grid=grid(3072), stream=stream0)
        buf384 = buf369; del buf369  # reuse
        buf385 = reinterpret_tensor(buf361, (1, 1024, 768), (786432, 768, 1), 0); del buf361  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_8.run(buf384, buf376, primals_105, mul_10, div_55, getitem_23, buf385, 1024, 768, grid=grid(1024), stream=stream0)
        del div_55
        del getitem_23
        del primals_105
        buf382 = empty((768, ), device='cuda', dtype=torch.float32)
        buf383 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf376, mul_10, buf382, buf383, 768, 1024, grid=grid(768), stream=stream0)
        del mul_10
        buf386 = buf376; del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf385, (1024, 768), (768, 1), 0), permute_239, out=buf386)
        del permute_239
        buf387 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_240, reinterpret_tensor(buf385, (1024, 768), (768, 1), 0), out=buf387)
        del permute_240
        buf388 = buf373; del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf385, buf388, 6144, 128, grid=grid(6144), stream=stream0)
        buf389 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf388, buf389, 768, 8, grid=grid(768), stream=stream0)
        buf390 = reinterpret_tensor(buf385, (12, 1024, 64), (65536, 64, 1), 0); del buf385  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_242, reinterpret_tensor(buf386, (12, 1024, 64), (64, 768, 1), 0), out=buf390)
        del permute_242
        buf391 = reinterpret_tensor(buf357, (12, 1024, 1024), (1048576, 1024, 1), 0); del buf357  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf386, (12, 1024, 64), (64, 768, 1), 0), permute_243, out=buf391)
        del permute_243
        buf393 = reinterpret_tensor(buf355, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf355  # reuse
        # Source Nodes: [full], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.native_dropout_backward, aten.scalar_tensor, aten.where]
        triton_per_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_10.run(buf391, getitem_21, alias_45, primals_151, buf393, 12288, 1024, grid=grid(12288), stream=stream0)
        del alias_45
        del getitem_21
        del primals_151
        buf394 = reinterpret_tensor(buf386, (12, 64, 1024), (65536, 1024, 1), 0); del buf386  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_244, reinterpret_tensor(buf393, (12, 1024, 1024), (1048576, 1024, 1), 0), out=buf394)
        del permute_244
        buf395 = buf354; del buf354  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf393, (12, 1024, 1024), (1048576, 1024, 1), 0), permute_245, out=buf395)
        del permute_245
        buf396 = empty((1, 1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_11.run(buf395, tangents_4, buf394, tangents_5, buf390, buf396, 1024, 2304, grid=grid(1024, 2304), stream=stream0)
        del tangents_4
        del tangents_5
        buf397 = reinterpret_tensor(buf395, (1024, 768), (768, 1), 0); del buf395  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf396, (1024, 2304), (2304, 1), 0), permute_250, out=buf397)
        del permute_250
        buf398 = empty((768, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_251, reinterpret_tensor(buf396, (1024, 2304), (2304, 1), 0), out=buf398)
        del permute_251
        buf399 = buf363; del buf363  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf396, buf399, 18432, 128, grid=grid(18432), stream=stream0)
        buf400 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf399, buf400, 2304, 8, grid=grid(2304), stream=stream0)
        buf405 = buf384; del buf384  # reuse
        buf406 = reinterpret_tensor(buf394, (1, 1024, 768), (786432, 768, 1), 0); del buf394  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_8.run(buf405, buf397, primals_103, mul_8, div_57, getitem_14, buf406, 1024, 768, grid=grid(1024), stream=stream0)
        del div_57
        del getitem_14
        del primals_103
        buf403 = empty((768, ), device='cuda', dtype=torch.float32)
        buf404 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf397, mul_8, buf403, buf404, 768, 1024, grid=grid(768), stream=stream0)
        del mul_8
        buf407 = reinterpret_tensor(buf375, (1024, 3072), (3072, 1), 0); del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf406, (1024, 768), (768, 1), 0), permute_252, out=buf407)
        del permute_252
        buf408 = reinterpret_tensor(buf396, (3072, 768), (768, 1), 0); del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_253, reinterpret_tensor(buf406, (1024, 768), (768, 1), 0), out=buf408)
        del permute_253
        buf409 = buf388; del buf388  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf406, buf409, 6144, 128, grid=grid(6144), stream=stream0)
        buf410 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf409, buf410, 768, 8, grid=grid(768), stream=stream0)
        buf411 = reinterpret_tensor(buf407, (1, 1024, 3072), (3145728, 3072, 1), 0); del buf407  # reuse
        # Source Nodes: [add_3, mul], Original ATen: [aten.add, aten.mul, aten.pow, aten.tanh_backward]
        triton_poi_fused_add_mul_pow_tanh_backward_5.run(buf411, addmm_2, tanh, 3145728, grid=grid(3145728), stream=stream0)
        del addmm_2
        del tanh
        buf412 = reinterpret_tensor(buf406, (1024, 768), (768, 1), 0); del buf406  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf411, (1024, 3072), (3072, 1), 0), permute_254, out=buf412)
        del permute_254
        buf413 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_255, reinterpret_tensor(buf411, (1024, 3072), (3072, 1), 0), out=buf413)
        del permute_255
        buf414 = buf378; del buf378  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_6.run(buf411, buf414, 24576, 128, grid=grid(24576), stream=stream0)
        del buf411
        buf415 = empty((1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_7.run(buf414, buf415, 3072, 8, grid=grid(3072), stream=stream0)
        del buf414
        buf420 = buf405; del buf405  # reuse
        buf421 = reinterpret_tensor(buf397, (1, 1024, 768), (786432, 768, 1), 0); del buf397  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_8.run(buf420, buf412, primals_101, mul_2, div_58, getitem_10, buf421, 1024, 768, grid=grid(1024), stream=stream0)
        del div_58
        del getitem_10
        del primals_101
        buf418 = empty((768, ), device='cuda', dtype=torch.float32)
        buf419 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf412, mul_2, buf418, buf419, 768, 1024, grid=grid(768), stream=stream0)
        del mul_2
        buf422 = buf412; del buf412  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf421, (1024, 768), (768, 1), 0), permute_256, out=buf422)
        del permute_256
        buf423 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_257, reinterpret_tensor(buf421, (1024, 768), (768, 1), 0), out=buf423)
        del permute_257
        buf424 = buf409; del buf409  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_3.run(buf421, buf424, 6144, 128, grid=grid(6144), stream=stream0)
        buf425 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(buf424, buf425, 768, 8, grid=grid(768), stream=stream0)
        del buf424
        buf426 = reinterpret_tensor(buf421, (12, 1024, 64), (65536, 64, 1), 0); del buf421  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_259, reinterpret_tensor(buf422, (12, 1024, 64), (64, 768, 1), 0), out=buf426)
        del permute_259
        buf427 = reinterpret_tensor(buf393, (12, 1024, 1024), (1048576, 1024, 1), 0); del buf393  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf422, (12, 1024, 64), (64, 768, 1), 0), permute_260, out=buf427)
        del permute_260
        buf429 = reinterpret_tensor(buf391, (1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), 0); del buf391  # reuse
        # Source Nodes: [full], Original ATen: [aten._softmax_backward_data, aten.div, aten.full, aten.native_dropout_backward, aten.scalar_tensor, aten.where]
        triton_per_fused__softmax_backward_data_div_full_native_dropout_backward_scalar_tensor_where_10.run(buf427, getitem_8, alias_47, primals_150, buf429, 12288, 1024, grid=grid(12288), stream=stream0)
        del alias_47
        del buf427
        del getitem_8
        del primals_150
        buf430 = reinterpret_tensor(buf422, (12, 64, 1024), (65536, 1024, 1), 0); del buf422  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_261, reinterpret_tensor(buf429, (12, 1024, 1024), (1048576, 1024, 1), 0), out=buf430)
        del permute_261
        buf431 = buf390; del buf390  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf429, (12, 1024, 1024), (1048576, 1024, 1), 0), permute_262, out=buf431)
        del buf429
        del permute_262
        buf432 = empty((1, 1024, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_11.run(buf431, tangents_2, buf430, tangents_3, buf426, buf432, 1024, 2304, grid=grid(1024, 2304), stream=stream0)
        del tangents_2
        del tangents_3
        buf433 = reinterpret_tensor(buf431, (1024, 768), (768, 1), 0); del buf431  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf432, (1024, 2304), (2304, 1), 0), permute_267, out=buf433)
        del permute_267
        buf434 = empty((768, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute_268, reinterpret_tensor(buf432, (1024, 2304), (2304, 1), 0), out=buf434)
        del permute_268
        buf435 = buf399; del buf399  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_12.run(buf432, buf435, 18432, 128, grid=grid(18432), stream=stream0)
        del buf432
        buf436 = empty((1, 2304), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf435, buf436, 2304, 8, grid=grid(2304), stream=stream0)
        del buf435
        buf441 = buf420; del buf420  # reuse
        buf443 = reinterpret_tensor(buf430, (1, 1024, 768), (786432, 768, 1), 0); del buf430  # reuse
        buf447 = reinterpret_tensor(buf426, (1, 1024, 768), (786432, 768, 1), 0); del buf426  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.scalar_tensor]
        triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_scalar_tensor_14.run(buf441, buf433, primals_99, mul, div_60, getitem_1, view, buf443, buf447, 1024, 768, grid=grid(1024), stream=stream0)
        del buf441
        del div_60
        del getitem_1
        del primals_99
        buf439 = empty((768, ), device='cuda', dtype=torch.float32)
        buf440 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_9.run(buf433, mul, buf439, buf440, 768, 1024, grid=grid(768), stream=stream0)
        del mul
        buf442 = buf433; del buf433  # reuse
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_15.run(buf442, 786432, grid=grid(786432), stream=stream0)
        aten.index_put_(buf442, [view_1], buf443, True)
        del buf443
        del view_1
        buf446 = empty((50257, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_16.run(buf446, 38597376, grid=grid(38597376), stream=stream0)
        aten.index_put_(buf446, [view], buf447, True)
        del buf447
        del view
        return (reinterpret_tensor(buf436, (2304, ), (1, ), 0), buf434, reinterpret_tensor(buf425, (768, ), (1, ), 0), buf423, reinterpret_tensor(buf415, (3072, ), (1, ), 0), buf413, reinterpret_tensor(buf410, (768, ), (1, ), 0), buf408, reinterpret_tensor(buf400, (2304, ), (1, ), 0), buf398, reinterpret_tensor(buf389, (768, ), (1, ), 0), buf387, reinterpret_tensor(buf379, (3072, ), (1, ), 0), buf377, reinterpret_tensor(buf374, (768, ), (1, ), 0), buf372, reinterpret_tensor(buf364, (2304, ), (1, ), 0), buf362, reinterpret_tensor(buf353, (768, ), (1, ), 0), buf351, reinterpret_tensor(buf343, (3072, ), (1, ), 0), buf341, reinterpret_tensor(buf338, (768, ), (1, ), 0), buf336, reinterpret_tensor(buf328, (2304, ), (1, ), 0), buf326, reinterpret_tensor(buf317, (768, ), (1, ), 0), buf315, reinterpret_tensor(buf307, (3072, ), (1, ), 0), buf305, reinterpret_tensor(buf302, (768, ), (1, ), 0), buf300, reinterpret_tensor(buf292, (2304, ), (1, ), 0), buf290, reinterpret_tensor(buf281, (768, ), (1, ), 0), buf279, reinterpret_tensor(buf271, (3072, ), (1, ), 0), buf269, reinterpret_tensor(buf266, (768, ), (1, ), 0), buf264, reinterpret_tensor(buf256, (2304, ), (1, ), 0), buf254, reinterpret_tensor(buf245, (768, ), (1, ), 0), buf243, reinterpret_tensor(buf235, (3072, ), (1, ), 0), buf233, reinterpret_tensor(buf230, (768, ), (1, ), 0), buf228, reinterpret_tensor(buf220, (2304, ), (1, ), 0), buf218, reinterpret_tensor(buf209, (768, ), (1, ), 0), buf207, reinterpret_tensor(buf199, (3072, ), (1, ), 0), buf197, reinterpret_tensor(buf194, (768, ), (1, ), 0), buf192, reinterpret_tensor(buf184, (2304, ), (1, ), 0), buf182, reinterpret_tensor(buf173, (768, ), (1, ), 0), buf171, reinterpret_tensor(buf163, (3072, ), (1, ), 0), buf161, reinterpret_tensor(buf158, (768, ), (1, ), 0), buf156, reinterpret_tensor(buf148, (2304, ), (1, ), 0), buf146, reinterpret_tensor(buf137, (768, ), (1, ), 0), buf135, reinterpret_tensor(buf127, (3072, ), (1, ), 0), buf125, reinterpret_tensor(buf122, (768, ), (1, ), 0), buf120, reinterpret_tensor(buf112, (2304, ), (1, ), 0), buf110, reinterpret_tensor(buf101, (768, ), (1, ), 0), buf99, reinterpret_tensor(buf91, (3072, ), (1, ), 0), buf89, reinterpret_tensor(buf86, (768, ), (1, ), 0), buf84, reinterpret_tensor(buf76, (2304, ), (1, ), 0), buf74, reinterpret_tensor(buf65, (768, ), (1, ), 0), buf63, reinterpret_tensor(buf55, (3072, ), (1, ), 0), buf53, reinterpret_tensor(buf50, (768, ), (1, ), 0), buf48, reinterpret_tensor(buf40, (2304, ), (1, ), 0), buf38, reinterpret_tensor(buf29, (768, ), (1, ), 0), buf27, reinterpret_tensor(buf19, (3072, ), (1, ), 0), buf17, reinterpret_tensor(buf14, (768, ), (1, ), 0), buf12, buf446, buf442, buf439, buf440, buf418, buf419, buf403, buf404, buf382, buf383, buf367, buf368, buf346, buf347, buf331, buf332, buf310, buf311, buf295, buf296, buf274, buf275, buf259, buf260, buf238, buf239, buf223, buf224, buf202, buf203, buf187, buf188, buf166, buf167, buf151, buf152, buf130, buf131, buf115, buf116, buf94, buf95, buf79, buf80, buf58, buf59, buf43, buf44, buf22, buf23, buf8, buf9, reinterpret_tensor(buf3, (2, 768), (768, 1), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, )


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
    primals_150 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_151 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_152 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_153 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_154 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_155 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_156 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_157 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_158 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_159 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_160 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    primals_161 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    view = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    view_1 = rand_strided((1, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    getitem_1 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_8 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    getitem_10 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_2 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_2 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh = rand_strided((1, 1024, 3072), (3145728, 3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_14 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_8 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_21 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    getitem_23 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_10 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_6 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_1 = rand_strided((1, 1024, 3072), (3145728, 3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_27 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_16 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_34 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    getitem_36 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_18 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_2 = rand_strided((1, 1024, 3072), (3145728, 3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_40 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_24 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_47 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    getitem_49 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_26 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_14 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_3 = rand_strided((1, 1024, 3072), (3145728, 3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_53 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_32 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_60 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    getitem_62 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_34 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_18 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_4 = rand_strided((1, 1024, 3072), (3145728, 3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_66 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_40 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_73 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    getitem_75 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_42 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_5 = rand_strided((1, 1024, 3072), (3145728, 3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_79 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_48 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_86 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    getitem_88 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_50 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_26 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_6 = rand_strided((1, 1024, 3072), (3145728, 3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_92 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_56 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_99 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    getitem_101 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_58 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_30 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_7 = rand_strided((1, 1024, 3072), (3145728, 3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_105 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_64 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_112 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    getitem_114 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_66 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_8 = rand_strided((1, 1024, 3072), (3145728, 3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_118 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_72 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_125 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    getitem_127 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_74 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_38 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_9 = rand_strided((1, 1024, 3072), (3145728, 3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_131 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_80 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_138 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    getitem_140 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_82 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_42 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_10 = rand_strided((1, 1024, 3072), (3145728, 3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_144 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_88 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_151 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    getitem_153 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_90 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    addmm_46 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    tanh_11 = rand_strided((1, 1024, 3072), (3145728, 3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_157 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_96 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    view_219 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    sub_37 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int64)
    full_default_24 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int64)
    permute_63 = rand_strided((2, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_24 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_65 = rand_strided((768, 3072), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_66 = rand_strided((3072, 1024), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_67 = rand_strided((3072, 768), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_68 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_25 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_69 = rand_strided((768, 768), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_70 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_72 = rand_strided((12, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_73 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    alias_25 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_74 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    permute_75 = rand_strided((12, 1024, 64), (64, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_80 = rand_strided((2304, 768), (1, 2304), device='cuda:0', dtype=torch.float32)
    permute_81 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_27 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_82 = rand_strided((768, 3072), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_83 = rand_strided((3072, 1024), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_84 = rand_strided((3072, 768), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_85 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_28 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_86 = rand_strided((768, 768), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_87 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_89 = rand_strided((12, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_90 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    alias_27 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_91 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    permute_92 = rand_strided((12, 1024, 64), (64, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_97 = rand_strided((2304, 768), (1, 2304), device='cuda:0', dtype=torch.float32)
    permute_98 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_30 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_99 = rand_strided((768, 3072), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_100 = rand_strided((3072, 1024), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_101 = rand_strided((3072, 768), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_102 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_31 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_103 = rand_strided((768, 768), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_104 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_106 = rand_strided((12, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_107 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    alias_29 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_108 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    permute_109 = rand_strided((12, 1024, 64), (64, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_114 = rand_strided((2304, 768), (1, 2304), device='cuda:0', dtype=torch.float32)
    permute_115 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_33 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_116 = rand_strided((768, 3072), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_117 = rand_strided((3072, 1024), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_118 = rand_strided((3072, 768), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_119 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_34 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_120 = rand_strided((768, 768), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_121 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_123 = rand_strided((12, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_124 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    alias_31 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_125 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    permute_126 = rand_strided((12, 1024, 64), (64, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_131 = rand_strided((2304, 768), (1, 2304), device='cuda:0', dtype=torch.float32)
    permute_132 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_36 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_133 = rand_strided((768, 3072), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_134 = rand_strided((3072, 1024), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_135 = rand_strided((3072, 768), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_136 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_37 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_137 = rand_strided((768, 768), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_138 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_140 = rand_strided((12, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_141 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    alias_33 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_142 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    permute_143 = rand_strided((12, 1024, 64), (64, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_148 = rand_strided((2304, 768), (1, 2304), device='cuda:0', dtype=torch.float32)
    permute_149 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_39 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_150 = rand_strided((768, 3072), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_151 = rand_strided((3072, 1024), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_152 = rand_strided((3072, 768), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_153 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_40 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_154 = rand_strided((768, 768), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_155 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_157 = rand_strided((12, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_158 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    alias_35 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_159 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    permute_160 = rand_strided((12, 1024, 64), (64, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_165 = rand_strided((2304, 768), (1, 2304), device='cuda:0', dtype=torch.float32)
    permute_166 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_42 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_167 = rand_strided((768, 3072), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_168 = rand_strided((3072, 1024), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_169 = rand_strided((3072, 768), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_170 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_43 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_171 = rand_strided((768, 768), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_172 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_174 = rand_strided((12, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_175 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    alias_37 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_176 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    permute_177 = rand_strided((12, 1024, 64), (64, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_182 = rand_strided((2304, 768), (1, 2304), device='cuda:0', dtype=torch.float32)
    permute_183 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_45 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_184 = rand_strided((768, 3072), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_185 = rand_strided((3072, 1024), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_186 = rand_strided((3072, 768), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_187 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_46 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_188 = rand_strided((768, 768), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_189 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_191 = rand_strided((12, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_192 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    alias_39 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_193 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    permute_194 = rand_strided((12, 1024, 64), (64, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_199 = rand_strided((2304, 768), (1, 2304), device='cuda:0', dtype=torch.float32)
    permute_200 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_48 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_201 = rand_strided((768, 3072), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_202 = rand_strided((3072, 1024), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_203 = rand_strided((3072, 768), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_204 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_49 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_205 = rand_strided((768, 768), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_206 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_208 = rand_strided((12, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_209 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    alias_41 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_210 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    permute_211 = rand_strided((12, 1024, 64), (64, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_216 = rand_strided((2304, 768), (1, 2304), device='cuda:0', dtype=torch.float32)
    permute_217 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_51 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_218 = rand_strided((768, 3072), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_219 = rand_strided((3072, 1024), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_220 = rand_strided((3072, 768), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_221 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_52 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_222 = rand_strided((768, 768), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_223 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_225 = rand_strided((12, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_226 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    alias_43 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_227 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    permute_228 = rand_strided((12, 1024, 64), (64, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_233 = rand_strided((2304, 768), (1, 2304), device='cuda:0', dtype=torch.float32)
    permute_234 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_54 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_235 = rand_strided((768, 3072), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_236 = rand_strided((3072, 1024), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_237 = rand_strided((3072, 768), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_238 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_55 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_239 = rand_strided((768, 768), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_240 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_242 = rand_strided((12, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_243 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    alias_45 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_244 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    permute_245 = rand_strided((12, 1024, 64), (64, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_250 = rand_strided((2304, 768), (1, 2304), device='cuda:0', dtype=torch.float32)
    permute_251 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_57 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_252 = rand_strided((768, 3072), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_253 = rand_strided((3072, 1024), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_254 = rand_strided((3072, 768), (1, 3072), device='cuda:0', dtype=torch.float32)
    permute_255 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_58 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_256 = rand_strided((768, 768), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_257 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    permute_259 = rand_strided((12, 1024, 1024), (1048576, 1, 1024), device='cuda:0', dtype=torch.float32)
    permute_260 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    alias_47 = rand_strided((1, 12, 1024, 1024), (12582912, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    permute_261 = rand_strided((12, 64, 1024), (64, 1, 2304), device='cuda:0', dtype=torch.float32)
    permute_262 = rand_strided((12, 1024, 64), (64, 2304, 1), device='cuda:0', dtype=torch.float32)
    permute_267 = rand_strided((2304, 768), (1, 2304), device='cuda:0', dtype=torch.float32)
    permute_268 = rand_strided((768, 1024), (1, 768), device='cuda:0', dtype=torch.float32)
    div_60 = rand_strided((1, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((1, 1024, 768), (786432, 768, 1), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_3 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_4 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_5 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_6 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_7 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_8 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_9 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_10 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_11 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_12 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_13 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_14 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_15 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_16 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_17 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_18 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_19 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_20 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_21 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_22 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_23 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_24 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_25 = rand_strided((1, 12, 1024, 64), (786432, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    tangents_26 = rand_strided((1, 2), (2, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, view, view_1, getitem_1, mul, getitem_8, getitem_10, mul_2, addmm_2, tanh, getitem_14, mul_8, getitem_21, getitem_23, mul_10, addmm_6, tanh_1, getitem_27, mul_16, getitem_34, getitem_36, mul_18, addmm_10, tanh_2, getitem_40, mul_24, getitem_47, getitem_49, mul_26, addmm_14, tanh_3, getitem_53, mul_32, getitem_60, getitem_62, mul_34, addmm_18, tanh_4, getitem_66, mul_40, getitem_73, getitem_75, mul_42, addmm_22, tanh_5, getitem_79, mul_48, getitem_86, getitem_88, mul_50, addmm_26, tanh_6, getitem_92, mul_56, getitem_99, getitem_101, mul_58, addmm_30, tanh_7, getitem_105, mul_64, getitem_112, getitem_114, mul_66, addmm_34, tanh_8, getitem_118, mul_72, getitem_125, getitem_127, mul_74, addmm_38, tanh_9, getitem_131, mul_80, getitem_138, getitem_140, mul_82, addmm_42, tanh_10, getitem_144, mul_88, getitem_151, getitem_153, mul_90, addmm_46, tanh_11, getitem_157, mul_96, view_219, sub_37, full_default_24, permute_63, div_24, permute_65, permute_66, permute_67, permute_68, div_25, permute_69, permute_70, permute_72, permute_73, alias_25, permute_74, permute_75, permute_80, permute_81, div_27, permute_82, permute_83, permute_84, permute_85, div_28, permute_86, permute_87, permute_89, permute_90, alias_27, permute_91, permute_92, permute_97, permute_98, div_30, permute_99, permute_100, permute_101, permute_102, div_31, permute_103, permute_104, permute_106, permute_107, alias_29, permute_108, permute_109, permute_114, permute_115, div_33, permute_116, permute_117, permute_118, permute_119, div_34, permute_120, permute_121, permute_123, permute_124, alias_31, permute_125, permute_126, permute_131, permute_132, div_36, permute_133, permute_134, permute_135, permute_136, div_37, permute_137, permute_138, permute_140, permute_141, alias_33, permute_142, permute_143, permute_148, permute_149, div_39, permute_150, permute_151, permute_152, permute_153, div_40, permute_154, permute_155, permute_157, permute_158, alias_35, permute_159, permute_160, permute_165, permute_166, div_42, permute_167, permute_168, permute_169, permute_170, div_43, permute_171, permute_172, permute_174, permute_175, alias_37, permute_176, permute_177, permute_182, permute_183, div_45, permute_184, permute_185, permute_186, permute_187, div_46, permute_188, permute_189, permute_191, permute_192, alias_39, permute_193, permute_194, permute_199, permute_200, div_48, permute_201, permute_202, permute_203, permute_204, div_49, permute_205, permute_206, permute_208, permute_209, alias_41, permute_210, permute_211, permute_216, permute_217, div_51, permute_218, permute_219, permute_220, permute_221, div_52, permute_222, permute_223, permute_225, permute_226, alias_43, permute_227, permute_228, permute_233, permute_234, div_54, permute_235, permute_236, permute_237, permute_238, div_55, permute_239, permute_240, permute_242, permute_243, alias_45, permute_244, permute_245, permute_250, permute_251, div_57, permute_252, permute_253, permute_254, permute_255, div_58, permute_256, permute_257, permute_259, permute_260, alias_47, permute_261, permute_262, permute_267, permute_268, div_60, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('GPT2ForSequenceClassification', benchmark_compiled_module)
