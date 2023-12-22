
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


# kernel path: /tmp/torchinductor_youkaichao/3n/c3ntxglseeqcqxfguqqwuhehgwi7qhqk2n3zdiguoi63i2gbbcpr.py
# Source Nodes: [], Original ATen: [aten.div]

triton_poi_fused_div_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 2.0
    tmp2 = tmp0 / tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/o4/co4vjtjqksra33vcpqh3r32ryzxgffsm3jb6gm3lzhuguxlcxyjn.py
# Source Nodes: [l__mod___head_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
# l__mod___head_bn => var_mean_62
triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0 + (384*r1)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (x0 + (384*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp18 = tl.full([XBLOCK, 1], 8, tl.int32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 / tmp19
    tmp21 = tmp11 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None]
    tmp27 = tmp10 - tmp20
    tmp28 = tmp5 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.where(rmask & xmask, tmp29, 0)
    tmp32 = tl.sum(tmp31, 1)[:, None]
    tmp33 = tmp0 * tmp27
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, RBLOCK])
    tmp36 = tl.where(rmask & xmask, tmp34, 0)
    tmp37 = tl.sum(tmp36, 1)[:, None]
    tmp38 = 8.0
    tmp39 = tmp26 / tmp38
    tmp40 = 1e-05
    tmp41 = tmp39 + tmp40
    tmp42 = tl.math.rsqrt(tmp41)
    tmp43 = tmp32 * tmp42
    tmp44 = tmp37 * tmp42
    tl.store(out_ptr6 + (x0), tmp43, xmask)
    tl.store(out_ptr7 + (x0), tmp44, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp9, xmask)
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr3 + (x0), tmp32, xmask)
    tl.store(out_ptr4 + (x0), tmp37, xmask)
    tl.store(out_ptr5 + (x0), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uo/cuopyso4ddnzoxzhz5zphcz43df6oqnycac6l6fmvxqyuxhf5lef.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_2', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/7m/c7mj424p32sdsawxg4qsdntnn6ziqgqmdbz2ml2mpripe7xtomqv.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]

triton_poi_fused_add_native_batch_norm_backward_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_3', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp24 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.125
    tmp6 = tmp4 * tmp5
    tmp8 = 8.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp12 * tmp12
    tmp14 = tmp6 * tmp13
    tmp15 = tmp3 * tmp14
    tmp16 = tmp0 - tmp15
    tmp18 = tmp17 * tmp5
    tmp19 = tmp16 - tmp18
    tmp21 = tmp12 * tmp20
    tmp22 = tmp19 * tmp21
    tmp25 = tmp24 * tmp5
    tmp26 = tmp25 * tmp13
    tmp27 = tmp3 * tmp26
    tmp28 = tmp23 - tmp27
    tmp30 = tmp29 * tmp5
    tmp31 = tmp28 - tmp30
    tmp33 = tmp12 * tmp32
    tmp34 = tmp31 * tmp33
    tmp35 = tmp22 + tmp34
    tl.store(in_out_ptr0 + (x2), tmp35, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z5/cz5bcyksp6h7aogzc5hwrfy7a7ga4dp4vamd67iqsvgen2s7vfl3.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp7 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*(r1 // 16))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr1 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 16.0
        tmp2 = tmp0 / tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp11, xmask)
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tmp11 * tmp13
    tl.store(out_ptr2 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ri/criyyiaf3dit2cjmzqnhcpz44hqu4quyyyvnvfhiwkyqedshlmlj.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 384
    x1 = (xindex // 384)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*(x1 // 16))), None)
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = 16.0
    tmp2 = tmp0 / tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.0078125
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(out_ptr0 + (x2), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/uy/cuyled4xmbuafosbis3mqrdognsggs6f4rzmnicsqdpsmw5kjdrm.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp17 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr2 + (x0 + (768*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = tmp0 / tmp3
        tmp7 = 0.5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.where(tmp4, tmp9, tmp5)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, tmp10)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
        tmp18 = tmp16 - tmp17
        tmp19 = tmp12 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask & xmask, tmp22, _tmp21)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp14, xmask)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp21, xmask)
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tmp21 * tmp23
    tl.store(out_ptr2 + (x0), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a7/ca73uo4qsoh555sjdunmzygawwjbwe3xvw3t7nzmvbvgbmzrudrb.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp5 = tl.load(in_out_ptr0 + (x2), None)
    tmp13 = tl.load(in_ptr1 + (x2), None)
    tmp14 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = tmp0 / tmp3
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp4, tmp9, tmp5)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp11, tmp10)
    tmp15 = tmp13 - tmp14
    tmp17 = 0.0078125
    tmp18 = tmp16 * tmp17
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 * tmp20
    tmp22 = tmp15 * tmp21
    tmp23 = tmp12 - tmp22
    tmp25 = tmp24 * tmp17
    tmp26 = tmp23 - tmp25
    tmp28 = tmp19 * tmp27
    tmp29 = tmp26 * tmp28
    tl.store(in_out_ptr0 + (x2), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6s/c6sbs4tbv6u5ffino6saenkin6y5d5hd7zbl7bzc2iphwdvfarwm.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*(r1 // 16))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 16.0
        tmp2 = tmp0 / tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ti/ctitkdikstudwhzf6ojswlo3cfnne6dscoeg36illr2jgmvhhyic.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 384
    x1 = (xindex // 384)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*(x1 // 16))), None)
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp5 = tl.load(in_ptr2 + (x2), None)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp1 = 16.0
    tmp2 = tmp0 / tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0078125
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/z4/cz4vrvphhdrrxwvxtynjzzb5igcx2nznfsntqehnt2sodggz6bku.py
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
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x0 = xindex % 32
    x1 = (xindex // 32) % 12
    x2 = (xindex // 384) % 16
    x3 = (xindex // 6144)
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp5 = tl.load(in_ptr1 + (x4), None)
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = tmp0 / tmp3
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp4, tmp9, tmp5)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp11, tmp10)
    tl.store(out_ptr0 + (x0 + (32*x2) + (512*x1) + (6144*x3)), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6y/c6ytq5gkeouyuikaphvfyhans5w2mnvlr47s3wjat2awcn3nnqak.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]

triton_per_fused__softmax_backward_data_mul_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_mul_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (16*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (16*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tmp1 * tmp6
    tmp8 = tmp2 - tmp7
    tmp9 = 0.25
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr1 + (r1 + (16*x0)), tmp10, rmask & xmask)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lc/clc2sz3a44eflnqwbcnexltspntb7s27jkqiwcycoe77v3dyw3ob.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]

triton_per_fused__softmax_backward_data_sum_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_sum_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x1 = (xindex // 16)
    tmp0 = tl.load(in_ptr0 + (x3 + (3072*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x3 + (3072*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x1 + (192*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp1 * tmp3
    tmp5 = tmp2 - tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bd/cbdylmdienjffbnop2nzoj7lqy47osgv4polmd26ih3n4oa2s5lr.py
# Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]

triton_poi_fused_index_put_new_zeros_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_new_zeros_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/by/cbyplry4yix7co64kiclczjd5tgxm5m6tq4tdojm65adpjpffnur.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp27 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp31 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp26 = tl.load(in_ptr3 + (x0 + (768*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp0 = x0 % 64
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 16, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + ((16*(r1 % 16)) + (256*(x0 // 64)) + (3072*(r1 // 16)) + (x0 % 64)), rmask & tmp4 & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
        tmp7 = tl.where(tmp4, tmp5, tmp6)
        tmp8 = tmp0 >= tmp3
        tmp9 = tl.full([1, 1], 32, tl.int64)
        tmp10 = tmp0 < tmp9
        tmp11 = tmp8 & tmp10
        tmp12 = tl.load(in_ptr1 + ((-256) + (16*(x0 % 64)) + (256*(x0 // 64)) + (3072*(r1 // 16)) + (r1 % 16)), rmask & tmp11 & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
        tmp14 = tl.where(tmp11, tmp12, tmp13)
        tmp15 = tmp0 >= tmp9
        tmp16 = tl.full([1, 1], 64, tl.int64)
        tmp17 = tmp0 < tmp16
        tmp18 = tl.load(in_ptr2 + ((-32) + (32*(r1 % 16)) + (512*(x0 // 64)) + (6144*(r1 // 16)) + (x0 % 64)), rmask & tmp15 & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
        tmp20 = tl.where(tmp15, tmp18, tmp19)
        tmp21 = tl.where(tmp11, tmp14, tmp20)
        tmp22 = tl.where(tmp4, tmp7, tmp21)
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
        tmp28 = tmp26 - tmp27
        tmp29 = tmp22 * tmp28
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp32 = _tmp31 + tmp30
        _tmp31 = tl.where(rmask & xmask, tmp32, _tmp31)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp24, xmask)
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp31, xmask)
    tmp33 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp34 = tmp31 * tmp33
    tl.store(out_ptr2 + (x0), tmp34, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kw/ckw4teurvnqfflvdy2evg4l7qlzzppcv5mmqmbwqy2ljqi5smeah.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp23 = tl.load(in_ptr3 + (x1 + (768*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1 % 64
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((16*(y0 % 16)) + (256*(x1 // 64)) + (3072*(y0 // 16)) + (x1 % 64)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 32, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-256) + (16*(x1 % 64)) + (256*(x1 // 64)) + (3072*(y0 // 16)) + (y0 % 16)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 64, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-32) + (32*(y0 % 16)) + (512*(x1 // 64)) + (6144*(y0 // 16)) + (x1 % 64)), tmp15 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tl.where(tmp11, tmp14, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tmp25 = tmp23 - tmp24
    tmp27 = 0.0078125
    tmp28 = tmp26 * tmp27
    tmp30 = tmp29 * tmp29
    tmp31 = tmp28 * tmp30
    tmp32 = tmp25 * tmp31
    tmp33 = tmp22 - tmp32
    tmp35 = tmp34 * tmp27
    tmp36 = tmp33 - tmp35
    tmp38 = tmp29 * tmp37
    tmp39 = tmp36 * tmp38
    tl.store(out_ptr0 + (x1 + (768*y0)), tmp39, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jo/cjovnfpcloatfuibbzdoi4ygpunkov43uialqaxfabobe34exh4k.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*(r1 // 16))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr3 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 16.0
        tmp2 = tmp0 / tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp6 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp15, xmask)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tmp15 * tmp17
    tl.store(out_ptr2 + (x0), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ac/cackwtieulihnrhmohb6rzn3v3kidr5xim4fiasuscqf4uinfylx.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 384
    x1 = (xindex // 384)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*(x1 // 16))), None)
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp5 = tl.load(in_ptr2 + (x2), None)
    tmp7 = tl.load(in_ptr3 + (x2), None)
    tmp8 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp1 = 16.0
    tmp2 = tmp0 / tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp9 = tmp7 - tmp8
    tmp11 = 0.0078125
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(out_ptr0 + (x2), tmp23, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fx/cfxmn2m33itcwapadkffnpzdqucy36ocmg7usc7myi7x4n7x2be4.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp13 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*(r1 // 16))), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.load(in_ptr3 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr4 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 16.0
        tmp2 = tmp0 / tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 + tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
        tmp14 = tmp12 - tmp13
        tmp15 = tmp8 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp17, xmask)
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tmp17 * tmp19
    tl.store(out_ptr2 + (x0), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2h/c2hyhia22tzeeioi5hs5cvjxup65jbvb3ynpc3y5yp44ntctbkkv.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 384
    x1 = (xindex // 384)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*(x1 // 16))), None)
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp5 = tl.load(in_ptr2 + (x2), None)
    tmp7 = tl.load(in_ptr3 + (x2), None)
    tmp9 = tl.load(in_ptr4 + (x2), None)
    tmp10 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp1 = 16.0
    tmp2 = tmp0 / tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp11 = tmp9 - tmp10
    tmp13 = 0.0078125
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15 * tmp15
    tmp17 = tmp14 * tmp16
    tmp18 = tmp11 * tmp17
    tmp19 = tmp8 - tmp18
    tmp21 = tmp20 * tmp13
    tmp22 = tmp19 - tmp21
    tmp24 = tmp15 * tmp23
    tmp25 = tmp22 * tmp24
    tl.store(in_out_ptr0 + (x2), tmp25, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5o/c5o2whhs4pxnu3bhqi6cif7ituabzplawtnhf4lnacjxmemx2vyx.py
# Source Nodes: [], Original ATen: [aten.add, aten.div]

triton_poi_fused_add_div_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_20', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 384
    x2 = (xindex // 6144)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp7 = tl.load(in_ptr2 + (x3), None)
    tmp9 = tl.load(in_ptr3 + (x3), None)
    tmp1 = 16.0
    tmp2 = tmp0 / tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tl.store(in_out_ptr0 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xz/cxz5bwx3a7gqf67nqiw2i6tbynucputyxv7dcvlaxqvwdfxzmqkh.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_21', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp9, xmask)
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tmp9 * tmp11
    tl.store(out_ptr2 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ze/cze3fulntlasdwbj6zyt62rtipkdhe7ysmbu5hqgp6hxq6urildd.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.0078125
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/my/cmyshl4hcvc2qrtrhjt2aeksi7qqiunlwtgwgnwhjbjirjjhdhts.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp11, xmask)
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tmp11 * tmp13
    tl.store(out_ptr2 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sm/csmksvwq47ecapkzian2c4qamige3dz5y3e2tlfkcqhdjeve7xyx.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.0078125
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(out_ptr0 + (x2), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ov/covgod3jdzgqfot6jvcxasfoz3r2hrnheg2jkjccz5fjemvucpac.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp13, xmask)
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp13 * tmp15
    tl.store(out_ptr2 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ex/cex4seoxs6xwwb77p5lgqqhf3r4mjzqebgnu73jzcqo7e2qnpv47.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp5 = tl.load(in_ptr3 + (x2), None)
    tmp6 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0078125
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2i/c2id7iazflhcvhwispr3q3ldy35obi4yobsfs5mvpsmypamqbwlj.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp11 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr3 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr4 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp6 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp15, xmask)
    tmp17 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tmp15 * tmp17
    tl.store(out_ptr2 + (x0), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bx/cbxbgbwx57s4j2rzxdcgfnzkyad2aixcdkkuy7atc3745vc2tzkv.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_28', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp5 = tl.load(in_ptr3 + (x2), None)
    tmp7 = tl.load(in_ptr4 + (x2), None)
    tmp8 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp9 = tmp7 - tmp8
    tmp11 = 0.0078125
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(in_out_ptr0 + (x2), tmp23, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ey/cey5aqfvfe5rlcnsr3i24kbskosg57dnux36caaztj6fmvk7e7o4.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_29', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp3 = tl.load(in_out_ptr0 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp7 = tl.load(in_ptr3 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/n6/cn6gj7hinc25gqnnvj5rraqxne5e7mstqbdemaq2mig2azq7xhto.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_30', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.0078125
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/uf/cuffztn7xeva72eqgj5j3q7vwkg3acxvejkppx4lrox46f6edfge.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64) % 16
    x2 = (xindex // 1024) % 16
    x3 = (xindex // 16384)
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp5 = tl.load(in_ptr1 + (x4), None)
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = tmp0 / tmp3
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp4, tmp9, tmp5)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp11, tmp10)
    tl.store(out_ptr0 + (x0 + (64*x2) + (1024*x1) + (16384*x3)), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tz/ctz4ibyhcht374p2dg3pk6kj3yvxzpm6iukwi5qiatnrlviuwcie.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]

triton_per_fused__softmax_backward_data_mul_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_mul_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (49*x0)), rmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tmp1 * tmp6
    tmp8 = tmp2 - tmp7
    tmp9 = 0.25
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr1 + (r1 + (49*x0)), tmp10, rmask)
    tl.store(out_ptr0 + (x0), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/d3/cd36bgnp6qvc5lj4uqwi25sbg5z4gbcwk75k6e7ftpp7h2325fsb.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]

triton_per_fused__softmax_backward_data_sum_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_sum_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x1 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (x3 + (12544*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x3 + (12544*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x1 + (256*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp1 * tmp3
    tmp5 = tmp2 - tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z3/cz35jpq4kguyahb4frkedglgajneddswm3jhcmmtr2myfstlvz4f.py
# Source Nodes: [], Original ATen: [aten.new_zeros]

triton_poi_fused_new_zeros_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_zeros_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ob/cobbw6cewf66ho3a6sqvafe2rbrxlefuiec2iafjcrvcf3y6f4hq.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + ((16*(r1 % 16)) + (256*(x0 // 16)) + (4096*(r1 // 16)) + (x0 % 16)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (256*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp9, xmask)
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tmp9 * tmp11
    tl.store(out_ptr2 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cn/ccnia4kjxuh3sqme6ld6smvps5himtnt75o6ah24k5ywzfunlebt.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((16*(x1 % 16)) + (256*(x0 // 16)) + (4096*(x1 // 16)) + (x0 % 16)), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.0078125
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zx/czxtfbapfzxtscesl7kbyrfh5xuetqghqg3z6qrs7gegmajzcxws.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_37', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5120
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1280
    x1 = (xindex // 1280)
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp19 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp18 = tl.load(in_ptr2 + (x0 + (1280*r2) + (125440*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp0 = x0 % 80
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 16, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + ((49*(x0 % 80)) + (784*(x0 // 80)) + (12544*(r2 // 49)) + (25088*x1) + (r2 % 49)), rmask & tmp4 & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
        tmp7 = tl.where(tmp4, tmp5, tmp6)
        tmp8 = tmp0 >= tmp3
        tmp9 = tl.full([1, 1], 80, tl.int64)
        tmp10 = tmp0 < tmp9
        tmp11 = tl.load(in_ptr1 + ((-16) + (64*(r2 % 49)) + (3136*(x0 // 80)) + (50176*(r2 // 49)) + (100352*x1) + (x0 % 80)), rmask & tmp8 & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
        tmp13 = tl.where(tmp8, tmp11, tmp12)
        tmp14 = tl.where(tmp4, tmp7, tmp13)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
        tmp20 = tmp18 - tmp19
        tmp21 = tmp14 * tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp16, xmask)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2x/c2xlony2werei3feccpb53rbsgojoeougc2wwwutqagcefxsb3op.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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


# kernel path: /tmp/torchinductor_youkaichao/qa/cqalsyqnrqldxmn25gag3da75p47cswqjb2fp3jzs44ukukxoqpo.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1280*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q4/cq4afngkssrdod2fyzaqssknbizonv63ju53khgxl64g5yeaqisi.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 1280
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp15 = tl.load(in_ptr2 + (x1 + (1280*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1 % 80
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((49*(x1 % 80)) + (784*(x1 // 80)) + (12544*(y0 // 49)) + (y0 % 49)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 80, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-16) + (64*(y0 % 49)) + (3136*(x1 // 80)) + (50176*(y0 // 49)) + (x1 % 80)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp17 = tmp15 - tmp16
    tmp19 = 0.002551020408163265
    tmp20 = tmp18 * tmp19
    tmp22 = tmp21 * tmp21
    tmp23 = tmp20 * tmp22
    tmp24 = tmp17 * tmp23
    tmp25 = tmp14 - tmp24
    tmp27 = tmp26 * tmp19
    tmp28 = tmp25 - tmp27
    tmp30 = tmp21 * tmp29
    tmp31 = tmp28 * tmp30
    tl.store(out_ptr0 + (x1 + (1280*y0)), tmp31, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6m/c6mrhovg6zvyokt6hy33i2mr35wcvsgmwnpvdncaltcig5lxvz34.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp20 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp14 = tl.load(in_ptr1 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr2 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp0 = ((r2 % 49) // 7) % 2
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 == tmp1
        tmp3 = tl.broadcast_to(((r2 % 49) % 7) % 2, [XBLOCK, RBLOCK])
        tmp4 = tmp3 == tmp1
        tmp5 = tmp4 & tmp2
        tmp6 = tl.load(in_ptr0 + (x0 + (256*(((r2 % 49) % 7) // 2)) + (1024*((r2 % 49) // 14)) + (4096*(r2 // 49)) + (8192*x1)), rmask & tmp5 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = 0.0
        tmp10 = tl.where(tmp4, tmp8, tmp9)
        tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.where(tmp2, tmp12, tmp9)
        tmp15 = tmp13 + tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
        tmp21 = tmp19 - tmp20
        tmp22 = tmp15 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask & xmask, tmp25, _tmp24)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kl/cklpbtdnp26jsknzxaaici4uhhectk6uz35scpns2py2aqi44fsl.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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


# kernel path: /tmp/torchinductor_youkaichao/me/cme2jk7ekc2kysfkrbxqda4mnck6rul7s5li7ukqc7eonn2og73r.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jj/cjjpp5lckgsy32xrr4tanoxcajnysozpiulmhvax5gcnnrlxod4p.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 256)
    x0 = xindex % 256
    x2 = xindex
    tmp14 = tl.load(in_ptr1 + (x2), None)
    tmp16 = tl.load(in_ptr2 + (x2), None)
    tmp17 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp0 = ((x1 % 49) // 7) % 2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp3 = ((x1 % 49) % 7) % 2
    tmp4 = tmp3 == tmp1
    tmp5 = tmp4 & tmp2
    tmp6 = tl.load(in_ptr0 + (x0 + (256*(((x1 % 49) % 7) // 2)) + (1024*((x1 % 49) // 14)) + (4096*(x1 // 49))), tmp5, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = 0.0
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp2, tmp10, tmp11)
    tmp13 = tl.where(tmp2, tmp12, tmp9)
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 0.002551020408163265
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tmp31 = tmp22 * tmp30
    tmp32 = tmp29 * tmp31
    tl.store(out_ptr0 + (x2), tmp32, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/a4/ca4x62rrugqrbt47nujlg4heia7xrflzpcugj6q7p3p6ynj26xhg.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp17 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (50176*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr1 + (x0 + (512*r2) + (50176*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr2 + (x0 + (512*r2) + (50176*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = tmp0 / tmp3
        tmp7 = 0.5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.where(tmp4, tmp9, tmp5)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, tmp10)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask, tmp15, _tmp14)
        tmp18 = tmp16 - tmp17
        tmp19 = tmp12 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask, tmp22, _tmp21)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, None)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hm/chm3hc4kd4lwdmdmo6y7lmbkt3aiddvtx6ymuq7coxlao27sqeht.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_46', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/iz/cizy4qdjb7qfhlcmoiuql7rgsskqtsxzqmhji2n2wa73grsgogas.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cg/ccgxx77z4j2f7mc55bltmieg32lw5lsiwedvdwcxorofq2wsgysd.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_48', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp5 = tl.load(in_out_ptr0 + (x2), None)
    tmp13 = tl.load(in_ptr1 + (x2), None)
    tmp14 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = tmp0 / tmp3
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp4, tmp9, tmp5)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp11, tmp10)
    tmp15 = tmp13 - tmp14
    tmp17 = 0.002551020408163265
    tmp18 = tmp16 * tmp17
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 * tmp20
    tmp22 = tmp15 * tmp21
    tmp23 = tmp12 - tmp22
    tmp25 = tmp24 * tmp17
    tmp26 = tmp23 - tmp25
    tmp28 = tmp19 * tmp27
    tmp29 = tmp26 * tmp28
    tl.store(in_out_ptr0 + (x2), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dk/cdkianv26f7hl5obludh3nzhodehmkeibitg5rzt5dvalazi6qfm.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp22 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp14 = tl.load(in_ptr1 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr2 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr3 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp0 = ((r2 % 49) // 7) % 2
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 == tmp1
        tmp3 = tl.broadcast_to(((r2 % 49) % 7) % 2, [XBLOCK, RBLOCK])
        tmp4 = tmp3 == tmp1
        tmp5 = tmp4 & tmp2
        tmp6 = tl.load(in_ptr0 + (x0 + (256*(((r2 % 49) % 7) // 2)) + (1024*((r2 % 49) // 14)) + (4096*(r2 // 49)) + (8192*x1)), rmask & tmp5 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = 0.0
        tmp10 = tl.where(tmp4, tmp8, tmp9)
        tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.where(tmp2, tmp12, tmp9)
        tmp15 = tmp13 + tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
        tmp23 = tmp21 - tmp22
        tmp24 = tmp17 * tmp23
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp19, xmask)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5x/c5xz7ocwatqw6sesjtbj4fauhwwtuzetesncwj4esn3dxqhux54w.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 256)
    x0 = xindex % 256
    x2 = xindex
    tmp14 = tl.load(in_ptr1 + (x2), None)
    tmp16 = tl.load(in_ptr2 + (x2), None)
    tmp18 = tl.load(in_ptr3 + (x2), None)
    tmp19 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp0 = ((x1 % 49) // 7) % 2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp3 = ((x1 % 49) % 7) % 2
    tmp4 = tmp3 == tmp1
    tmp5 = tmp4 & tmp2
    tmp6 = tl.load(in_ptr0 + (x0 + (256*(((x1 % 49) % 7) // 2)) + (1024*((x1 % 49) // 14)) + (4096*(x1 // 49))), tmp5, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = 0.0
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp2, tmp10, tmp11)
    tmp13 = tl.where(tmp2, tmp12, tmp9)
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = 0.002551020408163265
    tmp23 = tmp21 * tmp22
    tmp25 = tmp24 * tmp24
    tmp26 = tmp23 * tmp25
    tmp27 = tmp20 * tmp26
    tmp28 = tmp17 - tmp27
    tmp30 = tmp29 * tmp22
    tmp31 = tmp28 - tmp30
    tmp33 = tmp24 * tmp32
    tmp34 = tmp31 * tmp33
    tl.store(out_ptr0 + (x2), tmp34, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ue/cueytxs4p5t5vvoxigqv6a6ex36bdkkuxrvsi3xtshjjkojvn5pw.py
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
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x0 = xindex % 32
    x1 = (xindex // 32) % 8
    x2 = (xindex // 256) % 49
    x3 = (xindex // 12544)
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp5 = tl.load(in_ptr1 + (x4), None)
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = tmp0 / tmp3
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp4, tmp9, tmp5)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp11, tmp10)
    tl.store(out_ptr0 + (x0 + (32*x2) + (1568*x1) + (12544*x3)), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lp/clpjlp6agkmexkcfecbm3k52xkl6daraff6534jzxrccx62zt3qj.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]

triton_per_fused__softmax_backward_data_mul_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_mul_52', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3136
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
    tmp1 = tl.load(in_ptr1 + (r1 + (49*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tmp1 * tmp6
    tmp8 = tmp2 - tmp7
    tmp9 = 0.25
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr1 + (r1 + (49*x0)), tmp10, rmask & xmask)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ye/cye4od4agptnoxjfoktlwbjsuw7mlytllts6rcg337apo24l4wpu.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]

triton_per_fused__softmax_backward_data_sum_53 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_sum_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 19208
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x1 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (x3 + (19208*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x3 + (19208*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x1 + (392*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp1 * tmp3
    tmp5 = tmp2 - tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4e/c4efjdrgbztrnnkcmlsdcxvrum2wqkpggw2d4kzxw2nun5yczg5g.py
# Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]

triton_poi_fused_index_put_new_zeros_54 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0,), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_new_zeros_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mf/cmfchvnxazolg4d7ddaqt4bhlytq6p26qvygyjoa6npgkgmsklol.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_55 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_55', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp27 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp31 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp26 = tl.load(in_ptr3 + (x0 + (512*r2) + (50176*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp0 = x0 % 64
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 16, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + ((16*(r2 % 49)) + (784*(x0 // 64)) + (6272*(r2 // 49)) + (12544*x1) + (x0 % 64)), rmask & tmp4, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
        tmp7 = tl.where(tmp4, tmp5, tmp6)
        tmp8 = tmp0 >= tmp3
        tmp9 = tl.full([1, 1], 32, tl.int64)
        tmp10 = tmp0 < tmp9
        tmp11 = tmp8 & tmp10
        tmp12 = tl.load(in_ptr1 + ((-784) + (49*(x0 % 64)) + (784*(x0 // 64)) + (6272*(r2 // 49)) + (12544*x1) + (r2 % 49)), rmask & tmp11, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
        tmp14 = tl.where(tmp11, tmp12, tmp13)
        tmp15 = tmp0 >= tmp9
        tmp16 = tl.full([1, 1], 64, tl.int64)
        tmp17 = tmp0 < tmp16
        tmp18 = tl.load(in_ptr2 + ((-32) + (32*(r2 % 49)) + (1568*(x0 // 64)) + (12544*(r2 // 49)) + (25088*x1) + (x0 % 64)), rmask & tmp15, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
        tmp20 = tl.where(tmp15, tmp18, tmp19)
        tmp21 = tl.where(tmp11, tmp14, tmp20)
        tmp22 = tl.where(tmp4, tmp7, tmp21)
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask, tmp25, _tmp24)
        tmp28 = tmp26 - tmp27
        tmp29 = tmp22 * tmp28
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp32 = _tmp31 + tmp30
        _tmp31 = tl.where(rmask, tmp32, _tmp31)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp24, None)
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp31, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wa/cwa2b4xovtanqo33cq7tggolxcl6fvpsxjyjxu6rafl6cd3lmea5.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_56 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_56', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp23 = tl.load(in_ptr3 + (x1 + (512*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1 % 64
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((16*(y0 % 49)) + (784*(x1 // 64)) + (6272*(y0 // 49)) + (x1 % 64)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 32, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-784) + (49*(x1 % 64)) + (784*(x1 // 64)) + (6272*(y0 // 49)) + (y0 % 49)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 64, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-32) + (32*(y0 % 49)) + (1568*(x1 // 64)) + (12544*(y0 // 49)) + (x1 % 64)), tmp15 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tl.where(tmp11, tmp14, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tmp25 = tmp23 - tmp24
    tmp27 = 0.002551020408163265
    tmp28 = tmp26 * tmp27
    tmp30 = tmp29 * tmp29
    tmp31 = tmp28 * tmp30
    tmp32 = tmp25 * tmp31
    tmp33 = tmp22 - tmp32
    tmp35 = tmp34 * tmp27
    tmp36 = tmp33 - tmp35
    tmp38 = tmp29 * tmp37
    tmp39 = tmp36 * tmp38
    tl.store(out_ptr0 + (x1 + (512*y0)), tmp39, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wh/cwhopvvtn3fhnbfzgk26lybq4iwrdithlwn6nvf3ycderoz6b26n.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_57 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_57', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp24 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp28 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp14 = tl.load(in_ptr1 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr2 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr3 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tl.load(in_ptr4 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp0 = ((r2 % 49) // 7) % 2
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 == tmp1
        tmp3 = tl.broadcast_to(((r2 % 49) % 7) % 2, [XBLOCK, RBLOCK])
        tmp4 = tmp3 == tmp1
        tmp5 = tmp4 & tmp2
        tmp6 = tl.load(in_ptr0 + (x0 + (256*(((r2 % 49) % 7) // 2)) + (1024*((r2 % 49) // 14)) + (4096*(r2 // 49)) + (8192*x1)), rmask & tmp5 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = 0.0
        tmp10 = tl.where(tmp4, tmp8, tmp9)
        tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
        tmp12 = tl.where(tmp2, tmp10, tmp11)
        tmp13 = tl.where(tmp2, tmp12, tmp9)
        tmp15 = tmp13 + tmp14
        tmp17 = tmp15 + tmp16
        tmp19 = tmp17 + tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask & xmask, tmp22, _tmp21)
        tmp25 = tmp23 - tmp24
        tmp26 = tmp19 * tmp25
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(rmask & xmask, tmp29, _tmp28)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp21, xmask)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d3/cd3d7ahpqeiolt5snal3qxrlde7va5zqootgggp5vncmlbjz2dck.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_58 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_58', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 256)
    x0 = xindex % 256
    x2 = xindex
    tmp14 = tl.load(in_ptr1 + (x2), None)
    tmp16 = tl.load(in_ptr2 + (x2), None)
    tmp18 = tl.load(in_ptr3 + (x2), None)
    tmp20 = tl.load(in_ptr4 + (x2), None)
    tmp21 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp0 = ((x1 % 49) // 7) % 2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp3 = ((x1 % 49) % 7) % 2
    tmp4 = tmp3 == tmp1
    tmp5 = tmp4 & tmp2
    tmp6 = tl.load(in_ptr0 + (x0 + (256*(((x1 % 49) % 7) // 2)) + (1024*((x1 % 49) // 14)) + (4096*(x1 // 49))), tmp5, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = 0.0
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp2, tmp10, tmp11)
    tmp13 = tl.where(tmp2, tmp12, tmp9)
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp19 = tmp17 + tmp18
    tmp22 = tmp20 - tmp21
    tmp24 = 0.002551020408163265
    tmp25 = tmp23 * tmp24
    tmp27 = tmp26 * tmp26
    tmp28 = tmp25 * tmp27
    tmp29 = tmp22 * tmp28
    tmp30 = tmp19 - tmp29
    tmp32 = tmp31 * tmp24
    tmp33 = tmp30 - tmp32
    tmp35 = tmp26 * tmp34
    tmp36 = tmp33 * tmp35
    tl.store(in_out_ptr0 + (x2), tmp36, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jm/cjmgzenqb4wabtg3ndydty74fpjaqassalflx3erlqvm2lksrrev.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_59 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_59', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 256) % 49
    x0 = xindex % 256
    x2 = (xindex // 12544)
    x3 = xindex
    tmp14 = tl.load(in_out_ptr0 + (x3), None)
    tmp16 = tl.load(in_ptr1 + (x3), None)
    tmp18 = tl.load(in_ptr2 + (x3), None)
    tmp20 = tl.load(in_ptr3 + (x3), None)
    tmp0 = (x1 // 7) % 2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp3 = (x1 % 7) % 2
    tmp4 = tmp3 == tmp1
    tmp5 = tmp4 & tmp2
    tmp6 = tl.load(in_ptr0 + (x0 + (256*((x1 % 7) // 2)) + (1024*(x1 // 14)) + (4096*x2)), tmp5, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = 0.0
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp2, tmp10, tmp11)
    tmp13 = tl.where(tmp2, tmp12, tmp9)
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp19 = tmp17 + tmp18
    tmp21 = tmp19 + tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/a7/ca7nixswseams43tykxtajpynuhxwl7sfx7yiuu7g3hfhglckyjk.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_60 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_60', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oe/coeopfxxgxcia44rmsw533ya527dwzp4lpoeydcynmaowymjvd3q.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_61 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_61', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.002551020408163265
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gu/cgufhncp5umlr6c5wywoepg74evqto44qsvetslibvarsxfja3uc.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_62 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_62', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp7 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tmp8 = tmp6 - tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/by/cbysibfm2zhbxy4qqydvvouxdrypgntrnjz3qmye7atk4t4hop7c.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_63 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_63', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.002551020408163265
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(out_ptr0 + (x2), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/j5/cj5y2qlxagftowquwvbry47cbl3lgxlzhzk4naoevqztegdn5uw2.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_64 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_64', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp9 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr3 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 - tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bm/cbmt4meqypdsf3obmghvjilqul6h3soik5y7sqncjx76bh5rp64l.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_65 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_65', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp5 = tl.load(in_ptr3 + (x2), None)
    tmp6 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 - tmp6
    tmp9 = 0.002551020408163265
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/re/cre3srcmqylurrvpppbwgyligdn6ermzkmnieowzofyzq3uxgphk.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_66 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_66', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp11 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr3 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr4 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
        tmp12 = tmp10 - tmp11
        tmp13 = tmp6 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ll/cllpp3u52bnn6w25lq5ugmgsqciu5v6bkgouw7xfyhcwlqyi4uol.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_67 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_67', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp5 = tl.load(in_ptr3 + (x2), None)
    tmp7 = tl.load(in_ptr4 + (x2), None)
    tmp8 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp9 = tmp7 - tmp8
    tmp11 = 0.002551020408163265
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(in_out_ptr0 + (x2), tmp23, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vj/cvjjo4uf7fwh3b7jlz6bfmfeqoikr7bx7b323nkucaqe6xlhjrox.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_68 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_68', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp7 = tl.load(in_ptr3 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gv/cgv4z5povs657q3olnea7w64o5ej3wlwj7eauzqvwtsl5fg2eypp.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_69 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_69', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp5 = tl.load(in_ptr2 + (x2), None)
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 - tmp6
    tmp9 = 0.002551020408163265
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x2), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rw/crwx24tngyjhpeg7zfybyaaevjcvh732nzc7ayt57y5bems6jefe.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_70 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_70', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64) % 8
    x2 = (xindex // 512) % 49
    x3 = (xindex // 25088)
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp5 = tl.load(in_ptr1 + (x4), None)
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = tmp0 / tmp3
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp4, tmp9, tmp5)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp11, tmp10)
    tl.store(out_ptr0 + (x0 + (64*x2) + (3136*x1) + (25088*x3)), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wx/cwxxtsrrmjkyukpzozrldb4szcvpxlqeq7l7x5nkuow7hprdueud.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]

triton_per_fused__softmax_backward_data_mul_71 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_mul_71', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3136
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
    tmp9 = 0.25
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr1 + (r1 + (196*x0)), tmp10, rmask & xmask)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4e/c4ev5xapcfmvihhpqiv22pxrjoirhox2ityvikyqdlejhwk7evah.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]

triton_per_fused__softmax_backward_data_sum_72 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_sum_72', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 76832
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (x3 + (76832*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x3 + (76832*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x1 + (392*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp1 * tmp3
    tmp5 = tmp2 - tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u4/cu4sr7fj6ulyzohmaoly6feykswrwm2uwuas23em2upsw4fpqqaw.py
# Source Nodes: [], Original ATen: [aten.new_zeros]

triton_poi_fused_new_zeros_73 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_zeros_73', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ox/cox43uis6ar7ktmkptmqxcutt5yx5inc3ksq7enlzfho5tyaxfsj.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_74 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_74', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((16*(r2 % 49)) + (784*(x0 // 16)) + (6272*(r2 // 49)) + (12544*x1) + (x0 % 16)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (128*r2) + (12544*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
        tmp6 = tmp4 - tmp5
        tmp7 = tmp0 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/td/ctdsoyk2hniqflixuwolm6qck2uihnpjqidoxjsrnz5rfbbzb2ie.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_75 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_75', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rs/crsvzbsnd5rcuixgit5o5yjoeu52wek5z44nxllu3ty7rmp34rqp.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_76 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_76', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dy/cdykbkctw5eywrd4th4z2eocgd2xgsaatvsys7qnseblraemdglh.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_77 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_77', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((16*(x1 % 49)) + (784*(x0 // 16)) + (6272*(x1 // 49)) + (x0 % 16)), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.002551020408163265
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hl/chlfz2syh6odnqq5ynb5ubk4o7kbahg634cplq5ppvzdivlnylkr.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_78 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_78', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8320
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x0)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.broadcast_to(x1 % 80, [XBLOCK, RBLOCK])
        tmp4 = tl.full([1, 1], 0, tl.int64)
        tmp5 = tmp3 >= tmp4
        tmp6 = tl.full([1, 1], 16, tl.int64)
        tmp7 = tmp3 < tmp6
        tmp8 = tmp7 & tmp2
        tmp9 = tl.load(in_ptr0 + ((196*(x1 % 80)) + (3136*(x1 // 80)) + (25088*(((r2 + (121*x0)) // 196) % 8)) + ((r2 + (121*x0)) % 196)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
        tmp11 = tl.where(tmp8, tmp9, tmp10)
        tmp12 = tmp3 >= tmp6
        tmp13 = tl.full([1, 1], 80, tl.int64)
        tmp14 = tmp3 < tmp13
        tmp15 = tmp12 & tmp2
        tmp16 = tl.load(in_ptr1 + ((-16) + (64*((r2 + (121*x0)) % 196)) + (12544*(x1 // 80)) + (100352*(((r2 + (121*x0)) // 196) % 8)) + (x1 % 80)), rmask & tmp15 & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
        tmp18 = tl.where(tmp15, tmp16, tmp17)
        tmp19 = tl.where(tmp7, tmp11, tmp18)
        tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
        tmp21 = tl.where(tmp2, tmp19, tmp20)
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zi/czijkdld6cpkvekbex7qbww4pqnb7avndix7ksookru3tbfkloii.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_79 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_79', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 640
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


# kernel path: /tmp/torchinductor_youkaichao/jo/cjo3hwr7yqiiuvq23gfhesdotqvvtumptp7y5mzvza2yizdoudee.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_80 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_80', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8320
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 640)
    x0 = xindex % 640
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.broadcast_to(x0 % 80, [XBLOCK, RBLOCK])
        tmp4 = tl.full([1, 1], 0, tl.int64)
        tmp5 = tmp3 >= tmp4
        tmp6 = tl.full([1, 1], 16, tl.int64)
        tmp7 = tmp3 < tmp6
        tmp8 = tmp7 & tmp2
        tmp9 = tl.load(in_ptr0 + ((196*(x0 % 80)) + (3136*(x0 // 80)) + (25088*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp8 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
        tmp11 = tl.where(tmp8, tmp9, tmp10)
        tmp12 = tmp3 >= tmp6
        tmp13 = tl.full([1, 1], 80, tl.int64)
        tmp14 = tmp3 < tmp13
        tmp15 = tmp12 & tmp2
        tmp16 = tl.load(in_ptr1 + ((-16) + (64*((r2 + (121*x1)) % 196)) + (12544*(x0 // 80)) + (100352*(((r2 + (121*x1)) // 196) % 8)) + (x0 % 80)), rmask & tmp15 & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
        tmp18 = tl.where(tmp15, tmp16, tmp17)
        tmp19 = tl.where(tmp7, tmp11, tmp18)
        tmp20 = tl.load(in_ptr2 + (x0 + (640*r2) + (77440*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tmp20 - tmp21
        tmp23 = tmp19 * tmp22
        tmp24 = tl.full(tmp23.shape, 0, tmp23.dtype)
        tmp25 = tl.where(tmp2, tmp23, tmp24)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(rmask & xmask, tmp28, _tmp27)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp27, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sp/cspgctswu43rtprfslkbxy7hqfsjoklqw4vewenzv54pe7iqnk6u.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_81 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_81', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 640
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (640*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iv/civcmczcktqzp7aeatvp4eimxtnajbwvywdc3xtlynzraeuqqjsc.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_82 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_82', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 640
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp15 = tl.load(in_ptr2 + (x1 + (640*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1 % 80
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((196*(x1 % 80)) + (3136*(x1 // 80)) + (25088*(y0 // 196)) + (y0 % 196)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 80, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-16) + (64*(y0 % 196)) + (12544*(x1 // 80)) + (100352*(y0 // 196)) + (x1 % 80)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp17 = tmp15 - tmp16
    tmp19 = 0.0006377551020408163
    tmp20 = tmp18 * tmp19
    tmp22 = tmp21 * tmp21
    tmp23 = tmp20 * tmp22
    tmp24 = tmp17 * tmp23
    tmp25 = tmp14 - tmp24
    tmp27 = tmp26 * tmp19
    tmp28 = tmp25 - tmp27
    tmp30 = tmp21 * tmp29
    tmp31 = tmp28 * tmp30
    tl.store(out_ptr0 + (x1 + (640*y0)), tmp31, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u3/cu3iurmym44ae3gta4rik57ppwdgecrhslj5p2epn3r2wpnsovac.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_83 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_83', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1664
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 128)
    x0 = xindex % 128
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp32 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = (((r2 + (121*x1)) % 196) // 14) % 2
        tmp4 = tl.full([1, 1], 0, tl.int64)
        tmp5 = tmp3 == tmp4
        tmp6 = tmp5 & tmp2
        tmp7 = (((r2 + (121*x1)) % 196) % 14) % 2
        tmp8 = tmp7 == tmp4
        tmp9 = tmp8 & tmp6
        tmp10 = tl.load(in_ptr0 + (x0 + (128*((((r2 + (121*x1)) % 196) % 14) // 2)) + (896*(((r2 + (121*x1)) % 196) // 28)) + (6272*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp9 & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
        tmp12 = tl.where(tmp9, tmp10, tmp11)
        tmp13 = 0.0
        tmp14 = tl.where(tmp8, tmp12, tmp13)
        tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
        tmp16 = tl.where(tmp6, tmp14, tmp15)
        tmp17 = tl.where(tmp5, tmp16, tmp13)
        tmp18 = tl.load(in_ptr1 + (x0 + (128*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tmp17 + tmp18
        tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
        tmp21 = tl.where(tmp2, tmp19, tmp20)
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
        tmp25 = tl.load(in_ptr2 + (x0 + (128*r2) + (15488*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp26 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp27 = tmp25 - tmp26
        tmp28 = tmp19 * tmp27
        tmp29 = tl.full(tmp28.shape, 0, tmp28.dtype)
        tmp30 = tl.where(tmp2, tmp28, tmp29)
        tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
        tmp33 = _tmp32 + tmp31
        _tmp32 = tl.where(rmask & xmask, tmp33, _tmp32)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp23, xmask)
    tmp32 = tl.sum(_tmp32, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7b/c7bzlvpawo4z6q3jxs3eyf6g5dzgx5ofottjyk4ethjg2iz5m2uq.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_84 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_84', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 13
    RBLOCK: tl.constexpr = 16
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


# kernel path: /tmp/torchinductor_youkaichao/dp/cdpzeo2ef4bfzrknzusranrwsgghlxa7ttywvdloazoxhoi7tiu4.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_85 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_85', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cb/ccbt54xzxelkrjus2nvedggg7tzoipxhwpehenks6oimas6xn6vy.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_86 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_86', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 128)
    x0 = xindex % 128
    x2 = xindex
    tmp14 = tl.load(in_ptr1 + (x2), None)
    tmp16 = tl.load(in_ptr2 + (x2), None)
    tmp17 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp0 = ((x1 % 196) // 14) % 2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp3 = ((x1 % 196) % 14) % 2
    tmp4 = tmp3 == tmp1
    tmp5 = tmp4 & tmp2
    tmp6 = tl.load(in_ptr0 + (x0 + (128*(((x1 % 196) % 14) // 2)) + (896*((x1 % 196) // 28)) + (6272*(x1 // 196))), tmp5, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = 0.0
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp2, tmp10, tmp11)
    tmp13 = tl.where(tmp2, tmp12, tmp9)
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = 0.0006377551020408163
    tmp21 = tmp19 * tmp20
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp15 - tmp25
    tmp28 = tmp27 * tmp20
    tmp29 = tmp26 - tmp28
    tmp31 = tmp22 * tmp30
    tmp32 = tmp29 * tmp31
    tl.store(out_ptr0 + (x2), tmp32, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zk/czkakvzxljfuewee7qwcw4qakcpa3taqsrhkw4c6udlknbwc5w4f.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_87 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_87', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3328
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 256)
    x0 = xindex % 256
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp28 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (256*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + (x0 + (256*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tmp3 / tmp6
        tmp10 = 0.5
        tmp11 = tmp9 + tmp10
        tmp12 = tmp8 * tmp11
        tmp13 = tl.where(tmp7, tmp12, tmp8)
        tmp14 = 0.0
        tmp15 = tl.where(tmp5, tmp14, tmp13)
        tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
        tmp17 = tl.where(tmp2, tmp15, tmp16)
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
        tmp21 = tl.load(in_ptr2 + (x0 + (256*r2) + (30976*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp22 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tmp21 - tmp22
        tmp24 = tmp15 * tmp23
        tmp25 = tl.full(tmp24.shape, 0, tmp24.dtype)
        tmp26 = tl.where(tmp2, tmp24, tmp25)
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(rmask & xmask, tmp29, _tmp28)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp19, xmask)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qj/cqj4rgdxkgl2sni67erojdieoclsp6g2agtu6o6iegjfmnjl25u5.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_88 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_88', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 13
    RBLOCK: tl.constexpr = 16
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


# kernel path: /tmp/torchinductor_youkaichao/ud/cudmtdhiin2mgv6ajpn7piinncjqp2d53qdn3o6koumf2w66wfsi.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_per_fused_native_batch_norm_backward_89 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 16],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_89', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bx/cbxize27qrj2cqoh5cetxealmhvehhfojgf4z3bqpkjbh6pvfmng.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_90 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_90', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp5 = tl.load(in_out_ptr0 + (x2), None)
    tmp13 = tl.load(in_ptr1 + (x2), None)
    tmp14 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = tmp0 / tmp3
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp4, tmp9, tmp5)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp11, tmp10)
    tmp15 = tmp13 - tmp14
    tmp17 = 0.0006377551020408163
    tmp18 = tmp16 * tmp17
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 * tmp20
    tmp22 = tmp15 * tmp21
    tmp23 = tmp12 - tmp22
    tmp25 = tmp24 * tmp17
    tmp26 = tmp23 - tmp25
    tmp28 = tmp19 * tmp27
    tmp29 = tmp26 * tmp28
    tl.store(in_out_ptr0 + (x2), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/y7/cy7od4inl6aqk4n4z5nx3ittmf76u6uvgbttxt5sxtcpzuj77rdg.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_91 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_91', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1664
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 128)
    x0 = xindex % 128
    _tmp25 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp34 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = (((r2 + (121*x1)) % 196) // 14) % 2
        tmp4 = tl.full([1, 1], 0, tl.int64)
        tmp5 = tmp3 == tmp4
        tmp6 = tmp5 & tmp2
        tmp7 = (((r2 + (121*x1)) % 196) % 14) % 2
        tmp8 = tmp7 == tmp4
        tmp9 = tmp8 & tmp6
        tmp10 = tl.load(in_ptr0 + (x0 + (128*((((r2 + (121*x1)) % 196) % 14) // 2)) + (896*(((r2 + (121*x1)) % 196) // 28)) + (6272*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp9 & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
        tmp12 = tl.where(tmp9, tmp10, tmp11)
        tmp13 = 0.0
        tmp14 = tl.where(tmp8, tmp12, tmp13)
        tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
        tmp16 = tl.where(tmp6, tmp14, tmp15)
        tmp17 = tl.where(tmp5, tmp16, tmp13)
        tmp18 = tl.load(in_ptr1 + (x0 + (128*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tmp17 + tmp18
        tmp20 = tl.load(in_ptr2 + (x0 + (128*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tmp19 + tmp20
        tmp22 = tl.full(tmp21.shape, 0, tmp21.dtype)
        tmp23 = tl.where(tmp2, tmp21, tmp22)
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(rmask & xmask, tmp26, _tmp25)
        tmp27 = tl.load(in_ptr3 + (x0 + (128*r2) + (15488*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_ptr4 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp29 = tmp27 - tmp28
        tmp30 = tmp21 * tmp29
        tmp31 = tl.full(tmp30.shape, 0, tmp30.dtype)
        tmp32 = tl.where(tmp2, tmp30, tmp31)
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
        tmp35 = _tmp34 + tmp33
        _tmp34 = tl.where(rmask & xmask, tmp35, _tmp34)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp25, xmask)
    tmp34 = tl.sum(_tmp34, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp34, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3w/c3wuzvt6qimus5q5tjnliuwaqlke6veclgcmxhuxlcc2phpuil7d.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_92 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_92', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 128)
    x0 = xindex % 128
    x2 = xindex
    tmp14 = tl.load(in_ptr1 + (x2), None)
    tmp16 = tl.load(in_ptr2 + (x2), None)
    tmp18 = tl.load(in_ptr3 + (x2), None)
    tmp19 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp0 = ((x1 % 196) // 14) % 2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp3 = ((x1 % 196) % 14) % 2
    tmp4 = tmp3 == tmp1
    tmp5 = tmp4 & tmp2
    tmp6 = tl.load(in_ptr0 + (x0 + (128*(((x1 % 196) % 14) // 2)) + (896*((x1 % 196) // 28)) + (6272*(x1 // 196))), tmp5, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = 0.0
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp2, tmp10, tmp11)
    tmp13 = tl.where(tmp2, tmp12, tmp9)
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp20 = tmp18 - tmp19
    tmp22 = 0.0006377551020408163
    tmp23 = tmp21 * tmp22
    tmp25 = tmp24 * tmp24
    tmp26 = tmp23 * tmp25
    tmp27 = tmp20 * tmp26
    tmp28 = tmp17 - tmp27
    tmp30 = tmp29 * tmp22
    tmp31 = tmp28 - tmp30
    tmp33 = tmp24 * tmp32
    tmp34 = tmp31 * tmp33
    tl.store(out_ptr0 + (x2), tmp34, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kt/cktu3yey5foeclrodhnouio2lidrt73cxrngadwrzcjlfj7gokjv.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_93 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_93', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x0 = xindex % 32
    x1 = (xindex // 32) % 4
    x2 = (xindex // 128) % 196
    x3 = (xindex // 25088)
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp5 = tl.load(in_ptr1 + (x4), None)
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = tmp0 / tmp3
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp4, tmp9, tmp5)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp11, tmp10)
    tl.store(out_ptr0 + (x0 + (32*x2) + (6272*x1) + (25088*x3)), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cx/ccxttvazt7r4rcsheazi4illwugndldqw2qzpd5twb5w2j4sxch6.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]

triton_per_fused__softmax_backward_data_mul_94 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_mul_94', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
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
    tmp9 = 0.25
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr1 + (r1 + (196*x0)), tmp10, rmask & xmask)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zn/cznrb6ysn4xyznctjrr54gmyywnlcp7jza7c2p33ascol4iudcu2.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]

triton_per_fused__softmax_backward_data_sum_95 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_sum_95', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 153664
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (x3 + (153664*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x3 + (153664*r2)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x1 + (784*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp1 * tmp3
    tmp5 = tmp2 - tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(rmask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y5/cy5zw747c6fzljiqxi74romdpznjci4rk5pr3ddpse35r2x6y7cl.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_96 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_96', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3328
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 256)
    x0 = xindex % 256
    _tmp32 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp41 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.broadcast_to(x0 % 64, [XBLOCK, RBLOCK])
        tmp4 = tl.full([1, 1], 0, tl.int64)
        tmp5 = tmp3 >= tmp4
        tmp6 = tl.full([1, 1], 16, tl.int64)
        tmp7 = tmp3 < tmp6
        tmp8 = tmp7 & tmp2
        tmp9 = tl.load(in_ptr0 + ((16*((r2 + (121*x1)) % 196)) + (3136*(x0 // 64)) + (12544*(((r2 + (121*x1)) // 196) % 8)) + (x0 % 64)), rmask & tmp8 & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
        tmp11 = tl.where(tmp8, tmp9, tmp10)
        tmp12 = tmp3 >= tmp6
        tmp13 = tl.full([1, 1], 32, tl.int64)
        tmp14 = tmp3 < tmp13
        tmp15 = tmp12 & tmp14
        tmp16 = tmp15 & tmp2
        tmp17 = tl.load(in_ptr1 + ((-3136) + (196*(x0 % 64)) + (3136*(x0 // 64)) + (12544*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp16 & xmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
        tmp19 = tl.where(tmp16, tmp17, tmp18)
        tmp20 = tmp3 >= tmp13
        tmp21 = tl.full([1, 1], 64, tl.int64)
        tmp22 = tmp3 < tmp21
        tmp23 = tmp20 & tmp2
        tmp24 = tl.load(in_ptr2 + ((-32) + (32*((r2 + (121*x1)) % 196)) + (6272*(x0 // 64)) + (25088*(((r2 + (121*x1)) // 196) % 8)) + (x0 % 64)), rmask & tmp23 & xmask, eviction_policy='evict_first', other=0.0)
        tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
        tmp26 = tl.where(tmp23, tmp24, tmp25)
        tmp27 = tl.where(tmp15, tmp19, tmp26)
        tmp28 = tl.where(tmp7, tmp11, tmp27)
        tmp29 = tl.full(tmp28.shape, 0, tmp28.dtype)
        tmp30 = tl.where(tmp2, tmp28, tmp29)
        tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
        tmp33 = _tmp32 + tmp31
        _tmp32 = tl.where(rmask & xmask, tmp33, _tmp32)
        tmp34 = tl.load(in_ptr3 + (x0 + (256*r2) + (30976*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp35 = tl.load(in_ptr4 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp36 = tmp34 - tmp35
        tmp37 = tmp28 * tmp36
        tmp38 = tl.full(tmp37.shape, 0, tmp37.dtype)
        tmp39 = tl.where(tmp2, tmp37, tmp38)
        tmp40 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
        tmp42 = _tmp41 + tmp40
        _tmp41 = tl.where(rmask & xmask, tmp42, _tmp41)
    tmp32 = tl.sum(_tmp32, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp32, xmask)
    tmp41 = tl.sum(_tmp41, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp41, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hw/chwcu66yaduhfwcs5yoz3ek2k3efrastzfjojjmqfwchppuwrlfv.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_97 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_97', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp23 = tl.load(in_ptr3 + (x1 + (256*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1 % 64
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((16*(y0 % 196)) + (3136*(x1 // 64)) + (12544*(y0 // 196)) + (x1 % 64)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 32, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-3136) + (196*(x1 % 64)) + (3136*(x1 // 64)) + (12544*(y0 // 196)) + (y0 % 196)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 64, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-32) + (32*(y0 % 196)) + (6272*(x1 // 64)) + (25088*(y0 // 196)) + (x1 % 64)), tmp15 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tl.where(tmp11, tmp14, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tmp25 = tmp23 - tmp24
    tmp27 = 0.0006377551020408163
    tmp28 = tmp26 * tmp27
    tmp30 = tmp29 * tmp29
    tmp31 = tmp28 * tmp30
    tmp32 = tmp25 * tmp31
    tmp33 = tmp22 - tmp32
    tmp35 = tmp34 * tmp27
    tmp36 = tmp33 - tmp35
    tmp38 = tmp29 * tmp37
    tmp39 = tmp36 * tmp38
    tl.store(out_ptr0 + (x1 + (256*y0)), tmp39, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r5/cr5zvzvaqlwh5i4b7gt2gvcosf5jywksahnf6nvyoeiha74hd5pw.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_98 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_98', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1664
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 128)
    x0 = xindex % 128
    _tmp27 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp36 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = (((r2 + (121*x1)) % 196) // 14) % 2
        tmp4 = tl.full([1, 1], 0, tl.int64)
        tmp5 = tmp3 == tmp4
        tmp6 = tmp5 & tmp2
        tmp7 = (((r2 + (121*x1)) % 196) % 14) % 2
        tmp8 = tmp7 == tmp4
        tmp9 = tmp8 & tmp6
        tmp10 = tl.load(in_ptr0 + (x0 + (128*((((r2 + (121*x1)) % 196) % 14) // 2)) + (896*(((r2 + (121*x1)) % 196) // 28)) + (6272*(((r2 + (121*x1)) // 196) % 8))), rmask & tmp9 & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
        tmp12 = tl.where(tmp9, tmp10, tmp11)
        tmp13 = 0.0
        tmp14 = tl.where(tmp8, tmp12, tmp13)
        tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
        tmp16 = tl.where(tmp6, tmp14, tmp15)
        tmp17 = tl.where(tmp5, tmp16, tmp13)
        tmp18 = tl.load(in_ptr1 + (x0 + (128*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tmp17 + tmp18
        tmp20 = tl.load(in_ptr2 + (x0 + (128*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tmp19 + tmp20
        tmp22 = tl.load(in_ptr3 + (x0 + (128*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tmp21 + tmp22
        tmp24 = tl.full(tmp23.shape, 0, tmp23.dtype)
        tmp25 = tl.where(tmp2, tmp23, tmp24)
        tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
        tmp28 = _tmp27 + tmp26
        _tmp27 = tl.where(rmask & xmask, tmp28, _tmp27)
        tmp29 = tl.load(in_ptr4 + (x0 + (128*r2) + (15488*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp30 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp31 = tmp29 - tmp30
        tmp32 = tmp23 * tmp31
        tmp33 = tl.full(tmp32.shape, 0, tmp32.dtype)
        tmp34 = tl.where(tmp2, tmp32, tmp33)
        tmp35 = tl.broadcast_to(tmp34, [XBLOCK, RBLOCK])
        tmp37 = _tmp36 + tmp35
        _tmp36 = tl.where(rmask & xmask, tmp37, _tmp36)
    tmp27 = tl.sum(_tmp27, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp27, xmask)
    tmp36 = tl.sum(_tmp36, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zg/czgn4emoexvwo7iuds7obsxaw7omxdxl4ya6l6ol3rblp2v3ccbx.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_99 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_99', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 128)
    x0 = xindex % 128
    x2 = xindex
    tmp14 = tl.load(in_ptr1 + (x2), None)
    tmp16 = tl.load(in_ptr2 + (x2), None)
    tmp18 = tl.load(in_ptr3 + (x2), None)
    tmp20 = tl.load(in_ptr4 + (x2), None)
    tmp21 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp0 = ((x1 % 196) // 14) % 2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp3 = ((x1 % 196) % 14) % 2
    tmp4 = tmp3 == tmp1
    tmp5 = tmp4 & tmp2
    tmp6 = tl.load(in_ptr0 + (x0 + (128*(((x1 % 196) % 14) // 2)) + (896*((x1 % 196) // 28)) + (6272*(x1 // 196))), tmp5, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = 0.0
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp2, tmp10, tmp11)
    tmp13 = tl.where(tmp2, tmp12, tmp9)
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp19 = tmp17 + tmp18
    tmp22 = tmp20 - tmp21
    tmp24 = 0.0006377551020408163
    tmp25 = tmp23 * tmp24
    tmp27 = tmp26 * tmp26
    tmp28 = tmp25 * tmp27
    tmp29 = tmp22 * tmp28
    tmp30 = tmp19 - tmp29
    tmp32 = tmp31 * tmp24
    tmp33 = tmp30 - tmp32
    tmp35 = tmp26 * tmp34
    tmp36 = tmp33 * tmp35
    tl.store(in_out_ptr0 + (x2), tmp36, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/c7/cc7u7i23cdixg23nswoz4hniqhsz2nmmjdmkxbfxu62xfne6cbp6.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_100 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_100', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 128) % 196
    x0 = xindex % 128
    x2 = (xindex // 25088)
    x3 = xindex
    tmp14 = tl.load(in_out_ptr0 + (x3), None)
    tmp16 = tl.load(in_ptr1 + (x3), None)
    tmp18 = tl.load(in_ptr2 + (x3), None)
    tmp20 = tl.load(in_ptr3 + (x3), None)
    tmp0 = (x1 // 14) % 2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp3 = (x1 % 14) % 2
    tmp4 = tmp3 == tmp1
    tmp5 = tmp4 & tmp2
    tmp6 = tl.load(in_ptr0 + (x0 + (128*((x1 % 14) // 2)) + (896*(x1 // 28)) + (6272*x2)), tmp5, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = 0.0
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp2, tmp10, tmp11)
    tmp13 = tl.where(tmp2, tmp12, tmp9)
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp19 = tmp17 + tmp18
    tmp21 = tmp19 + tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2d/c2djm75xgvdaml5p3x5drmvhs5fnj72ezettgp4eneuzkm7iwrec.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_101 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_101', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1664
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 128)
    x0 = xindex % 128
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (128*r2) + (15488*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
        tmp9 = tl.load(in_ptr1 + (x0 + (128*r2) + (15488*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 - tmp10
        tmp12 = tmp3 * tmp11
        tmp13 = tl.full(tmp12.shape, 0, tmp12.dtype)
        tmp14 = tl.where(tmp2, tmp12, tmp13)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/44/c44chbeswyi77bjrsrocrbbh67g5u4cchh4ib7llh2obczxzlzws.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_102 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_102', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 0.0006377551020408163
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 - tmp10
    tmp13 = tmp12 * tmp5
    tmp14 = tmp11 - tmp13
    tmp16 = tmp7 * tmp15
    tmp17 = tmp14 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4a/c4a7bamokchnv2xezfldyc7jaue6igrouvhur6fw224r4t7nvqj3.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_103 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_103', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1664
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 128)
    x0 = xindex % 128
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (128*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (128*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp11 = tl.load(in_ptr2 + (x0 + (128*r2) + (15488*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 - tmp12
        tmp14 = tmp5 * tmp13
        tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
        tmp16 = tl.where(tmp2, tmp14, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6y/c6y5ryzar5wi7uysn5bxyogkcg3x7pmvwcrkeqenaauu7mrjmljo.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_104 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_104', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.0006377551020408163
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.store(out_ptr0 + (x2), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2t/c2tinrww6f2k6zgdkbppshmxhl6ykgguuhgwscye3hqqpecsty72.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_105 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_105', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1664
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 128)
    x0 = xindex % 128
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (128*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (128*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.load(in_ptr2 + (x0 + (128*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
        tmp13 = tl.load(in_ptr3 + (x0 + (128*r2) + (15488*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr4 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 - tmp14
        tmp16 = tmp7 * tmp15
        tmp17 = tl.full(tmp16.shape, 0, tmp16.dtype)
        tmp18 = tl.where(tmp2, tmp16, tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uf/cuf7w6vekck36yhmzw43rffnzzzusxtkpo224xpetkxwyhfb6hfs.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_106 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_106', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp5 = tl.load(in_ptr3 + (x2), None)
    tmp6 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp5 - tmp6
    tmp9 = 0.0006377551020408163
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr0 + (x2), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/47/c477ixj7a7cie7kcki3ipzlswah7kllh3yi5rz2o6i2cj3f6bb66.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_107 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_107', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1664
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 128)
    x0 = xindex % 128
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (128*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (128*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.load(in_ptr2 + (x0 + (128*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.load(in_ptr3 + (x0 + (128*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tmp7 + tmp8
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
        tmp15 = tl.load(in_ptr4 + (x0 + (128*r2) + (15488*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr5 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tmp15 - tmp16
        tmp18 = tmp9 * tmp17
        tmp19 = tl.full(tmp18.shape, 0, tmp18.dtype)
        tmp20 = tl.where(tmp2, tmp18, tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask & xmask, tmp23, _tmp22)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z3/cz33heh7wauy5hzi7tdrtwdx3y3myreonhguxfiqwud2qykq7jfk.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_108 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_108', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp5 = tl.load(in_ptr3 + (x2), None)
    tmp7 = tl.load(in_ptr4 + (x2), None)
    tmp8 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp9 = tmp7 - tmp8
    tmp11 = 0.0006377551020408163
    tmp12 = tmp10 * tmp11
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 * tmp14
    tmp16 = tmp9 * tmp15
    tmp17 = tmp6 - tmp16
    tmp19 = tmp18 * tmp11
    tmp20 = tmp17 - tmp19
    tmp22 = tmp13 * tmp21
    tmp23 = tmp20 * tmp22
    tl.store(in_out_ptr0 + (x2), tmp23, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/i3/ci3o3qjslmmicbxi23xr6hjj7df5d4qkyjhornp5pyarh2ncvftm.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_109 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_109', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp7 = tl.load(in_ptr3 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/py/cpybkegvakjv6gwwugw6qgskguivsoperl5ebu4qmzkr3xidsppb.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_red_fused_native_batch_norm_backward_110 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_110', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1664
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 128)
    x0 = xindex % 128
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (128*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (128*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp11 = tl.load(in_ptr2 + ((196*x0) + (25088*(((r2 + (121*x1)) // 196) % 8)) + ((r2 + (121*x1)) % 196)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tmp11 - tmp12
        tmp14 = tmp5 * tmp13
        tmp15 = tl.full(tmp14.shape, 0, tmp14.dtype)
        tmp16 = tl.where(tmp2, tmp14, tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp9, xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yw/cywkhqzxpevx5b6cx76hfi7cbzzkakvc57q63sqtubslpon4dbnp.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_111 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_111', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 128
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + (196*x2) + (25088*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 - tmp4
    tmp7 = 0.0006377551020408163
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 * tmp10
    tmp12 = tmp5 * tmp11
    tmp13 = tmp2 - tmp12
    tmp15 = tmp14 * tmp7
    tmp16 = tmp13 - tmp15
    tmp18 = tmp9 * tmp17
    tmp19 = tmp16 * tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (128*y3)), tmp19, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oz/cozp4lc5cl4o4uc3jrv57rcei6m5joclvva7chrdvk3bicto3n3b.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_112 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[64, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_112', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp17 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (50176*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr1 + (r1 + (784*x0) + (50176*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr2 + (r1 + (784*x0) + (50176*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = tmp0 / tmp3
        tmp7 = 0.5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.where(tmp4, tmp9, tmp5)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, tmp10)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
        tmp18 = tmp16 - tmp17
        tmp19 = tmp12 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask & xmask, tmp22, _tmp21)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp14, xmask)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp21, xmask)
    tmp23 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tmp21 * tmp23
    tl.store(out_ptr2 + (x0), tmp24, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zz/czzo4574zeqbzbdi26xurzhd2nzezgbnjnnj3tki4o5yulyoczxb.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_113 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_113', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 64
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp5 = tl.load(in_out_ptr0 + (x3), None)
    tmp13 = tl.load(in_ptr1 + (x3), None)
    tmp14 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = tmp0 / tmp3
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp4, tmp9, tmp5)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp11, tmp10)
    tmp15 = tmp13 - tmp14
    tmp17 = 0.00015943877551020407
    tmp18 = tmp16 * tmp17
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 * tmp20
    tmp22 = tmp15 * tmp21
    tmp23 = tmp12 - tmp22
    tmp25 = tmp24 * tmp17
    tmp26 = tmp23 - tmp25
    tmp28 = tmp19 * tmp27
    tmp29 = tmp26 * tmp28
    tl.store(in_out_ptr0 + (x3), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pr/cprruz7kqoqkjjekqgdj2zvccwjwmhfuq76o3vf3qbh2wkg3jqe2.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_114 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_114', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp17 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (100352*(r2 // 3136)) + (200704*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr1 + ((3136*x0) + (100352*(r2 // 3136)) + (200704*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr2 + ((3136*x0) + (100352*(r2 // 3136)) + (200704*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = -3.0
        tmp2 = tmp0 < tmp1
        tmp3 = 3.0
        tmp4 = tmp0 <= tmp3
        tmp6 = tmp0 / tmp3
        tmp7 = 0.5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tl.where(tmp4, tmp9, tmp5)
        tmp11 = 0.0
        tmp12 = tl.where(tmp2, tmp11, tmp10)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
        tmp18 = tmp16 - tmp17
        tmp19 = tmp12 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask & xmask, tmp22, _tmp21)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, xmask)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xk/cxkei5w6ywbcgnd3tfmlawgmulefjzrb5pfgipyzxq7f2qsmwi4d.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_115 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_115', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t6/ct6qsmdu6sk3peqq2g3gtx6easyv44f325bu3vgl5dpkydfi5v4a.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_116 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_116', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ns/cns4zpcxogeje3l5jcqo6gdmbnckxgq25pntmtrrqnwuoj2v2mzu.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_117 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_117', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 32
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp5 = tl.load(in_out_ptr0 + (x3), None)
    tmp13 = tl.load(in_ptr1 + (x3), None)
    tmp14 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = tmp0 / tmp3
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp4, tmp9, tmp5)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp11, tmp10)
    tmp15 = tmp13 - tmp14
    tmp17 = 3.985969387755102e-05
    tmp18 = tmp16 * tmp17
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 * tmp20
    tmp22 = tmp15 * tmp21
    tmp23 = tmp12 - tmp22
    tmp25 = tmp24 * tmp17
    tmp26 = tmp23 - tmp25
    tmp28 = tmp19 * tmp27
    tmp29 = tmp26 * tmp28
    tl.store(in_out_ptr0 + (x3), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/k4/ck4cml4hcbdmw3wfzf4atc6dafutfphfitc2ds6nxysz7f3ovzz7.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_red_fused_hardswish_backward_native_batch_norm_backward_118 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_hardswish_backward_native_batch_norm_backward_118', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 208
    rnumel = 7720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp28 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7720*x0)
        tmp1 = tl.full([1, 1], 100352, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((12544*x1) + (200704*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = -3.0
        tmp5 = tmp3 < tmp4
        tmp6 = 3.0
        tmp7 = tmp3 <= tmp6
        tmp8 = tl.load(in_ptr1 + ((12544*x1) + (200704*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tmp3 / tmp6
        tmp10 = 0.5
        tmp11 = tmp9 + tmp10
        tmp12 = tmp8 * tmp11
        tmp13 = tl.where(tmp7, tmp12, tmp8)
        tmp14 = 0.0
        tmp15 = tl.where(tmp5, tmp14, tmp13)
        tmp16 = tl.full(tmp15.shape, 0, tmp15.dtype)
        tmp17 = tl.where(tmp2, tmp15, tmp16)
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
        tmp21 = tl.load(in_ptr2 + ((12544*x1) + (200704*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tl.load(in_ptr3 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tmp21 - tmp22
        tmp24 = tmp15 * tmp23
        tmp25 = tl.full(tmp24.shape, 0, tmp24.dtype)
        tmp26 = tl.where(tmp2, tmp24, tmp25)
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(rmask & xmask, tmp29, _tmp28)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp19, xmask)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y6/cy676bd4q3t4otg4pqt7ukcu2zl6qk7rvzjphyrv7eva4xhj5cy7.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_119 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_119', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
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


# kernel path: /tmp/torchinductor_youkaichao/pl/cplum7vfhuw4fp5n3fw6c6tm5adh7qqykpbsicvlnkxiuedwnywi.py
# Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]

triton_per_fused_hardswish_backward_native_batch_norm_backward_120 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_backward_native_batch_norm_backward_120', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
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
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xv/cxvwdnl4jp2ncuamh5l47spb7ssaffa4z2yev6xlastmudm7qieq.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]

triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_121 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_121', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 16
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp5 = tl.load(in_out_ptr0 + (x3), None)
    tmp13 = tl.load(in_ptr1 + (x3), None)
    tmp14 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = -3.0
    tmp2 = tmp0 < tmp1
    tmp3 = 3.0
    tmp4 = tmp0 <= tmp3
    tmp6 = tmp0 / tmp3
    tmp7 = 0.5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = tl.where(tmp4, tmp9, tmp5)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp11, tmp10)
    tmp15 = tmp13 - tmp14
    tmp17 = 9.964923469387754e-06
    tmp18 = tmp16 * tmp17
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 * tmp20
    tmp22 = tmp15 * tmp21
    tmp23 = tmp12 - tmp22
    tmp25 = tmp24 * tmp17
    tmp26 = tmp23 - tmp25
    tmp28 = tmp19 * tmp27
    tmp29 = tmp26 * tmp28
    tl.store(in_out_ptr0 + (x3), tmp29, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_15, primals_16, primals_18, primals_19, primals_21, primals_22, primals_24, primals_25, primals_28, primals_31, primals_34, primals_37, primals_40, primals_43, primals_46, primals_49, primals_52, primals_55, primals_58, primals_61, primals_64, primals_67, primals_70, primals_73, primals_76, primals_79, primals_82, primals_85, primals_88, primals_91, primals_94, primals_97, primals_100, primals_103, primals_106, primals_109, primals_112, primals_115, primals_118, primals_121, primals_124, primals_127, primals_130, primals_133, primals_136, primals_139, primals_142, primals_145, primals_148, primals_151, primals_154, primals_157, primals_160, primals_163, primals_166, primals_169, primals_172, primals_175, primals_178, primals_181, primals_184, primals_187, primals_190, primals_193, primals_196, primals_199, primals_201, primals_205, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_415, convolution, squeeze_1, add_4, div, convolution_1, squeeze_4, add_10, div_1, convolution_2, squeeze_7, add_16, div_2, convolution_3, squeeze_10, view_1, mm, squeeze_13, view_12, view_13, mm_1, squeeze_16, view_17, mm_2, squeeze_19, view_20, view_21, mm_3, squeeze_22, view_25, mm_4, squeeze_25, view_36, view_37, mm_5, squeeze_28, view_41, mm_6, squeeze_31, view_44, view_45, mm_7, squeeze_34, view_49, mm_8, squeeze_37, view_60, view_61, mm_9, squeeze_40, view_65, mm_10, squeeze_43, view_68, view_69, mm_11, squeeze_46, view_73, mm_12, squeeze_49, view_84, view_85, mm_13, squeeze_52, view_89, mm_14, squeeze_55, view_92, view_93, mm_15, squeeze_58, view_97, mm_16, squeeze_61, view_104, mm_17, squeeze_64, view_115, view_116, mm_18, squeeze_67, view_120, mm_19, squeeze_70, view_123, view_124, mm_20, squeeze_73, view_128, mm_21, squeeze_76, view_139, view_140, mm_22, squeeze_79, view_144, mm_23, squeeze_82, view_147, view_148, mm_24, squeeze_85, view_152, mm_25, squeeze_88, view_163, view_164, mm_26, squeeze_91, view_168, mm_27, squeeze_94, view_171, view_172, mm_28, squeeze_97, view_176, mm_29, squeeze_100, view_187, view_188, mm_30, squeeze_103, view_192, mm_31, squeeze_106, view_195, view_196, mm_32, squeeze_109, view_200, mm_33, squeeze_112, view_211, view_212, mm_34, squeeze_115, view_216, mm_35, squeeze_118, view_219, view_220, mm_36, squeeze_121, view_224, mm_37, squeeze_124, view_231, mm_38, squeeze_127, view_242, view_243, mm_39, squeeze_130, view_247, mm_40, squeeze_133, view_250, view_251, mm_41, squeeze_136, view_255, mm_42, squeeze_139, view_266, view_267, mm_43, squeeze_142, view_271, mm_44, squeeze_145, view_274, view_275, mm_45, squeeze_148, view_279, mm_46, squeeze_151, view_290, view_291, mm_47, squeeze_154, view_295, mm_48, squeeze_157, view_298, view_299, mm_49, squeeze_160, view_303, mm_50, squeeze_163, view_314, view_315, mm_51, squeeze_166, view_319, mm_52, squeeze_169, view_322, view_323, mm_53, squeeze_172, view_327, mm_54, squeeze_175, view_338, view_339, mm_55, squeeze_178, view_343, mm_56, squeeze_181, view_346, view_347, mm_57, squeeze_184, mean, clone_81, clone_82, permute_117, permute_121, unsqueeze_25, permute_127, unsqueeze_29, permute_131, unsqueeze_33, permute_135, permute_138, permute_139, alias_14, permute_140, permute_141, unsqueeze_37, permute_147, unsqueeze_41, permute_151, unsqueeze_45, permute_155, unsqueeze_49, permute_159, permute_162, permute_163, alias_15, permute_164, permute_165, unsqueeze_53, permute_171, unsqueeze_57, permute_175, unsqueeze_61, permute_179, unsqueeze_65, permute_183, permute_186, permute_187, alias_16, permute_188, permute_189, unsqueeze_69, permute_195, unsqueeze_73, permute_199, unsqueeze_77, permute_203, unsqueeze_81, permute_207, permute_210, permute_211, alias_17, permute_212, permute_213, unsqueeze_85, permute_219, unsqueeze_89, permute_223, unsqueeze_93, permute_227, unsqueeze_97, permute_231, permute_234, permute_235, alias_18, permute_236, permute_237, unsqueeze_101, permute_241, unsqueeze_105, permute_247, unsqueeze_109, permute_251, unsqueeze_113, permute_255, unsqueeze_117, permute_259, permute_262, permute_263, alias_19, permute_264, permute_265, unsqueeze_121, permute_271, unsqueeze_125, permute_275, unsqueeze_129, permute_279, unsqueeze_133, permute_283, permute_286, permute_287, alias_20, permute_288, permute_289, unsqueeze_137, permute_295, unsqueeze_141, permute_299, unsqueeze_145, permute_303, unsqueeze_149, permute_307, permute_310, permute_311, alias_21, permute_312, permute_313, unsqueeze_153, permute_319, unsqueeze_157, permute_323, unsqueeze_161, permute_327, unsqueeze_165, permute_331, permute_334, permute_335, alias_22, permute_336, permute_337, unsqueeze_169, permute_343, unsqueeze_173, permute_347, unsqueeze_177, permute_351, unsqueeze_181, permute_355, permute_358, permute_359, alias_23, permute_360, permute_361, unsqueeze_185, permute_365, unsqueeze_189, permute_371, unsqueeze_193, permute_375, unsqueeze_197, permute_379, unsqueeze_201, permute_383, permute_386, permute_387, alias_24, permute_388, permute_389, unsqueeze_205, permute_395, unsqueeze_209, permute_399, unsqueeze_213, permute_403, unsqueeze_217, permute_407, permute_410, permute_411, alias_25, permute_412, permute_413, unsqueeze_221, permute_419, unsqueeze_225, permute_423, unsqueeze_229, permute_427, unsqueeze_233, permute_431, permute_434, permute_435, alias_26, permute_436, permute_437, unsqueeze_237, permute_443, unsqueeze_241, permute_447, unsqueeze_245, permute_451, unsqueeze_249, permute_455, permute_458, permute_459, alias_27, permute_460, permute_461, unsqueeze_253, permute_467, unsqueeze_259, unsqueeze_271, unsqueeze_283, unsqueeze_295, tangents_1 = args
    args.clear()
    assert_size_stride(primals_15, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_16, (16, ), (1, ))
    assert_size_stride(primals_18, (32, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_19, (32, ), (1, ))
    assert_size_stride(primals_21, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_22, (64, ), (1, ))
    assert_size_stride(primals_24, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_25, (128, ), (1, ))
    assert_size_stride(primals_28, (256, ), (1, ))
    assert_size_stride(primals_31, (128, ), (1, ))
    assert_size_stride(primals_34, (256, ), (1, ))
    assert_size_stride(primals_37, (128, ), (1, ))
    assert_size_stride(primals_40, (256, ), (1, ))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_46, (256, ), (1, ))
    assert_size_stride(primals_49, (128, ), (1, ))
    assert_size_stride(primals_52, (256, ), (1, ))
    assert_size_stride(primals_55, (128, ), (1, ))
    assert_size_stride(primals_58, (256, ), (1, ))
    assert_size_stride(primals_61, (128, ), (1, ))
    assert_size_stride(primals_64, (256, ), (1, ))
    assert_size_stride(primals_67, (128, ), (1, ))
    assert_size_stride(primals_70, (256, ), (1, ))
    assert_size_stride(primals_73, (128, ), (1, ))
    assert_size_stride(primals_76, (640, ), (1, ))
    assert_size_stride(primals_79, (128, ), (1, ))
    assert_size_stride(primals_82, (256, ), (1, ))
    assert_size_stride(primals_85, (512, ), (1, ))
    assert_size_stride(primals_88, (256, ), (1, ))
    assert_size_stride(primals_91, (512, ), (1, ))
    assert_size_stride(primals_94, (256, ), (1, ))
    assert_size_stride(primals_97, (512, ), (1, ))
    assert_size_stride(primals_100, (256, ), (1, ))
    assert_size_stride(primals_103, (512, ), (1, ))
    assert_size_stride(primals_106, (256, ), (1, ))
    assert_size_stride(primals_109, (512, ), (1, ))
    assert_size_stride(primals_112, (256, ), (1, ))
    assert_size_stride(primals_115, (512, ), (1, ))
    assert_size_stride(primals_118, (256, ), (1, ))
    assert_size_stride(primals_121, (512, ), (1, ))
    assert_size_stride(primals_124, (256, ), (1, ))
    assert_size_stride(primals_127, (512, ), (1, ))
    assert_size_stride(primals_130, (256, ), (1, ))
    assert_size_stride(primals_133, (512, ), (1, ))
    assert_size_stride(primals_136, (256, ), (1, ))
    assert_size_stride(primals_139, (1280, ), (1, ))
    assert_size_stride(primals_142, (256, ), (1, ))
    assert_size_stride(primals_145, (384, ), (1, ))
    assert_size_stride(primals_148, (768, ), (1, ))
    assert_size_stride(primals_151, (384, ), (1, ))
    assert_size_stride(primals_154, (768, ), (1, ))
    assert_size_stride(primals_157, (384, ), (1, ))
    assert_size_stride(primals_160, (768, ), (1, ))
    assert_size_stride(primals_163, (384, ), (1, ))
    assert_size_stride(primals_166, (768, ), (1, ))
    assert_size_stride(primals_169, (384, ), (1, ))
    assert_size_stride(primals_172, (768, ), (1, ))
    assert_size_stride(primals_175, (384, ), (1, ))
    assert_size_stride(primals_178, (768, ), (1, ))
    assert_size_stride(primals_181, (384, ), (1, ))
    assert_size_stride(primals_184, (768, ), (1, ))
    assert_size_stride(primals_187, (384, ), (1, ))
    assert_size_stride(primals_190, (768, ), (1, ))
    assert_size_stride(primals_193, (384, ), (1, ))
    assert_size_stride(primals_196, (768, ), (1, ))
    assert_size_stride(primals_199, (384, ), (1, ))
    assert_size_stride(primals_201, (384, ), (1, ))
    assert_size_stride(primals_205, (384, ), (1, ))
    assert_size_stride(primals_209, (196, 196), (196, 1))
    assert_size_stride(primals_210, (196, 196), (196, 1))
    assert_size_stride(primals_211, (196, 196), (196, 1))
    assert_size_stride(primals_212, (196, 196), (196, 1))
    assert_size_stride(primals_213, (49, 196), (196, 1))
    assert_size_stride(primals_214, (49, 49), (49, 1))
    assert_size_stride(primals_215, (49, 49), (49, 1))
    assert_size_stride(primals_216, (49, 49), (49, 1))
    assert_size_stride(primals_217, (49, 49), (49, 1))
    assert_size_stride(primals_218, (16, 49), (49, 1))
    assert_size_stride(primals_219, (16, 16), (16, 1))
    assert_size_stride(primals_220, (16, 16), (16, 1))
    assert_size_stride(primals_221, (16, 16), (16, 1))
    assert_size_stride(primals_222, (16, 16), (16, 1))
    assert_size_stride(primals_415, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(convolution, (8, 16, 112, 112), (200704, 12544, 112, 1))
    assert_size_stride(squeeze_1, (16, ), (1, ))
    assert_size_stride(add_4, (8, 16, 112, 112), (200704, 12544, 112, 1))
    assert_size_stride(div, (8, 16, 112, 112), (200704, 12544, 112, 1))
    assert_size_stride(convolution_1, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(squeeze_4, (32, ), (1, ))
    assert_size_stride(add_10, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(div_1, (8, 32, 56, 56), (100352, 3136, 56, 1))
    assert_size_stride(convolution_2, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(squeeze_7, (64, ), (1, ))
    assert_size_stride(add_16, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(div_2, (8, 64, 28, 28), (50176, 784, 28, 1))
    assert_size_stride(convolution_3, (8, 128, 14, 14), (25088, 196, 14, 1))
    assert_size_stride(squeeze_10, (128, ), (1, ))
    assert_size_stride(view_1, (1568, 128), (128, 1))
    assert_size_stride(mm, (1568, 256), (256, 1))
    assert_size_stride(squeeze_13, (256, ), (1, ))
    assert_size_stride(view_12, (8, 196, 128), (25088, 128, 1))
    assert_size_stride(view_13, (1568, 128), (128, 1))
    assert_size_stride(mm_1, (1568, 128), (128, 1))
    assert_size_stride(squeeze_16, (128, ), (1, ))
    assert_size_stride(view_17, (1568, 128), (128, 1))
    assert_size_stride(mm_2, (1568, 256), (256, 1))
    assert_size_stride(squeeze_19, (256, ), (1, ))
    assert_size_stride(view_20, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_21, (1568, 256), (256, 1))
    assert_size_stride(mm_3, (1568, 128), (128, 1))
    assert_size_stride(squeeze_22, (128, ), (1, ))
    assert_size_stride(view_25, (1568, 128), (128, 1))
    assert_size_stride(mm_4, (1568, 256), (256, 1))
    assert_size_stride(squeeze_25, (256, ), (1, ))
    assert_size_stride(view_36, (8, 196, 128), (25088, 128, 1))
    assert_size_stride(view_37, (1568, 128), (128, 1))
    assert_size_stride(mm_5, (1568, 128), (128, 1))
    assert_size_stride(squeeze_28, (128, ), (1, ))
    assert_size_stride(view_41, (1568, 128), (128, 1))
    assert_size_stride(mm_6, (1568, 256), (256, 1))
    assert_size_stride(squeeze_31, (256, ), (1, ))
    assert_size_stride(view_44, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_45, (1568, 256), (256, 1))
    assert_size_stride(mm_7, (1568, 128), (128, 1))
    assert_size_stride(squeeze_34, (128, ), (1, ))
    assert_size_stride(view_49, (1568, 128), (128, 1))
    assert_size_stride(mm_8, (1568, 256), (256, 1))
    assert_size_stride(squeeze_37, (256, ), (1, ))
    assert_size_stride(view_60, (8, 196, 128), (25088, 128, 1))
    assert_size_stride(view_61, (1568, 128), (128, 1))
    assert_size_stride(mm_9, (1568, 128), (128, 1))
    assert_size_stride(squeeze_40, (128, ), (1, ))
    assert_size_stride(view_65, (1568, 128), (128, 1))
    assert_size_stride(mm_10, (1568, 256), (256, 1))
    assert_size_stride(squeeze_43, (256, ), (1, ))
    assert_size_stride(view_68, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_69, (1568, 256), (256, 1))
    assert_size_stride(mm_11, (1568, 128), (128, 1))
    assert_size_stride(squeeze_46, (128, ), (1, ))
    assert_size_stride(view_73, (1568, 128), (128, 1))
    assert_size_stride(mm_12, (1568, 256), (256, 1))
    assert_size_stride(squeeze_49, (256, ), (1, ))
    assert_size_stride(view_84, (8, 196, 128), (25088, 128, 1))
    assert_size_stride(view_85, (1568, 128), (128, 1))
    assert_size_stride(mm_13, (1568, 128), (128, 1))
    assert_size_stride(squeeze_52, (128, ), (1, ))
    assert_size_stride(view_89, (1568, 128), (128, 1))
    assert_size_stride(mm_14, (1568, 256), (256, 1))
    assert_size_stride(squeeze_55, (256, ), (1, ))
    assert_size_stride(view_92, (8, 196, 256), (50176, 256, 1))
    assert_size_stride(view_93, (1568, 256), (256, 1))
    assert_size_stride(mm_15, (1568, 128), (128, 1))
    assert_size_stride(squeeze_58, (128, ), (1, ))
    assert_size_stride(view_97, (1568, 128), (128, 1))
    assert_size_stride(mm_16, (1568, 640), (640, 1))
    assert_size_stride(squeeze_61, (640, ), (1, ))
    assert_size_stride(view_104, (392, 128), (128, 1))
    assert_size_stride(mm_17, (392, 128), (128, 1))
    assert_size_stride(squeeze_64, (128, ), (1, ))
    assert_size_stride(view_115, (8, 49, 512), (25088, 512, 1))
    assert_size_stride(view_116, (392, 512), (512, 1))
    assert_size_stride(mm_18, (392, 256), (256, 1))
    assert_size_stride(squeeze_67, (256, ), (1, ))
    assert_size_stride(view_120, (392, 256), (256, 1))
    assert_size_stride(mm_19, (392, 512), (512, 1))
    assert_size_stride(squeeze_70, (512, ), (1, ))
    assert_size_stride(view_123, (8, 49, 512), (25088, 512, 1))
    assert_size_stride(view_124, (392, 512), (512, 1))
    assert_size_stride(mm_20, (392, 256), (256, 1))
    assert_size_stride(squeeze_73, (256, ), (1, ))
    assert_size_stride(view_128, (392, 256), (256, 1))
    assert_size_stride(mm_21, (392, 512), (512, 1))
    assert_size_stride(squeeze_76, (512, ), (1, ))
    assert_size_stride(view_139, (8, 49, 256), (12544, 256, 1))
    assert_size_stride(view_140, (392, 256), (256, 1))
    assert_size_stride(mm_22, (392, 256), (256, 1))
    assert_size_stride(squeeze_79, (256, ), (1, ))
    assert_size_stride(view_144, (392, 256), (256, 1))
    assert_size_stride(mm_23, (392, 512), (512, 1))
    assert_size_stride(squeeze_82, (512, ), (1, ))
    assert_size_stride(view_147, (8, 49, 512), (25088, 512, 1))
    assert_size_stride(view_148, (392, 512), (512, 1))
    assert_size_stride(mm_24, (392, 256), (256, 1))
    assert_size_stride(squeeze_85, (256, ), (1, ))
    assert_size_stride(view_152, (392, 256), (256, 1))
    assert_size_stride(mm_25, (392, 512), (512, 1))
    assert_size_stride(squeeze_88, (512, ), (1, ))
    assert_size_stride(view_163, (8, 49, 256), (12544, 256, 1))
    assert_size_stride(view_164, (392, 256), (256, 1))
    assert_size_stride(mm_26, (392, 256), (256, 1))
    assert_size_stride(squeeze_91, (256, ), (1, ))
    assert_size_stride(view_168, (392, 256), (256, 1))
    assert_size_stride(mm_27, (392, 512), (512, 1))
    assert_size_stride(squeeze_94, (512, ), (1, ))
    assert_size_stride(view_171, (8, 49, 512), (25088, 512, 1))
    assert_size_stride(view_172, (392, 512), (512, 1))
    assert_size_stride(mm_28, (392, 256), (256, 1))
    assert_size_stride(squeeze_97, (256, ), (1, ))
    assert_size_stride(view_176, (392, 256), (256, 1))
    assert_size_stride(mm_29, (392, 512), (512, 1))
    assert_size_stride(squeeze_100, (512, ), (1, ))
    assert_size_stride(view_187, (8, 49, 256), (12544, 256, 1))
    assert_size_stride(view_188, (392, 256), (256, 1))
    assert_size_stride(mm_30, (392, 256), (256, 1))
    assert_size_stride(squeeze_103, (256, ), (1, ))
    assert_size_stride(view_192, (392, 256), (256, 1))
    assert_size_stride(mm_31, (392, 512), (512, 1))
    assert_size_stride(squeeze_106, (512, ), (1, ))
    assert_size_stride(view_195, (8, 49, 512), (25088, 512, 1))
    assert_size_stride(view_196, (392, 512), (512, 1))
    assert_size_stride(mm_32, (392, 256), (256, 1))
    assert_size_stride(squeeze_109, (256, ), (1, ))
    assert_size_stride(view_200, (392, 256), (256, 1))
    assert_size_stride(mm_33, (392, 512), (512, 1))
    assert_size_stride(squeeze_112, (512, ), (1, ))
    assert_size_stride(view_211, (8, 49, 256), (12544, 256, 1))
    assert_size_stride(view_212, (392, 256), (256, 1))
    assert_size_stride(mm_34, (392, 256), (256, 1))
    assert_size_stride(squeeze_115, (256, ), (1, ))
    assert_size_stride(view_216, (392, 256), (256, 1))
    assert_size_stride(mm_35, (392, 512), (512, 1))
    assert_size_stride(squeeze_118, (512, ), (1, ))
    assert_size_stride(view_219, (8, 49, 512), (25088, 512, 1))
    assert_size_stride(view_220, (392, 512), (512, 1))
    assert_size_stride(mm_36, (392, 256), (256, 1))
    assert_size_stride(squeeze_121, (256, ), (1, ))
    assert_size_stride(view_224, (392, 256), (256, 1))
    assert_size_stride(mm_37, (392, 1280), (1280, 1))
    assert_size_stride(squeeze_124, (1280, ), (1, ))
    assert_size_stride(view_231, (128, 256), (256, 1))
    assert_size_stride(mm_38, (128, 256), (256, 1))
    assert_size_stride(squeeze_127, (256, ), (1, ))
    assert_size_stride(view_242, (8, 16, 1024), (16384, 1024, 1))
    assert_size_stride(view_243, (128, 1024), (1024, 1))
    assert_size_stride(mm_39, (128, 384), (384, 1))
    assert_size_stride(squeeze_130, (384, ), (1, ))
    assert_size_stride(view_247, (128, 384), (384, 1))
    assert_size_stride(mm_40, (128, 768), (768, 1))
    assert_size_stride(squeeze_133, (768, ), (1, ))
    assert_size_stride(view_250, (8, 16, 768), (12288, 768, 1))
    assert_size_stride(view_251, (128, 768), (768, 1))
    assert_size_stride(mm_41, (128, 384), (384, 1))
    assert_size_stride(squeeze_136, (384, ), (1, ))
    assert_size_stride(view_255, (128, 384), (384, 1))
    assert_size_stride(mm_42, (128, 768), (768, 1))
    assert_size_stride(squeeze_139, (768, ), (1, ))
    assert_size_stride(view_266, (8, 16, 384), (6144, 384, 1))
    assert_size_stride(view_267, (128, 384), (384, 1))
    assert_size_stride(mm_43, (128, 384), (384, 1))
    assert_size_stride(squeeze_142, (384, ), (1, ))
    assert_size_stride(view_271, (128, 384), (384, 1))
    assert_size_stride(mm_44, (128, 768), (768, 1))
    assert_size_stride(squeeze_145, (768, ), (1, ))
    assert_size_stride(view_274, (8, 16, 768), (12288, 768, 1))
    assert_size_stride(view_275, (128, 768), (768, 1))
    assert_size_stride(mm_45, (128, 384), (384, 1))
    assert_size_stride(squeeze_148, (384, ), (1, ))
    assert_size_stride(view_279, (128, 384), (384, 1))
    assert_size_stride(mm_46, (128, 768), (768, 1))
    assert_size_stride(squeeze_151, (768, ), (1, ))
    assert_size_stride(view_290, (8, 16, 384), (6144, 384, 1))
    assert_size_stride(view_291, (128, 384), (384, 1))
    assert_size_stride(mm_47, (128, 384), (384, 1))
    assert_size_stride(squeeze_154, (384, ), (1, ))
    assert_size_stride(view_295, (128, 384), (384, 1))
    assert_size_stride(mm_48, (128, 768), (768, 1))
    assert_size_stride(squeeze_157, (768, ), (1, ))
    assert_size_stride(view_298, (8, 16, 768), (12288, 768, 1))
    assert_size_stride(view_299, (128, 768), (768, 1))
    assert_size_stride(mm_49, (128, 384), (384, 1))
    assert_size_stride(squeeze_160, (384, ), (1, ))
    assert_size_stride(view_303, (128, 384), (384, 1))
    assert_size_stride(mm_50, (128, 768), (768, 1))
    assert_size_stride(squeeze_163, (768, ), (1, ))
    assert_size_stride(view_314, (8, 16, 384), (6144, 384, 1))
    assert_size_stride(view_315, (128, 384), (384, 1))
    assert_size_stride(mm_51, (128, 384), (384, 1))
    assert_size_stride(squeeze_166, (384, ), (1, ))
    assert_size_stride(view_319, (128, 384), (384, 1))
    assert_size_stride(mm_52, (128, 768), (768, 1))
    assert_size_stride(squeeze_169, (768, ), (1, ))
    assert_size_stride(view_322, (8, 16, 768), (12288, 768, 1))
    assert_size_stride(view_323, (128, 768), (768, 1))
    assert_size_stride(mm_53, (128, 384), (384, 1))
    assert_size_stride(squeeze_172, (384, ), (1, ))
    assert_size_stride(view_327, (128, 384), (384, 1))
    assert_size_stride(mm_54, (128, 768), (768, 1))
    assert_size_stride(squeeze_175, (768, ), (1, ))
    assert_size_stride(view_338, (8, 16, 384), (6144, 384, 1))
    assert_size_stride(view_339, (128, 384), (384, 1))
    assert_size_stride(mm_55, (128, 384), (384, 1))
    assert_size_stride(squeeze_178, (384, ), (1, ))
    assert_size_stride(view_343, (128, 384), (384, 1))
    assert_size_stride(mm_56, (128, 768), (768, 1))
    assert_size_stride(squeeze_181, (768, ), (1, ))
    assert_size_stride(view_346, (8, 16, 768), (12288, 768, 1))
    assert_size_stride(view_347, (128, 768), (768, 1))
    assert_size_stride(mm_57, (128, 384), (384, 1))
    assert_size_stride(squeeze_184, (384, ), (1, ))
    assert_size_stride(mean, (8, 384), (384, 1))
    assert_size_stride(clone_81, (8, 384), (384, 1))
    assert_size_stride(clone_82, (8, 384), (384, 1))
    assert_size_stride(permute_117, (1000, 384), (384, 1))
    assert_size_stride(permute_121, (1000, 384), (384, 1))
    assert_size_stride(unsqueeze_25, (1, 384), (384, 1))
    assert_size_stride(permute_127, (384, 768), (768, 1))
    assert_size_stride(unsqueeze_29, (1, 768), (768, 1))
    assert_size_stride(permute_131, (768, 384), (384, 1))
    assert_size_stride(unsqueeze_33, (1, 384), (384, 1))
    assert_size_stride(permute_135, (384, 384), (384, 1))
    assert_size_stride(permute_138, (96, 16, 16), (256, 1, 16))
    assert_size_stride(permute_139, (96, 32, 16), (512, 1, 32))
    assert_size_stride(alias_14, (8, 12, 16, 16), (3072, 256, 16, 1))
    assert_size_stride(permute_140, (96, 16, 16), (256, 1, 16))
    assert_size_stride(permute_141, (96, 16, 16), (256, 1, 16))
    assert_size_stride(unsqueeze_37, (1, 768), (768, 1))
    assert_size_stride(permute_147, (768, 384), (384, 1))
    assert_size_stride(unsqueeze_41, (1, 384), (384, 1))
    assert_size_stride(permute_151, (384, 768), (768, 1))
    assert_size_stride(unsqueeze_45, (1, 768), (768, 1))
    assert_size_stride(permute_155, (768, 384), (384, 1))
    assert_size_stride(unsqueeze_49, (1, 384), (384, 1))
    assert_size_stride(permute_159, (384, 384), (384, 1))
    assert_size_stride(permute_162, (96, 16, 16), (256, 1, 16))
    assert_size_stride(permute_163, (96, 32, 16), (512, 1, 32))
    assert_size_stride(alias_15, (8, 12, 16, 16), (3072, 256, 16, 1))
    assert_size_stride(permute_164, (96, 16, 16), (256, 1, 16))
    assert_size_stride(permute_165, (96, 16, 16), (256, 1, 16))
    assert_size_stride(unsqueeze_53, (1, 768), (768, 1))
    assert_size_stride(permute_171, (768, 384), (384, 1))
    assert_size_stride(unsqueeze_57, (1, 384), (384, 1))
    assert_size_stride(permute_175, (384, 768), (768, 1))
    assert_size_stride(unsqueeze_61, (1, 768), (768, 1))
    assert_size_stride(permute_179, (768, 384), (384, 1))
    assert_size_stride(unsqueeze_65, (1, 384), (384, 1))
    assert_size_stride(permute_183, (384, 384), (384, 1))
    assert_size_stride(permute_186, (96, 16, 16), (256, 1, 16))
    assert_size_stride(permute_187, (96, 32, 16), (512, 1, 32))
    assert_size_stride(alias_16, (8, 12, 16, 16), (3072, 256, 16, 1))
    assert_size_stride(permute_188, (96, 16, 16), (256, 1, 16))
    assert_size_stride(permute_189, (96, 16, 16), (256, 1, 16))
    assert_size_stride(unsqueeze_69, (1, 768), (768, 1))
    assert_size_stride(permute_195, (768, 384), (384, 1))
    assert_size_stride(unsqueeze_73, (1, 384), (384, 1))
    assert_size_stride(permute_199, (384, 768), (768, 1))
    assert_size_stride(unsqueeze_77, (1, 768), (768, 1))
    assert_size_stride(permute_203, (768, 384), (384, 1))
    assert_size_stride(unsqueeze_81, (1, 384), (384, 1))
    assert_size_stride(permute_207, (384, 384), (384, 1))
    assert_size_stride(permute_210, (96, 16, 16), (256, 1, 16))
    assert_size_stride(permute_211, (96, 32, 16), (512, 1, 32))
    assert_size_stride(alias_17, (8, 12, 16, 16), (3072, 256, 16, 1))
    assert_size_stride(permute_212, (96, 16, 16), (256, 1, 16))
    assert_size_stride(permute_213, (96, 16, 16), (256, 1, 16))
    assert_size_stride(unsqueeze_85, (1, 768), (768, 1))
    assert_size_stride(permute_219, (768, 384), (384, 1))
    assert_size_stride(unsqueeze_89, (1, 384), (384, 1))
    assert_size_stride(permute_223, (384, 768), (768, 1))
    assert_size_stride(unsqueeze_93, (1, 768), (768, 1))
    assert_size_stride(permute_227, (768, 384), (384, 1))
    assert_size_stride(unsqueeze_97, (1, 384), (384, 1))
    assert_size_stride(permute_231, (384, 1024), (1024, 1))
    assert_size_stride(permute_234, (128, 49, 16), (784, 1, 49))
    assert_size_stride(permute_235, (128, 64, 49), (3136, 1, 64))
    assert_size_stride(alias_18, (8, 16, 16, 49), (12544, 784, 49, 1))
    assert_size_stride(permute_236, (128, 16, 16), (256, 1, 16))
    assert_size_stride(permute_237, (128, 49, 16), (784, 1, 49))
    assert_size_stride(unsqueeze_101, (1, 256), (256, 1))
    assert_size_stride(permute_241, (256, 256), (256, 1))
    assert_size_stride(unsqueeze_105, (1, 1280), (1280, 1))
    assert_size_stride(permute_247, (1280, 256), (256, 1))
    assert_size_stride(unsqueeze_109, (1, 256), (256, 1))
    assert_size_stride(permute_251, (256, 512), (512, 1))
    assert_size_stride(unsqueeze_113, (1, 512), (512, 1))
    assert_size_stride(permute_255, (512, 256), (256, 1))
    assert_size_stride(unsqueeze_117, (1, 256), (256, 1))
    assert_size_stride(permute_259, (256, 256), (256, 1))
    assert_size_stride(permute_262, (64, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_263, (64, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_19, (8, 8, 49, 49), (19208, 2401, 49, 1))
    assert_size_stride(permute_264, (64, 16, 49), (784, 1, 16))
    assert_size_stride(permute_265, (64, 49, 16), (784, 1, 49))
    assert_size_stride(unsqueeze_121, (1, 512), (512, 1))
    assert_size_stride(permute_271, (512, 256), (256, 1))
    assert_size_stride(unsqueeze_125, (1, 256), (256, 1))
    assert_size_stride(permute_275, (256, 512), (512, 1))
    assert_size_stride(unsqueeze_129, (1, 512), (512, 1))
    assert_size_stride(permute_279, (512, 256), (256, 1))
    assert_size_stride(unsqueeze_133, (1, 256), (256, 1))
    assert_size_stride(permute_283, (256, 256), (256, 1))
    assert_size_stride(permute_286, (64, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_287, (64, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_20, (8, 8, 49, 49), (19208, 2401, 49, 1))
    assert_size_stride(permute_288, (64, 16, 49), (784, 1, 16))
    assert_size_stride(permute_289, (64, 49, 16), (784, 1, 49))
    assert_size_stride(unsqueeze_137, (1, 512), (512, 1))
    assert_size_stride(permute_295, (512, 256), (256, 1))
    assert_size_stride(unsqueeze_141, (1, 256), (256, 1))
    assert_size_stride(permute_299, (256, 512), (512, 1))
    assert_size_stride(unsqueeze_145, (1, 512), (512, 1))
    assert_size_stride(permute_303, (512, 256), (256, 1))
    assert_size_stride(unsqueeze_149, (1, 256), (256, 1))
    assert_size_stride(permute_307, (256, 256), (256, 1))
    assert_size_stride(permute_310, (64, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_311, (64, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_21, (8, 8, 49, 49), (19208, 2401, 49, 1))
    assert_size_stride(permute_312, (64, 16, 49), (784, 1, 16))
    assert_size_stride(permute_313, (64, 49, 16), (784, 1, 49))
    assert_size_stride(unsqueeze_153, (1, 512), (512, 1))
    assert_size_stride(permute_319, (512, 256), (256, 1))
    assert_size_stride(unsqueeze_157, (1, 256), (256, 1))
    assert_size_stride(permute_323, (256, 512), (512, 1))
    assert_size_stride(unsqueeze_161, (1, 512), (512, 1))
    assert_size_stride(permute_327, (512, 256), (256, 1))
    assert_size_stride(unsqueeze_165, (1, 256), (256, 1))
    assert_size_stride(permute_331, (256, 256), (256, 1))
    assert_size_stride(permute_334, (64, 49, 49), (2401, 1, 49))
    assert_size_stride(permute_335, (64, 32, 49), (1568, 1, 32))
    assert_size_stride(alias_22, (8, 8, 49, 49), (19208, 2401, 49, 1))
    assert_size_stride(permute_336, (64, 16, 49), (784, 1, 16))
    assert_size_stride(permute_337, (64, 49, 16), (784, 1, 49))
    assert_size_stride(unsqueeze_169, (1, 512), (512, 1))
    assert_size_stride(permute_343, (512, 256), (256, 1))
    assert_size_stride(unsqueeze_173, (1, 256), (256, 1))
    assert_size_stride(permute_347, (256, 512), (512, 1))
    assert_size_stride(unsqueeze_177, (1, 512), (512, 1))
    assert_size_stride(permute_351, (512, 256), (256, 1))
    assert_size_stride(unsqueeze_181, (1, 256), (256, 1))
    assert_size_stride(permute_355, (256, 512), (512, 1))
    assert_size_stride(permute_358, (64, 196, 49), (9604, 1, 196))
    assert_size_stride(permute_359, (64, 64, 196), (12544, 1, 64))
    assert_size_stride(alias_23, (8, 8, 49, 196), (76832, 9604, 196, 1))
    assert_size_stride(permute_360, (64, 16, 49), (784, 1, 16))
    assert_size_stride(permute_361, (64, 196, 16), (3136, 1, 196))
    assert_size_stride(unsqueeze_185, (1, 128), (128, 1))
    assert_size_stride(permute_365, (128, 128), (128, 1))
    assert_size_stride(unsqueeze_189, (1, 640), (640, 1))
    assert_size_stride(permute_371, (640, 128), (128, 1))
    assert_size_stride(unsqueeze_193, (1, 128), (128, 1))
    assert_size_stride(permute_375, (128, 256), (256, 1))
    assert_size_stride(unsqueeze_197, (1, 256), (256, 1))
    assert_size_stride(permute_379, (256, 128), (128, 1))
    assert_size_stride(unsqueeze_201, (1, 128), (128, 1))
    assert_size_stride(permute_383, (128, 128), (128, 1))
    assert_size_stride(permute_386, (32, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_387, (32, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_24, (8, 4, 196, 196), (153664, 38416, 196, 1))
    assert_size_stride(permute_388, (32, 16, 196), (3136, 1, 16))
    assert_size_stride(permute_389, (32, 196, 16), (3136, 1, 196))
    assert_size_stride(unsqueeze_205, (1, 256), (256, 1))
    assert_size_stride(permute_395, (256, 128), (128, 1))
    assert_size_stride(unsqueeze_209, (1, 128), (128, 1))
    assert_size_stride(permute_399, (128, 256), (256, 1))
    assert_size_stride(unsqueeze_213, (1, 256), (256, 1))
    assert_size_stride(permute_403, (256, 128), (128, 1))
    assert_size_stride(unsqueeze_217, (1, 128), (128, 1))
    assert_size_stride(permute_407, (128, 128), (128, 1))
    assert_size_stride(permute_410, (32, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_411, (32, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_25, (8, 4, 196, 196), (153664, 38416, 196, 1))
    assert_size_stride(permute_412, (32, 16, 196), (3136, 1, 16))
    assert_size_stride(permute_413, (32, 196, 16), (3136, 1, 196))
    assert_size_stride(unsqueeze_221, (1, 256), (256, 1))
    assert_size_stride(permute_419, (256, 128), (128, 1))
    assert_size_stride(unsqueeze_225, (1, 128), (128, 1))
    assert_size_stride(permute_423, (128, 256), (256, 1))
    assert_size_stride(unsqueeze_229, (1, 256), (256, 1))
    assert_size_stride(permute_427, (256, 128), (128, 1))
    assert_size_stride(unsqueeze_233, (1, 128), (128, 1))
    assert_size_stride(permute_431, (128, 128), (128, 1))
    assert_size_stride(permute_434, (32, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_435, (32, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_26, (8, 4, 196, 196), (153664, 38416, 196, 1))
    assert_size_stride(permute_436, (32, 16, 196), (3136, 1, 16))
    assert_size_stride(permute_437, (32, 196, 16), (3136, 1, 196))
    assert_size_stride(unsqueeze_237, (1, 256), (256, 1))
    assert_size_stride(permute_443, (256, 128), (128, 1))
    assert_size_stride(unsqueeze_241, (1, 128), (128, 1))
    assert_size_stride(permute_447, (128, 256), (256, 1))
    assert_size_stride(unsqueeze_245, (1, 256), (256, 1))
    assert_size_stride(permute_451, (256, 128), (128, 1))
    assert_size_stride(unsqueeze_249, (1, 128), (128, 1))
    assert_size_stride(permute_455, (128, 128), (128, 1))
    assert_size_stride(permute_458, (32, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_459, (32, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_27, (8, 4, 196, 196), (153664, 38416, 196, 1))
    assert_size_stride(permute_460, (32, 16, 196), (3136, 1, 16))
    assert_size_stride(permute_461, (32, 196, 16), (3136, 1, 196))
    assert_size_stride(unsqueeze_253, (1, 256), (256, 1))
    assert_size_stride(permute_467, (256, 128), (128, 1))
    assert_size_stride(unsqueeze_259, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_271, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_283, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(unsqueeze_295, (1, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf3 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.div]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_div_0.run(tangents_1, buf3, 8000, grid=grid(8000), stream=stream0)
        del tangents_1
        buf10 = empty((8, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf3, permute_121, out=buf10)
        del permute_121
        buf4 = empty((8, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf3, permute_117, out=buf4)
        del permute_117
        buf12 = empty((384, ), device='cuda', dtype=torch.float32)
        buf7 = empty((384, ), device='cuda', dtype=torch.float32)
        buf0 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf8 = empty((384, ), device='cuda', dtype=torch.float32)
        buf13 = empty((384, ), device='cuda', dtype=torch.float32)
        buf1 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf9 = empty((384, ), device='cuda', dtype=torch.float32)
        buf14 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___head_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward]
        triton_per_fused__native_batch_norm_legit_functional_native_batch_norm_backward_1.run(buf10, buf4, mean, buf12, buf7, buf0, buf8, buf13, buf1, buf9, buf14, 384, 8, grid=grid(384), stream=stream0)
        buf5 = empty((1000, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf3, (1000, 8), (1, 1000), 0), clone_82, out=buf5)
        del clone_82
        buf6 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_2.run(buf3, buf6, 1000, 8, grid=grid(1000), stream=stream0)
        buf11 = empty((1000, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf3, (1000, 8), (1, 1000), 0), clone_81, out=buf11)
        del buf3
        del clone_81
        buf15 = buf10; del buf10  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_batch_norm_backward]
        triton_poi_fused_add_native_batch_norm_backward_3.run(buf15, buf4, mean, buf0, buf8, buf1, buf7, primals_205, buf13, buf12, primals_201, 3072, grid=grid(3072), stream=stream0)
        del mean
        del primals_201
        del primals_205
        buf16 = buf8; del buf8  # reuse
        buf17 = buf13; del buf13  # reuse
        buf18 = reinterpret_tensor(buf1, (384, ), (1, ), 0); del buf1  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_4.run(buf15, mm_57, unsqueeze_25, squeeze_184, buf16, buf17, buf18, 384, 128, grid=grid(384), stream=stream0)
        buf19 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_5.run(buf15, mm_57, unsqueeze_25, buf17, squeeze_184, buf16, primals_199, buf19, 49152, grid=grid(49152), stream=stream0)
        del mm_57
        del primals_199
        del squeeze_184
        del unsqueeze_25
        buf20 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf19, (384, 128), (1, 384), 0), view_347, out=buf20)
        del view_347
        buf21 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf19, (128, 384), (384, 1), 0), permute_127, out=buf21)
        del permute_127
        buf22 = empty((768, ), device='cuda', dtype=torch.float32)
        buf23 = empty((768, ), device='cuda', dtype=torch.float32)
        buf24 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_6.run(view_346, buf21, mm_56, unsqueeze_29, squeeze_181, buf22, buf23, buf24, 768, 128, grid=grid(768), stream=stream0)
        buf25 = buf21; del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_7.run(buf25, view_346, mm_56, unsqueeze_29, buf23, squeeze_181, buf22, primals_196, 98304, grid=grid(98304), stream=stream0)
        del mm_56
        del primals_196
        del squeeze_181
        del unsqueeze_29
        del view_346
        buf26 = empty((768, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf25, (768, 128), (1, 768), 0), view_343, out=buf26)
        del view_343
        buf27 = buf19; del buf19  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf25, (128, 768), (768, 1), 0), permute_131, out=buf27)
        del permute_131
        buf28 = buf17; del buf17  # reuse
        buf29 = reinterpret_tensor(buf0, (384, ), (1, ), 0); del buf0  # reuse
        buf30 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_8.run(buf15, buf27, mm_55, unsqueeze_33, squeeze_178, buf28, buf29, buf30, 384, 128, grid=grid(384), stream=stream0)
        buf31 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_9.run(buf15, buf27, mm_55, unsqueeze_33, buf29, squeeze_178, buf28, primals_193, buf31, 49152, grid=grid(49152), stream=stream0)
        del mm_55
        del primals_193
        del squeeze_178
        del unsqueeze_33
        buf32 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf31, (384, 128), (1, 384), 0), view_339, out=buf32)
        del view_339
        buf33 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf31, (128, 384), (384, 1), 0), permute_135, out=buf33)
        del permute_135
        buf34 = reinterpret_tensor(buf31, (8, 12, 16, 32), (6144, 512, 32, 1), 0); del buf31  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(view_338, buf33, buf34, 49152, grid=grid(49152), stream=stream0)
        del view_338
        buf35 = reinterpret_tensor(buf33, (96, 16, 32), (512, 32, 1), 0); del buf33  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_138, reinterpret_tensor(buf34, (96, 16, 32), (512, 32, 1), 0), out=buf35)
        del permute_138
        buf36 = empty((96, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf34, (96, 16, 32), (512, 32, 1), 0), permute_139, out=buf36)
        del permute_139
        buf37 = empty_strided((8, 12, 16, 1), (192, 16, 1, 1536), device='cuda', dtype=torch.float32)
        buf42 = empty((8, 12, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_11.run(buf36, alias_14, buf37, buf42, 1536, 16, grid=grid(1536), stream=stream0)
        buf38 = reinterpret_tensor(buf4, (1, 12, 16, 16), (3072, 256, 16, 1), 0); del buf4  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]
        triton_per_fused__softmax_backward_data_sum_12.run(buf36, alias_14, buf37, buf38, 3072, 8, grid=grid(3072), stream=stream0)
        del alias_14
        buf39 = empty((12, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_13.run(buf39, 192, grid=grid(192), stream=stream0)
        aten.index_put_(buf39, [None, primals_222], reinterpret_tensor(buf38, (12, 16, 16), (256, 16, 1), 0), True)
        del primals_222
        buf43 = buf36; del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_140, reinterpret_tensor(buf42, (96, 16, 16), (256, 16, 1), 0), out=buf43)
        del permute_140
        buf44 = empty((96, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf42, (96, 16, 16), (256, 16, 1), 0), permute_141, out=buf44)
        del permute_141
        buf45 = buf23; del buf23  # reuse
        buf46 = empty((768, ), device='cuda', dtype=torch.float32)
        buf48 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_14.run(buf44, buf43, buf35, mm_54, unsqueeze_37, squeeze_175, buf45, buf46, buf48, 768, 128, grid=grid(768), stream=stream0)
        buf47 = buf25; del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_15.run(buf44, buf43, buf35, mm_54, unsqueeze_37, buf46, squeeze_175, buf45, primals_190, buf47, 128, 768, grid=grid(128, 768), stream=stream0)
        del mm_54
        del primals_190
        del squeeze_175
        del unsqueeze_37
        buf49 = empty((768, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf47, (768, 128), (1, 768), 0), view_327, out=buf49)
        del view_327
        buf50 = reinterpret_tensor(buf35, (128, 384), (384, 1), 0); del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf47, (128, 768), (768, 1), 0), permute_147, out=buf50)
        del permute_147
        buf51 = buf29; del buf29  # reuse
        buf52 = empty((384, ), device='cuda', dtype=torch.float32)
        buf54 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_16.run(buf15, buf27, buf50, mm_53, unsqueeze_41, squeeze_172, buf51, buf52, buf54, 384, 128, grid=grid(384), stream=stream0)
        buf53 = reinterpret_tensor(buf34, (128, 384), (384, 1), 0); del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_17.run(buf15, buf27, buf50, mm_53, unsqueeze_41, buf52, squeeze_172, buf51, primals_187, buf53, 49152, grid=grid(49152), stream=stream0)
        del mm_53
        del primals_187
        del squeeze_172
        del unsqueeze_41
        buf55 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf53, (384, 128), (1, 384), 0), view_323, out=buf55)
        del view_323
        buf56 = buf47; del buf47  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf53, (128, 384), (384, 1), 0), permute_151, out=buf56)
        del permute_151
        buf57 = buf46; del buf46  # reuse
        buf58 = empty((768, ), device='cuda', dtype=torch.float32)
        buf59 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_6.run(view_322, buf56, mm_52, unsqueeze_45, squeeze_169, buf57, buf58, buf59, 768, 128, grid=grid(768), stream=stream0)
        buf60 = buf56; del buf56  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_7.run(buf60, view_322, mm_52, unsqueeze_45, buf58, squeeze_169, buf57, primals_184, 98304, grid=grid(98304), stream=stream0)
        del mm_52
        del primals_184
        del squeeze_169
        del unsqueeze_45
        del view_322
        buf61 = empty((768, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf60, (768, 128), (1, 768), 0), view_319, out=buf61)
        del view_319
        buf62 = buf53; del buf53  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf60, (128, 768), (768, 1), 0), permute_155, out=buf62)
        del permute_155
        buf63 = buf52; del buf52  # reuse
        buf64 = empty((384, ), device='cuda', dtype=torch.float32)
        buf66 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_18.run(buf15, buf27, buf50, buf62, mm_51, unsqueeze_49, squeeze_166, buf63, buf64, buf66, 384, 128, grid=grid(384), stream=stream0)
        buf65 = empty((128, 384), device='cuda', dtype=torch.float32)
        buf67 = buf65; del buf65  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_19.run(buf67, buf15, buf27, buf50, buf62, mm_51, unsqueeze_49, buf64, squeeze_166, buf63, primals_181, 49152, grid=grid(49152), stream=stream0)
        del mm_51
        del primals_181
        del squeeze_166
        del unsqueeze_49
        buf68 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf67, (384, 128), (1, 384), 0), view_315, out=buf68)
        del view_315
        buf69 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf67, (128, 384), (384, 1), 0), permute_159, out=buf69)
        del permute_159
        buf70 = reinterpret_tensor(buf67, (8, 12, 16, 32), (6144, 512, 32, 1), 0); del buf67  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(view_314, buf69, buf70, 49152, grid=grid(49152), stream=stream0)
        del view_314
        buf71 = reinterpret_tensor(buf69, (96, 16, 32), (512, 32, 1), 0); del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_162, reinterpret_tensor(buf70, (96, 16, 32), (512, 32, 1), 0), out=buf71)
        del permute_162
        buf72 = buf44; del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf70, (96, 16, 32), (512, 32, 1), 0), permute_163, out=buf72)
        del permute_163
        buf73 = buf37; del buf37  # reuse
        buf78 = reinterpret_tensor(buf43, (8, 12, 16, 16), (3072, 256, 16, 1), 0); del buf43  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_11.run(buf72, alias_15, buf73, buf78, 1536, 16, grid=grid(1536), stream=stream0)
        buf74 = buf38; del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]
        triton_per_fused__softmax_backward_data_sum_12.run(buf72, alias_15, buf73, buf74, 3072, 8, grid=grid(3072), stream=stream0)
        del alias_15
        buf75 = empty((12, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_13.run(buf75, 192, grid=grid(192), stream=stream0)
        aten.index_put_(buf75, [None, primals_221], reinterpret_tensor(buf74, (12, 16, 16), (256, 16, 1), 0), True)
        del buf74
        del primals_221
        buf79 = buf72; del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_164, reinterpret_tensor(buf78, (96, 16, 16), (256, 16, 1), 0), out=buf79)
        del permute_164
        buf80 = reinterpret_tensor(buf42, (96, 16, 16), (256, 16, 1), 0); del buf42  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf78, (96, 16, 16), (256, 16, 1), 0), permute_165, out=buf80)
        del permute_165
        buf81 = buf58; del buf58  # reuse
        buf82 = empty((768, ), device='cuda', dtype=torch.float32)
        buf84 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_14.run(buf80, buf79, buf71, mm_50, unsqueeze_53, squeeze_163, buf81, buf82, buf84, 768, 128, grid=grid(768), stream=stream0)
        buf83 = buf60; del buf60  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_15.run(buf80, buf79, buf71, mm_50, unsqueeze_53, buf82, squeeze_163, buf81, primals_178, buf83, 128, 768, grid=grid(128, 768), stream=stream0)
        del mm_50
        del primals_178
        del squeeze_163
        del unsqueeze_53
        buf85 = empty((768, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf83, (768, 128), (1, 768), 0), view_303, out=buf85)
        del view_303
        buf86 = reinterpret_tensor(buf71, (128, 384), (384, 1), 0); del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf83, (128, 768), (768, 1), 0), permute_171, out=buf86)
        del permute_171
        buf87 = reinterpret_tensor(buf27, (8, 16, 384), (6144, 384, 1), 0); del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.div]
        triton_poi_fused_add_div_20.run(buf87, buf15, buf50, buf62, buf86, 49152, grid=grid(49152), stream=stream0)
        buf88 = buf64; del buf64  # reuse
        buf89 = empty((384, ), device='cuda', dtype=torch.float32)
        buf90 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_21.run(buf87, mm_49, unsqueeze_57, squeeze_160, buf88, buf89, buf90, 384, 128, grid=grid(384), stream=stream0)
        buf91 = buf86; del buf86  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_22.run(buf87, mm_49, unsqueeze_57, buf89, squeeze_160, buf88, primals_175, buf91, 49152, grid=grid(49152), stream=stream0)
        del mm_49
        del primals_175
        del squeeze_160
        del unsqueeze_57
        buf92 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf91, (384, 128), (1, 384), 0), view_299, out=buf92)
        del view_299
        buf93 = buf83; del buf83  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf91, (128, 384), (384, 1), 0), permute_175, out=buf93)
        del permute_175
        buf94 = buf82; del buf82  # reuse
        buf95 = empty((768, ), device='cuda', dtype=torch.float32)
        buf96 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_6.run(view_298, buf93, mm_48, unsqueeze_61, squeeze_157, buf94, buf95, buf96, 768, 128, grid=grid(768), stream=stream0)
        buf97 = buf93; del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_7.run(buf97, view_298, mm_48, unsqueeze_61, buf95, squeeze_157, buf94, primals_172, 98304, grid=grid(98304), stream=stream0)
        del mm_48
        del primals_172
        del squeeze_157
        del unsqueeze_61
        del view_298
        buf98 = empty((768, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf97, (768, 128), (1, 768), 0), view_295, out=buf98)
        del view_295
        buf99 = buf91; del buf91  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf97, (128, 768), (768, 1), 0), permute_179, out=buf99)
        del permute_179
        buf100 = buf89; del buf89  # reuse
        buf101 = empty((384, ), device='cuda', dtype=torch.float32)
        buf102 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_23.run(buf87, buf99, mm_47, unsqueeze_65, squeeze_154, buf100, buf101, buf102, 384, 128, grid=grid(384), stream=stream0)
        buf103 = buf62; del buf62  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_24.run(buf87, buf99, mm_47, unsqueeze_65, buf101, squeeze_154, buf100, primals_169, buf103, 49152, grid=grid(49152), stream=stream0)
        del mm_47
        del primals_169
        del squeeze_154
        del unsqueeze_65
        buf104 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf103, (384, 128), (1, 384), 0), view_291, out=buf104)
        del view_291
        buf105 = buf50; del buf50  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf103, (128, 384), (384, 1), 0), permute_183, out=buf105)
        del permute_183
        buf106 = reinterpret_tensor(buf103, (8, 12, 16, 32), (6144, 512, 32, 1), 0); del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(view_290, buf105, buf106, 49152, grid=grid(49152), stream=stream0)
        del view_290
        buf107 = reinterpret_tensor(buf105, (96, 16, 32), (512, 32, 1), 0); del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_186, reinterpret_tensor(buf106, (96, 16, 32), (512, 32, 1), 0), out=buf107)
        del permute_186
        buf108 = buf80; del buf80  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf106, (96, 16, 32), (512, 32, 1), 0), permute_187, out=buf108)
        del permute_187
        buf109 = buf73; del buf73  # reuse
        buf114 = reinterpret_tensor(buf79, (8, 12, 16, 16), (3072, 256, 16, 1), 0); del buf79  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_11.run(buf108, alias_16, buf109, buf114, 1536, 16, grid=grid(1536), stream=stream0)
        buf110 = reinterpret_tensor(buf15, (1, 12, 16, 16), (3072, 256, 16, 1), 0); del buf15  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]
        triton_per_fused__softmax_backward_data_sum_12.run(buf108, alias_16, buf109, buf110, 3072, 8, grid=grid(3072), stream=stream0)
        del alias_16
        buf111 = empty((12, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_13.run(buf111, 192, grid=grid(192), stream=stream0)
        aten.index_put_(buf111, [None, primals_220], reinterpret_tensor(buf110, (12, 16, 16), (256, 16, 1), 0), True)
        del primals_220
        buf115 = buf108; del buf108  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_188, reinterpret_tensor(buf114, (96, 16, 16), (256, 16, 1), 0), out=buf115)
        del permute_188
        buf116 = reinterpret_tensor(buf78, (96, 16, 16), (256, 16, 1), 0); del buf78  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf114, (96, 16, 16), (256, 16, 1), 0), permute_189, out=buf116)
        del permute_189
        buf117 = buf95; del buf95  # reuse
        buf118 = empty((768, ), device='cuda', dtype=torch.float32)
        buf120 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_14.run(buf116, buf115, buf107, mm_46, unsqueeze_69, squeeze_151, buf117, buf118, buf120, 768, 128, grid=grid(768), stream=stream0)
        buf119 = buf97; del buf97  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_15.run(buf116, buf115, buf107, mm_46, unsqueeze_69, buf118, squeeze_151, buf117, primals_166, buf119, 128, 768, grid=grid(128, 768), stream=stream0)
        del mm_46
        del primals_166
        del squeeze_151
        del unsqueeze_69
        buf121 = empty((768, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf119, (768, 128), (1, 768), 0), view_279, out=buf121)
        del view_279
        buf122 = reinterpret_tensor(buf107, (128, 384), (384, 1), 0); del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf119, (128, 768), (768, 1), 0), permute_195, out=buf122)
        del permute_195
        buf123 = buf101; del buf101  # reuse
        buf124 = empty((384, ), device='cuda', dtype=torch.float32)
        buf126 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_25.run(buf87, buf99, buf122, mm_45, unsqueeze_73, squeeze_148, buf123, buf124, buf126, 384, 128, grid=grid(384), stream=stream0)
        buf125 = reinterpret_tensor(buf106, (128, 384), (384, 1), 0); del buf106  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_26.run(buf87, buf99, buf122, mm_45, unsqueeze_73, buf124, squeeze_148, buf123, primals_163, buf125, 49152, grid=grid(49152), stream=stream0)
        del mm_45
        del primals_163
        del squeeze_148
        del unsqueeze_73
        buf127 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (384, 128), (1, 384), 0), view_275, out=buf127)
        del view_275
        buf128 = buf119; del buf119  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (128, 384), (384, 1), 0), permute_199, out=buf128)
        del permute_199
        buf129 = buf118; del buf118  # reuse
        buf130 = empty((768, ), device='cuda', dtype=torch.float32)
        buf131 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_6.run(view_274, buf128, mm_44, unsqueeze_77, squeeze_145, buf129, buf130, buf131, 768, 128, grid=grid(768), stream=stream0)
        buf132 = buf128; del buf128  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_7.run(buf132, view_274, mm_44, unsqueeze_77, buf130, squeeze_145, buf129, primals_160, 98304, grid=grid(98304), stream=stream0)
        del mm_44
        del primals_160
        del squeeze_145
        del unsqueeze_77
        del view_274
        buf133 = empty((768, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf132, (768, 128), (1, 768), 0), view_271, out=buf133)
        del view_271
        buf134 = buf125; del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf132, (128, 768), (768, 1), 0), permute_203, out=buf134)
        del permute_203
        buf135 = buf124; del buf124  # reuse
        buf136 = empty((384, ), device='cuda', dtype=torch.float32)
        buf138 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_27.run(buf87, buf99, buf122, buf134, mm_43, unsqueeze_81, squeeze_142, buf135, buf136, buf138, 384, 128, grid=grid(384), stream=stream0)
        buf137 = reinterpret_tensor(buf70, (128, 384), (384, 1), 0); del buf70  # reuse
        buf139 = buf137; del buf137  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_28.run(buf139, buf87, buf99, buf122, buf134, mm_43, unsqueeze_81, buf136, squeeze_142, buf135, primals_157, 49152, grid=grid(49152), stream=stream0)
        del mm_43
        del primals_157
        del squeeze_142
        del unsqueeze_81
        buf140 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf139, (384, 128), (1, 384), 0), view_267, out=buf140)
        del view_267
        buf141 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf139, (128, 384), (384, 1), 0), permute_207, out=buf141)
        del permute_207
        buf142 = reinterpret_tensor(buf139, (8, 12, 16, 32), (6144, 512, 32, 1), 0); del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(view_266, buf141, buf142, 49152, grid=grid(49152), stream=stream0)
        del view_266
        buf143 = reinterpret_tensor(buf141, (96, 16, 32), (512, 32, 1), 0); del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_210, reinterpret_tensor(buf142, (96, 16, 32), (512, 32, 1), 0), out=buf143)
        del permute_210
        buf144 = buf116; del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf142, (96, 16, 32), (512, 32, 1), 0), permute_211, out=buf144)
        del buf142
        del permute_211
        buf145 = buf109; del buf109  # reuse
        buf150 = reinterpret_tensor(buf115, (8, 12, 16, 16), (3072, 256, 16, 1), 0); del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_11.run(buf144, alias_17, buf145, buf150, 1536, 16, grid=grid(1536), stream=stream0)
        buf146 = buf110; del buf110  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]
        triton_per_fused__softmax_backward_data_sum_12.run(buf144, alias_17, buf145, buf146, 3072, 8, grid=grid(3072), stream=stream0)
        del alias_17
        del buf145
        buf147 = empty((12, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_13.run(buf147, 192, grid=grid(192), stream=stream0)
        aten.index_put_(buf147, [None, primals_219], reinterpret_tensor(buf146, (12, 16, 16), (256, 16, 1), 0), True)
        del buf146
        del primals_219
        buf151 = buf144; del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_212, reinterpret_tensor(buf150, (96, 16, 16), (256, 16, 1), 0), out=buf151)
        del permute_212
        buf152 = reinterpret_tensor(buf114, (96, 16, 16), (256, 16, 1), 0); del buf114  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf150, (96, 16, 16), (256, 16, 1), 0), permute_213, out=buf152)
        del buf150
        del permute_213
        buf153 = buf130; del buf130  # reuse
        buf154 = empty((768, ), device='cuda', dtype=torch.float32)
        buf156 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_14.run(buf152, buf151, buf143, mm_42, unsqueeze_85, squeeze_139, buf153, buf154, buf156, 768, 128, grid=grid(768), stream=stream0)
        buf155 = buf132; del buf132  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_15.run(buf152, buf151, buf143, mm_42, unsqueeze_85, buf154, squeeze_139, buf153, primals_154, buf155, 128, 768, grid=grid(128, 768), stream=stream0)
        del buf151
        del buf152
        del mm_42
        del primals_154
        del squeeze_139
        del unsqueeze_85
        buf157 = empty((768, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf155, (768, 128), (1, 768), 0), view_255, out=buf157)
        del view_255
        buf158 = reinterpret_tensor(buf143, (128, 384), (384, 1), 0); del buf143  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf155, (128, 768), (768, 1), 0), permute_219, out=buf158)
        del permute_219
        buf159 = reinterpret_tensor(buf122, (8, 16, 384), (6144, 384, 1), 0); del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_29.run(buf159, buf87, buf99, buf134, buf158, 49152, grid=grid(49152), stream=stream0)
        del buf134
        del buf158
        del buf87
        buf160 = buf136; del buf136  # reuse
        buf161 = empty((384, ), device='cuda', dtype=torch.float32)
        buf162 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_21.run(buf159, mm_41, unsqueeze_89, squeeze_136, buf160, buf161, buf162, 384, 128, grid=grid(384), stream=stream0)
        buf163 = buf99; del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_22.run(buf159, mm_41, unsqueeze_89, buf161, squeeze_136, buf160, primals_151, buf163, 49152, grid=grid(49152), stream=stream0)
        del mm_41
        del primals_151
        del squeeze_136
        del unsqueeze_89
        buf164 = empty((384, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (384, 128), (1, 384), 0), view_251, out=buf164)
        del view_251
        buf165 = buf155; del buf155  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (128, 384), (384, 1), 0), permute_223, out=buf165)
        del permute_223
        buf166 = buf154; del buf154  # reuse
        buf167 = empty((768, ), device='cuda', dtype=torch.float32)
        buf168 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_6.run(view_250, buf165, mm_40, unsqueeze_93, squeeze_133, buf166, buf167, buf168, 768, 128, grid=grid(768), stream=stream0)
        buf169 = buf165; del buf165  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_7.run(buf169, view_250, mm_40, unsqueeze_93, buf167, squeeze_133, buf166, primals_148, 98304, grid=grid(98304), stream=stream0)
        del buf167
        del mm_40
        del primals_148
        del squeeze_133
        del unsqueeze_93
        del view_250
        buf170 = empty((768, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf169, (768, 128), (1, 768), 0), view_247, out=buf170)
        del view_247
        buf171 = buf163; del buf163  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf169, (128, 768), (768, 1), 0), permute_227, out=buf171)
        del buf169
        del permute_227
        buf172 = buf161; del buf161  # reuse
        buf173 = empty((384, ), device='cuda', dtype=torch.float32)
        buf174 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_23.run(buf159, buf171, mm_39, unsqueeze_97, squeeze_130, buf172, buf173, buf174, 384, 128, grid=grid(384), stream=stream0)
        buf175 = reinterpret_tensor(buf159, (128, 384), (384, 1), 0); del buf159  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_30.run(buf175, buf171, mm_39, unsqueeze_97, buf173, squeeze_130, buf172, primals_145, 49152, grid=grid(49152), stream=stream0)
        del buf171
        del buf173
        del mm_39
        del primals_145
        del squeeze_130
        del unsqueeze_97
        buf176 = empty((384, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf175, (384, 128), (1, 384), 0), view_243, out=buf176)
        del view_243
        buf177 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf175, (128, 384), (384, 1), 0), permute_231, out=buf177)
        del buf175
        del permute_231
        buf178 = empty((8, 16, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(view_242, buf177, buf178, 131072, grid=grid(131072), stream=stream0)
        del view_242
        buf179 = empty((128, 49, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_234, reinterpret_tensor(buf178, (128, 16, 64), (1024, 64, 1), 0), out=buf179)
        del permute_234
        buf180 = empty((128, 16, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf178, (128, 16, 64), (1024, 64, 1), 0), permute_235, out=buf180)
        del permute_235
        buf181 = empty_strided((8, 16, 16, 1), (256, 16, 1, 2048), device='cuda', dtype=torch.float32)
        buf186 = empty((8, 16, 16, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_32.run(buf180, alias_18, buf181, buf186, 2048, 49, grid=grid(2048), stream=stream0)
        buf182 = empty((1, 16, 16, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]
        triton_per_fused__softmax_backward_data_sum_33.run(buf180, alias_18, buf181, buf182, 12544, 8, grid=grid(12544), stream=stream0)
        del alias_18
        buf183 = empty((16, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_34.run(buf183, 784, grid=grid(784), stream=stream0)
        aten.index_put_(buf183, [None, primals_218], reinterpret_tensor(buf182, (16, 16, 49), (784, 49, 1), 0), True)
        del buf182
        del primals_218
        buf187 = buf180; del buf180  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_236, reinterpret_tensor(buf186, (128, 16, 49), (784, 49, 1), 0), out=buf187)
        del permute_236
        buf188 = empty((128, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf186, (128, 16, 49), (784, 49, 1), 0), permute_237, out=buf188)
        del permute_237
        buf189 = empty((256, ), device='cuda', dtype=torch.float32)
        buf190 = empty((256, ), device='cuda', dtype=torch.float32)
        buf191 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_35.run(buf188, mm_38, unsqueeze_101, squeeze_127, buf189, buf190, buf191, 256, 128, grid=grid(256), stream=stream0)
        buf192 = empty((128, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_36.run(buf188, mm_38, unsqueeze_101, buf190, squeeze_127, buf189, primals_142, buf192, 32768, grid=grid(32768), stream=stream0)
        del mm_38
        del primals_142
        del squeeze_127
        del unsqueeze_101
        buf193 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf192, (256, 128), (1, 256), 0), view_231, out=buf193)
        del view_231
        buf194 = reinterpret_tensor(buf188, (128, 256), (256, 1), 0); del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf192, (128, 256), (256, 1), 0), permute_241, out=buf194)
        del permute_241
        buf195 = empty_strided((1280, 4), (1, 1280), device='cuda', dtype=torch.float32)
        buf197 = empty_strided((1280, 4), (1, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_37.run(buf187, buf179, mm_37, unsqueeze_105, buf195, buf197, 5120, 98, grid=grid(5120), stream=stream0)
        buf196 = empty((1280, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_38.run(buf195, buf196, 1280, 4, grid=grid(1280), stream=stream0)
        del buf195
        buf198 = empty((1280, ), device='cuda', dtype=torch.float32)
        buf199 = empty((1280, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_39.run(buf197, squeeze_124, buf198, buf199, 1280, 4, grid=grid(1280), stream=stream0)
        del buf197
        buf200 = empty((392, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_40.run(buf187, buf179, mm_37, unsqueeze_105, buf198, squeeze_124, buf196, primals_139, buf200, 392, 1280, grid=grid(392, 1280), stream=stream0)
        del buf198
        del mm_37
        del primals_139
        del squeeze_124
        del unsqueeze_105
        buf201 = empty((1280, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf200, (1280, 392), (1, 1280), 0), view_224, out=buf201)
        del view_224
        buf202 = reinterpret_tensor(buf187, (392, 256), (256, 1), 0); del buf187  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf200, (392, 1280), (1280, 1), 0), permute_247, out=buf202)
        del buf200
        del permute_247
        buf203 = empty_strided((256, 4), (1, 256), device='cuda', dtype=torch.float32)
        buf205 = empty_strided((256, 4), (1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_41.run(buf194, buf202, mm_36, unsqueeze_109, buf203, buf205, 1024, 98, grid=grid(1024), stream=stream0)
        buf204 = buf190; del buf190  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_42.run(buf203, buf204, 256, 4, grid=grid(256), stream=stream0)
        buf206 = empty((256, ), device='cuda', dtype=torch.float32)
        buf207 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_43.run(buf205, squeeze_121, buf206, buf207, 256, 4, grid=grid(256), stream=stream0)
        buf208 = reinterpret_tensor(buf186, (392, 256), (256, 1), 0); del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_44.run(buf194, buf202, mm_36, unsqueeze_109, buf206, squeeze_121, buf204, primals_136, buf208, 100352, grid=grid(100352), stream=stream0)
        del mm_36
        del primals_136
        del squeeze_121
        del unsqueeze_109
        buf209 = reinterpret_tensor(buf178, (256, 512), (512, 1), 0); del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (256, 392), (1, 256), 0), view_220, out=buf209)
        del view_220
        buf210 = empty((392, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (392, 256), (256, 1), 0), permute_251, out=buf210)
        del permute_251
        buf211 = reinterpret_tensor(buf181, (512, 4), (1, 512), 0); del buf181  # reuse
        buf213 = empty_strided((512, 4), (1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_45.run(view_219, buf210, mm_35, unsqueeze_113, buf211, buf213, 2048, 98, grid=grid(2048), stream=stream0)
        buf212 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_46.run(buf211, buf212, 512, 4, grid=grid(512), stream=stream0)
        buf214 = empty((512, ), device='cuda', dtype=torch.float32)
        buf215 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_47.run(buf213, squeeze_118, buf214, buf215, 512, 4, grid=grid(512), stream=stream0)
        buf216 = buf210; del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_48.run(buf216, view_219, mm_35, unsqueeze_113, buf214, squeeze_118, buf212, primals_133, 200704, grid=grid(200704), stream=stream0)
        del mm_35
        del primals_133
        del squeeze_118
        del unsqueeze_113
        del view_219
        buf217 = reinterpret_tensor(buf177, (512, 256), (256, 1), 0); del buf177  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf216, (512, 392), (1, 512), 0), view_216, out=buf217)
        del view_216
        buf218 = buf208; del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf216, (392, 512), (512, 1), 0), permute_255, out=buf218)
        del permute_255
        buf219 = buf205; del buf205  # reuse
        buf221 = buf203; del buf203  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_49.run(buf194, buf202, buf218, mm_34, unsqueeze_117, buf219, buf221, 1024, 98, grid=grid(1024), stream=stream0)
        buf220 = buf206; del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_42.run(buf219, buf220, 256, 4, grid=grid(256), stream=stream0)
        buf222 = empty((256, ), device='cuda', dtype=torch.float32)
        buf224 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_43.run(buf221, squeeze_115, buf222, buf224, 256, 4, grid=grid(256), stream=stream0)
        buf223 = empty((392, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_50.run(buf194, buf202, buf218, mm_34, unsqueeze_117, buf222, squeeze_115, buf220, primals_130, buf223, 100352, grid=grid(100352), stream=stream0)
        del mm_34
        del primals_130
        del squeeze_115
        del unsqueeze_117
        buf225 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf223, (256, 392), (1, 256), 0), view_212, out=buf225)
        del view_212
        buf226 = empty((392, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf223, (392, 256), (256, 1), 0), permute_259, out=buf226)
        del permute_259
        buf227 = reinterpret_tensor(buf223, (8, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf223  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_51.run(view_211, buf226, buf227, 100352, grid=grid(100352), stream=stream0)
        del view_211
        buf228 = reinterpret_tensor(buf226, (64, 49, 32), (1568, 32, 1), 0); del buf226  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_262, reinterpret_tensor(buf227, (64, 49, 32), (1568, 32, 1), 0), out=buf228)
        del permute_262
        buf229 = empty((64, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf227, (64, 49, 32), (1568, 32, 1), 0), permute_263, out=buf229)
        del permute_263
        buf230 = empty_strided((8, 8, 49, 1), (392, 49, 1, 3136), device='cuda', dtype=torch.float32)
        buf235 = empty((8, 8, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_52.run(buf229, alias_19, buf230, buf235, 3136, 49, grid=grid(3136), stream=stream0)
        buf231 = empty((1, 8, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]
        triton_per_fused__softmax_backward_data_sum_53.run(buf229, alias_19, buf230, buf231, 19208, 8, grid=grid(19208), stream=stream0)
        del alias_19
        buf232 = empty((8, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_54.run(buf232, 392, grid=grid(392), stream=stream0)
        aten.index_put_(buf232, [None, primals_217], reinterpret_tensor(buf231, (8, 49, 49), (2401, 49, 1), 0), True)
        del primals_217
        buf236 = empty((64, 16, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_264, reinterpret_tensor(buf235, (64, 49, 49), (2401, 49, 1), 0), out=buf236)
        del permute_264
        buf237 = empty((64, 49, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf235, (64, 49, 49), (2401, 49, 1), 0), permute_265, out=buf237)
        del permute_265
        buf238 = buf213; del buf213  # reuse
        buf240 = buf211; del buf211  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_55.run(buf237, buf236, buf228, mm_33, unsqueeze_121, buf238, buf240, 2048, 98, grid=grid(2048), stream=stream0)
        buf239 = buf214; del buf214  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_46.run(buf238, buf239, 512, 4, grid=grid(512), stream=stream0)
        buf241 = empty((512, ), device='cuda', dtype=torch.float32)
        buf243 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_47.run(buf240, squeeze_112, buf241, buf243, 512, 4, grid=grid(512), stream=stream0)
        buf242 = buf216; del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_56.run(buf237, buf236, buf228, mm_33, unsqueeze_121, buf241, squeeze_112, buf239, primals_127, buf242, 392, 512, grid=grid(392, 512), stream=stream0)
        del mm_33
        del primals_127
        del squeeze_112
        del unsqueeze_121
        buf244 = empty((512, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf242, (512, 392), (1, 512), 0), view_200, out=buf244)
        del view_200
        buf245 = reinterpret_tensor(buf228, (392, 256), (256, 1), 0); del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf242, (392, 512), (512, 1), 0), permute_271, out=buf245)
        del permute_271
        buf246 = buf221; del buf221  # reuse
        buf248 = buf219; del buf219  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_57.run(buf194, buf202, buf218, buf245, mm_32, unsqueeze_125, buf246, buf248, 1024, 98, grid=grid(1024), stream=stream0)
        buf247 = buf222; del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_42.run(buf246, buf247, 256, 4, grid=grid(256), stream=stream0)
        buf249 = empty((256, ), device='cuda', dtype=torch.float32)
        buf251 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_43.run(buf248, squeeze_109, buf249, buf251, 256, 4, grid=grid(256), stream=stream0)
        buf250 = reinterpret_tensor(buf227, (392, 256), (256, 1), 0); del buf227  # reuse
        buf252 = buf250; del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_58.run(buf252, buf194, buf202, buf218, buf245, mm_32, unsqueeze_125, buf249, squeeze_109, buf247, primals_124, 100352, grid=grid(100352), stream=stream0)
        del mm_32
        del primals_124
        del squeeze_109
        del unsqueeze_125
        buf253 = empty((256, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf252, (256, 392), (1, 256), 0), view_196, out=buf253)
        del view_196
        buf254 = buf242; del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf252, (392, 256), (256, 1), 0), permute_275, out=buf254)
        del permute_275
        buf255 = buf240; del buf240  # reuse
        buf257 = buf238; del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_45.run(view_195, buf254, mm_31, unsqueeze_129, buf255, buf257, 2048, 98, grid=grid(2048), stream=stream0)
        buf256 = buf241; del buf241  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_46.run(buf255, buf256, 512, 4, grid=grid(512), stream=stream0)
        buf258 = empty((512, ), device='cuda', dtype=torch.float32)
        buf259 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_47.run(buf257, squeeze_106, buf258, buf259, 512, 4, grid=grid(512), stream=stream0)
        buf260 = buf254; del buf254  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_48.run(buf260, view_195, mm_31, unsqueeze_129, buf258, squeeze_106, buf256, primals_121, 200704, grid=grid(200704), stream=stream0)
        del mm_31
        del primals_121
        del squeeze_106
        del unsqueeze_129
        del view_195
        buf261 = empty((512, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf260, (512, 392), (1, 512), 0), view_192, out=buf261)
        del view_192
        buf262 = buf252; del buf252  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf260, (392, 512), (512, 1), 0), permute_279, out=buf262)
        del permute_279
        buf263 = reinterpret_tensor(buf202, (8, 49, 256), (12544, 256, 1), 0); del buf202  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_59.run(buf263, buf194, buf218, buf245, buf262, 100352, grid=grid(100352), stream=stream0)
        buf264 = buf248; del buf248  # reuse
        buf266 = buf246; del buf246  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_60.run(buf263, mm_30, unsqueeze_133, buf264, buf266, 1024, 98, grid=grid(1024), stream=stream0)
        buf265 = buf249; del buf249  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_42.run(buf264, buf265, 256, 4, grid=grid(256), stream=stream0)
        buf267 = empty((256, ), device='cuda', dtype=torch.float32)
        buf268 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_43.run(buf266, squeeze_103, buf267, buf268, 256, 4, grid=grid(256), stream=stream0)
        buf269 = buf262; del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_61.run(buf263, mm_30, unsqueeze_133, buf267, squeeze_103, buf265, primals_118, buf269, 100352, grid=grid(100352), stream=stream0)
        del mm_30
        del primals_118
        del squeeze_103
        del unsqueeze_133
        buf270 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf269, (256, 392), (1, 256), 0), view_188, out=buf270)
        del view_188
        buf271 = buf245; del buf245  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf269, (392, 256), (256, 1), 0), permute_283, out=buf271)
        del permute_283
        buf272 = reinterpret_tensor(buf269, (8, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf269  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_51.run(view_187, buf271, buf272, 100352, grid=grid(100352), stream=stream0)
        del view_187
        buf273 = reinterpret_tensor(buf271, (64, 49, 32), (1568, 32, 1), 0); del buf271  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_286, reinterpret_tensor(buf272, (64, 49, 32), (1568, 32, 1), 0), out=buf273)
        del permute_286
        buf274 = reinterpret_tensor(buf235, (64, 49, 49), (2401, 49, 1), 0); del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf272, (64, 49, 32), (1568, 32, 1), 0), permute_287, out=buf274)
        del permute_287
        buf275 = buf230; del buf230  # reuse
        buf280 = reinterpret_tensor(buf229, (8, 8, 49, 49), (19208, 2401, 49, 1), 0); del buf229  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_52.run(buf274, alias_20, buf275, buf280, 3136, 49, grid=grid(3136), stream=stream0)
        buf276 = buf231; del buf231  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]
        triton_per_fused__softmax_backward_data_sum_53.run(buf274, alias_20, buf275, buf276, 19208, 8, grid=grid(19208), stream=stream0)
        del alias_20
        buf277 = empty((8, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_54.run(buf277, 392, grid=grid(392), stream=stream0)
        aten.index_put_(buf277, [None, primals_216], reinterpret_tensor(buf276, (8, 49, 49), (2401, 49, 1), 0), True)
        del primals_216
        buf281 = reinterpret_tensor(buf237, (64, 16, 49), (784, 49, 1), 0); del buf237  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_288, reinterpret_tensor(buf280, (64, 49, 49), (2401, 49, 1), 0), out=buf281)
        del permute_288
        buf282 = reinterpret_tensor(buf236, (64, 49, 16), (784, 16, 1), 0); del buf236  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf280, (64, 49, 49), (2401, 49, 1), 0), permute_289, out=buf282)
        del permute_289
        buf283 = buf257; del buf257  # reuse
        buf285 = buf255; del buf255  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_55.run(buf282, buf281, buf273, mm_29, unsqueeze_137, buf283, buf285, 2048, 98, grid=grid(2048), stream=stream0)
        buf284 = buf258; del buf258  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_46.run(buf283, buf284, 512, 4, grid=grid(512), stream=stream0)
        buf286 = empty((512, ), device='cuda', dtype=torch.float32)
        buf288 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_47.run(buf285, squeeze_100, buf286, buf288, 512, 4, grid=grid(512), stream=stream0)
        buf287 = buf260; del buf260  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_56.run(buf282, buf281, buf273, mm_29, unsqueeze_137, buf286, squeeze_100, buf284, primals_115, buf287, 392, 512, grid=grid(392, 512), stream=stream0)
        del mm_29
        del primals_115
        del squeeze_100
        del unsqueeze_137
        buf289 = empty((512, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf287, (512, 392), (1, 512), 0), view_176, out=buf289)
        del view_176
        buf290 = reinterpret_tensor(buf273, (392, 256), (256, 1), 0); del buf273  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf287, (392, 512), (512, 1), 0), permute_295, out=buf290)
        del permute_295
        buf291 = buf266; del buf266  # reuse
        buf293 = buf264; del buf264  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_62.run(buf263, buf290, mm_28, unsqueeze_141, buf291, buf293, 1024, 98, grid=grid(1024), stream=stream0)
        buf292 = buf267; del buf267  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_42.run(buf291, buf292, 256, 4, grid=grid(256), stream=stream0)
        buf294 = empty((256, ), device='cuda', dtype=torch.float32)
        buf295 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_43.run(buf293, squeeze_97, buf294, buf295, 256, 4, grid=grid(256), stream=stream0)
        buf296 = reinterpret_tensor(buf272, (392, 256), (256, 1), 0); del buf272  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_63.run(buf263, buf290, mm_28, unsqueeze_141, buf294, squeeze_97, buf292, primals_112, buf296, 100352, grid=grid(100352), stream=stream0)
        del mm_28
        del primals_112
        del squeeze_97
        del unsqueeze_141
        buf297 = empty((256, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf296, (256, 392), (1, 256), 0), view_172, out=buf297)
        del view_172
        buf298 = buf287; del buf287  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf296, (392, 256), (256, 1), 0), permute_299, out=buf298)
        del permute_299
        buf299 = buf285; del buf285  # reuse
        buf301 = buf283; del buf283  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_45.run(view_171, buf298, mm_27, unsqueeze_145, buf299, buf301, 2048, 98, grid=grid(2048), stream=stream0)
        buf300 = buf286; del buf286  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_46.run(buf299, buf300, 512, 4, grid=grid(512), stream=stream0)
        buf302 = empty((512, ), device='cuda', dtype=torch.float32)
        buf303 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_47.run(buf301, squeeze_94, buf302, buf303, 512, 4, grid=grid(512), stream=stream0)
        buf304 = buf298; del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_48.run(buf304, view_171, mm_27, unsqueeze_145, buf302, squeeze_94, buf300, primals_109, 200704, grid=grid(200704), stream=stream0)
        del mm_27
        del primals_109
        del squeeze_94
        del unsqueeze_145
        del view_171
        buf305 = empty((512, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (512, 392), (1, 512), 0), view_168, out=buf305)
        del view_168
        buf306 = buf296; del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (392, 512), (512, 1), 0), permute_303, out=buf306)
        del permute_303
        buf307 = buf293; del buf293  # reuse
        buf309 = buf291; del buf291  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_64.run(buf263, buf290, buf306, mm_26, unsqueeze_149, buf307, buf309, 1024, 98, grid=grid(1024), stream=stream0)
        buf308 = buf294; del buf294  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_42.run(buf307, buf308, 256, 4, grid=grid(256), stream=stream0)
        buf310 = empty((256, ), device='cuda', dtype=torch.float32)
        buf312 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_43.run(buf309, squeeze_91, buf310, buf312, 256, 4, grid=grid(256), stream=stream0)
        buf311 = buf218; del buf218  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_65.run(buf263, buf290, buf306, mm_26, unsqueeze_149, buf310, squeeze_91, buf308, primals_106, buf311, 100352, grid=grid(100352), stream=stream0)
        del mm_26
        del primals_106
        del squeeze_91
        del unsqueeze_149
        buf313 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf311, (256, 392), (1, 256), 0), view_164, out=buf313)
        del view_164
        buf314 = empty((392, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf311, (392, 256), (256, 1), 0), permute_307, out=buf314)
        del permute_307
        buf315 = reinterpret_tensor(buf311, (8, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf311  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_51.run(view_163, buf314, buf315, 100352, grid=grid(100352), stream=stream0)
        del view_163
        buf316 = reinterpret_tensor(buf314, (64, 49, 32), (1568, 32, 1), 0); del buf314  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_310, reinterpret_tensor(buf315, (64, 49, 32), (1568, 32, 1), 0), out=buf316)
        del permute_310
        buf317 = reinterpret_tensor(buf280, (64, 49, 49), (2401, 49, 1), 0); del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf315, (64, 49, 32), (1568, 32, 1), 0), permute_311, out=buf317)
        del permute_311
        buf318 = buf275; del buf275  # reuse
        buf323 = reinterpret_tensor(buf274, (8, 8, 49, 49), (19208, 2401, 49, 1), 0); del buf274  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_52.run(buf317, alias_21, buf318, buf323, 3136, 49, grid=grid(3136), stream=stream0)
        buf319 = buf276; del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]
        triton_per_fused__softmax_backward_data_sum_53.run(buf317, alias_21, buf318, buf319, 19208, 8, grid=grid(19208), stream=stream0)
        del alias_21
        buf320 = empty((8, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_54.run(buf320, 392, grid=grid(392), stream=stream0)
        aten.index_put_(buf320, [None, primals_215], reinterpret_tensor(buf319, (8, 49, 49), (2401, 49, 1), 0), True)
        del primals_215
        buf324 = reinterpret_tensor(buf282, (64, 16, 49), (784, 49, 1), 0); del buf282  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_312, reinterpret_tensor(buf323, (64, 49, 49), (2401, 49, 1), 0), out=buf324)
        del permute_312
        buf325 = reinterpret_tensor(buf281, (64, 49, 16), (784, 16, 1), 0); del buf281  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf323, (64, 49, 49), (2401, 49, 1), 0), permute_313, out=buf325)
        del permute_313
        buf326 = buf301; del buf301  # reuse
        buf328 = buf299; del buf299  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_55.run(buf325, buf324, buf316, mm_25, unsqueeze_153, buf326, buf328, 2048, 98, grid=grid(2048), stream=stream0)
        buf327 = buf302; del buf302  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_46.run(buf326, buf327, 512, 4, grid=grid(512), stream=stream0)
        buf329 = empty((512, ), device='cuda', dtype=torch.float32)
        buf331 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_47.run(buf328, squeeze_88, buf329, buf331, 512, 4, grid=grid(512), stream=stream0)
        buf330 = buf304; del buf304  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_56.run(buf325, buf324, buf316, mm_25, unsqueeze_153, buf329, squeeze_88, buf327, primals_103, buf330, 392, 512, grid=grid(392, 512), stream=stream0)
        del mm_25
        del primals_103
        del squeeze_88
        del unsqueeze_153
        buf332 = empty((512, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf330, (512, 392), (1, 512), 0), view_152, out=buf332)
        del view_152
        buf333 = reinterpret_tensor(buf316, (392, 256), (256, 1), 0); del buf316  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf330, (392, 512), (512, 1), 0), permute_319, out=buf333)
        del permute_319
        buf334 = buf309; del buf309  # reuse
        buf336 = buf307; del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_66.run(buf263, buf290, buf306, buf333, mm_24, unsqueeze_157, buf334, buf336, 1024, 98, grid=grid(1024), stream=stream0)
        buf335 = buf310; del buf310  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_42.run(buf334, buf335, 256, 4, grid=grid(256), stream=stream0)
        buf337 = empty((256, ), device='cuda', dtype=torch.float32)
        buf339 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_43.run(buf336, squeeze_85, buf337, buf339, 256, 4, grid=grid(256), stream=stream0)
        buf338 = reinterpret_tensor(buf315, (392, 256), (256, 1), 0); del buf315  # reuse
        buf340 = buf338; del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_67.run(buf340, buf263, buf290, buf306, buf333, mm_24, unsqueeze_157, buf337, squeeze_85, buf335, primals_100, 100352, grid=grid(100352), stream=stream0)
        del mm_24
        del primals_100
        del squeeze_85
        del unsqueeze_157
        buf341 = empty((256, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf340, (256, 392), (1, 256), 0), view_148, out=buf341)
        del view_148
        buf342 = buf330; del buf330  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf340, (392, 256), (256, 1), 0), permute_323, out=buf342)
        del permute_323
        buf343 = buf328; del buf328  # reuse
        buf345 = buf326; del buf326  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_45.run(view_147, buf342, mm_23, unsqueeze_161, buf343, buf345, 2048, 98, grid=grid(2048), stream=stream0)
        buf344 = buf329; del buf329  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_46.run(buf343, buf344, 512, 4, grid=grid(512), stream=stream0)
        buf346 = empty((512, ), device='cuda', dtype=torch.float32)
        buf347 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_47.run(buf345, squeeze_82, buf346, buf347, 512, 4, grid=grid(512), stream=stream0)
        buf348 = buf342; del buf342  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_48.run(buf348, view_147, mm_23, unsqueeze_161, buf346, squeeze_82, buf344, primals_97, 200704, grid=grid(200704), stream=stream0)
        del mm_23
        del primals_97
        del squeeze_82
        del unsqueeze_161
        del view_147
        buf349 = empty((512, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf348, (512, 392), (1, 512), 0), view_144, out=buf349)
        del view_144
        buf350 = buf340; del buf340  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf348, (392, 512), (512, 1), 0), permute_327, out=buf350)
        del permute_327
        buf351 = buf263; del buf263  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(buf351, buf290, buf306, buf333, buf350, 100352, grid=grid(100352), stream=stream0)
        del buf290
        del buf306
        buf352 = buf336; del buf336  # reuse
        buf354 = buf334; del buf334  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_60.run(buf351, mm_22, unsqueeze_165, buf352, buf354, 1024, 98, grid=grid(1024), stream=stream0)
        buf353 = buf337; del buf337  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_42.run(buf352, buf353, 256, 4, grid=grid(256), stream=stream0)
        buf355 = empty((256, ), device='cuda', dtype=torch.float32)
        buf356 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_43.run(buf354, squeeze_79, buf355, buf356, 256, 4, grid=grid(256), stream=stream0)
        buf357 = buf350; del buf350  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_61.run(buf351, mm_22, unsqueeze_165, buf355, squeeze_79, buf353, primals_94, buf357, 100352, grid=grid(100352), stream=stream0)
        del mm_22
        del primals_94
        del squeeze_79
        del unsqueeze_165
        buf358 = empty((256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf357, (256, 392), (1, 256), 0), view_140, out=buf358)
        del view_140
        buf359 = buf333; del buf333  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf357, (392, 256), (256, 1), 0), permute_331, out=buf359)
        del permute_331
        buf360 = reinterpret_tensor(buf357, (8, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf357  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_51.run(view_139, buf359, buf360, 100352, grid=grid(100352), stream=stream0)
        del view_139
        buf361 = reinterpret_tensor(buf359, (64, 49, 32), (1568, 32, 1), 0); del buf359  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_334, reinterpret_tensor(buf360, (64, 49, 32), (1568, 32, 1), 0), out=buf361)
        del permute_334
        buf362 = reinterpret_tensor(buf323, (64, 49, 49), (2401, 49, 1), 0); del buf323  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf360, (64, 49, 32), (1568, 32, 1), 0), permute_335, out=buf362)
        del permute_335
        buf363 = buf318; del buf318  # reuse
        buf368 = reinterpret_tensor(buf317, (8, 8, 49, 49), (19208, 2401, 49, 1), 0); del buf317  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_52.run(buf362, alias_22, buf363, buf368, 3136, 49, grid=grid(3136), stream=stream0)
        buf364 = buf319; del buf319  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]
        triton_per_fused__softmax_backward_data_sum_53.run(buf362, alias_22, buf363, buf364, 19208, 8, grid=grid(19208), stream=stream0)
        del alias_22
        del buf362
        buf365 = empty((8, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.new_zeros]
        triton_poi_fused_index_put_new_zeros_54.run(buf365, 392, grid=grid(392), stream=stream0)
        aten.index_put_(buf365, [None, primals_214], reinterpret_tensor(buf364, (8, 49, 49), (2401, 49, 1), 0), True)
        del buf364
        del primals_214
        buf369 = reinterpret_tensor(buf325, (64, 16, 49), (784, 49, 1), 0); del buf325  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_336, reinterpret_tensor(buf368, (64, 49, 49), (2401, 49, 1), 0), out=buf369)
        del permute_336
        buf370 = reinterpret_tensor(buf324, (64, 49, 16), (784, 16, 1), 0); del buf324  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf368, (64, 49, 49), (2401, 49, 1), 0), permute_337, out=buf370)
        del permute_337
        buf371 = buf345; del buf345  # reuse
        buf373 = buf343; del buf343  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_55.run(buf370, buf369, buf361, mm_21, unsqueeze_169, buf371, buf373, 2048, 98, grid=grid(2048), stream=stream0)
        buf372 = buf346; del buf346  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_46.run(buf371, buf372, 512, 4, grid=grid(512), stream=stream0)
        buf374 = empty((512, ), device='cuda', dtype=torch.float32)
        buf376 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_47.run(buf373, squeeze_76, buf374, buf376, 512, 4, grid=grid(512), stream=stream0)
        buf375 = buf348; del buf348  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_56.run(buf370, buf369, buf361, mm_21, unsqueeze_169, buf374, squeeze_76, buf372, primals_91, buf375, 392, 512, grid=grid(392, 512), stream=stream0)
        del mm_21
        del primals_91
        del squeeze_76
        del unsqueeze_169
        buf377 = empty((512, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf375, (512, 392), (1, 512), 0), view_128, out=buf377)
        del view_128
        buf378 = reinterpret_tensor(buf361, (392, 256), (256, 1), 0); del buf361  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf375, (392, 512), (512, 1), 0), permute_343, out=buf378)
        del permute_343
        buf379 = buf354; del buf354  # reuse
        buf381 = buf352; del buf352  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_62.run(buf351, buf378, mm_20, unsqueeze_173, buf379, buf381, 1024, 98, grid=grid(1024), stream=stream0)
        buf380 = buf355; del buf355  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_42.run(buf379, buf380, 256, 4, grid=grid(256), stream=stream0)
        buf382 = empty((256, ), device='cuda', dtype=torch.float32)
        buf383 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_43.run(buf381, squeeze_73, buf382, buf383, 256, 4, grid=grid(256), stream=stream0)
        buf384 = reinterpret_tensor(buf360, (392, 256), (256, 1), 0); del buf360  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_63.run(buf351, buf378, mm_20, unsqueeze_173, buf382, squeeze_73, buf380, primals_88, buf384, 100352, grid=grid(100352), stream=stream0)
        del mm_20
        del primals_88
        del squeeze_73
        del unsqueeze_173
        buf385 = empty((256, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf384, (256, 392), (1, 256), 0), view_124, out=buf385)
        del view_124
        buf386 = buf375; del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf384, (392, 256), (256, 1), 0), permute_347, out=buf386)
        del permute_347
        buf387 = buf373; del buf373  # reuse
        buf389 = buf371; del buf371  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_45.run(view_123, buf386, mm_19, unsqueeze_177, buf387, buf389, 2048, 98, grid=grid(2048), stream=stream0)
        buf388 = buf374; del buf374  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_46.run(buf387, buf388, 512, 4, grid=grid(512), stream=stream0)
        del buf387
        buf390 = empty((512, ), device='cuda', dtype=torch.float32)
        buf391 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_47.run(buf389, squeeze_70, buf390, buf391, 512, 4, grid=grid(512), stream=stream0)
        del buf389
        buf392 = buf386; del buf386  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_48.run(buf392, view_123, mm_19, unsqueeze_177, buf390, squeeze_70, buf388, primals_85, 200704, grid=grid(200704), stream=stream0)
        del mm_19
        del primals_85
        del squeeze_70
        del unsqueeze_177
        del view_123
        buf393 = empty((512, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf392, (512, 392), (1, 512), 0), view_120, out=buf393)
        del view_120
        buf394 = buf384; del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf392, (392, 512), (512, 1), 0), permute_351, out=buf394)
        del permute_351
        buf395 = buf381; del buf381  # reuse
        buf397 = buf379; del buf379  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_64.run(buf351, buf378, buf394, mm_18, unsqueeze_181, buf395, buf397, 1024, 98, grid=grid(1024), stream=stream0)
        buf396 = buf382; del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_42.run(buf395, buf396, 256, 4, grid=grid(256), stream=stream0)
        del buf395
        buf398 = empty((256, ), device='cuda', dtype=torch.float32)
        buf400 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_43.run(buf397, squeeze_67, buf398, buf400, 256, 4, grid=grid(256), stream=stream0)
        del buf397
        buf399 = reinterpret_tensor(buf351, (392, 256), (256, 1), 0); del buf351  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_69.run(buf399, buf378, buf394, mm_18, unsqueeze_181, buf398, squeeze_67, buf396, primals_82, 100352, grid=grid(100352), stream=stream0)
        del buf378
        del mm_18
        del primals_82
        del squeeze_67
        del unsqueeze_181
        buf401 = empty((256, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf399, (256, 392), (1, 256), 0), view_116, out=buf401)
        del view_116
        buf402 = buf392; del buf392  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf399, (392, 256), (256, 1), 0), permute_355, out=buf402)
        del permute_355
        buf403 = empty((8, 8, 49, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_70.run(view_115, buf402, buf403, 200704, grid=grid(200704), stream=stream0)
        del view_115
        buf404 = empty((64, 196, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_358, reinterpret_tensor(buf403, (64, 49, 64), (3136, 64, 1), 0), out=buf404)
        del permute_358
        buf405 = empty((64, 49, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf403, (64, 49, 64), (3136, 64, 1), 0), permute_359, out=buf405)
        del permute_359
        buf406 = buf363; del buf363  # reuse
        buf411 = empty((8, 8, 49, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_71.run(buf405, alias_23, buf406, buf411, 3136, 196, grid=grid(3136), stream=stream0)
        buf407 = empty((1, 8, 49, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]
        triton_per_fused__softmax_backward_data_sum_72.run(buf405, alias_23, buf406, buf407, 76832, 8, grid=grid(76832), stream=stream0)
        del alias_23
        del buf405
        del buf406
        buf408 = empty((8, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_73.run(buf408, 1568, grid=grid(1568), stream=stream0)
        aten.index_put_(buf408, [None, primals_213], reinterpret_tensor(buf407, (8, 49, 196), (9604, 196, 1), 0), True)
        del buf407
        del primals_213
        buf412 = reinterpret_tensor(buf403, (64, 16, 196), (3136, 196, 1), 0); del buf403  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_360, reinterpret_tensor(buf411, (64, 49, 196), (9604, 196, 1), 0), out=buf412)
        del permute_360
        buf413 = buf370; del buf370  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf411, (64, 49, 196), (9604, 196, 1), 0), permute_361, out=buf413)
        del buf411
        del permute_361
        buf414 = reinterpret_tensor(buf390, (128, 4), (1, 128), 0); del buf390  # reuse
        buf416 = empty_strided((128, 4), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_74.run(buf413, mm_17, unsqueeze_185, buf414, buf416, 512, 98, grid=grid(512), stream=stream0)
        buf415 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_75.run(buf414, buf415, 128, 4, grid=grid(128), stream=stream0)
        del buf414
        buf417 = empty((128, ), device='cuda', dtype=torch.float32)
        buf418 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_76.run(buf416, squeeze_64, buf417, buf418, 128, 4, grid=grid(128), stream=stream0)
        del buf416
        buf419 = reinterpret_tensor(buf369, (392, 128), (128, 1), 0); del buf369  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_77.run(buf413, mm_17, unsqueeze_185, buf417, squeeze_64, buf415, primals_79, buf419, 50176, grid=grid(50176), stream=stream0)
        del mm_17
        del primals_79
        del squeeze_64
        del unsqueeze_185
        buf420 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf419, (128, 392), (1, 128), 0), view_104, out=buf420)
        del view_104
        buf421 = reinterpret_tensor(buf413, (392, 128), (128, 1), 0); del buf413  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf419, (392, 128), (128, 1), 0), permute_365, out=buf421)
        del buf419
        del permute_365
        buf422 = empty((640, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_78.run(buf412, buf404, buf422, 8320, 121, grid=grid(8320), stream=stream0)
        buf423 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_79.run(buf422, buf423, 640, 13, grid=grid(640), stream=stream0)
        buf424 = reinterpret_tensor(buf422, (640, 13), (1, 640), 0); del buf422  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_80.run(buf412, buf404, mm_16, unsqueeze_189, buf424, 8320, 121, grid=grid(8320), stream=stream0)
        buf425 = empty((640, ), device='cuda', dtype=torch.float32)
        buf426 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_81.run(buf424, squeeze_61, buf425, buf426, 640, 13, grid=grid(640), stream=stream0)
        del buf424
        buf427 = empty((1568, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_82.run(buf412, buf404, mm_16, unsqueeze_189, buf425, squeeze_61, buf423, primals_76, buf427, 1568, 640, grid=grid(1568, 640), stream=stream0)
        del buf404
        del buf425
        del mm_16
        del primals_76
        del squeeze_61
        del unsqueeze_189
        buf428 = empty((640, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf427, (640, 1568), (1, 640), 0), view_97, out=buf428)
        del view_97
        buf429 = reinterpret_tensor(buf412, (1568, 128), (128, 1), 0); del buf412  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf427, (1568, 640), (640, 1), 0), permute_371, out=buf429)
        del buf427
        del permute_371
        buf430 = empty_strided((128, 13), (1, 128), device='cuda', dtype=torch.float32)
        buf432 = empty_strided((128, 13), (1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_83.run(buf421, buf429, mm_15, unsqueeze_193, buf430, buf432, 1664, 121, grid=grid(1664), stream=stream0)
        buf431 = buf417; del buf417  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_84.run(buf430, buf431, 128, 13, grid=grid(128), stream=stream0)
        buf433 = empty((128, ), device='cuda', dtype=torch.float32)
        buf434 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_85.run(buf432, squeeze_58, buf433, buf434, 128, 13, grid=grid(128), stream=stream0)
        buf435 = reinterpret_tensor(buf402, (1568, 128), (128, 1), 0); del buf402  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_86.run(buf421, buf429, mm_15, unsqueeze_193, buf433, squeeze_58, buf431, primals_73, buf435, 200704, grid=grid(200704), stream=stream0)
        del mm_15
        del primals_73
        del squeeze_58
        del unsqueeze_193
        buf436 = buf194; del buf194  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf435, (128, 1568), (1, 128), 0), view_93, out=buf436)
        del view_93
        buf437 = reinterpret_tensor(buf179, (1568, 256), (256, 1), 0); del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf435, (1568, 128), (128, 1), 0), permute_375, out=buf437)
        del permute_375
        buf438 = empty_strided((256, 13), (1, 256), device='cuda', dtype=torch.float32)
        buf440 = empty_strided((256, 13), (1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_87.run(view_92, buf437, mm_14, unsqueeze_197, buf438, buf440, 3328, 121, grid=grid(3328), stream=stream0)
        buf439 = buf398; del buf398  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_88.run(buf438, buf439, 256, 13, grid=grid(256), stream=stream0)
        buf441 = empty((256, ), device='cuda', dtype=torch.float32)
        buf442 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_89.run(buf440, squeeze_55, buf441, buf442, 256, 13, grid=grid(256), stream=stream0)
        buf443 = buf437; del buf437  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_90.run(buf443, view_92, mm_14, unsqueeze_197, buf441, squeeze_55, buf439, primals_70, 401408, grid=grid(401408), stream=stream0)
        del mm_14
        del primals_70
        del squeeze_55
        del unsqueeze_197
        del view_92
        buf444 = reinterpret_tensor(buf192, (256, 128), (128, 1), 0); del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf443, (256, 1568), (1, 256), 0), view_89, out=buf444)
        del view_89
        buf445 = buf435; del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf443, (1568, 256), (256, 1), 0), permute_379, out=buf445)
        del permute_379
        buf446 = buf432; del buf432  # reuse
        buf448 = buf430; del buf430  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_91.run(buf421, buf429, buf445, mm_13, unsqueeze_201, buf446, buf448, 1664, 121, grid=grid(1664), stream=stream0)
        buf447 = buf433; del buf433  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_84.run(buf446, buf447, 128, 13, grid=grid(128), stream=stream0)
        buf449 = empty((128, ), device='cuda', dtype=torch.float32)
        buf451 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_85.run(buf448, squeeze_52, buf449, buf451, 128, 13, grid=grid(128), stream=stream0)
        buf450 = empty((1568, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_92.run(buf421, buf429, buf445, mm_13, unsqueeze_201, buf449, squeeze_52, buf447, primals_67, buf450, 200704, grid=grid(200704), stream=stream0)
        del mm_13
        del primals_67
        del squeeze_52
        del unsqueeze_201
        buf452 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf450, (128, 1568), (1, 128), 0), view_85, out=buf452)
        del view_85
        buf453 = empty((1568, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf450, (1568, 128), (128, 1), 0), permute_383, out=buf453)
        del permute_383
        buf454 = reinterpret_tensor(buf450, (8, 4, 196, 32), (25088, 6272, 32, 1), 0); del buf450  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_93.run(view_84, buf453, buf454, 200704, grid=grid(200704), stream=stream0)
        del view_84
        buf455 = reinterpret_tensor(buf453, (32, 196, 32), (6272, 32, 1), 0); del buf453  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_386, reinterpret_tensor(buf454, (32, 196, 32), (6272, 32, 1), 0), out=buf455)
        del permute_386
        buf456 = empty((32, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf454, (32, 196, 32), (6272, 32, 1), 0), permute_387, out=buf456)
        del permute_387
        buf457 = empty_strided((8, 4, 196, 1), (784, 196, 1, 6272), device='cuda', dtype=torch.float32)
        buf462 = empty((8, 4, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_94.run(buf456, alias_24, buf457, buf462, 6272, 196, grid=grid(6272), stream=stream0)
        buf458 = reinterpret_tensor(buf368, (1, 4, 196, 196), (153664, 38416, 196, 1), 0); del buf368  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]
        triton_per_fused__softmax_backward_data_sum_95.run(buf456, alias_24, buf457, buf458, 153664, 8, grid=grid(153664), stream=stream0)
        del alias_24
        buf459 = empty((4, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_new_zeros_34.run(buf459, 784, grid=grid(784), stream=stream0)
        aten.index_put_(buf459, [None, primals_212], reinterpret_tensor(buf458, (4, 196, 196), (38416, 196, 1), 0), True)
        del primals_212
        buf463 = reinterpret_tensor(buf399, (32, 16, 196), (3136, 196, 1), 0); del buf399  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_388, reinterpret_tensor(buf462, (32, 196, 196), (38416, 196, 1), 0), out=buf463)
        del permute_388
        buf464 = reinterpret_tensor(buf394, (32, 196, 16), (3136, 16, 1), 0); del buf394  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf462, (32, 196, 196), (38416, 196, 1), 0), permute_389, out=buf464)
        del permute_389
        buf465 = buf440; del buf440  # reuse
        buf467 = buf438; del buf438  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_96.run(buf464, buf463, buf455, mm_12, unsqueeze_205, buf465, buf467, 3328, 121, grid=grid(3328), stream=stream0)
        buf466 = buf441; del buf441  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_88.run(buf465, buf466, 256, 13, grid=grid(256), stream=stream0)
        buf468 = empty((256, ), device='cuda', dtype=torch.float32)
        buf470 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_89.run(buf467, squeeze_49, buf468, buf470, 256, 13, grid=grid(256), stream=stream0)
        buf469 = buf443; del buf443  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_97.run(buf464, buf463, buf455, mm_12, unsqueeze_205, buf468, squeeze_49, buf466, primals_64, buf469, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del mm_12
        del primals_64
        del squeeze_49
        del unsqueeze_205
        buf471 = empty((256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf469, (256, 1568), (1, 256), 0), view_73, out=buf471)
        del view_73
        buf472 = reinterpret_tensor(buf455, (1568, 128), (128, 1), 0); del buf455  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf469, (1568, 256), (256, 1), 0), permute_395, out=buf472)
        del permute_395
        buf473 = buf448; del buf448  # reuse
        buf475 = buf446; del buf446  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_98.run(buf421, buf429, buf445, buf472, mm_11, unsqueeze_209, buf473, buf475, 1664, 121, grid=grid(1664), stream=stream0)
        buf474 = buf449; del buf449  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_84.run(buf473, buf474, 128, 13, grid=grid(128), stream=stream0)
        buf476 = empty((128, ), device='cuda', dtype=torch.float32)
        buf478 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_85.run(buf475, squeeze_46, buf476, buf478, 128, 13, grid=grid(128), stream=stream0)
        buf477 = reinterpret_tensor(buf454, (1568, 128), (128, 1), 0); del buf454  # reuse
        buf479 = buf477; del buf477  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_99.run(buf479, buf421, buf429, buf445, buf472, mm_11, unsqueeze_209, buf476, squeeze_46, buf474, primals_61, 200704, grid=grid(200704), stream=stream0)
        del mm_11
        del primals_61
        del squeeze_46
        del unsqueeze_209
        buf480 = empty((128, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf479, (128, 1568), (1, 128), 0), view_69, out=buf480)
        del view_69
        buf481 = buf469; del buf469  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf479, (1568, 128), (128, 1), 0), permute_399, out=buf481)
        del permute_399
        buf482 = buf467; del buf467  # reuse
        buf484 = buf465; del buf465  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_87.run(view_68, buf481, mm_10, unsqueeze_213, buf482, buf484, 3328, 121, grid=grid(3328), stream=stream0)
        buf483 = buf468; del buf468  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_88.run(buf482, buf483, 256, 13, grid=grid(256), stream=stream0)
        buf485 = empty((256, ), device='cuda', dtype=torch.float32)
        buf486 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_89.run(buf484, squeeze_43, buf485, buf486, 256, 13, grid=grid(256), stream=stream0)
        buf487 = buf481; del buf481  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_90.run(buf487, view_68, mm_10, unsqueeze_213, buf485, squeeze_43, buf483, primals_58, 401408, grid=grid(401408), stream=stream0)
        del mm_10
        del primals_58
        del squeeze_43
        del unsqueeze_213
        del view_68
        buf488 = empty((256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf487, (256, 1568), (1, 256), 0), view_65, out=buf488)
        del view_65
        buf489 = buf479; del buf479  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf487, (1568, 256), (256, 1), 0), permute_403, out=buf489)
        del permute_403
        buf490 = reinterpret_tensor(buf429, (8, 196, 128), (25088, 128, 1), 0); del buf429  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_100.run(buf490, buf421, buf445, buf472, buf489, 200704, grid=grid(200704), stream=stream0)
        del buf421
        buf491 = buf475; del buf475  # reuse
        buf493 = buf473; del buf473  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_101.run(buf490, mm_9, unsqueeze_217, buf491, buf493, 1664, 121, grid=grid(1664), stream=stream0)
        buf492 = buf476; del buf476  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_84.run(buf491, buf492, 128, 13, grid=grid(128), stream=stream0)
        buf494 = empty((128, ), device='cuda', dtype=torch.float32)
        buf495 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_85.run(buf493, squeeze_40, buf494, buf495, 128, 13, grid=grid(128), stream=stream0)
        buf496 = buf489; del buf489  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_102.run(buf490, mm_9, unsqueeze_217, buf494, squeeze_40, buf492, primals_55, buf496, 200704, grid=grid(200704), stream=stream0)
        del mm_9
        del primals_55
        del squeeze_40
        del unsqueeze_217
        buf497 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf496, (128, 1568), (1, 128), 0), view_61, out=buf497)
        del view_61
        buf498 = buf472; del buf472  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf496, (1568, 128), (128, 1), 0), permute_407, out=buf498)
        del permute_407
        buf499 = reinterpret_tensor(buf496, (8, 4, 196, 32), (25088, 6272, 32, 1), 0); del buf496  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_93.run(view_60, buf498, buf499, 200704, grid=grid(200704), stream=stream0)
        del view_60
        buf500 = reinterpret_tensor(buf498, (32, 196, 32), (6272, 32, 1), 0); del buf498  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_410, reinterpret_tensor(buf499, (32, 196, 32), (6272, 32, 1), 0), out=buf500)
        del permute_410
        buf501 = reinterpret_tensor(buf462, (32, 196, 196), (38416, 196, 1), 0); del buf462  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf499, (32, 196, 32), (6272, 32, 1), 0), permute_411, out=buf501)
        del permute_411
        buf502 = buf457; del buf457  # reuse
        buf507 = reinterpret_tensor(buf456, (8, 4, 196, 196), (153664, 38416, 196, 1), 0); del buf456  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_94.run(buf501, alias_25, buf502, buf507, 6272, 196, grid=grid(6272), stream=stream0)
        buf503 = buf458; del buf458  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]
        triton_per_fused__softmax_backward_data_sum_95.run(buf501, alias_25, buf502, buf503, 153664, 8, grid=grid(153664), stream=stream0)
        del alias_25
        buf504 = empty((4, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_new_zeros_34.run(buf504, 784, grid=grid(784), stream=stream0)
        aten.index_put_(buf504, [None, primals_211], reinterpret_tensor(buf503, (4, 196, 196), (38416, 196, 1), 0), True)
        del primals_211
        buf508 = reinterpret_tensor(buf464, (32, 16, 196), (3136, 196, 1), 0); del buf464  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_412, reinterpret_tensor(buf507, (32, 196, 196), (38416, 196, 1), 0), out=buf508)
        del permute_412
        buf509 = reinterpret_tensor(buf463, (32, 196, 16), (3136, 16, 1), 0); del buf463  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf507, (32, 196, 196), (38416, 196, 1), 0), permute_413, out=buf509)
        del permute_413
        buf510 = buf484; del buf484  # reuse
        buf512 = buf482; del buf482  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_96.run(buf509, buf508, buf500, mm_8, unsqueeze_221, buf510, buf512, 3328, 121, grid=grid(3328), stream=stream0)
        buf511 = buf485; del buf485  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_88.run(buf510, buf511, 256, 13, grid=grid(256), stream=stream0)
        buf513 = empty((256, ), device='cuda', dtype=torch.float32)
        buf515 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_89.run(buf512, squeeze_37, buf513, buf515, 256, 13, grid=grid(256), stream=stream0)
        buf514 = buf487; del buf487  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_97.run(buf509, buf508, buf500, mm_8, unsqueeze_221, buf513, squeeze_37, buf511, primals_52, buf514, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del mm_8
        del primals_52
        del squeeze_37
        del unsqueeze_221
        buf516 = empty((256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf514, (256, 1568), (1, 256), 0), view_49, out=buf516)
        del view_49
        buf517 = reinterpret_tensor(buf500, (1568, 128), (128, 1), 0); del buf500  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf514, (1568, 256), (256, 1), 0), permute_419, out=buf517)
        del permute_419
        buf518 = buf493; del buf493  # reuse
        buf520 = buf491; del buf491  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_103.run(buf490, buf517, mm_7, unsqueeze_225, buf518, buf520, 1664, 121, grid=grid(1664), stream=stream0)
        buf519 = buf494; del buf494  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_84.run(buf518, buf519, 128, 13, grid=grid(128), stream=stream0)
        buf521 = empty((128, ), device='cuda', dtype=torch.float32)
        buf522 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_85.run(buf520, squeeze_34, buf521, buf522, 128, 13, grid=grid(128), stream=stream0)
        buf523 = reinterpret_tensor(buf499, (1568, 128), (128, 1), 0); del buf499  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_104.run(buf490, buf517, mm_7, unsqueeze_225, buf521, squeeze_34, buf519, primals_49, buf523, 200704, grid=grid(200704), stream=stream0)
        del mm_7
        del primals_49
        del squeeze_34
        del unsqueeze_225
        buf524 = empty((128, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf523, (128, 1568), (1, 128), 0), view_45, out=buf524)
        del view_45
        buf525 = buf514; del buf514  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf523, (1568, 128), (128, 1), 0), permute_423, out=buf525)
        del permute_423
        buf526 = buf512; del buf512  # reuse
        buf528 = buf510; del buf510  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_87.run(view_44, buf525, mm_6, unsqueeze_229, buf526, buf528, 3328, 121, grid=grid(3328), stream=stream0)
        buf527 = buf513; del buf513  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_88.run(buf526, buf527, 256, 13, grid=grid(256), stream=stream0)
        buf529 = empty((256, ), device='cuda', dtype=torch.float32)
        buf530 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_89.run(buf528, squeeze_31, buf529, buf530, 256, 13, grid=grid(256), stream=stream0)
        buf531 = buf525; del buf525  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_90.run(buf531, view_44, mm_6, unsqueeze_229, buf529, squeeze_31, buf527, primals_46, 401408, grid=grid(401408), stream=stream0)
        del mm_6
        del primals_46
        del squeeze_31
        del unsqueeze_229
        del view_44
        buf532 = empty((256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf531, (256, 1568), (1, 256), 0), view_41, out=buf532)
        del view_41
        buf533 = buf523; del buf523  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf531, (1568, 256), (256, 1), 0), permute_427, out=buf533)
        del permute_427
        buf534 = buf520; del buf520  # reuse
        buf536 = buf518; del buf518  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_105.run(buf490, buf517, buf533, mm_5, unsqueeze_233, buf534, buf536, 1664, 121, grid=grid(1664), stream=stream0)
        buf535 = buf521; del buf521  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_84.run(buf534, buf535, 128, 13, grid=grid(128), stream=stream0)
        buf537 = empty((128, ), device='cuda', dtype=torch.float32)
        buf539 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_85.run(buf536, squeeze_28, buf537, buf539, 128, 13, grid=grid(128), stream=stream0)
        buf538 = buf445; del buf445  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_106.run(buf490, buf517, buf533, mm_5, unsqueeze_233, buf537, squeeze_28, buf535, primals_43, buf538, 200704, grid=grid(200704), stream=stream0)
        del mm_5
        del primals_43
        del squeeze_28
        del unsqueeze_233
        buf540 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf538, (128, 1568), (1, 128), 0), view_37, out=buf540)
        del view_37
        buf541 = empty((1568, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf538, (1568, 128), (128, 1), 0), permute_431, out=buf541)
        del permute_431
        buf542 = reinterpret_tensor(buf538, (8, 4, 196, 32), (25088, 6272, 32, 1), 0); del buf538  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_93.run(view_36, buf541, buf542, 200704, grid=grid(200704), stream=stream0)
        del view_36
        buf543 = reinterpret_tensor(buf541, (32, 196, 32), (6272, 32, 1), 0); del buf541  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_434, reinterpret_tensor(buf542, (32, 196, 32), (6272, 32, 1), 0), out=buf543)
        del permute_434
        buf544 = reinterpret_tensor(buf507, (32, 196, 196), (38416, 196, 1), 0); del buf507  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf542, (32, 196, 32), (6272, 32, 1), 0), permute_435, out=buf544)
        del permute_435
        buf545 = buf502; del buf502  # reuse
        buf550 = reinterpret_tensor(buf501, (8, 4, 196, 196), (153664, 38416, 196, 1), 0); del buf501  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_94.run(buf544, alias_26, buf545, buf550, 6272, 196, grid=grid(6272), stream=stream0)
        buf546 = buf503; del buf503  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]
        triton_per_fused__softmax_backward_data_sum_95.run(buf544, alias_26, buf545, buf546, 153664, 8, grid=grid(153664), stream=stream0)
        del alias_26
        buf547 = empty((4, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.index_put, aten.new_zeros]
        triton_poi_fused_new_zeros_34.run(buf547, 784, grid=grid(784), stream=stream0)
        aten.index_put_(buf547, [None, primals_210], reinterpret_tensor(buf546, (4, 196, 196), (38416, 196, 1), 0), True)
        del primals_210
        buf551 = reinterpret_tensor(buf509, (32, 16, 196), (3136, 196, 1), 0); del buf509  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_436, reinterpret_tensor(buf550, (32, 196, 196), (38416, 196, 1), 0), out=buf551)
        del permute_436
        buf552 = reinterpret_tensor(buf508, (32, 196, 16), (3136, 16, 1), 0); del buf508  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf550, (32, 196, 196), (38416, 196, 1), 0), permute_437, out=buf552)
        del permute_437
        buf553 = buf528; del buf528  # reuse
        buf555 = buf526; del buf526  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_96.run(buf552, buf551, buf543, mm_4, unsqueeze_237, buf553, buf555, 3328, 121, grid=grid(3328), stream=stream0)
        buf554 = buf529; del buf529  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_88.run(buf553, buf554, 256, 13, grid=grid(256), stream=stream0)
        buf556 = empty((256, ), device='cuda', dtype=torch.float32)
        buf558 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_89.run(buf555, squeeze_25, buf556, buf558, 256, 13, grid=grid(256), stream=stream0)
        buf557 = buf531; del buf531  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_97.run(buf552, buf551, buf543, mm_4, unsqueeze_237, buf556, squeeze_25, buf554, primals_40, buf557, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del mm_4
        del primals_40
        del squeeze_25
        del unsqueeze_237
        buf559 = empty((256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf557, (256, 1568), (1, 256), 0), view_25, out=buf559)
        del view_25
        buf560 = reinterpret_tensor(buf543, (1568, 128), (128, 1), 0); del buf543  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf557, (1568, 256), (256, 1), 0), permute_443, out=buf560)
        del permute_443
        buf561 = buf536; del buf536  # reuse
        buf563 = buf534; del buf534  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_107.run(buf490, buf517, buf533, buf560, mm_3, unsqueeze_241, buf561, buf563, 1664, 121, grid=grid(1664), stream=stream0)
        buf562 = buf537; del buf537  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_84.run(buf561, buf562, 128, 13, grid=grid(128), stream=stream0)
        buf564 = empty((128, ), device='cuda', dtype=torch.float32)
        buf566 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_85.run(buf563, squeeze_22, buf564, buf566, 128, 13, grid=grid(128), stream=stream0)
        buf565 = reinterpret_tensor(buf542, (1568, 128), (128, 1), 0); del buf542  # reuse
        buf567 = buf565; del buf565  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_108.run(buf567, buf490, buf517, buf533, buf560, mm_3, unsqueeze_241, buf564, squeeze_22, buf562, primals_37, 200704, grid=grid(200704), stream=stream0)
        del mm_3
        del primals_37
        del squeeze_22
        del unsqueeze_241
        buf568 = empty((128, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf567, (128, 1568), (1, 128), 0), view_21, out=buf568)
        del view_21
        buf569 = buf557; del buf557  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf567, (1568, 128), (128, 1), 0), permute_447, out=buf569)
        del permute_447
        buf570 = buf555; del buf555  # reuse
        buf572 = buf553; del buf553  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_87.run(view_20, buf569, mm_2, unsqueeze_245, buf570, buf572, 3328, 121, grid=grid(3328), stream=stream0)
        buf571 = buf556; del buf556  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_88.run(buf570, buf571, 256, 13, grid=grid(256), stream=stream0)
        buf573 = empty((256, ), device='cuda', dtype=torch.float32)
        buf574 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_89.run(buf572, squeeze_19, buf573, buf574, 256, 13, grid=grid(256), stream=stream0)
        buf575 = buf569; del buf569  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_90.run(buf575, view_20, mm_2, unsqueeze_245, buf573, squeeze_19, buf571, primals_34, 401408, grid=grid(401408), stream=stream0)
        del mm_2
        del primals_34
        del squeeze_19
        del unsqueeze_245
        del view_20
        buf576 = empty((256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf575, (256, 1568), (1, 256), 0), view_17, out=buf576)
        del view_17
        buf577 = buf567; del buf567  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf575, (1568, 256), (256, 1), 0), permute_451, out=buf577)
        del permute_451
        buf578 = buf490; del buf490  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_109.run(buf578, buf517, buf533, buf560, buf577, 200704, grid=grid(200704), stream=stream0)
        del buf517
        del buf533
        buf579 = buf563; del buf563  # reuse
        buf581 = buf561; del buf561  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_101.run(buf578, mm_1, unsqueeze_249, buf579, buf581, 1664, 121, grid=grid(1664), stream=stream0)
        buf580 = buf564; del buf564  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_84.run(buf579, buf580, 128, 13, grid=grid(128), stream=stream0)
        buf582 = empty((128, ), device='cuda', dtype=torch.float32)
        buf583 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_85.run(buf581, squeeze_16, buf582, buf583, 128, 13, grid=grid(128), stream=stream0)
        buf584 = buf577; del buf577  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_102.run(buf578, mm_1, unsqueeze_249, buf582, squeeze_16, buf580, primals_31, buf584, 200704, grid=grid(200704), stream=stream0)
        del mm_1
        del primals_31
        del squeeze_16
        del unsqueeze_249
        buf585 = empty((128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf584, (128, 1568), (1, 128), 0), view_13, out=buf585)
        del view_13
        buf586 = buf560; del buf560  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf584, (1568, 128), (128, 1), 0), permute_455, out=buf586)
        del permute_455
        buf587 = reinterpret_tensor(buf584, (8, 4, 196, 32), (25088, 6272, 32, 1), 0); del buf584  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_93.run(view_12, buf586, buf587, 200704, grid=grid(200704), stream=stream0)
        del view_12
        buf588 = reinterpret_tensor(buf586, (32, 196, 32), (6272, 32, 1), 0); del buf586  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_458, reinterpret_tensor(buf587, (32, 196, 32), (6272, 32, 1), 0), out=buf588)
        del permute_458
        buf589 = reinterpret_tensor(buf550, (32, 196, 196), (38416, 196, 1), 0); del buf550  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf587, (32, 196, 32), (6272, 32, 1), 0), permute_459, out=buf589)
        del buf587
        del permute_459
        buf590 = buf545; del buf545  # reuse
        buf595 = reinterpret_tensor(buf544, (8, 4, 196, 196), (153664, 38416, 196, 1), 0); del buf544  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_94.run(buf589, alias_27, buf590, buf595, 6272, 196, grid=grid(6272), stream=stream0)
        buf591 = buf546; del buf546  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.sum]
        triton_per_fused__softmax_backward_data_sum_95.run(buf589, alias_27, buf590, buf591, 153664, 8, grid=grid(153664), stream=stream0)
        del alias_27
        del buf589
        del buf590
        buf592 = empty((4, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.new_zeros]
        triton_poi_fused_new_zeros_34.run(buf592, 784, grid=grid(784), stream=stream0)
        aten.index_put_(buf592, [None, primals_209], reinterpret_tensor(buf591, (4, 196, 196), (38416, 196, 1), 0), True)
        del buf591
        del primals_209
        buf596 = reinterpret_tensor(buf552, (32, 16, 196), (3136, 196, 1), 0); del buf552  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_460, reinterpret_tensor(buf595, (32, 196, 196), (38416, 196, 1), 0), out=buf596)
        del permute_460
        buf597 = reinterpret_tensor(buf551, (32, 196, 16), (3136, 16, 1), 0); del buf551  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf595, (32, 196, 196), (38416, 196, 1), 0), permute_461, out=buf597)
        del buf595
        del permute_461
        buf598 = buf572; del buf572  # reuse
        buf600 = buf570; del buf570  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_96.run(buf597, buf596, buf588, mm, unsqueeze_253, buf598, buf600, 3328, 121, grid=grid(3328), stream=stream0)
        buf599 = buf573; del buf573  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_88.run(buf598, buf599, 256, 13, grid=grid(256), stream=stream0)
        del buf598
        buf601 = empty((256, ), device='cuda', dtype=torch.float32)
        buf603 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_89.run(buf600, squeeze_13, buf601, buf603, 256, 13, grid=grid(256), stream=stream0)
        del buf600
        buf602 = buf575; del buf575  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_97.run(buf597, buf596, buf588, mm, unsqueeze_253, buf601, squeeze_13, buf599, primals_28, buf602, 1568, 256, grid=grid(1568, 256), stream=stream0)
        del buf596
        del buf597
        del buf601
        del mm
        del primals_28
        del squeeze_13
        del unsqueeze_253
        buf604 = empty((256, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf602, (256, 1568), (1, 256), 0), view_1, out=buf604)
        del view_1
        buf605 = reinterpret_tensor(buf588, (1568, 128), (128, 1), 0); del buf588  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf602, (1568, 256), (256, 1), 0), permute_467, out=buf605)
        del buf602
        del permute_467
        buf606 = buf581; del buf581  # reuse
        buf608 = buf579; del buf579  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_red_fused_native_batch_norm_backward_110.run(buf578, buf605, convolution_3, unsqueeze_259, buf606, buf608, 1664, 121, grid=grid(1664), stream=stream0)
        buf607 = buf582; del buf582  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_84.run(buf606, buf607, 128, 13, grid=grid(128), stream=stream0)
        del buf606
        buf609 = empty((128, ), device='cuda', dtype=torch.float32)
        buf610 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_per_fused_native_batch_norm_backward_85.run(buf608, squeeze_10, buf609, buf610, 128, 13, grid=grid(128), stream=stream0)
        del buf608
        buf611 = reinterpret_tensor(buf578, (8, 128, 14, 14), (25088, 1, 1792, 128), 0); del buf578  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_111.run(buf611, buf605, convolution_3, unsqueeze_259, buf609, squeeze_10, buf607, primals_25, 1568, 128, grid=grid(1568, 128), stream=stream0)
        del buf605
        del convolution_3
        del primals_25
        del squeeze_10
        del unsqueeze_259
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
        buf612 = aten.convolution_backward(buf611, div_2, primals_24, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf611
        del div_2
        del primals_24
        buf613 = buf612[0]
        buf614 = buf612[1]
        del buf612
        buf615 = empty((64, ), device='cuda', dtype=torch.float32)
        buf616 = empty((64, ), device='cuda', dtype=torch.float32)
        buf617 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_112.run(add_16, buf613, convolution_2, unsqueeze_271, squeeze_7, buf615, buf616, buf617, 64, 6272, grid=grid(64), stream=stream0)
        buf618 = buf613; del buf613  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_113.run(buf618, add_16, convolution_2, unsqueeze_271, buf616, squeeze_7, buf615, primals_22, 401408, grid=grid(401408), stream=stream0)
        del add_16
        del buf616
        del convolution_2
        del primals_22
        del squeeze_7
        del unsqueeze_271
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf619 = aten.convolution_backward(buf618, div_1, primals_21, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf618
        del div_1
        del primals_21
        buf620 = buf619[0]
        buf621 = buf619[1]
        del buf619
        buf622 = reinterpret_tensor(buf609, (32, 4), (1, 32), 0); del buf609  # reuse
        buf624 = empty_strided((32, 4), (1, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_114.run(add_10, buf620, convolution_1, unsqueeze_283, buf622, buf624, 128, 6272, grid=grid(128), stream=stream0)
        buf623 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_115.run(buf622, buf623, 32, 4, grid=grid(32), stream=stream0)
        del buf622
        buf625 = empty((32, ), device='cuda', dtype=torch.float32)
        buf626 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_116.run(buf624, squeeze_4, buf625, buf626, 32, 4, grid=grid(32), stream=stream0)
        del buf624
        buf627 = buf620; del buf620  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_117.run(buf627, add_10, convolution_1, unsqueeze_283, buf625, squeeze_4, buf623, primals_19, 802816, grid=grid(802816), stream=stream0)
        del add_10
        del buf625
        del convolution_1
        del primals_19
        del squeeze_4
        del unsqueeze_283
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf628 = aten.convolution_backward(buf627, div, primals_18, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf627
        del div
        del primals_18
        buf629 = buf628[0]
        buf630 = buf628[1]
        del buf628
        buf631 = empty((16, 13), device='cuda', dtype=torch.float32)
        buf633 = empty((16, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_red_fused_hardswish_backward_native_batch_norm_backward_118.run(add_4, buf629, convolution, unsqueeze_295, buf631, buf633, 208, 7720, grid=grid(208), stream=stream0)
        buf632 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_119.run(buf631, buf632, 16, 13, grid=grid(16), stream=stream0)
        del buf631
        buf634 = empty((16, ), device='cuda', dtype=torch.float32)
        buf635 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_per_fused_hardswish_backward_native_batch_norm_backward_120.run(buf633, squeeze_1, buf634, buf635, 16, 13, grid=grid(16), stream=stream0)
        del buf633
        buf636 = buf629; del buf629  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        triton_poi_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_121.run(buf636, add_4, convolution, unsqueeze_295, buf634, squeeze_1, buf632, primals_16, 1605632, grid=grid(1605632), stream=stream0)
        del add_4
        del buf634
        del convolution
        del primals_16
        del squeeze_1
        del unsqueeze_295
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
        buf637 = aten.convolution_backward(buf636, primals_415, primals_15, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf636
        del primals_15
        del primals_415
        buf638 = buf637[1]
        return (buf592, buf547, buf504, buf459, buf408, buf365, buf320, buf277, buf232, buf183, buf147, buf111, buf75, buf39, buf638, buf635, buf632, buf630, buf626, buf623, buf621, buf617, buf615, buf614, buf610, buf607, reinterpret_tensor(buf604, (256, 128), (128, 1), 0), buf603, buf599, reinterpret_tensor(buf585, (128, 128), (128, 1), 0), buf583, buf580, reinterpret_tensor(buf576, (256, 128), (128, 1), 0), buf574, buf571, reinterpret_tensor(buf568, (128, 256), (256, 1), 0), buf566, buf562, reinterpret_tensor(buf559, (256, 128), (128, 1), 0), buf558, buf554, reinterpret_tensor(buf540, (128, 128), (128, 1), 0), buf539, buf535, reinterpret_tensor(buf532, (256, 128), (128, 1), 0), buf530, buf527, reinterpret_tensor(buf524, (128, 256), (256, 1), 0), buf522, buf519, reinterpret_tensor(buf516, (256, 128), (128, 1), 0), buf515, buf511, reinterpret_tensor(buf497, (128, 128), (128, 1), 0), buf495, buf492, reinterpret_tensor(buf488, (256, 128), (128, 1), 0), buf486, buf483, reinterpret_tensor(buf480, (128, 256), (256, 1), 0), buf478, buf474, reinterpret_tensor(buf471, (256, 128), (128, 1), 0), buf470, buf466, reinterpret_tensor(buf452, (128, 128), (128, 1), 0), buf451, buf447, reinterpret_tensor(buf444, (256, 128), (128, 1), 0), buf442, buf439, reinterpret_tensor(buf436, (128, 256), (256, 1), 0), buf434, buf431, reinterpret_tensor(buf428, (640, 128), (128, 1), 0), buf426, buf423, reinterpret_tensor(buf420, (128, 128), (128, 1), 0), buf418, buf415, reinterpret_tensor(buf401, (256, 512), (512, 1), 0), buf400, buf396, reinterpret_tensor(buf393, (512, 256), (256, 1), 0), buf391, buf388, reinterpret_tensor(buf385, (256, 512), (512, 1), 0), buf383, buf380, reinterpret_tensor(buf377, (512, 256), (256, 1), 0), buf376, buf372, reinterpret_tensor(buf358, (256, 256), (256, 1), 0), buf356, buf353, reinterpret_tensor(buf349, (512, 256), (256, 1), 0), buf347, buf344, reinterpret_tensor(buf341, (256, 512), (512, 1), 0), buf339, buf335, reinterpret_tensor(buf332, (512, 256), (256, 1), 0), buf331, buf327, reinterpret_tensor(buf313, (256, 256), (256, 1), 0), buf312, buf308, reinterpret_tensor(buf305, (512, 256), (256, 1), 0), buf303, buf300, reinterpret_tensor(buf297, (256, 512), (512, 1), 0), buf295, buf292, reinterpret_tensor(buf289, (512, 256), (256, 1), 0), buf288, buf284, reinterpret_tensor(buf270, (256, 256), (256, 1), 0), buf268, buf265, reinterpret_tensor(buf261, (512, 256), (256, 1), 0), buf259, buf256, reinterpret_tensor(buf253, (256, 512), (512, 1), 0), buf251, buf247, reinterpret_tensor(buf244, (512, 256), (256, 1), 0), buf243, buf239, reinterpret_tensor(buf225, (256, 256), (256, 1), 0), buf224, buf220, reinterpret_tensor(buf217, (512, 256), (256, 1), 0), buf215, buf212, reinterpret_tensor(buf209, (256, 512), (512, 1), 0), buf207, buf204, reinterpret_tensor(buf201, (1280, 256), (256, 1), 0), buf199, buf196, reinterpret_tensor(buf193, (256, 256), (256, 1), 0), buf191, buf189, reinterpret_tensor(buf176, (384, 1024), (1024, 1), 0), buf174, buf172, reinterpret_tensor(buf170, (768, 384), (384, 1), 0), buf168, buf166, reinterpret_tensor(buf164, (384, 768), (768, 1), 0), buf162, buf160, reinterpret_tensor(buf157, (768, 384), (384, 1), 0), buf156, buf153, reinterpret_tensor(buf140, (384, 384), (384, 1), 0), buf138, buf135, reinterpret_tensor(buf133, (768, 384), (384, 1), 0), buf131, buf129, reinterpret_tensor(buf127, (384, 768), (768, 1), 0), buf126, buf123, reinterpret_tensor(buf121, (768, 384), (384, 1), 0), buf120, buf117, reinterpret_tensor(buf104, (384, 384), (384, 1), 0), buf102, buf100, reinterpret_tensor(buf98, (768, 384), (384, 1), 0), buf96, buf94, reinterpret_tensor(buf92, (384, 768), (768, 1), 0), buf90, buf88, reinterpret_tensor(buf85, (768, 384), (384, 1), 0), buf84, buf81, reinterpret_tensor(buf68, (384, 384), (384, 1), 0), buf66, buf63, reinterpret_tensor(buf61, (768, 384), (384, 1), 0), buf59, buf57, reinterpret_tensor(buf55, (384, 768), (768, 1), 0), buf54, buf51, reinterpret_tensor(buf49, (768, 384), (384, 1), 0), buf48, buf45, reinterpret_tensor(buf32, (384, 384), (384, 1), 0), buf30, buf28, reinterpret_tensor(buf26, (768, 384), (384, 1), 0), buf24, buf22, reinterpret_tensor(buf20, (384, 768), (768, 1), 0), buf18, buf16, buf14, buf12, reinterpret_tensor(buf11, (1000, 384), (384, 1), 0), reinterpret_tensor(buf6, (1000, ), (1, ), 0), buf9, buf7, reinterpret_tensor(buf5, (1000, 384), (384, 1), 0), reinterpret_tensor(buf6, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_15 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((32, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.int64)
    primals_210 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.int64)
    primals_211 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.int64)
    primals_212 = rand_strided((196, 196), (196, 1), device='cuda:0', dtype=torch.int64)
    primals_213 = rand_strided((49, 196), (196, 1), device='cuda:0', dtype=torch.int64)
    primals_214 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    primals_215 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    primals_216 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    primals_217 = rand_strided((49, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    primals_218 = rand_strided((16, 49), (49, 1), device='cuda:0', dtype=torch.int64)
    primals_219 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.int64)
    primals_220 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.int64)
    primals_221 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.int64)
    primals_222 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.int64)
    primals_415 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 16, 112, 112), (200704, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_4 = rand_strided((8, 16, 112, 112), (200704, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    div = rand_strided((8, 16, 112, 112), (200704, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_10 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    div_1 = rand_strided((8, 32, 56, 56), (100352, 3136, 56, 1), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_16 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    div_2 = rand_strided((8, 64, 28, 28), (50176, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    convolution_3 = rand_strided((8, 128, 14, 14), (25088, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    squeeze_10 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_1 = rand_strided((1568, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mm = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    squeeze_13 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_12 = rand_strided((8, 196, 128), (25088, 128, 1), device='cuda:0', dtype=torch.float32)
    view_13 = rand_strided((1568, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mm_1 = rand_strided((1568, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    squeeze_16 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_17 = rand_strided((1568, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mm_2 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    squeeze_19 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_20 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_21 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mm_3 = rand_strided((1568, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    squeeze_22 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_25 = rand_strided((1568, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mm_4 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    squeeze_25 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_36 = rand_strided((8, 196, 128), (25088, 128, 1), device='cuda:0', dtype=torch.float32)
    view_37 = rand_strided((1568, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mm_5 = rand_strided((1568, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    squeeze_28 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_41 = rand_strided((1568, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mm_6 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    squeeze_31 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_44 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_45 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mm_7 = rand_strided((1568, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    squeeze_34 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_49 = rand_strided((1568, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mm_8 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    squeeze_37 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_60 = rand_strided((8, 196, 128), (25088, 128, 1), device='cuda:0', dtype=torch.float32)
    view_61 = rand_strided((1568, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mm_9 = rand_strided((1568, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    squeeze_40 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_65 = rand_strided((1568, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mm_10 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    squeeze_43 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_68 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_69 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mm_11 = rand_strided((1568, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    squeeze_46 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_73 = rand_strided((1568, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mm_12 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    squeeze_49 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_84 = rand_strided((8, 196, 128), (25088, 128, 1), device='cuda:0', dtype=torch.float32)
    view_85 = rand_strided((1568, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mm_13 = rand_strided((1568, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    squeeze_52 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_89 = rand_strided((1568, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mm_14 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    squeeze_55 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_92 = rand_strided((8, 196, 256), (50176, 256, 1), device='cuda:0', dtype=torch.float32)
    view_93 = rand_strided((1568, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mm_15 = rand_strided((1568, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    squeeze_58 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_97 = rand_strided((1568, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mm_16 = rand_strided((1568, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    squeeze_61 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_104 = rand_strided((392, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    mm_17 = rand_strided((392, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    squeeze_64 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_115 = rand_strided((8, 49, 512), (25088, 512, 1), device='cuda:0', dtype=torch.float32)
    view_116 = rand_strided((392, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_18 = rand_strided((392, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    squeeze_67 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_120 = rand_strided((392, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mm_19 = rand_strided((392, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    squeeze_70 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_123 = rand_strided((8, 49, 512), (25088, 512, 1), device='cuda:0', dtype=torch.float32)
    view_124 = rand_strided((392, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_20 = rand_strided((392, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    squeeze_73 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_128 = rand_strided((392, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mm_21 = rand_strided((392, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    squeeze_76 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_139 = rand_strided((8, 49, 256), (12544, 256, 1), device='cuda:0', dtype=torch.float32)
    view_140 = rand_strided((392, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mm_22 = rand_strided((392, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    squeeze_79 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_144 = rand_strided((392, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mm_23 = rand_strided((392, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    squeeze_82 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_147 = rand_strided((8, 49, 512), (25088, 512, 1), device='cuda:0', dtype=torch.float32)
    view_148 = rand_strided((392, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_24 = rand_strided((392, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    squeeze_85 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_152 = rand_strided((392, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mm_25 = rand_strided((392, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    squeeze_88 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_163 = rand_strided((8, 49, 256), (12544, 256, 1), device='cuda:0', dtype=torch.float32)
    view_164 = rand_strided((392, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mm_26 = rand_strided((392, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    squeeze_91 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_168 = rand_strided((392, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mm_27 = rand_strided((392, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    squeeze_94 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_171 = rand_strided((8, 49, 512), (25088, 512, 1), device='cuda:0', dtype=torch.float32)
    view_172 = rand_strided((392, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_28 = rand_strided((392, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    squeeze_97 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_176 = rand_strided((392, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mm_29 = rand_strided((392, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    squeeze_100 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_187 = rand_strided((8, 49, 256), (12544, 256, 1), device='cuda:0', dtype=torch.float32)
    view_188 = rand_strided((392, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mm_30 = rand_strided((392, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    squeeze_103 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_192 = rand_strided((392, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mm_31 = rand_strided((392, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    squeeze_106 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_195 = rand_strided((8, 49, 512), (25088, 512, 1), device='cuda:0', dtype=torch.float32)
    view_196 = rand_strided((392, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_32 = rand_strided((392, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    squeeze_109 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_200 = rand_strided((392, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mm_33 = rand_strided((392, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    squeeze_112 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_211 = rand_strided((8, 49, 256), (12544, 256, 1), device='cuda:0', dtype=torch.float32)
    view_212 = rand_strided((392, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mm_34 = rand_strided((392, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    squeeze_115 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_216 = rand_strided((392, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mm_35 = rand_strided((392, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    squeeze_118 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_219 = rand_strided((8, 49, 512), (25088, 512, 1), device='cuda:0', dtype=torch.float32)
    view_220 = rand_strided((392, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    mm_36 = rand_strided((392, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    squeeze_121 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_224 = rand_strided((392, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mm_37 = rand_strided((392, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    squeeze_124 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_231 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    mm_38 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    squeeze_127 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_242 = rand_strided((8, 16, 1024), (16384, 1024, 1), device='cuda:0', dtype=torch.float32)
    view_243 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    mm_39 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    squeeze_130 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_247 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mm_40 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    squeeze_133 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_250 = rand_strided((8, 16, 768), (12288, 768, 1), device='cuda:0', dtype=torch.float32)
    view_251 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mm_41 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    squeeze_136 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_255 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mm_42 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    squeeze_139 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_266 = rand_strided((8, 16, 384), (6144, 384, 1), device='cuda:0', dtype=torch.float32)
    view_267 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mm_43 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    squeeze_142 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_271 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mm_44 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    squeeze_145 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_274 = rand_strided((8, 16, 768), (12288, 768, 1), device='cuda:0', dtype=torch.float32)
    view_275 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mm_45 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    squeeze_148 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_279 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mm_46 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    squeeze_151 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_290 = rand_strided((8, 16, 384), (6144, 384, 1), device='cuda:0', dtype=torch.float32)
    view_291 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mm_47 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    squeeze_154 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_295 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mm_48 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    squeeze_157 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_298 = rand_strided((8, 16, 768), (12288, 768, 1), device='cuda:0', dtype=torch.float32)
    view_299 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mm_49 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    squeeze_160 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_303 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mm_50 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    squeeze_163 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_314 = rand_strided((8, 16, 384), (6144, 384, 1), device='cuda:0', dtype=torch.float32)
    view_315 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mm_51 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    squeeze_166 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_319 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mm_52 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    squeeze_169 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_322 = rand_strided((8, 16, 768), (12288, 768, 1), device='cuda:0', dtype=torch.float32)
    view_323 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mm_53 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    squeeze_172 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_327 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mm_54 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    squeeze_175 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_338 = rand_strided((8, 16, 384), (6144, 384, 1), device='cuda:0', dtype=torch.float32)
    view_339 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mm_55 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    squeeze_178 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_343 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mm_56 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    squeeze_181 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    view_346 = rand_strided((8, 16, 768), (12288, 768, 1), device='cuda:0', dtype=torch.float32)
    view_347 = rand_strided((128, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    mm_57 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    squeeze_184 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    mean = rand_strided((8, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    clone_81 = rand_strided((8, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    clone_82 = rand_strided((8, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_117 = rand_strided((1000, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_121 = rand_strided((1000, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_25 = rand_strided((1, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_127 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_29 = rand_strided((1, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_131 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_33 = rand_strided((1, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_135 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_138 = rand_strided((96, 16, 16), (256, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_139 = rand_strided((96, 32, 16), (512, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_14 = rand_strided((8, 12, 16, 16), (3072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    permute_140 = rand_strided((96, 16, 16), (256, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_141 = rand_strided((96, 16, 16), (256, 1, 16), device='cuda:0', dtype=torch.float32)
    unsqueeze_37 = rand_strided((1, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_147 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_41 = rand_strided((1, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_151 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_45 = rand_strided((1, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_155 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_49 = rand_strided((1, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_159 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_162 = rand_strided((96, 16, 16), (256, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_163 = rand_strided((96, 32, 16), (512, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_15 = rand_strided((8, 12, 16, 16), (3072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    permute_164 = rand_strided((96, 16, 16), (256, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_165 = rand_strided((96, 16, 16), (256, 1, 16), device='cuda:0', dtype=torch.float32)
    unsqueeze_53 = rand_strided((1, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_171 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_57 = rand_strided((1, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_175 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_61 = rand_strided((1, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_179 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_65 = rand_strided((1, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_183 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_186 = rand_strided((96, 16, 16), (256, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_187 = rand_strided((96, 32, 16), (512, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_16 = rand_strided((8, 12, 16, 16), (3072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    permute_188 = rand_strided((96, 16, 16), (256, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_189 = rand_strided((96, 16, 16), (256, 1, 16), device='cuda:0', dtype=torch.float32)
    unsqueeze_69 = rand_strided((1, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_195 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_73 = rand_strided((1, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_199 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_77 = rand_strided((1, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_203 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_81 = rand_strided((1, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_207 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_210 = rand_strided((96, 16, 16), (256, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_211 = rand_strided((96, 32, 16), (512, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_17 = rand_strided((8, 12, 16, 16), (3072, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    permute_212 = rand_strided((96, 16, 16), (256, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_213 = rand_strided((96, 16, 16), (256, 1, 16), device='cuda:0', dtype=torch.float32)
    unsqueeze_85 = rand_strided((1, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_219 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_89 = rand_strided((1, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_223 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_93 = rand_strided((1, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_227 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_97 = rand_strided((1, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_231 = rand_strided((384, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    permute_234 = rand_strided((128, 49, 16), (784, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_235 = rand_strided((128, 64, 49), (3136, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_18 = rand_strided((8, 16, 16, 49), (12544, 784, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_236 = rand_strided((128, 16, 16), (256, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_237 = rand_strided((128, 49, 16), (784, 1, 49), device='cuda:0', dtype=torch.float32)
    unsqueeze_101 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_241 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_105 = rand_strided((1, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    permute_247 = rand_strided((1280, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_109 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_251 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_113 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_255 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_117 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_259 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_262 = rand_strided((64, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_263 = rand_strided((64, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_19 = rand_strided((8, 8, 49, 49), (19208, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_264 = rand_strided((64, 16, 49), (784, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_265 = rand_strided((64, 49, 16), (784, 1, 49), device='cuda:0', dtype=torch.float32)
    unsqueeze_121 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_271 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_125 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_275 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_129 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_279 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_133 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_283 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_286 = rand_strided((64, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_287 = rand_strided((64, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_20 = rand_strided((8, 8, 49, 49), (19208, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_288 = rand_strided((64, 16, 49), (784, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_289 = rand_strided((64, 49, 16), (784, 1, 49), device='cuda:0', dtype=torch.float32)
    unsqueeze_137 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_295 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_141 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_299 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_145 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_303 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_149 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_307 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_310 = rand_strided((64, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_311 = rand_strided((64, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_21 = rand_strided((8, 8, 49, 49), (19208, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_312 = rand_strided((64, 16, 49), (784, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_313 = rand_strided((64, 49, 16), (784, 1, 49), device='cuda:0', dtype=torch.float32)
    unsqueeze_153 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_319 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_157 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_323 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_161 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_327 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_165 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_331 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_334 = rand_strided((64, 49, 49), (2401, 1, 49), device='cuda:0', dtype=torch.float32)
    permute_335 = rand_strided((64, 32, 49), (1568, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_22 = rand_strided((8, 8, 49, 49), (19208, 2401, 49, 1), device='cuda:0', dtype=torch.float32)
    permute_336 = rand_strided((64, 16, 49), (784, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_337 = rand_strided((64, 49, 16), (784, 1, 49), device='cuda:0', dtype=torch.float32)
    unsqueeze_169 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_343 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_173 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_347 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_177 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_351 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_181 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_355 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    permute_358 = rand_strided((64, 196, 49), (9604, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_359 = rand_strided((64, 64, 196), (12544, 1, 64), device='cuda:0', dtype=torch.float32)
    alias_23 = rand_strided((8, 8, 49, 196), (76832, 9604, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_360 = rand_strided((64, 16, 49), (784, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_361 = rand_strided((64, 196, 16), (3136, 1, 196), device='cuda:0', dtype=torch.float32)
    unsqueeze_185 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_365 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_189 = rand_strided((1, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    permute_371 = rand_strided((640, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_193 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_375 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_197 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_379 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_201 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_383 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_386 = rand_strided((32, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_387 = rand_strided((32, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_24 = rand_strided((8, 4, 196, 196), (153664, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_388 = rand_strided((32, 16, 196), (3136, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_389 = rand_strided((32, 196, 16), (3136, 1, 196), device='cuda:0', dtype=torch.float32)
    unsqueeze_205 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_395 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_209 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_399 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_213 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_403 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_217 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_407 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_410 = rand_strided((32, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_411 = rand_strided((32, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_25 = rand_strided((8, 4, 196, 196), (153664, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_412 = rand_strided((32, 16, 196), (3136, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_413 = rand_strided((32, 196, 16), (3136, 1, 196), device='cuda:0', dtype=torch.float32)
    unsqueeze_221 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_419 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_225 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_423 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_229 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_427 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_233 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_431 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_434 = rand_strided((32, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_435 = rand_strided((32, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_26 = rand_strided((8, 4, 196, 196), (153664, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_436 = rand_strided((32, 16, 196), (3136, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_437 = rand_strided((32, 196, 16), (3136, 1, 196), device='cuda:0', dtype=torch.float32)
    unsqueeze_237 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_443 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_241 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_447 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_245 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_451 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_249 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_455 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    permute_458 = rand_strided((32, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_459 = rand_strided((32, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_27 = rand_strided((8, 4, 196, 196), (153664, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_460 = rand_strided((32, 16, 196), (3136, 1, 16), device='cuda:0', dtype=torch.float32)
    permute_461 = rand_strided((32, 196, 16), (3136, 1, 196), device='cuda:0', dtype=torch.float32)
    unsqueeze_253 = rand_strided((1, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    permute_467 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_259 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_271 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_283 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_295 = rand_strided((1, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_15, primals_16, primals_18, primals_19, primals_21, primals_22, primals_24, primals_25, primals_28, primals_31, primals_34, primals_37, primals_40, primals_43, primals_46, primals_49, primals_52, primals_55, primals_58, primals_61, primals_64, primals_67, primals_70, primals_73, primals_76, primals_79, primals_82, primals_85, primals_88, primals_91, primals_94, primals_97, primals_100, primals_103, primals_106, primals_109, primals_112, primals_115, primals_118, primals_121, primals_124, primals_127, primals_130, primals_133, primals_136, primals_139, primals_142, primals_145, primals_148, primals_151, primals_154, primals_157, primals_160, primals_163, primals_166, primals_169, primals_172, primals_175, primals_178, primals_181, primals_184, primals_187, primals_190, primals_193, primals_196, primals_199, primals_201, primals_205, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_415, convolution, squeeze_1, add_4, div, convolution_1, squeeze_4, add_10, div_1, convolution_2, squeeze_7, add_16, div_2, convolution_3, squeeze_10, view_1, mm, squeeze_13, view_12, view_13, mm_1, squeeze_16, view_17, mm_2, squeeze_19, view_20, view_21, mm_3, squeeze_22, view_25, mm_4, squeeze_25, view_36, view_37, mm_5, squeeze_28, view_41, mm_6, squeeze_31, view_44, view_45, mm_7, squeeze_34, view_49, mm_8, squeeze_37, view_60, view_61, mm_9, squeeze_40, view_65, mm_10, squeeze_43, view_68, view_69, mm_11, squeeze_46, view_73, mm_12, squeeze_49, view_84, view_85, mm_13, squeeze_52, view_89, mm_14, squeeze_55, view_92, view_93, mm_15, squeeze_58, view_97, mm_16, squeeze_61, view_104, mm_17, squeeze_64, view_115, view_116, mm_18, squeeze_67, view_120, mm_19, squeeze_70, view_123, view_124, mm_20, squeeze_73, view_128, mm_21, squeeze_76, view_139, view_140, mm_22, squeeze_79, view_144, mm_23, squeeze_82, view_147, view_148, mm_24, squeeze_85, view_152, mm_25, squeeze_88, view_163, view_164, mm_26, squeeze_91, view_168, mm_27, squeeze_94, view_171, view_172, mm_28, squeeze_97, view_176, mm_29, squeeze_100, view_187, view_188, mm_30, squeeze_103, view_192, mm_31, squeeze_106, view_195, view_196, mm_32, squeeze_109, view_200, mm_33, squeeze_112, view_211, view_212, mm_34, squeeze_115, view_216, mm_35, squeeze_118, view_219, view_220, mm_36, squeeze_121, view_224, mm_37, squeeze_124, view_231, mm_38, squeeze_127, view_242, view_243, mm_39, squeeze_130, view_247, mm_40, squeeze_133, view_250, view_251, mm_41, squeeze_136, view_255, mm_42, squeeze_139, view_266, view_267, mm_43, squeeze_142, view_271, mm_44, squeeze_145, view_274, view_275, mm_45, squeeze_148, view_279, mm_46, squeeze_151, view_290, view_291, mm_47, squeeze_154, view_295, mm_48, squeeze_157, view_298, view_299, mm_49, squeeze_160, view_303, mm_50, squeeze_163, view_314, view_315, mm_51, squeeze_166, view_319, mm_52, squeeze_169, view_322, view_323, mm_53, squeeze_172, view_327, mm_54, squeeze_175, view_338, view_339, mm_55, squeeze_178, view_343, mm_56, squeeze_181, view_346, view_347, mm_57, squeeze_184, mean, clone_81, clone_82, permute_117, permute_121, unsqueeze_25, permute_127, unsqueeze_29, permute_131, unsqueeze_33, permute_135, permute_138, permute_139, alias_14, permute_140, permute_141, unsqueeze_37, permute_147, unsqueeze_41, permute_151, unsqueeze_45, permute_155, unsqueeze_49, permute_159, permute_162, permute_163, alias_15, permute_164, permute_165, unsqueeze_53, permute_171, unsqueeze_57, permute_175, unsqueeze_61, permute_179, unsqueeze_65, permute_183, permute_186, permute_187, alias_16, permute_188, permute_189, unsqueeze_69, permute_195, unsqueeze_73, permute_199, unsqueeze_77, permute_203, unsqueeze_81, permute_207, permute_210, permute_211, alias_17, permute_212, permute_213, unsqueeze_85, permute_219, unsqueeze_89, permute_223, unsqueeze_93, permute_227, unsqueeze_97, permute_231, permute_234, permute_235, alias_18, permute_236, permute_237, unsqueeze_101, permute_241, unsqueeze_105, permute_247, unsqueeze_109, permute_251, unsqueeze_113, permute_255, unsqueeze_117, permute_259, permute_262, permute_263, alias_19, permute_264, permute_265, unsqueeze_121, permute_271, unsqueeze_125, permute_275, unsqueeze_129, permute_279, unsqueeze_133, permute_283, permute_286, permute_287, alias_20, permute_288, permute_289, unsqueeze_137, permute_295, unsqueeze_141, permute_299, unsqueeze_145, permute_303, unsqueeze_149, permute_307, permute_310, permute_311, alias_21, permute_312, permute_313, unsqueeze_153, permute_319, unsqueeze_157, permute_323, unsqueeze_161, permute_327, unsqueeze_165, permute_331, permute_334, permute_335, alias_22, permute_336, permute_337, unsqueeze_169, permute_343, unsqueeze_173, permute_347, unsqueeze_177, permute_351, unsqueeze_181, permute_355, permute_358, permute_359, alias_23, permute_360, permute_361, unsqueeze_185, permute_365, unsqueeze_189, permute_371, unsqueeze_193, permute_375, unsqueeze_197, permute_379, unsqueeze_201, permute_383, permute_386, permute_387, alias_24, permute_388, permute_389, unsqueeze_205, permute_395, unsqueeze_209, permute_399, unsqueeze_213, permute_403, unsqueeze_217, permute_407, permute_410, permute_411, alias_25, permute_412, permute_413, unsqueeze_221, permute_419, unsqueeze_225, permute_423, unsqueeze_229, permute_427, unsqueeze_233, permute_431, permute_434, permute_435, alias_26, permute_436, permute_437, unsqueeze_237, permute_443, unsqueeze_241, permute_447, unsqueeze_245, permute_451, unsqueeze_249, permute_455, permute_458, permute_459, alias_27, permute_460, permute_461, unsqueeze_253, permute_467, unsqueeze_259, unsqueeze_271, unsqueeze_283, unsqueeze_295, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('levit_128', benchmark_compiled_module)
