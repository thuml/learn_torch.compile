
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


# kernel path: /tmp/torchinductor_youkaichao/56/c56yxhplee6c3jxt7n4ps4hmanays7pkvuqgj42i76fftjbd77nj.py
# Source Nodes: [], Original ATen: [aten.sum, aten.view]

triton_poi_fused_sum_view_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sum_view_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/iw/ciwn456cmysbmkhgbcpmq2wdpq4wyz2rzcn4at2mefu3lntdcunm.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.sum, aten.tanh_backward, aten.view]

triton_poi_fused_add_native_dropout_backward_sum_tanh_backward_view_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_dropout_backward_sum_tanh_backward_view_1', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask).to(tl.int1)
    tmp8 = tl.load(in_ptr2 + (x0), xmask)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 1.1111111111111112
    tmp5 = tmp3 * tmp4
    tmp6 = tmp1 * tmp5
    tmp7 = tmp0 + tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = 1.0
    tmp11 = tmp10 - tmp9
    tmp12 = tmp7 * tmp11
    tl.store(in_out_ptr0 + (x0), tmp12, xmask)
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zl/czlgx4754xdpahmpacwyrpyqdc6tn3ny6ky6j5v73e5ir4hfuhb7.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.select_backward]

triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_select_backward_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_select_backward_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 512
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
    tmp4 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp1 = x0
    tmp2 = tl.full([1], 0, tl.int32)
    tmp3 = tmp1 == tmp2
    tmp5 = 0.0
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tmp0 + tmp6
    tmp9 = tmp7 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp15 = tmp9 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp21 = 768.0
    tmp22 = tmp9 * tmp21
    tmp23 = tmp22 - tmp13
    tmp24 = tmp14 * tmp19
    tmp25 = tmp23 - tmp24
    tmp26 = tmp20 * tmp25
    tmp28 = tmp27.to(tl.float32)
    tmp29 = 1.1111111111111112
    tmp30 = tmp28 * tmp29
    tmp31 = tmp26 * tmp30
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp26, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp31, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5r/c5rhq25c5gs4v6ub5asd6sm674tf3lngizhgnu7ngtlpjmvs7lrs.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.select_backward]

triton_per_fused_add_native_layer_norm_backward_select_backward_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_select_backward_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = r1
    tmp2 = tl.full([1], 0, tl.int32)
    tmp3 = tmp1 == tmp2
    tmp5 = 0.0
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tmp0 + tmp6
    tmp9 = tmp7 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask & xmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lp/clpckpjyh7rjo4ioepuapbu6v3ztojni5dyrunpwfjjcggjcucjv.py
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
    size_hints=[4096, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
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


# kernel path: /tmp/torchinductor_youkaichao/nw/cnwqb2wyo6f7q6sz7dzsyrnnlnitzvhrvpg7i2xzh4vn2w7jq65e.py
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
    size_hints=[1024, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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


# kernel path: /tmp/torchinductor_youkaichao/tw/ctw5jabqzlyzt6z2wm5bm2c7wlqinwh6tiowl4x7rryacmo3cwqi.py
# Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu, aten.gelu_backward]
# intermediate_output_11 => add_102, erf_11, mul_83
triton_poi_fused_gelu_gelu_backward_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
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


# kernel path: /tmp/torchinductor_youkaichao/4y/c4ydamd225auikjie2zt77dklnsagld2z3hgorqk2j3txc4iw3ns.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12288
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


# kernel path: /tmp/torchinductor_youkaichao/65/c65anrmw6y26xc6vesgigedhymttq5wmcghca5sforgzpugestel.py
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
    size_hints=[4096, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 4
    RBLOCK: tl.constexpr = 4
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


# kernel path: /tmp/torchinductor_youkaichao/rd/crdzc7otyfja4jrtsgwrx2hkv5mqi6pzhn5i6qxlj6m3am5cejx3.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 512
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


# kernel path: /tmp/torchinductor_youkaichao/li/cliy5beiyrvw3tceohjxwq3i6ka6jnyhm5foes2par2t2l3r4lt2.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 768
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


# kernel path: /tmp/torchinductor_youkaichao/sa/csavvbqo44inpw2qjlsv3fstkftqbxkphyqpuphsxwkc2b5lk2ku.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 512
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
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr7 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
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


# kernel path: /tmp/torchinductor_youkaichao/rp/crp3yz3kxtwwjtgttrocgjnsvdyt2c6l5wuca6rsplkmoccd5peb.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr4 + (x0 + (768*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
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


# kernel path: /tmp/torchinductor_youkaichao/ay/caym7u3473jut3ua5efkjx7dmbg7y7taotz7vsi3hsuhdn5lwzv4.py
# Source Nodes: [], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i64', 9: '*i64', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_13', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
    xnumel = 512
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
    tmp0 = tl.load(in_out_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask & xmask).to(tl.int1)
    tmp12 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr5 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = 1.1111111111111112
    tmp10 = tmp8 * tmp9
    tmp11 = tmp6 * tmp10
    tmp13 = tmp11 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask & xmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp19 = tmp13 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask & xmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp25 = 768.0
    tmp26 = tmp13 * tmp25
    tmp27 = tmp26 - tmp17
    tmp28 = tmp18 * tmp23
    tmp29 = tmp27 - tmp28
    tmp30 = tmp24 * tmp29
    tmp31 = tl.full([1], False, tl.int1)
    tmp32 = 0.0
    tmp33 = tl.where(tmp31, tmp32, tmp30)
    tmp35 = tl.full([1], -1, tl.int64)
    tmp36 = tmp34 == tmp35
    tmp37 = tl.where(tmp36, tmp32, tmp30)
    tmp39 = tl.full([1], 0, tl.int64)
    tmp40 = tmp38 == tmp39
    tmp41 = tl.where(tmp40, tmp32, tmp30)
    tl.store(in_out_ptr0 + (r1 + (768*x0)), tmp11, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp33, rmask & xmask)
    tl.store(out_ptr4 + (r1 + (768*x0)), tmp37, rmask & xmask)
    tl.store(out_ptr5 + (r1 + (768*x0)), tmp41, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mv/cmve36wrwa4lbsvwefc3yzhptv4cij5se64wgm56l3h3dajuo7sc.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 768
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


# kernel path: /tmp/torchinductor_youkaichao/gj/cgj6uomiobsot4srjjbmachax5nkppd3tnxgcoqwwkhss4qqpeiw.py
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
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ua/cua5j34wucnjabgppswrelg7pe436vknck22sbszbnur6fqblwyi.py
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
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_16', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/66/c66pqjlp65hxny4oyaqmbmpwycgock7a7dufay6e772xs4b27agg.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_17', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/v6/cv6cm6hrosmytfguclps7g7bplef4ljrfyfexl43h6kzaucd6vsr.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fw/cfwbditxojbbu6sqo52z3ntlp6nzvyz65sxhtgsabzjxohi47s2m.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 23440896
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
    primals_8, primals_18, primals_24, primals_34, primals_40, primals_50, primals_56, primals_66, primals_72, primals_82, primals_88, primals_98, primals_104, primals_114, primals_120, primals_130, primals_136, primals_146, primals_152, primals_162, primals_168, primals_178, primals_184, primals_194, primals_200, primals_207, full_default, slice_1, select, select_1, select_2, select_3, mul_1, getitem_3, view, clone_default_33, clone_default_34, clone_default_35, getitem_204, getitem_205, getitem_206, alias_default_23, view_16, getitem_7, mul_3, view_18, addmm_4, view_20, getitem_11, mul_8, view_22, clone_default_30, clone_default_31, clone_default_32, getitem_197, getitem_198, getitem_199, alias_default_21, view_38, getitem_17, mul_10, view_40, addmm_10, view_42, getitem_21, mul_15, view_44, clone_default_27, clone_default_28, clone_default_29, getitem_190, getitem_191, getitem_192, alias_default_19, view_60, getitem_27, mul_17, view_62, addmm_16, view_64, getitem_31, mul_22, view_66, clone_default_24, clone_default_25, clone_default_26, getitem_183, getitem_184, getitem_185, alias_default_17, view_82, getitem_37, mul_24, view_84, addmm_22, view_86, getitem_41, mul_29, view_88, clone_default_21, clone_default_22, clone_default_23, getitem_176, getitem_177, getitem_178, alias_default_15, view_104, getitem_47, mul_31, view_106, addmm_28, view_108, getitem_51, mul_36, view_110, clone_default_18, clone_default_19, clone_default_20, getitem_169, getitem_170, getitem_171, alias_default_13, view_126, getitem_57, mul_38, view_128, addmm_34, view_130, getitem_61, mul_43, view_132, clone_default_15, clone_default_16, clone_default_17, getitem_162, getitem_163, getitem_164, alias_default_11, view_148, getitem_67, mul_45, view_150, addmm_40, view_152, getitem_71, mul_50, view_154, clone_default_12, clone_default_13, clone_default_14, getitem_155, getitem_156, getitem_157, alias_default_9, view_170, getitem_77, mul_52, view_172, addmm_46, view_174, getitem_81, mul_57, view_176, clone_default_9, clone_default_10, clone_default_11, getitem_148, getitem_149, getitem_150, alias_default_7, view_192, getitem_87, mul_59, view_194, addmm_52, view_196, getitem_91, mul_64, view_198, clone_default_6, clone_default_7, clone_default_8, getitem_141, getitem_142, getitem_143, alias_default_5, view_214, getitem_97, mul_66, view_216, addmm_58, view_218, getitem_101, mul_71, view_220, clone_default_3, clone_default_4, clone_default_5, getitem_134, getitem_135, getitem_136, alias_default_3, view_236, getitem_107, mul_73, view_238, addmm_64, view_240, getitem_111, mul_78, view_242, clone_default, clone_default_1, clone_default_2, getitem_127, getitem_128, getitem_129, alias_default_1, view_258, getitem_117, mul_80, view_260, addmm_70, view_262, getitem_121, mul_85, select_8, tanh, getitem_124, getitem_125, permute_134, permute_138, div_24, permute_142, permute_146, div_25, permute_150, permute_162, permute_167, permute_171, div_27, permute_175, permute_179, div_28, permute_183, permute_195, permute_200, permute_204, div_30, permute_208, permute_212, div_31, permute_216, permute_228, permute_233, permute_237, div_33, permute_241, permute_245, div_34, permute_249, permute_261, permute_266, permute_270, div_36, permute_274, permute_278, div_37, permute_282, permute_294, permute_299, permute_303, div_39, permute_307, permute_311, div_40, permute_315, permute_327, permute_332, permute_336, div_42, permute_340, permute_344, div_43, permute_348, permute_360, permute_365, permute_369, div_45, permute_373, permute_377, div_46, permute_381, permute_393, permute_398, permute_402, div_48, permute_406, permute_410, div_49, permute_414, permute_426, permute_431, permute_435, div_51, permute_439, permute_443, div_52, permute_447, permute_459, permute_464, permute_468, div_54, permute_472, permute_476, div_55, permute_480, permute_492, permute_497, permute_501, div_57, permute_505, permute_509, div_58, permute_513, permute_525, permute_530, permute_534, div_60, tangents_1, tangents_2, tangents_3 = args
    args.clear()
    assert_size_stride(primals_8, (768, ), (1, ))
    assert_size_stride(primals_18, (768, ), (1, ))
    assert_size_stride(primals_24, (768, ), (1, ))
    assert_size_stride(primals_34, (768, ), (1, ))
    assert_size_stride(primals_40, (768, ), (1, ))
    assert_size_stride(primals_50, (768, ), (1, ))
    assert_size_stride(primals_56, (768, ), (1, ))
    assert_size_stride(primals_66, (768, ), (1, ))
    assert_size_stride(primals_72, (768, ), (1, ))
    assert_size_stride(primals_82, (768, ), (1, ))
    assert_size_stride(primals_88, (768, ), (1, ))
    assert_size_stride(primals_98, (768, ), (1, ))
    assert_size_stride(primals_104, (768, ), (1, ))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_120, (768, ), (1, ))
    assert_size_stride(primals_130, (768, ), (1, ))
    assert_size_stride(primals_136, (768, ), (1, ))
    assert_size_stride(primals_146, (768, ), (1, ))
    assert_size_stride(primals_152, (768, ), (1, ))
    assert_size_stride(primals_162, (768, ), (1, ))
    assert_size_stride(primals_168, (768, ), (1, ))
    assert_size_stride(primals_178, (768, ), (1, ))
    assert_size_stride(primals_184, (768, ), (1, ))
    assert_size_stride(primals_194, (768, ), (1, ))
    assert_size_stride(primals_200, (768, ), (1, ))
    assert_size_stride(primals_207, (1, 512), (512, 1))
    assert_size_stride(full_default, (1, 512), (512, 1))
    assert_size_stride(slice_1, (1, 512), (512, 1))
    assert_size_stride(select, (1, 512), (0, 4))
    assert_size_stride(select_1, (1, 512), (0, 4))
    assert_size_stride(select_2, (1, 512), (0, 4))
    assert_size_stride(select_3, (1, 512), (0, 4))
    assert_size_stride(mul_1, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(getitem_3, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view, (512, 768), (768, 1))
    assert_size_stride(clone_default_33, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_34, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_35, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(getitem_204, (1, 12, 512), (6144, 512, 1))
    assert_size_stride(getitem_205, (), ())
    assert_size_stride(getitem_206, (), ())
    assert_size_stride(alias_default_23, (1, 12, 512, 64), (393216, 64, 768, 1))
    assert_size_stride(view_16, (512, 768), (768, 1))
    assert_size_stride(getitem_7, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_3, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_18, (512, 768), (768, 1))
    assert_size_stride(addmm_4, (512, 3072), (3072, 1))
    assert_size_stride(view_20, (512, 3072), (3072, 1))
    assert_size_stride(getitem_11, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_8, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_22, (512, 768), (768, 1))
    assert_size_stride(clone_default_30, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_31, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_32, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(getitem_197, (1, 12, 512), (6144, 512, 1))
    assert_size_stride(getitem_198, (), ())
    assert_size_stride(getitem_199, (), ())
    assert_size_stride(alias_default_21, (1, 12, 512, 64), (393216, 64, 768, 1))
    assert_size_stride(view_38, (512, 768), (768, 1))
    assert_size_stride(getitem_17, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_10, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_40, (512, 768), (768, 1))
    assert_size_stride(addmm_10, (512, 3072), (3072, 1))
    assert_size_stride(view_42, (512, 3072), (3072, 1))
    assert_size_stride(getitem_21, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_15, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_44, (512, 768), (768, 1))
    assert_size_stride(clone_default_27, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_28, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_29, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(getitem_190, (1, 12, 512), (6144, 512, 1))
    assert_size_stride(getitem_191, (), ())
    assert_size_stride(getitem_192, (), ())
    assert_size_stride(alias_default_19, (1, 12, 512, 64), (393216, 64, 768, 1))
    assert_size_stride(view_60, (512, 768), (768, 1))
    assert_size_stride(getitem_27, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_17, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_62, (512, 768), (768, 1))
    assert_size_stride(addmm_16, (512, 3072), (3072, 1))
    assert_size_stride(view_64, (512, 3072), (3072, 1))
    assert_size_stride(getitem_31, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_22, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_66, (512, 768), (768, 1))
    assert_size_stride(clone_default_24, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_25, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_26, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(getitem_183, (1, 12, 512), (6144, 512, 1))
    assert_size_stride(getitem_184, (), ())
    assert_size_stride(getitem_185, (), ())
    assert_size_stride(alias_default_17, (1, 12, 512, 64), (393216, 64, 768, 1))
    assert_size_stride(view_82, (512, 768), (768, 1))
    assert_size_stride(getitem_37, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_24, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_84, (512, 768), (768, 1))
    assert_size_stride(addmm_22, (512, 3072), (3072, 1))
    assert_size_stride(view_86, (512, 3072), (3072, 1))
    assert_size_stride(getitem_41, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_29, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_88, (512, 768), (768, 1))
    assert_size_stride(clone_default_21, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_22, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_23, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(getitem_176, (1, 12, 512), (6144, 512, 1))
    assert_size_stride(getitem_177, (), ())
    assert_size_stride(getitem_178, (), ())
    assert_size_stride(alias_default_15, (1, 12, 512, 64), (393216, 64, 768, 1))
    assert_size_stride(view_104, (512, 768), (768, 1))
    assert_size_stride(getitem_47, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_31, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_106, (512, 768), (768, 1))
    assert_size_stride(addmm_28, (512, 3072), (3072, 1))
    assert_size_stride(view_108, (512, 3072), (3072, 1))
    assert_size_stride(getitem_51, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_36, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_110, (512, 768), (768, 1))
    assert_size_stride(clone_default_18, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_19, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_20, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(getitem_169, (1, 12, 512), (6144, 512, 1))
    assert_size_stride(getitem_170, (), ())
    assert_size_stride(getitem_171, (), ())
    assert_size_stride(alias_default_13, (1, 12, 512, 64), (393216, 64, 768, 1))
    assert_size_stride(view_126, (512, 768), (768, 1))
    assert_size_stride(getitem_57, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_38, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_128, (512, 768), (768, 1))
    assert_size_stride(addmm_34, (512, 3072), (3072, 1))
    assert_size_stride(view_130, (512, 3072), (3072, 1))
    assert_size_stride(getitem_61, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_43, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_132, (512, 768), (768, 1))
    assert_size_stride(clone_default_15, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_16, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_17, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(getitem_162, (1, 12, 512), (6144, 512, 1))
    assert_size_stride(getitem_163, (), ())
    assert_size_stride(getitem_164, (), ())
    assert_size_stride(alias_default_11, (1, 12, 512, 64), (393216, 64, 768, 1))
    assert_size_stride(view_148, (512, 768), (768, 1))
    assert_size_stride(getitem_67, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_45, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_150, (512, 768), (768, 1))
    assert_size_stride(addmm_40, (512, 3072), (3072, 1))
    assert_size_stride(view_152, (512, 3072), (3072, 1))
    assert_size_stride(getitem_71, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_50, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_154, (512, 768), (768, 1))
    assert_size_stride(clone_default_12, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_13, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_14, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(getitem_155, (1, 12, 512), (6144, 512, 1))
    assert_size_stride(getitem_156, (), ())
    assert_size_stride(getitem_157, (), ())
    assert_size_stride(alias_default_9, (1, 12, 512, 64), (393216, 64, 768, 1))
    assert_size_stride(view_170, (512, 768), (768, 1))
    assert_size_stride(getitem_77, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_52, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_172, (512, 768), (768, 1))
    assert_size_stride(addmm_46, (512, 3072), (3072, 1))
    assert_size_stride(view_174, (512, 3072), (3072, 1))
    assert_size_stride(getitem_81, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_57, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_176, (512, 768), (768, 1))
    assert_size_stride(clone_default_9, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_10, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_11, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(getitem_148, (1, 12, 512), (6144, 512, 1))
    assert_size_stride(getitem_149, (), ())
    assert_size_stride(getitem_150, (), ())
    assert_size_stride(alias_default_7, (1, 12, 512, 64), (393216, 64, 768, 1))
    assert_size_stride(view_192, (512, 768), (768, 1))
    assert_size_stride(getitem_87, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_59, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_194, (512, 768), (768, 1))
    assert_size_stride(addmm_52, (512, 3072), (3072, 1))
    assert_size_stride(view_196, (512, 3072), (3072, 1))
    assert_size_stride(getitem_91, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_64, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_198, (512, 768), (768, 1))
    assert_size_stride(clone_default_6, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_7, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_8, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(getitem_141, (1, 12, 512), (6144, 512, 1))
    assert_size_stride(getitem_142, (), ())
    assert_size_stride(getitem_143, (), ())
    assert_size_stride(alias_default_5, (1, 12, 512, 64), (393216, 64, 768, 1))
    assert_size_stride(view_214, (512, 768), (768, 1))
    assert_size_stride(getitem_97, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_66, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_216, (512, 768), (768, 1))
    assert_size_stride(addmm_58, (512, 3072), (3072, 1))
    assert_size_stride(view_218, (512, 3072), (3072, 1))
    assert_size_stride(getitem_101, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_71, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_220, (512, 768), (768, 1))
    assert_size_stride(clone_default_3, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_4, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_5, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(getitem_134, (1, 12, 512), (6144, 512, 1))
    assert_size_stride(getitem_135, (), ())
    assert_size_stride(getitem_136, (), ())
    assert_size_stride(alias_default_3, (1, 12, 512, 64), (393216, 64, 768, 1))
    assert_size_stride(view_236, (512, 768), (768, 1))
    assert_size_stride(getitem_107, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_73, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_238, (512, 768), (768, 1))
    assert_size_stride(addmm_64, (512, 3072), (3072, 1))
    assert_size_stride(view_240, (512, 3072), (3072, 1))
    assert_size_stride(getitem_111, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_78, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_242, (512, 768), (768, 1))
    assert_size_stride(clone_default, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_1, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(clone_default_2, (1, 12, 512, 64), (393216, 32768, 64, 1))
    assert_size_stride(getitem_127, (1, 12, 512), (6144, 512, 1))
    assert_size_stride(getitem_128, (), ())
    assert_size_stride(getitem_129, (), ())
    assert_size_stride(alias_default_1, (1, 12, 512, 64), (393216, 64, 768, 1))
    assert_size_stride(view_258, (512, 768), (768, 1))
    assert_size_stride(getitem_117, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_80, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(view_260, (512, 768), (768, 1))
    assert_size_stride(addmm_70, (512, 3072), (3072, 1))
    assert_size_stride(view_262, (512, 3072), (3072, 1))
    assert_size_stride(getitem_121, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(mul_85, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(select_8, (1, 768), (768, 1))
    assert_size_stride(tanh, (1, 768), (768, 1))
    assert_size_stride(getitem_124, (1, 768), (768, 1))
    assert_size_stride(getitem_125, (1, 768), (768, 1))
    assert_size_stride(permute_134, (2, 768), (768, 1))
    assert_size_stride(permute_138, (768, 768), (768, 1))
    assert_size_stride(div_24, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_142, (768, 3072), (3072, 1))
    assert_size_stride(permute_146, (3072, 768), (768, 1))
    assert_size_stride(div_25, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_150, (768, 768), (768, 1))
    assert_size_stride(permute_162, (768, 768), (768, 1))
    assert_size_stride(permute_167, (768, 768), (768, 1))
    assert_size_stride(permute_171, (768, 768), (768, 1))
    assert_size_stride(div_27, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_175, (768, 3072), (3072, 1))
    assert_size_stride(permute_179, (3072, 768), (768, 1))
    assert_size_stride(div_28, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_183, (768, 768), (768, 1))
    assert_size_stride(permute_195, (768, 768), (768, 1))
    assert_size_stride(permute_200, (768, 768), (768, 1))
    assert_size_stride(permute_204, (768, 768), (768, 1))
    assert_size_stride(div_30, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_208, (768, 3072), (3072, 1))
    assert_size_stride(permute_212, (3072, 768), (768, 1))
    assert_size_stride(div_31, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_216, (768, 768), (768, 1))
    assert_size_stride(permute_228, (768, 768), (768, 1))
    assert_size_stride(permute_233, (768, 768), (768, 1))
    assert_size_stride(permute_237, (768, 768), (768, 1))
    assert_size_stride(div_33, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_241, (768, 3072), (3072, 1))
    assert_size_stride(permute_245, (3072, 768), (768, 1))
    assert_size_stride(div_34, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_249, (768, 768), (768, 1))
    assert_size_stride(permute_261, (768, 768), (768, 1))
    assert_size_stride(permute_266, (768, 768), (768, 1))
    assert_size_stride(permute_270, (768, 768), (768, 1))
    assert_size_stride(div_36, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_274, (768, 3072), (3072, 1))
    assert_size_stride(permute_278, (3072, 768), (768, 1))
    assert_size_stride(div_37, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_282, (768, 768), (768, 1))
    assert_size_stride(permute_294, (768, 768), (768, 1))
    assert_size_stride(permute_299, (768, 768), (768, 1))
    assert_size_stride(permute_303, (768, 768), (768, 1))
    assert_size_stride(div_39, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_307, (768, 3072), (3072, 1))
    assert_size_stride(permute_311, (3072, 768), (768, 1))
    assert_size_stride(div_40, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_315, (768, 768), (768, 1))
    assert_size_stride(permute_327, (768, 768), (768, 1))
    assert_size_stride(permute_332, (768, 768), (768, 1))
    assert_size_stride(permute_336, (768, 768), (768, 1))
    assert_size_stride(div_42, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_340, (768, 3072), (3072, 1))
    assert_size_stride(permute_344, (3072, 768), (768, 1))
    assert_size_stride(div_43, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_348, (768, 768), (768, 1))
    assert_size_stride(permute_360, (768, 768), (768, 1))
    assert_size_stride(permute_365, (768, 768), (768, 1))
    assert_size_stride(permute_369, (768, 768), (768, 1))
    assert_size_stride(div_45, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_373, (768, 3072), (3072, 1))
    assert_size_stride(permute_377, (3072, 768), (768, 1))
    assert_size_stride(div_46, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_381, (768, 768), (768, 1))
    assert_size_stride(permute_393, (768, 768), (768, 1))
    assert_size_stride(permute_398, (768, 768), (768, 1))
    assert_size_stride(permute_402, (768, 768), (768, 1))
    assert_size_stride(div_48, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_406, (768, 3072), (3072, 1))
    assert_size_stride(permute_410, (3072, 768), (768, 1))
    assert_size_stride(div_49, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_414, (768, 768), (768, 1))
    assert_size_stride(permute_426, (768, 768), (768, 1))
    assert_size_stride(permute_431, (768, 768), (768, 1))
    assert_size_stride(permute_435, (768, 768), (768, 1))
    assert_size_stride(div_51, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_439, (768, 3072), (3072, 1))
    assert_size_stride(permute_443, (3072, 768), (768, 1))
    assert_size_stride(div_52, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_447, (768, 768), (768, 1))
    assert_size_stride(permute_459, (768, 768), (768, 1))
    assert_size_stride(permute_464, (768, 768), (768, 1))
    assert_size_stride(permute_468, (768, 768), (768, 1))
    assert_size_stride(div_54, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_472, (768, 3072), (3072, 1))
    assert_size_stride(permute_476, (3072, 768), (768, 1))
    assert_size_stride(div_55, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_480, (768, 768), (768, 1))
    assert_size_stride(permute_492, (768, 768), (768, 1))
    assert_size_stride(permute_497, (768, 768), (768, 1))
    assert_size_stride(permute_501, (768, 768), (768, 1))
    assert_size_stride(div_57, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_505, (768, 3072), (3072, 1))
    assert_size_stride(permute_509, (3072, 768), (768, 1))
    assert_size_stride(div_58, (1, 512, 1), (512, 1, 1))
    assert_size_stride(permute_513, (768, 768), (768, 1))
    assert_size_stride(permute_525, (768, 768), (768, 1))
    assert_size_stride(permute_530, (768, 768), (768, 1))
    assert_size_stride(permute_534, (768, 768), (768, 1))
    assert_size_stride(div_60, (1, 512, 1), (512, 1, 1))
    assert_size_stride(tangents_1, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(tangents_2, (1, 768), (768, 1))
    assert_size_stride(tangents_3, (1, 2), (2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_3, permute_134, out=buf0)
        del permute_134
        buf1 = empty((2, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_3, (2, 1), (1, 2), 0), getitem_124, out=buf1)
        del getitem_124
        buf2 = empty((2, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum, aten.view]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_sum_view_0.run(tangents_3, buf2, 2, grid=grid(2), stream=stream0)
        del tangents_3
        buf3 = buf0; del buf0  # reuse
        buf6 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.sum, aten.tanh_backward, aten.view]
        triton_poi_fused_add_native_dropout_backward_sum_tanh_backward_view_1.run(buf3, tangents_2, getitem_125, tanh, buf6, 768, grid=grid(768), stream=stream0)
        del getitem_125
        del tangents_2
        del tanh
        buf4 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf3, permute_138, out=buf4)
        del permute_138
        buf5 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf3, (768, 1), (1, 768), 0), select_8, out=buf5)
        del select_8
        buf9 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf12 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.select_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_select_backward_2.run(tangents_1, buf4, primals_200, mul_85, div_24, getitem_121, buf9, buf12, 512, 768, grid=grid(512), stream=stream0)
        del div_24
        del getitem_121
        del primals_200
        buf10 = reinterpret_tensor(buf3, (768, ), (1, ), 0); del buf3  # reuse
        buf11 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.select_backward]
        triton_per_fused_add_native_layer_norm_backward_select_backward_3.run(tangents_1, buf4, mul_85, buf10, buf11, 768, 512, grid=grid(768), stream=stream0)
        del mul_85
        del tangents_1
        buf13 = empty((512, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf12, (512, 768), (768, 1), 0), permute_142, out=buf13)
        del permute_142
        buf14 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf12, (768, 512), (1, 768), 0), view_262, out=buf14)
        del view_262
        buf15 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf12, buf15, 3072, 128, grid=grid(3072), stream=stream0)
        buf16 = buf4; del buf4  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf15, buf16, 768, 4, grid=grid(768), stream=stream0)
        buf17 = reinterpret_tensor(buf13, (1, 512, 3072), (1572864, 3072, 1), 0); del buf13  # reuse
        # Source Nodes: [intermediate_output_11], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf17, addmm_70, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_70
        buf18 = reinterpret_tensor(buf12, (512, 768), (768, 1), 0); del buf12  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (512, 3072), (3072, 1), 0), permute_146, out=buf18)
        del permute_146
        buf19 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (3072, 512), (1, 3072), 0), view_260, out=buf19)
        del view_260
        buf20 = empty_strided((1, 3072, 4), (12288, 1, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf17, buf20, 12288, 128, grid=grid(12288), stream=stream0)
        buf21 = reinterpret_tensor(buf15, (1, 3072), (3072, 1), 0); del buf15  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf20, buf21, 3072, 4, grid=grid(3072), stream=stream0)
        buf24 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        buf27 = empty((1, 512, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_9.run(buf9, buf18, primals_194, mul_80, div_25, getitem_117, buf24, buf27, 512, 768, grid=grid(512), stream=stream0)
        del div_25
        del getitem_117
        del primals_194
        buf25 = empty((768, ), device='cuda', dtype=torch.float32)
        buf26 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf9, buf18, mul_80, buf25, buf26, 768, 512, grid=grid(768), stream=stream0)
        del buf18
        del mul_80
        buf28 = reinterpret_tensor(buf9, (512, 768), (768, 1), 0); del buf9  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (512, 768), (768, 1), 0), permute_150, out=buf28)
        del permute_150
        buf29 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (768, 512), (1, 768), 0), view_258, out=buf29)
        del view_258
        buf30 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf27, buf30, 3072, 128, grid=grid(3072), stream=stream0)
        buf31 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf30, buf31, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf32 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf28, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default, clone_default_1, clone_default_2, None, alias_default_1, getitem_127, getitem_128, getitem_129, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_1
        del clone_default
        del clone_default_1
        del clone_default_2
        del getitem_127
        del getitem_128
        del getitem_129
        buf33 = buf32[0]
        buf34 = buf32[1]
        buf35 = buf32[2]
        del buf32
        buf36 = buf28; del buf28  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf35, (512, 768), (768, 1), 0), permute_162, out=buf36)
        del permute_162
        buf37 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf35, (768, 512), (1, 768), 0), view_242, out=buf37)
        buf38 = buf30; del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf35, buf38, 3072, 128, grid=grid(3072), stream=stream0)
        buf39 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf38, buf39, 768, 4, grid=grid(768), stream=stream0)
        buf40 = reinterpret_tensor(buf35, (512, 768), (768, 1), 0); del buf35  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf34, (512, 768), (768, 1), 0), permute_167, out=buf40)
        del permute_167
        buf41 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf34, (768, 512), (1, 768), 0), view_242, out=buf41)
        buf42 = buf38; del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf34, buf42, 3072, 128, grid=grid(3072), stream=stream0)
        buf43 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf42, buf43, 768, 4, grid=grid(768), stream=stream0)
        buf44 = reinterpret_tensor(buf34, (512, 768), (768, 1), 0); del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf33, (512, 768), (768, 1), 0), permute_171, out=buf44)
        del permute_171
        buf45 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf33, (768, 512), (1, 768), 0), view_242, out=buf45)
        del view_242
        buf46 = buf42; del buf42  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf33, buf46, 3072, 128, grid=grid(3072), stream=stream0)
        buf47 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf46, buf47, 768, 4, grid=grid(768), stream=stream0)
        buf51 = reinterpret_tensor(buf33, (1, 512, 768), (393216, 768, 1), 0); del buf33  # reuse
        buf54 = buf27; del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf24, buf36, buf40, buf44, primals_184, mul_78, div_27, getitem_111, buf51, buf54, 512, 768, grid=grid(512), stream=stream0)
        del div_27
        del getitem_111
        del primals_184
        buf52 = empty((768, ), device='cuda', dtype=torch.float32)
        buf53 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf24, buf36, buf40, buf44, mul_78, buf52, buf53, 768, 512, grid=grid(768), stream=stream0)
        del buf24
        del buf36
        del mul_78
        buf55 = reinterpret_tensor(buf17, (512, 3072), (3072, 1), 0); del buf17  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf54, (512, 768), (768, 1), 0), permute_175, out=buf55)
        del permute_175
        buf56 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf54, (768, 512), (1, 768), 0), view_240, out=buf56)
        del view_240
        buf57 = buf46; del buf46  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf54, buf57, 3072, 128, grid=grid(3072), stream=stream0)
        buf58 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf57, buf58, 768, 4, grid=grid(768), stream=stream0)
        buf59 = reinterpret_tensor(buf55, (1, 512, 3072), (1572864, 3072, 1), 0); del buf55  # reuse
        # Source Nodes: [intermediate_output_10], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf59, addmm_64, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_64
        buf60 = reinterpret_tensor(buf54, (512, 768), (768, 1), 0); del buf54  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf59, (512, 3072), (3072, 1), 0), permute_179, out=buf60)
        del permute_179
        buf61 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf59, (3072, 512), (1, 3072), 0), view_238, out=buf61)
        del view_238
        buf62 = buf20; del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf59, buf62, 12288, 128, grid=grid(12288), stream=stream0)
        buf63 = reinterpret_tensor(buf57, (1, 3072), (3072, 1), 0); del buf57  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf62, buf63, 3072, 4, grid=grid(3072), stream=stream0)
        buf66 = reinterpret_tensor(buf44, (1, 512, 768), (393216, 768, 1), 0); del buf44  # reuse
        buf69 = reinterpret_tensor(buf40, (1, 512, 768), (393216, 768, 1), 0); del buf40  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_9.run(buf51, buf60, primals_178, mul_73, div_28, getitem_107, buf66, buf69, 512, 768, grid=grid(512), stream=stream0)
        del div_28
        del getitem_107
        del primals_178
        buf67 = empty((768, ), device='cuda', dtype=torch.float32)
        buf68 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf51, buf60, mul_73, buf67, buf68, 768, 512, grid=grid(768), stream=stream0)
        del buf51
        del mul_73
        buf70 = buf60; del buf60  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (512, 768), (768, 1), 0), permute_183, out=buf70)
        del permute_183
        buf71 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf69, (768, 512), (1, 768), 0), view_236, out=buf71)
        del view_236
        buf72 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf69, buf72, 3072, 128, grid=grid(3072), stream=stream0)
        buf73 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf72, buf73, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf74 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf70, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_3, clone_default_4, clone_default_5, None, alias_default_3, getitem_134, getitem_135, getitem_136, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_3
        del clone_default_3
        del clone_default_4
        del clone_default_5
        del getitem_134
        del getitem_135
        del getitem_136
        buf75 = buf74[0]
        buf76 = buf74[1]
        buf77 = buf74[2]
        del buf74
        buf78 = buf70; del buf70  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf77, (512, 768), (768, 1), 0), permute_195, out=buf78)
        del permute_195
        buf79 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf77, (768, 512), (1, 768), 0), view_220, out=buf79)
        buf80 = buf72; del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf77, buf80, 3072, 128, grid=grid(3072), stream=stream0)
        buf81 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf80, buf81, 768, 4, grid=grid(768), stream=stream0)
        buf82 = reinterpret_tensor(buf77, (512, 768), (768, 1), 0); del buf77  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf76, (512, 768), (768, 1), 0), permute_200, out=buf82)
        del permute_200
        buf83 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf76, (768, 512), (1, 768), 0), view_220, out=buf83)
        buf84 = buf80; del buf80  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf76, buf84, 3072, 128, grid=grid(3072), stream=stream0)
        buf85 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf84, buf85, 768, 4, grid=grid(768), stream=stream0)
        buf86 = reinterpret_tensor(buf76, (512, 768), (768, 1), 0); del buf76  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf75, (512, 768), (768, 1), 0), permute_204, out=buf86)
        del permute_204
        buf87 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf75, (768, 512), (1, 768), 0), view_220, out=buf87)
        del view_220
        buf88 = buf84; del buf84  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf75, buf88, 3072, 128, grid=grid(3072), stream=stream0)
        buf89 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf88, buf89, 768, 4, grid=grid(768), stream=stream0)
        buf93 = reinterpret_tensor(buf75, (1, 512, 768), (393216, 768, 1), 0); del buf75  # reuse
        buf96 = buf69; del buf69  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf66, buf78, buf82, buf86, primals_168, mul_71, div_30, getitem_101, buf93, buf96, 512, 768, grid=grid(512), stream=stream0)
        del div_30
        del getitem_101
        del primals_168
        buf94 = empty((768, ), device='cuda', dtype=torch.float32)
        buf95 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf66, buf78, buf82, buf86, mul_71, buf94, buf95, 768, 512, grid=grid(768), stream=stream0)
        del buf66
        del buf78
        del mul_71
        buf97 = reinterpret_tensor(buf59, (512, 3072), (3072, 1), 0); del buf59  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf96, (512, 768), (768, 1), 0), permute_208, out=buf97)
        del permute_208
        buf98 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf96, (768, 512), (1, 768), 0), view_218, out=buf98)
        del view_218
        buf99 = buf88; del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf96, buf99, 3072, 128, grid=grid(3072), stream=stream0)
        buf100 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf99, buf100, 768, 4, grid=grid(768), stream=stream0)
        buf101 = reinterpret_tensor(buf97, (1, 512, 3072), (1572864, 3072, 1), 0); del buf97  # reuse
        # Source Nodes: [intermediate_output_9], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf101, addmm_58, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_58
        buf102 = reinterpret_tensor(buf96, (512, 768), (768, 1), 0); del buf96  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf101, (512, 3072), (3072, 1), 0), permute_212, out=buf102)
        del permute_212
        buf103 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf101, (3072, 512), (1, 3072), 0), view_216, out=buf103)
        del view_216
        buf104 = buf62; del buf62  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf101, buf104, 12288, 128, grid=grid(12288), stream=stream0)
        buf105 = reinterpret_tensor(buf99, (1, 3072), (3072, 1), 0); del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf104, buf105, 3072, 4, grid=grid(3072), stream=stream0)
        buf108 = reinterpret_tensor(buf86, (1, 512, 768), (393216, 768, 1), 0); del buf86  # reuse
        buf111 = reinterpret_tensor(buf82, (1, 512, 768), (393216, 768, 1), 0); del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_9.run(buf93, buf102, primals_162, mul_66, div_31, getitem_97, buf108, buf111, 512, 768, grid=grid(512), stream=stream0)
        del div_31
        del getitem_97
        del primals_162
        buf109 = empty((768, ), device='cuda', dtype=torch.float32)
        buf110 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf93, buf102, mul_66, buf109, buf110, 768, 512, grid=grid(768), stream=stream0)
        del buf102
        del mul_66
        buf112 = reinterpret_tensor(buf93, (512, 768), (768, 1), 0); del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf111, (512, 768), (768, 1), 0), permute_216, out=buf112)
        del permute_216
        buf113 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf111, (768, 512), (1, 768), 0), view_214, out=buf113)
        del view_214
        buf114 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf111, buf114, 3072, 128, grid=grid(3072), stream=stream0)
        buf115 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf114, buf115, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf116 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf112, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_6, clone_default_7, clone_default_8, None, alias_default_5, getitem_141, getitem_142, getitem_143, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_5
        del clone_default_6
        del clone_default_7
        del clone_default_8
        del getitem_141
        del getitem_142
        del getitem_143
        buf117 = buf116[0]
        buf118 = buf116[1]
        buf119 = buf116[2]
        del buf116
        buf120 = buf112; del buf112  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf119, (512, 768), (768, 1), 0), permute_228, out=buf120)
        del permute_228
        buf121 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf119, (768, 512), (1, 768), 0), view_198, out=buf121)
        buf122 = buf114; del buf114  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf119, buf122, 3072, 128, grid=grid(3072), stream=stream0)
        buf123 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf122, buf123, 768, 4, grid=grid(768), stream=stream0)
        buf124 = reinterpret_tensor(buf119, (512, 768), (768, 1), 0); del buf119  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf118, (512, 768), (768, 1), 0), permute_233, out=buf124)
        del permute_233
        buf125 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf118, (768, 512), (1, 768), 0), view_198, out=buf125)
        buf126 = buf122; del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf118, buf126, 3072, 128, grid=grid(3072), stream=stream0)
        buf127 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf126, buf127, 768, 4, grid=grid(768), stream=stream0)
        buf128 = reinterpret_tensor(buf118, (512, 768), (768, 1), 0); del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf117, (512, 768), (768, 1), 0), permute_237, out=buf128)
        del permute_237
        buf129 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf117, (768, 512), (1, 768), 0), view_198, out=buf129)
        del view_198
        buf130 = buf126; del buf126  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf117, buf130, 3072, 128, grid=grid(3072), stream=stream0)
        buf131 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf130, buf131, 768, 4, grid=grid(768), stream=stream0)
        buf135 = reinterpret_tensor(buf117, (1, 512, 768), (393216, 768, 1), 0); del buf117  # reuse
        buf138 = buf111; del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf108, buf120, buf124, buf128, primals_152, mul_64, div_33, getitem_91, buf135, buf138, 512, 768, grid=grid(512), stream=stream0)
        del div_33
        del getitem_91
        del primals_152
        buf136 = empty((768, ), device='cuda', dtype=torch.float32)
        buf137 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf108, buf120, buf124, buf128, mul_64, buf136, buf137, 768, 512, grid=grid(768), stream=stream0)
        del buf108
        del buf120
        del mul_64
        buf139 = reinterpret_tensor(buf101, (512, 3072), (3072, 1), 0); del buf101  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf138, (512, 768), (768, 1), 0), permute_241, out=buf139)
        del permute_241
        buf140 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf138, (768, 512), (1, 768), 0), view_196, out=buf140)
        del view_196
        buf141 = buf130; del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf138, buf141, 3072, 128, grid=grid(3072), stream=stream0)
        buf142 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf141, buf142, 768, 4, grid=grid(768), stream=stream0)
        buf143 = reinterpret_tensor(buf139, (1, 512, 3072), (1572864, 3072, 1), 0); del buf139  # reuse
        # Source Nodes: [intermediate_output_8], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf143, addmm_52, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_52
        buf144 = reinterpret_tensor(buf138, (512, 768), (768, 1), 0); del buf138  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (512, 3072), (3072, 1), 0), permute_245, out=buf144)
        del permute_245
        buf145 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (3072, 512), (1, 3072), 0), view_194, out=buf145)
        del view_194
        buf146 = buf104; del buf104  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf143, buf146, 12288, 128, grid=grid(12288), stream=stream0)
        buf147 = reinterpret_tensor(buf141, (1, 3072), (3072, 1), 0); del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf146, buf147, 3072, 4, grid=grid(3072), stream=stream0)
        buf150 = reinterpret_tensor(buf128, (1, 512, 768), (393216, 768, 1), 0); del buf128  # reuse
        buf153 = reinterpret_tensor(buf124, (1, 512, 768), (393216, 768, 1), 0); del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_9.run(buf135, buf144, primals_146, mul_59, div_34, getitem_87, buf150, buf153, 512, 768, grid=grid(512), stream=stream0)
        del div_34
        del getitem_87
        del primals_146
        buf151 = empty((768, ), device='cuda', dtype=torch.float32)
        buf152 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf135, buf144, mul_59, buf151, buf152, 768, 512, grid=grid(768), stream=stream0)
        del buf135
        del mul_59
        buf154 = buf144; del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf153, (512, 768), (768, 1), 0), permute_249, out=buf154)
        del permute_249
        buf155 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf153, (768, 512), (1, 768), 0), view_192, out=buf155)
        del view_192
        buf156 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf153, buf156, 3072, 128, grid=grid(3072), stream=stream0)
        buf157 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf156, buf157, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf158 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf154, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_9, clone_default_10, clone_default_11, None, alias_default_7, getitem_148, getitem_149, getitem_150, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_7
        del clone_default_10
        del clone_default_11
        del clone_default_9
        del getitem_148
        del getitem_149
        del getitem_150
        buf159 = buf158[0]
        buf160 = buf158[1]
        buf161 = buf158[2]
        del buf158
        buf162 = buf154; del buf154  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf161, (512, 768), (768, 1), 0), permute_261, out=buf162)
        del permute_261
        buf163 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf161, (768, 512), (1, 768), 0), view_176, out=buf163)
        buf164 = buf156; del buf156  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf161, buf164, 3072, 128, grid=grid(3072), stream=stream0)
        buf165 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf164, buf165, 768, 4, grid=grid(768), stream=stream0)
        buf166 = reinterpret_tensor(buf161, (512, 768), (768, 1), 0); del buf161  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf160, (512, 768), (768, 1), 0), permute_266, out=buf166)
        del permute_266
        buf167 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf160, (768, 512), (1, 768), 0), view_176, out=buf167)
        buf168 = buf164; del buf164  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf160, buf168, 3072, 128, grid=grid(3072), stream=stream0)
        buf169 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf168, buf169, 768, 4, grid=grid(768), stream=stream0)
        buf170 = reinterpret_tensor(buf160, (512, 768), (768, 1), 0); del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf159, (512, 768), (768, 1), 0), permute_270, out=buf170)
        del permute_270
        buf171 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf159, (768, 512), (1, 768), 0), view_176, out=buf171)
        del view_176
        buf172 = buf168; del buf168  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf159, buf172, 3072, 128, grid=grid(3072), stream=stream0)
        buf173 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf172, buf173, 768, 4, grid=grid(768), stream=stream0)
        buf177 = reinterpret_tensor(buf159, (1, 512, 768), (393216, 768, 1), 0); del buf159  # reuse
        buf180 = buf153; del buf153  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf150, buf162, buf166, buf170, primals_136, mul_57, div_36, getitem_81, buf177, buf180, 512, 768, grid=grid(512), stream=stream0)
        del div_36
        del getitem_81
        del primals_136
        buf178 = empty((768, ), device='cuda', dtype=torch.float32)
        buf179 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf150, buf162, buf166, buf170, mul_57, buf178, buf179, 768, 512, grid=grid(768), stream=stream0)
        del buf150
        del buf162
        del mul_57
        buf181 = reinterpret_tensor(buf143, (512, 3072), (3072, 1), 0); del buf143  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf180, (512, 768), (768, 1), 0), permute_274, out=buf181)
        del permute_274
        buf182 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf180, (768, 512), (1, 768), 0), view_174, out=buf182)
        del view_174
        buf183 = buf172; del buf172  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf180, buf183, 3072, 128, grid=grid(3072), stream=stream0)
        buf184 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf183, buf184, 768, 4, grid=grid(768), stream=stream0)
        buf185 = reinterpret_tensor(buf181, (1, 512, 3072), (1572864, 3072, 1), 0); del buf181  # reuse
        # Source Nodes: [intermediate_output_7], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf185, addmm_46, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_46
        buf186 = reinterpret_tensor(buf180, (512, 768), (768, 1), 0); del buf180  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf185, (512, 3072), (3072, 1), 0), permute_278, out=buf186)
        del permute_278
        buf187 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf185, (3072, 512), (1, 3072), 0), view_172, out=buf187)
        del view_172
        buf188 = buf146; del buf146  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf185, buf188, 12288, 128, grid=grid(12288), stream=stream0)
        buf189 = reinterpret_tensor(buf183, (1, 3072), (3072, 1), 0); del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf188, buf189, 3072, 4, grid=grid(3072), stream=stream0)
        buf192 = reinterpret_tensor(buf170, (1, 512, 768), (393216, 768, 1), 0); del buf170  # reuse
        buf195 = reinterpret_tensor(buf166, (1, 512, 768), (393216, 768, 1), 0); del buf166  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_9.run(buf177, buf186, primals_130, mul_52, div_37, getitem_77, buf192, buf195, 512, 768, grid=grid(512), stream=stream0)
        del div_37
        del getitem_77
        del primals_130
        buf193 = empty((768, ), device='cuda', dtype=torch.float32)
        buf194 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf177, buf186, mul_52, buf193, buf194, 768, 512, grid=grid(768), stream=stream0)
        del buf177
        del mul_52
        buf196 = buf186; del buf186  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf195, (512, 768), (768, 1), 0), permute_282, out=buf196)
        del permute_282
        buf197 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf195, (768, 512), (1, 768), 0), view_170, out=buf197)
        del view_170
        buf198 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf195, buf198, 3072, 128, grid=grid(3072), stream=stream0)
        buf199 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf198, buf199, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf200 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf196, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_12, clone_default_13, clone_default_14, None, alias_default_9, getitem_155, getitem_156, getitem_157, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_9
        del clone_default_12
        del clone_default_13
        del clone_default_14
        del getitem_155
        del getitem_156
        del getitem_157
        buf201 = buf200[0]
        buf202 = buf200[1]
        buf203 = buf200[2]
        del buf200
        buf204 = buf196; del buf196  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf203, (512, 768), (768, 1), 0), permute_294, out=buf204)
        del permute_294
        buf205 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf203, (768, 512), (1, 768), 0), view_154, out=buf205)
        buf206 = buf198; del buf198  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf203, buf206, 3072, 128, grid=grid(3072), stream=stream0)
        buf207 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf206, buf207, 768, 4, grid=grid(768), stream=stream0)
        buf208 = reinterpret_tensor(buf203, (512, 768), (768, 1), 0); del buf203  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf202, (512, 768), (768, 1), 0), permute_299, out=buf208)
        del permute_299
        buf209 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf202, (768, 512), (1, 768), 0), view_154, out=buf209)
        buf210 = buf206; del buf206  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf202, buf210, 3072, 128, grid=grid(3072), stream=stream0)
        buf211 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf210, buf211, 768, 4, grid=grid(768), stream=stream0)
        buf212 = reinterpret_tensor(buf202, (512, 768), (768, 1), 0); del buf202  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf201, (512, 768), (768, 1), 0), permute_303, out=buf212)
        del permute_303
        buf213 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf201, (768, 512), (1, 768), 0), view_154, out=buf213)
        del view_154
        buf214 = buf210; del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf201, buf214, 3072, 128, grid=grid(3072), stream=stream0)
        buf215 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf214, buf215, 768, 4, grid=grid(768), stream=stream0)
        buf219 = reinterpret_tensor(buf201, (1, 512, 768), (393216, 768, 1), 0); del buf201  # reuse
        buf222 = buf195; del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf192, buf204, buf208, buf212, primals_120, mul_50, div_39, getitem_71, buf219, buf222, 512, 768, grid=grid(512), stream=stream0)
        del div_39
        del getitem_71
        del primals_120
        buf220 = empty((768, ), device='cuda', dtype=torch.float32)
        buf221 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf192, buf204, buf208, buf212, mul_50, buf220, buf221, 768, 512, grid=grid(768), stream=stream0)
        del buf192
        del buf204
        del mul_50
        buf223 = reinterpret_tensor(buf185, (512, 3072), (3072, 1), 0); del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf222, (512, 768), (768, 1), 0), permute_307, out=buf223)
        del permute_307
        buf224 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf222, (768, 512), (1, 768), 0), view_152, out=buf224)
        del view_152
        buf225 = buf214; del buf214  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf222, buf225, 3072, 128, grid=grid(3072), stream=stream0)
        buf226 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf225, buf226, 768, 4, grid=grid(768), stream=stream0)
        buf227 = reinterpret_tensor(buf223, (1, 512, 3072), (1572864, 3072, 1), 0); del buf223  # reuse
        # Source Nodes: [intermediate_output_6], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf227, addmm_40, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_40
        buf228 = reinterpret_tensor(buf222, (512, 768), (768, 1), 0); del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf227, (512, 3072), (3072, 1), 0), permute_311, out=buf228)
        del permute_311
        buf229 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf227, (3072, 512), (1, 3072), 0), view_150, out=buf229)
        del view_150
        buf230 = buf188; del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf227, buf230, 12288, 128, grid=grid(12288), stream=stream0)
        buf231 = reinterpret_tensor(buf225, (1, 3072), (3072, 1), 0); del buf225  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf230, buf231, 3072, 4, grid=grid(3072), stream=stream0)
        buf234 = reinterpret_tensor(buf212, (1, 512, 768), (393216, 768, 1), 0); del buf212  # reuse
        buf237 = reinterpret_tensor(buf208, (1, 512, 768), (393216, 768, 1), 0); del buf208  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_9.run(buf219, buf228, primals_114, mul_45, div_40, getitem_67, buf234, buf237, 512, 768, grid=grid(512), stream=stream0)
        del div_40
        del getitem_67
        del primals_114
        buf235 = empty((768, ), device='cuda', dtype=torch.float32)
        buf236 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf219, buf228, mul_45, buf235, buf236, 768, 512, grid=grid(768), stream=stream0)
        del buf219
        del mul_45
        buf238 = buf228; del buf228  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf237, (512, 768), (768, 1), 0), permute_315, out=buf238)
        del permute_315
        buf239 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf237, (768, 512), (1, 768), 0), view_148, out=buf239)
        del view_148
        buf240 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf237, buf240, 3072, 128, grid=grid(3072), stream=stream0)
        buf241 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf240, buf241, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf242 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf238, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_15, clone_default_16, clone_default_17, None, alias_default_11, getitem_162, getitem_163, getitem_164, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_11
        del clone_default_15
        del clone_default_16
        del clone_default_17
        del getitem_162
        del getitem_163
        del getitem_164
        buf243 = buf242[0]
        buf244 = buf242[1]
        buf245 = buf242[2]
        del buf242
        buf246 = buf238; del buf238  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf245, (512, 768), (768, 1), 0), permute_327, out=buf246)
        del permute_327
        buf247 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf245, (768, 512), (1, 768), 0), view_132, out=buf247)
        buf248 = buf240; del buf240  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf245, buf248, 3072, 128, grid=grid(3072), stream=stream0)
        buf249 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf248, buf249, 768, 4, grid=grid(768), stream=stream0)
        buf250 = reinterpret_tensor(buf245, (512, 768), (768, 1), 0); del buf245  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf244, (512, 768), (768, 1), 0), permute_332, out=buf250)
        del permute_332
        buf251 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf244, (768, 512), (1, 768), 0), view_132, out=buf251)
        buf252 = buf248; del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf244, buf252, 3072, 128, grid=grid(3072), stream=stream0)
        buf253 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf252, buf253, 768, 4, grid=grid(768), stream=stream0)
        buf254 = reinterpret_tensor(buf244, (512, 768), (768, 1), 0); del buf244  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf243, (512, 768), (768, 1), 0), permute_336, out=buf254)
        del permute_336
        buf255 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf243, (768, 512), (1, 768), 0), view_132, out=buf255)
        del view_132
        buf256 = buf252; del buf252  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf243, buf256, 3072, 128, grid=grid(3072), stream=stream0)
        buf257 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf256, buf257, 768, 4, grid=grid(768), stream=stream0)
        buf261 = reinterpret_tensor(buf243, (1, 512, 768), (393216, 768, 1), 0); del buf243  # reuse
        buf264 = buf237; del buf237  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf234, buf246, buf250, buf254, primals_104, mul_43, div_42, getitem_61, buf261, buf264, 512, 768, grid=grid(512), stream=stream0)
        del div_42
        del getitem_61
        del primals_104
        buf262 = empty((768, ), device='cuda', dtype=torch.float32)
        buf263 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf234, buf246, buf250, buf254, mul_43, buf262, buf263, 768, 512, grid=grid(768), stream=stream0)
        del buf234
        del buf246
        del mul_43
        buf265 = reinterpret_tensor(buf227, (512, 3072), (3072, 1), 0); del buf227  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf264, (512, 768), (768, 1), 0), permute_340, out=buf265)
        del permute_340
        buf266 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf264, (768, 512), (1, 768), 0), view_130, out=buf266)
        del view_130
        buf267 = buf256; del buf256  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf264, buf267, 3072, 128, grid=grid(3072), stream=stream0)
        buf268 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf267, buf268, 768, 4, grid=grid(768), stream=stream0)
        buf269 = reinterpret_tensor(buf265, (1, 512, 3072), (1572864, 3072, 1), 0); del buf265  # reuse
        # Source Nodes: [intermediate_output_5], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf269, addmm_34, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_34
        buf270 = reinterpret_tensor(buf264, (512, 768), (768, 1), 0); del buf264  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf269, (512, 3072), (3072, 1), 0), permute_344, out=buf270)
        del permute_344
        buf271 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf269, (3072, 512), (1, 3072), 0), view_128, out=buf271)
        del view_128
        buf272 = buf230; del buf230  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf269, buf272, 12288, 128, grid=grid(12288), stream=stream0)
        buf273 = reinterpret_tensor(buf267, (1, 3072), (3072, 1), 0); del buf267  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf272, buf273, 3072, 4, grid=grid(3072), stream=stream0)
        buf276 = reinterpret_tensor(buf254, (1, 512, 768), (393216, 768, 1), 0); del buf254  # reuse
        buf279 = reinterpret_tensor(buf250, (1, 512, 768), (393216, 768, 1), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_9.run(buf261, buf270, primals_98, mul_38, div_43, getitem_57, buf276, buf279, 512, 768, grid=grid(512), stream=stream0)
        del div_43
        del getitem_57
        del primals_98
        buf277 = empty((768, ), device='cuda', dtype=torch.float32)
        buf278 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf261, buf270, mul_38, buf277, buf278, 768, 512, grid=grid(768), stream=stream0)
        del buf261
        del mul_38
        buf280 = buf270; del buf270  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf279, (512, 768), (768, 1), 0), permute_348, out=buf280)
        del permute_348
        buf281 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf279, (768, 512), (1, 768), 0), view_126, out=buf281)
        del view_126
        buf282 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf279, buf282, 3072, 128, grid=grid(3072), stream=stream0)
        buf283 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf282, buf283, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf284 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf280, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_18, clone_default_19, clone_default_20, None, alias_default_13, getitem_169, getitem_170, getitem_171, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_13
        del clone_default_18
        del clone_default_19
        del clone_default_20
        del getitem_169
        del getitem_170
        del getitem_171
        buf285 = buf284[0]
        buf286 = buf284[1]
        buf287 = buf284[2]
        del buf284
        buf288 = buf280; del buf280  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf287, (512, 768), (768, 1), 0), permute_360, out=buf288)
        del permute_360
        buf289 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf287, (768, 512), (1, 768), 0), view_110, out=buf289)
        buf290 = buf282; del buf282  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf287, buf290, 3072, 128, grid=grid(3072), stream=stream0)
        buf291 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf290, buf291, 768, 4, grid=grid(768), stream=stream0)
        buf292 = reinterpret_tensor(buf287, (512, 768), (768, 1), 0); del buf287  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf286, (512, 768), (768, 1), 0), permute_365, out=buf292)
        del permute_365
        buf293 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf286, (768, 512), (1, 768), 0), view_110, out=buf293)
        buf294 = buf290; del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf286, buf294, 3072, 128, grid=grid(3072), stream=stream0)
        buf295 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf294, buf295, 768, 4, grid=grid(768), stream=stream0)
        buf296 = reinterpret_tensor(buf286, (512, 768), (768, 1), 0); del buf286  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf285, (512, 768), (768, 1), 0), permute_369, out=buf296)
        del permute_369
        buf297 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf285, (768, 512), (1, 768), 0), view_110, out=buf297)
        del view_110
        buf298 = buf294; del buf294  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf285, buf298, 3072, 128, grid=grid(3072), stream=stream0)
        buf299 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf298, buf299, 768, 4, grid=grid(768), stream=stream0)
        buf303 = reinterpret_tensor(buf285, (1, 512, 768), (393216, 768, 1), 0); del buf285  # reuse
        buf306 = buf279; del buf279  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf276, buf288, buf292, buf296, primals_88, mul_36, div_45, getitem_51, buf303, buf306, 512, 768, grid=grid(512), stream=stream0)
        del div_45
        del getitem_51
        del primals_88
        buf304 = empty((768, ), device='cuda', dtype=torch.float32)
        buf305 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf276, buf288, buf292, buf296, mul_36, buf304, buf305, 768, 512, grid=grid(768), stream=stream0)
        del buf276
        del buf288
        del mul_36
        buf307 = reinterpret_tensor(buf269, (512, 3072), (3072, 1), 0); del buf269  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf306, (512, 768), (768, 1), 0), permute_373, out=buf307)
        del permute_373
        buf308 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf306, (768, 512), (1, 768), 0), view_108, out=buf308)
        del view_108
        buf309 = buf298; del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf306, buf309, 3072, 128, grid=grid(3072), stream=stream0)
        buf310 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf309, buf310, 768, 4, grid=grid(768), stream=stream0)
        buf311 = reinterpret_tensor(buf307, (1, 512, 3072), (1572864, 3072, 1), 0); del buf307  # reuse
        # Source Nodes: [intermediate_output_4], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf311, addmm_28, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_28
        buf312 = reinterpret_tensor(buf306, (512, 768), (768, 1), 0); del buf306  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf311, (512, 3072), (3072, 1), 0), permute_377, out=buf312)
        del permute_377
        buf313 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf311, (3072, 512), (1, 3072), 0), view_106, out=buf313)
        del view_106
        buf314 = buf272; del buf272  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf311, buf314, 12288, 128, grid=grid(12288), stream=stream0)
        buf315 = reinterpret_tensor(buf309, (1, 3072), (3072, 1), 0); del buf309  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf314, buf315, 3072, 4, grid=grid(3072), stream=stream0)
        buf318 = reinterpret_tensor(buf296, (1, 512, 768), (393216, 768, 1), 0); del buf296  # reuse
        buf321 = reinterpret_tensor(buf292, (1, 512, 768), (393216, 768, 1), 0); del buf292  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_9.run(buf303, buf312, primals_82, mul_31, div_46, getitem_47, buf318, buf321, 512, 768, grid=grid(512), stream=stream0)
        del div_46
        del getitem_47
        del primals_82
        buf319 = empty((768, ), device='cuda', dtype=torch.float32)
        buf320 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf303, buf312, mul_31, buf319, buf320, 768, 512, grid=grid(768), stream=stream0)
        del buf303
        del mul_31
        buf322 = buf312; del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf321, (512, 768), (768, 1), 0), permute_381, out=buf322)
        del permute_381
        buf323 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf321, (768, 512), (1, 768), 0), view_104, out=buf323)
        del view_104
        buf324 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf321, buf324, 3072, 128, grid=grid(3072), stream=stream0)
        buf325 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf324, buf325, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf326 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf322, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_21, clone_default_22, clone_default_23, None, alias_default_15, getitem_176, getitem_177, getitem_178, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_15
        del clone_default_21
        del clone_default_22
        del clone_default_23
        del getitem_176
        del getitem_177
        del getitem_178
        buf327 = buf326[0]
        buf328 = buf326[1]
        buf329 = buf326[2]
        del buf326
        buf330 = buf322; del buf322  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf329, (512, 768), (768, 1), 0), permute_393, out=buf330)
        del permute_393
        buf331 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf329, (768, 512), (1, 768), 0), view_88, out=buf331)
        buf332 = buf324; del buf324  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf329, buf332, 3072, 128, grid=grid(3072), stream=stream0)
        buf333 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf332, buf333, 768, 4, grid=grid(768), stream=stream0)
        buf334 = reinterpret_tensor(buf329, (512, 768), (768, 1), 0); del buf329  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf328, (512, 768), (768, 1), 0), permute_398, out=buf334)
        del permute_398
        buf335 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf328, (768, 512), (1, 768), 0), view_88, out=buf335)
        buf336 = buf332; del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf328, buf336, 3072, 128, grid=grid(3072), stream=stream0)
        buf337 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf336, buf337, 768, 4, grid=grid(768), stream=stream0)
        buf338 = reinterpret_tensor(buf328, (512, 768), (768, 1), 0); del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf327, (512, 768), (768, 1), 0), permute_402, out=buf338)
        del permute_402
        buf339 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf327, (768, 512), (1, 768), 0), view_88, out=buf339)
        del view_88
        buf340 = buf336; del buf336  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf327, buf340, 3072, 128, grid=grid(3072), stream=stream0)
        buf341 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf340, buf341, 768, 4, grid=grid(768), stream=stream0)
        buf345 = reinterpret_tensor(buf327, (1, 512, 768), (393216, 768, 1), 0); del buf327  # reuse
        buf348 = buf321; del buf321  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf318, buf330, buf334, buf338, primals_72, mul_29, div_48, getitem_41, buf345, buf348, 512, 768, grid=grid(512), stream=stream0)
        del div_48
        del getitem_41
        del primals_72
        buf346 = empty((768, ), device='cuda', dtype=torch.float32)
        buf347 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf318, buf330, buf334, buf338, mul_29, buf346, buf347, 768, 512, grid=grid(768), stream=stream0)
        del buf318
        del buf330
        del mul_29
        buf349 = reinterpret_tensor(buf311, (512, 3072), (3072, 1), 0); del buf311  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf348, (512, 768), (768, 1), 0), permute_406, out=buf349)
        del permute_406
        buf350 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf348, (768, 512), (1, 768), 0), view_86, out=buf350)
        del view_86
        buf351 = buf340; del buf340  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf348, buf351, 3072, 128, grid=grid(3072), stream=stream0)
        buf352 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf351, buf352, 768, 4, grid=grid(768), stream=stream0)
        buf353 = reinterpret_tensor(buf349, (1, 512, 3072), (1572864, 3072, 1), 0); del buf349  # reuse
        # Source Nodes: [intermediate_output_3], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf353, addmm_22, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_22
        buf354 = reinterpret_tensor(buf348, (512, 768), (768, 1), 0); del buf348  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf353, (512, 3072), (3072, 1), 0), permute_410, out=buf354)
        del permute_410
        buf355 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf353, (3072, 512), (1, 3072), 0), view_84, out=buf355)
        del view_84
        buf356 = buf314; del buf314  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf353, buf356, 12288, 128, grid=grid(12288), stream=stream0)
        buf357 = reinterpret_tensor(buf351, (1, 3072), (3072, 1), 0); del buf351  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf356, buf357, 3072, 4, grid=grid(3072), stream=stream0)
        buf360 = reinterpret_tensor(buf338, (1, 512, 768), (393216, 768, 1), 0); del buf338  # reuse
        buf363 = reinterpret_tensor(buf334, (1, 512, 768), (393216, 768, 1), 0); del buf334  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_9.run(buf345, buf354, primals_66, mul_24, div_49, getitem_37, buf360, buf363, 512, 768, grid=grid(512), stream=stream0)
        del div_49
        del getitem_37
        del primals_66
        buf361 = empty((768, ), device='cuda', dtype=torch.float32)
        buf362 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf345, buf354, mul_24, buf361, buf362, 768, 512, grid=grid(768), stream=stream0)
        del buf345
        del mul_24
        buf364 = buf354; del buf354  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf363, (512, 768), (768, 1), 0), permute_414, out=buf364)
        del permute_414
        buf365 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf363, (768, 512), (1, 768), 0), view_82, out=buf365)
        del view_82
        buf366 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf363, buf366, 3072, 128, grid=grid(3072), stream=stream0)
        buf367 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf366, buf367, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf368 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf364, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_24, clone_default_25, clone_default_26, None, alias_default_17, getitem_183, getitem_184, getitem_185, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_17
        del clone_default_24
        del clone_default_25
        del clone_default_26
        del getitem_183
        del getitem_184
        del getitem_185
        buf369 = buf368[0]
        buf370 = buf368[1]
        buf371 = buf368[2]
        del buf368
        buf372 = buf364; del buf364  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf371, (512, 768), (768, 1), 0), permute_426, out=buf372)
        del permute_426
        buf373 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf371, (768, 512), (1, 768), 0), view_66, out=buf373)
        buf374 = buf366; del buf366  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf371, buf374, 3072, 128, grid=grid(3072), stream=stream0)
        buf375 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf374, buf375, 768, 4, grid=grid(768), stream=stream0)
        buf376 = reinterpret_tensor(buf371, (512, 768), (768, 1), 0); del buf371  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf370, (512, 768), (768, 1), 0), permute_431, out=buf376)
        del permute_431
        buf377 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf370, (768, 512), (1, 768), 0), view_66, out=buf377)
        buf378 = buf374; del buf374  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf370, buf378, 3072, 128, grid=grid(3072), stream=stream0)
        buf379 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf378, buf379, 768, 4, grid=grid(768), stream=stream0)
        buf380 = reinterpret_tensor(buf370, (512, 768), (768, 1), 0); del buf370  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf369, (512, 768), (768, 1), 0), permute_435, out=buf380)
        del permute_435
        buf381 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf369, (768, 512), (1, 768), 0), view_66, out=buf381)
        del view_66
        buf382 = buf378; del buf378  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf369, buf382, 3072, 128, grid=grid(3072), stream=stream0)
        buf383 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf382, buf383, 768, 4, grid=grid(768), stream=stream0)
        buf387 = reinterpret_tensor(buf369, (1, 512, 768), (393216, 768, 1), 0); del buf369  # reuse
        buf390 = buf363; del buf363  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf360, buf372, buf376, buf380, primals_56, mul_22, div_51, getitem_31, buf387, buf390, 512, 768, grid=grid(512), stream=stream0)
        del div_51
        del getitem_31
        del primals_56
        buf388 = empty((768, ), device='cuda', dtype=torch.float32)
        buf389 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf360, buf372, buf376, buf380, mul_22, buf388, buf389, 768, 512, grid=grid(768), stream=stream0)
        del buf360
        del buf372
        del mul_22
        buf391 = reinterpret_tensor(buf353, (512, 3072), (3072, 1), 0); del buf353  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf390, (512, 768), (768, 1), 0), permute_439, out=buf391)
        del permute_439
        buf392 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf390, (768, 512), (1, 768), 0), view_64, out=buf392)
        del view_64
        buf393 = buf382; del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf390, buf393, 3072, 128, grid=grid(3072), stream=stream0)
        buf394 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf393, buf394, 768, 4, grid=grid(768), stream=stream0)
        buf395 = reinterpret_tensor(buf391, (1, 512, 3072), (1572864, 3072, 1), 0); del buf391  # reuse
        # Source Nodes: [intermediate_output_2], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf395, addmm_16, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_16
        buf396 = reinterpret_tensor(buf390, (512, 768), (768, 1), 0); del buf390  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf395, (512, 3072), (3072, 1), 0), permute_443, out=buf396)
        del permute_443
        buf397 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf395, (3072, 512), (1, 3072), 0), view_62, out=buf397)
        del view_62
        buf398 = buf356; del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf395, buf398, 12288, 128, grid=grid(12288), stream=stream0)
        buf399 = reinterpret_tensor(buf393, (1, 3072), (3072, 1), 0); del buf393  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf398, buf399, 3072, 4, grid=grid(3072), stream=stream0)
        buf402 = reinterpret_tensor(buf380, (1, 512, 768), (393216, 768, 1), 0); del buf380  # reuse
        buf405 = reinterpret_tensor(buf376, (1, 512, 768), (393216, 768, 1), 0); del buf376  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_9.run(buf387, buf396, primals_50, mul_17, div_52, getitem_27, buf402, buf405, 512, 768, grid=grid(512), stream=stream0)
        del div_52
        del getitem_27
        del primals_50
        buf403 = empty((768, ), device='cuda', dtype=torch.float32)
        buf404 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf387, buf396, mul_17, buf403, buf404, 768, 512, grid=grid(768), stream=stream0)
        del buf387
        del mul_17
        buf406 = buf396; del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf405, (512, 768), (768, 1), 0), permute_447, out=buf406)
        del permute_447
        buf407 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf405, (768, 512), (1, 768), 0), view_60, out=buf407)
        del view_60
        buf408 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf405, buf408, 3072, 128, grid=grid(3072), stream=stream0)
        buf409 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf408, buf409, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf410 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf406, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_27, clone_default_28, clone_default_29, None, alias_default_19, getitem_190, getitem_191, getitem_192, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_19
        del clone_default_27
        del clone_default_28
        del clone_default_29
        del getitem_190
        del getitem_191
        del getitem_192
        buf411 = buf410[0]
        buf412 = buf410[1]
        buf413 = buf410[2]
        del buf410
        buf414 = buf406; del buf406  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf413, (512, 768), (768, 1), 0), permute_459, out=buf414)
        del permute_459
        buf415 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf413, (768, 512), (1, 768), 0), view_44, out=buf415)
        buf416 = buf408; del buf408  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf413, buf416, 3072, 128, grid=grid(3072), stream=stream0)
        buf417 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf416, buf417, 768, 4, grid=grid(768), stream=stream0)
        buf418 = reinterpret_tensor(buf413, (512, 768), (768, 1), 0); del buf413  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf412, (512, 768), (768, 1), 0), permute_464, out=buf418)
        del permute_464
        buf419 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf412, (768, 512), (1, 768), 0), view_44, out=buf419)
        buf420 = buf416; del buf416  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf412, buf420, 3072, 128, grid=grid(3072), stream=stream0)
        buf421 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf420, buf421, 768, 4, grid=grid(768), stream=stream0)
        buf422 = reinterpret_tensor(buf412, (512, 768), (768, 1), 0); del buf412  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf411, (512, 768), (768, 1), 0), permute_468, out=buf422)
        del permute_468
        buf423 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf411, (768, 512), (1, 768), 0), view_44, out=buf423)
        del view_44
        buf424 = buf420; del buf420  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf411, buf424, 3072, 128, grid=grid(3072), stream=stream0)
        buf425 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf424, buf425, 768, 4, grid=grid(768), stream=stream0)
        buf429 = reinterpret_tensor(buf411, (1, 512, 768), (393216, 768, 1), 0); del buf411  # reuse
        buf432 = buf405; del buf405  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf402, buf414, buf418, buf422, primals_40, mul_15, div_54, getitem_21, buf429, buf432, 512, 768, grid=grid(512), stream=stream0)
        del div_54
        del getitem_21
        del primals_40
        buf430 = empty((768, ), device='cuda', dtype=torch.float32)
        buf431 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf402, buf414, buf418, buf422, mul_15, buf430, buf431, 768, 512, grid=grid(768), stream=stream0)
        del buf402
        del buf414
        del mul_15
        buf433 = reinterpret_tensor(buf395, (512, 3072), (3072, 1), 0); del buf395  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf432, (512, 768), (768, 1), 0), permute_472, out=buf433)
        del permute_472
        buf434 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf432, (768, 512), (1, 768), 0), view_42, out=buf434)
        del view_42
        buf435 = buf424; del buf424  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf432, buf435, 3072, 128, grid=grid(3072), stream=stream0)
        buf436 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf435, buf436, 768, 4, grid=grid(768), stream=stream0)
        buf437 = reinterpret_tensor(buf433, (1, 512, 3072), (1572864, 3072, 1), 0); del buf433  # reuse
        # Source Nodes: [intermediate_output_1], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf437, addmm_10, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_10
        buf438 = reinterpret_tensor(buf432, (512, 768), (768, 1), 0); del buf432  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf437, (512, 3072), (3072, 1), 0), permute_476, out=buf438)
        del permute_476
        buf439 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf437, (3072, 512), (1, 3072), 0), view_40, out=buf439)
        del view_40
        buf440 = buf398; del buf398  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf437, buf440, 12288, 128, grid=grid(12288), stream=stream0)
        buf441 = reinterpret_tensor(buf435, (1, 3072), (3072, 1), 0); del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf440, buf441, 3072, 4, grid=grid(3072), stream=stream0)
        buf444 = reinterpret_tensor(buf422, (1, 512, 768), (393216, 768, 1), 0); del buf422  # reuse
        buf447 = reinterpret_tensor(buf418, (1, 512, 768), (393216, 768, 1), 0); del buf418  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_9.run(buf429, buf438, primals_34, mul_10, div_55, getitem_17, buf444, buf447, 512, 768, grid=grid(512), stream=stream0)
        del div_55
        del getitem_17
        del primals_34
        buf445 = empty((768, ), device='cuda', dtype=torch.float32)
        buf446 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf429, buf438, mul_10, buf445, buf446, 768, 512, grid=grid(768), stream=stream0)
        del buf429
        del mul_10
        buf448 = buf438; del buf438  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf447, (512, 768), (768, 1), 0), permute_480, out=buf448)
        del permute_480
        buf449 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf447, (768, 512), (1, 768), 0), view_38, out=buf449)
        del view_38
        buf450 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf447, buf450, 3072, 128, grid=grid(3072), stream=stream0)
        buf451 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf450, buf451, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf452 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf448, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_30, clone_default_31, clone_default_32, None, alias_default_21, getitem_197, getitem_198, getitem_199, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_21
        del clone_default_30
        del clone_default_31
        del clone_default_32
        del getitem_197
        del getitem_198
        del getitem_199
        buf453 = buf452[0]
        buf454 = buf452[1]
        buf455 = buf452[2]
        del buf452
        buf456 = buf448; del buf448  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf455, (512, 768), (768, 1), 0), permute_492, out=buf456)
        del permute_492
        buf457 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf455, (768, 512), (1, 768), 0), view_22, out=buf457)
        buf458 = buf450; del buf450  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf455, buf458, 3072, 128, grid=grid(3072), stream=stream0)
        buf459 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf458, buf459, 768, 4, grid=grid(768), stream=stream0)
        buf460 = reinterpret_tensor(buf455, (512, 768), (768, 1), 0); del buf455  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf454, (512, 768), (768, 1), 0), permute_497, out=buf460)
        del permute_497
        buf461 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf454, (768, 512), (1, 768), 0), view_22, out=buf461)
        buf462 = buf458; del buf458  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf454, buf462, 3072, 128, grid=grid(3072), stream=stream0)
        buf463 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf462, buf463, 768, 4, grid=grid(768), stream=stream0)
        buf464 = reinterpret_tensor(buf454, (512, 768), (768, 1), 0); del buf454  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf453, (512, 768), (768, 1), 0), permute_501, out=buf464)
        del permute_501
        buf465 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf453, (768, 512), (1, 768), 0), view_22, out=buf465)
        del view_22
        buf466 = buf462; del buf462  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf453, buf466, 3072, 128, grid=grid(3072), stream=stream0)
        buf467 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf466, buf467, 768, 4, grid=grid(768), stream=stream0)
        buf471 = reinterpret_tensor(buf453, (1, 512, 768), (393216, 768, 1), 0); del buf453  # reuse
        buf474 = buf447; del buf447  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_11.run(buf444, buf456, buf460, buf464, primals_24, mul_8, div_57, getitem_11, buf471, buf474, 512, 768, grid=grid(512), stream=stream0)
        del div_57
        del getitem_11
        del primals_24
        buf472 = empty((768, ), device='cuda', dtype=torch.float32)
        buf473 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_12.run(buf444, buf456, buf460, buf464, mul_8, buf472, buf473, 768, 512, grid=grid(768), stream=stream0)
        del buf444
        del buf456
        del mul_8
        buf475 = reinterpret_tensor(buf437, (512, 3072), (3072, 1), 0); del buf437  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf474, (512, 768), (768, 1), 0), permute_505, out=buf475)
        del permute_505
        buf476 = empty((768, 3072), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf474, (768, 512), (1, 768), 0), view_20, out=buf476)
        del view_20
        buf477 = buf466; del buf466  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf474, buf477, 3072, 128, grid=grid(3072), stream=stream0)
        buf478 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf477, buf478, 768, 4, grid=grid(768), stream=stream0)
        buf479 = reinterpret_tensor(buf475, (1, 512, 3072), (1572864, 3072, 1), 0); del buf475  # reuse
        # Source Nodes: [intermediate_output], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_6.run(buf479, addmm_4, 1572864, grid=grid(1572864), stream=stream0)
        del addmm_4
        buf480 = reinterpret_tensor(buf474, (512, 768), (768, 1), 0); del buf474  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf479, (512, 3072), (3072, 1), 0), permute_509, out=buf480)
        del permute_509
        buf481 = empty((3072, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf479, (3072, 512), (1, 3072), 0), view_18, out=buf481)
        del view_18
        buf482 = buf440; del buf440  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_7.run(buf479, buf482, 12288, 128, grid=grid(12288), stream=stream0)
        del buf479
        buf483 = reinterpret_tensor(buf477, (1, 3072), (3072, 1), 0); del buf477  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf482, buf483, 3072, 4, grid=grid(3072), stream=stream0)
        del buf482
        buf486 = reinterpret_tensor(buf464, (1, 512, 768), (393216, 768, 1), 0); del buf464  # reuse
        buf489 = reinterpret_tensor(buf460, (1, 512, 768), (393216, 768, 1), 0); del buf460  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_native_dropout_backward_native_layer_norm_backward_9.run(buf471, buf480, primals_18, mul_3, div_58, getitem_7, buf486, buf489, 512, 768, grid=grid(512), stream=stream0)
        del div_58
        del getitem_7
        del primals_18
        buf487 = empty((768, ), device='cuda', dtype=torch.float32)
        buf488 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_10.run(buf471, buf480, mul_3, buf487, buf488, 768, 512, grid=grid(768), stream=stream0)
        del mul_3
        buf490 = buf480; del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf489, (512, 768), (768, 1), 0), permute_513, out=buf490)
        del permute_513
        buf491 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf489, (768, 512), (1, 768), 0), view_16, out=buf491)
        del view_16
        buf492 = empty_strided((1, 768, 4), (3072, 1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf489, buf492, 3072, 128, grid=grid(3072), stream=stream0)
        buf493 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf492, buf493, 768, 4, grid=grid(768), stream=stream0)
        # Source Nodes: [], Original ATen: []
        buf494 = aten._scaled_dot_product_efficient_attention_backward(reinterpret_tensor(buf490, (1, 12, 512, 64), (393216, 64, 768, 1), 0), clone_default_33, clone_default_34, clone_default_35, None, alias_default_23, getitem_204, getitem_205, getitem_206, 0.1, [True, True, True, False], scale=0.125)
        del alias_default_23
        del clone_default_33
        del clone_default_34
        del clone_default_35
        del getitem_204
        del getitem_205
        del getitem_206
        buf495 = buf494[0]
        buf496 = buf494[1]
        buf497 = buf494[2]
        del buf494
        buf498 = buf490; del buf490  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf497, (512, 768), (768, 1), 0), permute_525, out=buf498)
        del permute_525
        buf499 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf497, (768, 512), (1, 768), 0), view, out=buf499)
        buf500 = buf492; del buf492  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf497, buf500, 3072, 128, grid=grid(3072), stream=stream0)
        buf501 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf500, buf501, 768, 4, grid=grid(768), stream=stream0)
        buf502 = reinterpret_tensor(buf497, (512, 768), (768, 1), 0); del buf497  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf496, (512, 768), (768, 1), 0), permute_530, out=buf502)
        del permute_530
        buf503 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf496, (768, 512), (1, 768), 0), view, out=buf503)
        buf504 = buf500; del buf500  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf496, buf504, 3072, 128, grid=grid(3072), stream=stream0)
        buf505 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf504, buf505, 768, 4, grid=grid(768), stream=stream0)
        buf506 = reinterpret_tensor(buf496, (512, 768), (768, 1), 0); del buf496  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf495, (512, 768), (768, 1), 0), permute_534, out=buf506)
        del permute_534
        buf507 = empty((768, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf495, (768, 512), (1, 768), 0), view, out=buf507)
        del view
        buf508 = buf504; del buf504  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_4.run(buf495, buf508, 3072, 128, grid=grid(3072), stream=stream0)
        buf509 = empty((1, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_5.run(buf508, buf509, 768, 4, grid=grid(768), stream=stream0)
        del buf508
        buf510 = buf486; del buf486  # reuse
        buf517 = reinterpret_tensor(buf495, (1, 512, 768), (393216, 768, 1), 0); del buf495  # reuse
        buf538 = buf489; del buf489  # reuse
        buf542 = buf471; del buf471  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.embedding_dense_backward, aten.native_dropout_backward, aten.native_layer_norm_backward]
        triton_per_fused_add_embedding_dense_backward_native_dropout_backward_native_layer_norm_backward_13.run(buf510, buf498, buf502, buf506, getitem_3, primals_8, mul_1, div_60, slice_1, primals_207, buf517, buf538, buf542, 512, 768, grid=grid(512), stream=stream0)
        del buf498
        del buf502
        del buf506
        del div_60
        del getitem_3
        del primals_8
        buf514 = empty((768, ), device='cuda', dtype=torch.float32)
        buf515 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_14.run(buf510, mul_1, buf514, buf515, 768, 512, grid=grid(768), stream=stream0)
        del buf510
        del mul_1
        buf516 = empty((2, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_15.run(buf516, 1536, grid=grid(1536), stream=stream0)
        aten.index_put_(buf516, [full_default], buf517, True)
        buf520 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_16.run(buf520, 786432, grid=grid(786432), stream=stream0)
        aten.index_put_(buf520, [full_default], buf517, True)
        del full_default
        buf523 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_16.run(buf523, 786432, grid=grid(786432), stream=stream0)
        aten.index_put_(buf523, [select_3], buf517, True)
        del select_3
        buf526 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_16.run(buf526, 786432, grid=grid(786432), stream=stream0)
        aten.index_put_(buf526, [select_2], buf517, True)
        del select_2
        buf529 = empty((1024, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_16.run(buf529, 786432, grid=grid(786432), stream=stream0)
        aten.index_put_(buf529, [select_1], buf517, True)
        del select_1
        buf525 = empty((1024, 768), device='cuda', dtype=torch.float32)
        buf532 = buf525; del buf525  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_17.run(buf532, buf523, buf529, 786432, grid=grid(786432), stream=stream0)
        buf533 = buf529; del buf529  # reuse
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_16.run(buf533, 786432, grid=grid(786432), stream=stream0)
        aten.index_put_(buf533, [select], buf517, True)
        del select
        buf528 = buf523; del buf523  # reuse
        buf536 = buf528; del buf528  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_17.run(buf536, buf526, buf533, 786432, grid=grid(786432), stream=stream0)
        del buf526
        del buf533
        buf537 = reinterpret_tensor(buf517, (512, 768), (768, 1), 0); del buf517  # reuse
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_18.run(buf537, 393216, grid=grid(393216), stream=stream0)
        aten.index_put_(buf537, [slice_1], buf538, True)
        del buf538
        del slice_1
        buf541 = empty((30522, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_19.run(buf541, 23440896, grid=grid(23440896), stream=stream0)
        aten.index_put_(buf541, [primals_207], buf542, True)
        del buf542
        del primals_207
        return (buf541, buf537, buf536, buf532, buf520, buf520, buf516, buf514, buf515, reinterpret_tensor(buf507, (768, 768), (768, 1), 0), reinterpret_tensor(buf509, (768, ), (1, ), 0), reinterpret_tensor(buf503, (768, 768), (768, 1), 0), reinterpret_tensor(buf505, (768, ), (1, ), 0), reinterpret_tensor(buf499, (768, 768), (768, 1), 0), reinterpret_tensor(buf501, (768, ), (1, ), 0), reinterpret_tensor(buf491, (768, 768), (768, 1), 0), reinterpret_tensor(buf493, (768, ), (1, ), 0), buf487, buf488, reinterpret_tensor(buf481, (3072, 768), (768, 1), 0), reinterpret_tensor(buf483, (3072, ), (1, ), 0), reinterpret_tensor(buf476, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf478, (768, ), (1, ), 0), buf472, buf473, reinterpret_tensor(buf465, (768, 768), (768, 1), 0), reinterpret_tensor(buf467, (768, ), (1, ), 0), reinterpret_tensor(buf461, (768, 768), (768, 1), 0), reinterpret_tensor(buf463, (768, ), (1, ), 0), reinterpret_tensor(buf457, (768, 768), (768, 1), 0), reinterpret_tensor(buf459, (768, ), (1, ), 0), reinterpret_tensor(buf449, (768, 768), (768, 1), 0), reinterpret_tensor(buf451, (768, ), (1, ), 0), buf445, buf446, reinterpret_tensor(buf439, (3072, 768), (768, 1), 0), reinterpret_tensor(buf441, (3072, ), (1, ), 0), reinterpret_tensor(buf434, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf436, (768, ), (1, ), 0), buf430, buf431, reinterpret_tensor(buf423, (768, 768), (768, 1), 0), reinterpret_tensor(buf425, (768, ), (1, ), 0), reinterpret_tensor(buf419, (768, 768), (768, 1), 0), reinterpret_tensor(buf421, (768, ), (1, ), 0), reinterpret_tensor(buf415, (768, 768), (768, 1), 0), reinterpret_tensor(buf417, (768, ), (1, ), 0), reinterpret_tensor(buf407, (768, 768), (768, 1), 0), reinterpret_tensor(buf409, (768, ), (1, ), 0), buf403, buf404, reinterpret_tensor(buf397, (3072, 768), (768, 1), 0), reinterpret_tensor(buf399, (3072, ), (1, ), 0), reinterpret_tensor(buf392, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf394, (768, ), (1, ), 0), buf388, buf389, reinterpret_tensor(buf381, (768, 768), (768, 1), 0), reinterpret_tensor(buf383, (768, ), (1, ), 0), reinterpret_tensor(buf377, (768, 768), (768, 1), 0), reinterpret_tensor(buf379, (768, ), (1, ), 0), reinterpret_tensor(buf373, (768, 768), (768, 1), 0), reinterpret_tensor(buf375, (768, ), (1, ), 0), reinterpret_tensor(buf365, (768, 768), (768, 1), 0), reinterpret_tensor(buf367, (768, ), (1, ), 0), buf361, buf362, reinterpret_tensor(buf355, (3072, 768), (768, 1), 0), reinterpret_tensor(buf357, (3072, ), (1, ), 0), reinterpret_tensor(buf350, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf352, (768, ), (1, ), 0), buf346, buf347, reinterpret_tensor(buf339, (768, 768), (768, 1), 0), reinterpret_tensor(buf341, (768, ), (1, ), 0), reinterpret_tensor(buf335, (768, 768), (768, 1), 0), reinterpret_tensor(buf337, (768, ), (1, ), 0), reinterpret_tensor(buf331, (768, 768), (768, 1), 0), reinterpret_tensor(buf333, (768, ), (1, ), 0), reinterpret_tensor(buf323, (768, 768), (768, 1), 0), reinterpret_tensor(buf325, (768, ), (1, ), 0), buf319, buf320, reinterpret_tensor(buf313, (3072, 768), (768, 1), 0), reinterpret_tensor(buf315, (3072, ), (1, ), 0), reinterpret_tensor(buf308, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf310, (768, ), (1, ), 0), buf304, buf305, reinterpret_tensor(buf297, (768, 768), (768, 1), 0), reinterpret_tensor(buf299, (768, ), (1, ), 0), reinterpret_tensor(buf293, (768, 768), (768, 1), 0), reinterpret_tensor(buf295, (768, ), (1, ), 0), reinterpret_tensor(buf289, (768, 768), (768, 1), 0), reinterpret_tensor(buf291, (768, ), (1, ), 0), reinterpret_tensor(buf281, (768, 768), (768, 1), 0), reinterpret_tensor(buf283, (768, ), (1, ), 0), buf277, buf278, reinterpret_tensor(buf271, (3072, 768), (768, 1), 0), reinterpret_tensor(buf273, (3072, ), (1, ), 0), reinterpret_tensor(buf266, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf268, (768, ), (1, ), 0), buf262, buf263, reinterpret_tensor(buf255, (768, 768), (768, 1), 0), reinterpret_tensor(buf257, (768, ), (1, ), 0), reinterpret_tensor(buf251, (768, 768), (768, 1), 0), reinterpret_tensor(buf253, (768, ), (1, ), 0), reinterpret_tensor(buf247, (768, 768), (768, 1), 0), reinterpret_tensor(buf249, (768, ), (1, ), 0), reinterpret_tensor(buf239, (768, 768), (768, 1), 0), reinterpret_tensor(buf241, (768, ), (1, ), 0), buf235, buf236, reinterpret_tensor(buf229, (3072, 768), (768, 1), 0), reinterpret_tensor(buf231, (3072, ), (1, ), 0), reinterpret_tensor(buf224, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf226, (768, ), (1, ), 0), buf220, buf221, reinterpret_tensor(buf213, (768, 768), (768, 1), 0), reinterpret_tensor(buf215, (768, ), (1, ), 0), reinterpret_tensor(buf209, (768, 768), (768, 1), 0), reinterpret_tensor(buf211, (768, ), (1, ), 0), reinterpret_tensor(buf205, (768, 768), (768, 1), 0), reinterpret_tensor(buf207, (768, ), (1, ), 0), reinterpret_tensor(buf197, (768, 768), (768, 1), 0), reinterpret_tensor(buf199, (768, ), (1, ), 0), buf193, buf194, reinterpret_tensor(buf187, (3072, 768), (768, 1), 0), reinterpret_tensor(buf189, (3072, ), (1, ), 0), reinterpret_tensor(buf182, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf184, (768, ), (1, ), 0), buf178, buf179, reinterpret_tensor(buf171, (768, 768), (768, 1), 0), reinterpret_tensor(buf173, (768, ), (1, ), 0), reinterpret_tensor(buf167, (768, 768), (768, 1), 0), reinterpret_tensor(buf169, (768, ), (1, ), 0), reinterpret_tensor(buf163, (768, 768), (768, 1), 0), reinterpret_tensor(buf165, (768, ), (1, ), 0), reinterpret_tensor(buf155, (768, 768), (768, 1), 0), reinterpret_tensor(buf157, (768, ), (1, ), 0), buf151, buf152, reinterpret_tensor(buf145, (3072, 768), (768, 1), 0), reinterpret_tensor(buf147, (3072, ), (1, ), 0), reinterpret_tensor(buf140, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf142, (768, ), (1, ), 0), buf136, buf137, reinterpret_tensor(buf129, (768, 768), (768, 1), 0), reinterpret_tensor(buf131, (768, ), (1, ), 0), reinterpret_tensor(buf125, (768, 768), (768, 1), 0), reinterpret_tensor(buf127, (768, ), (1, ), 0), reinterpret_tensor(buf121, (768, 768), (768, 1), 0), reinterpret_tensor(buf123, (768, ), (1, ), 0), reinterpret_tensor(buf113, (768, 768), (768, 1), 0), reinterpret_tensor(buf115, (768, ), (1, ), 0), buf109, buf110, reinterpret_tensor(buf103, (3072, 768), (768, 1), 0), reinterpret_tensor(buf105, (3072, ), (1, ), 0), reinterpret_tensor(buf98, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf100, (768, ), (1, ), 0), buf94, buf95, reinterpret_tensor(buf87, (768, 768), (768, 1), 0), reinterpret_tensor(buf89, (768, ), (1, ), 0), reinterpret_tensor(buf83, (768, 768), (768, 1), 0), reinterpret_tensor(buf85, (768, ), (1, ), 0), reinterpret_tensor(buf79, (768, 768), (768, 1), 0), reinterpret_tensor(buf81, (768, ), (1, ), 0), reinterpret_tensor(buf71, (768, 768), (768, 1), 0), reinterpret_tensor(buf73, (768, ), (1, ), 0), buf67, buf68, reinterpret_tensor(buf61, (3072, 768), (768, 1), 0), reinterpret_tensor(buf63, (3072, ), (1, ), 0), reinterpret_tensor(buf56, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf58, (768, ), (1, ), 0), buf52, buf53, reinterpret_tensor(buf45, (768, 768), (768, 1), 0), reinterpret_tensor(buf47, (768, ), (1, ), 0), reinterpret_tensor(buf41, (768, 768), (768, 1), 0), reinterpret_tensor(buf43, (768, ), (1, ), 0), reinterpret_tensor(buf37, (768, 768), (768, 1), 0), reinterpret_tensor(buf39, (768, ), (1, ), 0), reinterpret_tensor(buf29, (768, 768), (768, 1), 0), reinterpret_tensor(buf31, (768, ), (1, ), 0), buf25, buf26, reinterpret_tensor(buf19, (3072, 768), (768, 1), 0), reinterpret_tensor(buf21, (3072, ), (1, ), 0), reinterpret_tensor(buf14, (768, 3072), (3072, 1), 0), reinterpret_tensor(buf16, (768, ), (1, ), 0), buf10, buf11, reinterpret_tensor(buf5, (768, 768), (768, 1), 0), buf6, reinterpret_tensor(buf1, (2, 768), (768, 1), 0), buf2, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_8 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    full_default = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    slice_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    select = rand_strided((1, 512), (0, 4), device='cuda:0', dtype=torch.int64)
    select_1 = rand_strided((1, 512), (0, 4), device='cuda:0', dtype=torch.int64)
    select_2 = rand_strided((1, 512), (0, 4), device='cuda:0', dtype=torch.int64)
    select_3 = rand_strided((1, 512), (0, 4), device='cuda:0', dtype=torch.int64)
    mul_1 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    view = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    clone_default_33 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_34 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_35 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_204 = rand_strided((1, 12, 512), (6144, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_205 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_206 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_23 = rand_strided((1, 12, 512, 64), (393216, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    view_16 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_7 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_3 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_18 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_4 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_20 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_11 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_8 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_22 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    clone_default_30 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_31 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_32 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_197 = rand_strided((1, 12, 512), (6144, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_198 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_199 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_21 = rand_strided((1, 12, 512, 64), (393216, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    view_38 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_17 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_10 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_40 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_42 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_21 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_15 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_44 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    clone_default_27 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_28 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_29 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_190 = rand_strided((1, 12, 512), (6144, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_191 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_192 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_19 = rand_strided((1, 12, 512, 64), (393216, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    view_60 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_27 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_17 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_62 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_16 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_64 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_31 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_22 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_66 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    clone_default_24 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_25 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_26 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_183 = rand_strided((1, 12, 512), (6144, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_184 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_185 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_17 = rand_strided((1, 12, 512, 64), (393216, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    view_82 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_37 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_24 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_84 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_86 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_41 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_29 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_88 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    clone_default_21 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_22 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_23 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_176 = rand_strided((1, 12, 512), (6144, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_177 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_178 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_15 = rand_strided((1, 12, 512, 64), (393216, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    view_104 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_47 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_31 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_106 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_28 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_108 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_51 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_36 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_110 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    clone_default_18 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_19 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_20 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_169 = rand_strided((1, 12, 512), (6144, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_170 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_171 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_13 = rand_strided((1, 12, 512, 64), (393216, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    view_126 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_57 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_38 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_128 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_130 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_61 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_43 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_132 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    clone_default_15 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_16 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_17 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_162 = rand_strided((1, 12, 512), (6144, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_163 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_164 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_11 = rand_strided((1, 12, 512, 64), (393216, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    view_148 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_67 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_45 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_150 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_40 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_152 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_71 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_50 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_154 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    clone_default_12 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_13 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_14 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_155 = rand_strided((1, 12, 512), (6144, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_156 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_157 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_9 = rand_strided((1, 12, 512, 64), (393216, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    view_170 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_77 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_52 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_172 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_46 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_174 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_81 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_57 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_176 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    clone_default_9 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_10 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_11 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_148 = rand_strided((1, 12, 512), (6144, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_149 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_150 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_7 = rand_strided((1, 12, 512, 64), (393216, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    view_192 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_87 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_59 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_194 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_52 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_196 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_91 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_64 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_198 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    clone_default_6 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_7 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_8 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_141 = rand_strided((1, 12, 512), (6144, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_142 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_143 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_5 = rand_strided((1, 12, 512, 64), (393216, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    view_214 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_97 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_66 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_216 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_58 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_218 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_101 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_71 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_220 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    clone_default_3 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_4 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_5 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_134 = rand_strided((1, 12, 512), (6144, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_135 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_136 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_3 = rand_strided((1, 12, 512, 64), (393216, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    view_236 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_107 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_73 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_238 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_64 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_240 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_111 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_78 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_242 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    clone_default = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_1 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    clone_default_2 = rand_strided((1, 12, 512, 64), (393216, 32768, 64, 1), device='cuda:0', dtype=torch.float32)
    getitem_127 = rand_strided((1, 12, 512), (6144, 512, 1), device='cuda:0', dtype=torch.float32)
    getitem_128 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_129 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    alias_default_1 = rand_strided((1, 12, 512, 64), (393216, 64, 768, 1), device='cuda:0', dtype=torch.float32)
    view_258 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_117 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_80 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    view_260 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    addmm_70 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    view_262 = rand_strided((512, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    getitem_121 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.bool)
    mul_85 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    select_8 = rand_strided((1, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    tanh = rand_strided((1, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_124 = rand_strided((1, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    getitem_125 = rand_strided((1, 768), (768, 1), device='cuda:0', dtype=torch.bool)
    permute_134 = rand_strided((2, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_138 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_24 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_142 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_146 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_25 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_150 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_162 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_167 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_171 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_27 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_175 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_179 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_28 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_183 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_195 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_200 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_204 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_30 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_208 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_212 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_31 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_216 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_228 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_233 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_237 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_33 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_241 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_245 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_34 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_249 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_261 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_266 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_270 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_36 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_274 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_278 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_37 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_282 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_294 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_299 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_303 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_39 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_307 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_311 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_40 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_315 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_327 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_332 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_336 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_42 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_340 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_344 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_43 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_348 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_360 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_365 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_369 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_45 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_373 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_377 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_46 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_381 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_393 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_398 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_402 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_48 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_406 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_410 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_49 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_414 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_426 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_431 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_435 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_51 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_439 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_443 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_52 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_447 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_459 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_464 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_468 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_54 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_472 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_476 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_55 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_480 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_492 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_497 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_501 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_57 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_505 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    permute_509 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_58 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_513 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_525 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_530 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    permute_534 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    div_60 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((1, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((1, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    tangents_3 = rand_strided((1, 2), (2, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_8, primals_18, primals_24, primals_34, primals_40, primals_50, primals_56, primals_66, primals_72, primals_82, primals_88, primals_98, primals_104, primals_114, primals_120, primals_130, primals_136, primals_146, primals_152, primals_162, primals_168, primals_178, primals_184, primals_194, primals_200, primals_207, full_default, slice_1, select, select_1, select_2, select_3, mul_1, getitem_3, view, clone_default_33, clone_default_34, clone_default_35, getitem_204, getitem_205, getitem_206, alias_default_23, view_16, getitem_7, mul_3, view_18, addmm_4, view_20, getitem_11, mul_8, view_22, clone_default_30, clone_default_31, clone_default_32, getitem_197, getitem_198, getitem_199, alias_default_21, view_38, getitem_17, mul_10, view_40, addmm_10, view_42, getitem_21, mul_15, view_44, clone_default_27, clone_default_28, clone_default_29, getitem_190, getitem_191, getitem_192, alias_default_19, view_60, getitem_27, mul_17, view_62, addmm_16, view_64, getitem_31, mul_22, view_66, clone_default_24, clone_default_25, clone_default_26, getitem_183, getitem_184, getitem_185, alias_default_17, view_82, getitem_37, mul_24, view_84, addmm_22, view_86, getitem_41, mul_29, view_88, clone_default_21, clone_default_22, clone_default_23, getitem_176, getitem_177, getitem_178, alias_default_15, view_104, getitem_47, mul_31, view_106, addmm_28, view_108, getitem_51, mul_36, view_110, clone_default_18, clone_default_19, clone_default_20, getitem_169, getitem_170, getitem_171, alias_default_13, view_126, getitem_57, mul_38, view_128, addmm_34, view_130, getitem_61, mul_43, view_132, clone_default_15, clone_default_16, clone_default_17, getitem_162, getitem_163, getitem_164, alias_default_11, view_148, getitem_67, mul_45, view_150, addmm_40, view_152, getitem_71, mul_50, view_154, clone_default_12, clone_default_13, clone_default_14, getitem_155, getitem_156, getitem_157, alias_default_9, view_170, getitem_77, mul_52, view_172, addmm_46, view_174, getitem_81, mul_57, view_176, clone_default_9, clone_default_10, clone_default_11, getitem_148, getitem_149, getitem_150, alias_default_7, view_192, getitem_87, mul_59, view_194, addmm_52, view_196, getitem_91, mul_64, view_198, clone_default_6, clone_default_7, clone_default_8, getitem_141, getitem_142, getitem_143, alias_default_5, view_214, getitem_97, mul_66, view_216, addmm_58, view_218, getitem_101, mul_71, view_220, clone_default_3, clone_default_4, clone_default_5, getitem_134, getitem_135, getitem_136, alias_default_3, view_236, getitem_107, mul_73, view_238, addmm_64, view_240, getitem_111, mul_78, view_242, clone_default, clone_default_1, clone_default_2, getitem_127, getitem_128, getitem_129, alias_default_1, view_258, getitem_117, mul_80, view_260, addmm_70, view_262, getitem_121, mul_85, select_8, tanh, getitem_124, getitem_125, permute_134, permute_138, div_24, permute_142, permute_146, div_25, permute_150, permute_162, permute_167, permute_171, div_27, permute_175, permute_179, div_28, permute_183, permute_195, permute_200, permute_204, div_30, permute_208, permute_212, div_31, permute_216, permute_228, permute_233, permute_237, div_33, permute_241, permute_245, div_34, permute_249, permute_261, permute_266, permute_270, div_36, permute_274, permute_278, div_37, permute_282, permute_294, permute_299, permute_303, div_39, permute_307, permute_311, div_40, permute_315, permute_327, permute_332, permute_336, div_42, permute_340, permute_344, div_43, permute_348, permute_360, permute_365, permute_369, div_45, permute_373, permute_377, div_46, permute_381, permute_393, permute_398, permute_402, div_48, permute_406, permute_410, div_49, permute_414, permute_426, permute_431, permute_435, div_51, permute_439, permute_443, div_52, permute_447, permute_459, permute_464, permute_468, div_54, permute_472, permute_476, div_55, permute_480, permute_492, permute_497, permute_501, div_57, permute_505, permute_509, div_58, permute_513, permute_525, permute_530, permute_534, div_60, tangents_1, tangents_2, tangents_3]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('LayoutLMForSequenceClassification', benchmark_compiled_module)
