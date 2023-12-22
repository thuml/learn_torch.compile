
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


# kernel path: /tmp/torchinductor_youkaichao/ui/cuiwonof6lkph27msswmfp2cykmcthfd3ceacfqmuvbpquen4s3y.py
# Source Nodes: [], Original ATen: [aten.scatter, aten.zeros]

triton_poi_fused_scatter_zeros_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_scatter_zeros_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1568000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/23/c232ifs6gbc2zvtshtdyfkupspthwbgy2pv3ble7zfx3slwyezt5.py
# Source Nodes: [], Original ATen: [aten.mul]

triton_poi_fused_mul_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_1', 'mutated_arg_names': []},
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
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bh/cbhm4egws6jtd3i5joi4byiliwhnm2zrv66ixpt2kpnqxficdth6.py
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 13000
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 1000)
    x0 = xindex % 1000
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (1000*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ur/curctgntjivcg3i6qi2azdrc3cszw7caj6i2djfbnhtp3n4h2md3.py
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
    size_hints=[1024, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1000
    rnumel = 13
    RBLOCK: tl.constexpr = 16
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


# kernel path: /tmp/torchinductor_youkaichao/bl/cblyfakjqvh4q2o7r7pny3354sdzumtwdf4itmumxxacg4zqtqij.py
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
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_4', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/6m/c6m6bjnbqrgqaychz33ugqrouev3mczg2pablarsxxdbko2sjbks.py
# Source Nodes: [x_234], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
# x_234 => mul_181, sub_63
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 197
    r2 = rindex
    x1 = (xindex // 197)
    x3 = xindex
    tmp10 = tl.load(in_ptr1 + (r2 + (384*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.load(in_ptr3 + (r2 + (384*x3)), rmask & xmask, other=0.0)
    tmp20 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-384) + r2 + (384*x0) + (75264*x1)), rmask & tmp2 & xmask, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = tmp0 == tmp8
    tmp11 = tl.where(tmp9, tmp10, tmp6)
    tmp12 = tmp7 + tmp11
    tmp14 = tmp12 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp21 = tmp19 - tmp20
    tmp23 = tmp21 * tmp22
    tmp24 = tmp14 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp27 = tl.where(rmask & xmask, tmp25, 0)
    tmp28 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp29 = 384.0
    tmp30 = tmp22 / tmp29
    tmp31 = tmp14 * tmp29
    tmp32 = tmp31 - tmp18
    tmp33 = tmp23 * tmp28
    tmp34 = tmp32 - tmp33
    tmp35 = tmp30 * tmp34
    tl.store(out_ptr2 + (r2 + (384*x3)), tmp35, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/us/cuscugh6sypbshwbt32w4cfw6s2slmfg3hr4phaq5juhazplwixe.py
# Source Nodes: [x_234], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
# x_234 => mul_181, sub_63
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp31 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 1576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = (r2 + (122*x1)) % 197
        tmp4 = tl.full([1, 1], 1, tl.int64)
        tmp5 = tmp3 >= tmp4
        tmp6 = tmp5 & tmp2
        tmp7 = tl.load(in_ptr0 + ((-384) + x0 + (384*((r2 + (122*x1)) % 197)) + (75264*(((r2 + (122*x1)) // 197) % 8))), rmask & tmp6 & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
        tmp9 = tl.where(tmp6, tmp7, tmp8)
        tmp10 = 0.0
        tmp11 = tl.where(tmp5, tmp9, tmp10)
        tmp12 = tl.full([1, 1], 0, tl.int32)
        tmp13 = tmp3 == tmp12
        tmp14 = tl.load(in_ptr1 + (x0 + (384*(((r2 + (122*x1)) // 197) % 8))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.where(tmp13, tmp14, tmp10)
        tmp16 = tmp11 + tmp15
        tmp17 = tl.load(in_ptr2 + (x0 + (384*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr3 + ((r2 + (122*x1)) % 1576), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tmp17 - tmp18
        tmp20 = tl.load(in_ptr4 + ((r2 + (122*x1)) % 1576), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tmp19 * tmp20
        tmp22 = tmp16 * tmp21
        tmp23 = tl.full(tmp22.shape, 0, tmp22.dtype)
        tmp24 = tl.where(tmp2, tmp22, tmp23)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
        tmp28 = tl.full(tmp16.shape, 0, tmp16.dtype)
        tmp29 = tl.where(tmp2, tmp16, tmp28)
        tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
        tmp32 = _tmp31 + tmp30
        _tmp31 = tl.where(rmask & xmask, tmp32, _tmp31)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp26, xmask)
    tmp31 = tl.sum(_tmp31, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uv/cuvgme4w265uy5c4jfodgktprbkh7cyhqjxw6tbgmiiffuplwh5c.py
# Source Nodes: [x_234], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
# x_234 => mul_181, sub_63
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 13
    RBLOCK: tl.constexpr = 16
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


# kernel path: /tmp/torchinductor_youkaichao/fd/cfd23mnfhr2skmg4xsy4kt5jrh2mdi4agj4nsgb3gbtdtr725nil.py
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
    size_hints=[512, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x0 + (75648*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lc/clch5ky56fvofdkngtubkmuujpygrts3kyzle37qbrdkdji6ddow.py
# Source Nodes: [x_226], Original ATen: [aten.gelu, aten.gelu_backward]
# x_226 => add_174, erf_19, mul_179
triton_poi_fused_gelu_gelu_backward_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
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
    tl.store(in_out_ptr0 + (x0), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a4/ca4umcnxo3um4q6l5y267jtnisnilyzr2tch5vdavllgvjmyib4o.py
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
    size_hints=[2048, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1152*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k3/ck3en7ssv3wotes3at4gmryikppjtxzuo6w3jnrhfvd3hscz4uqo.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 8
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (r1 + (75648*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 384.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(out_ptr2 + (r1 + (384*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e5/ce56bve4rqy7pemewhbruwwry7nzqyhv4zclw2osfecgvz7zv3il.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_per_fused_native_layer_norm_backward_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_backward_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (384*r1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pv/cpvinw4ablf4kccxor2ubfsnkho7askrxes5tly6g54n2syo3i3j.py
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
    size_hints=[512, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mn/cmnijprk5yecn3rv354rogh2cyamuq3guy3w5rjuwh623rx3umvk.py
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
    size_hints=[128, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 197
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (197*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (197*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tmp1 * tmp6
    tmp8 = tmp2 - tmp7
    tl.store(out_ptr1 + (r1 + (197*x0)), tmp8, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7w/c7wbwpx2cys436zqnaias37ushondubggdxtayhav7nwemg77duu.py
# Source Nodes: [], Original ATen: [aten.view]

triton_poi_fused_view_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.1767766952966369
    tmp2 = tmp0 * tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/if/cifcajd6h72oijwed6ztqyoj5lmsxxk4jgqu73wgfc7ato2sqoae.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 37824
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y6 = (yindex // 2364)
    x4 = xindex
    y0 = yindex % 197
    y7 = (yindex // 197)
    y8 = yindex
    y1 = (yindex // 197) % 12
    y2 = (yindex // 2364) % 8
    y3 = (yindex // 18912)
    tmp0 = y6
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + (197*x4) + (6304*y7)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 16, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-605184) + x4 + (32*y8)), tmp8 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tl.store(out_ptr0 + (x4 + (32*y1) + (384*y3) + (768*y0) + (151296*y2)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gg/cggevidegm45hvceaiorskgoqepexmlroncut3sduv5zfwj64kk3.py
# Source Nodes: [l__mod___post_network_1_norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.slice_backward]
# l__mod___post_network_1_norm1 => mul_173, sub_60
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_slice_backward_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_slice_backward_17', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 197
    r2 = rindex
    x1 = (xindex // 197)
    x3 = xindex
    tmp8 = tl.load(in_ptr1 + (r2 + (384*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr3 + (r2 + (384*x3)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (r2 + (384*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp9 = tmp7 + tmp8
    tmp11 = tmp9 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp18 = tmp16 - tmp17
    tmp20 = tmp18 * tmp19
    tmp21 = tmp11 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tmp0 >= tmp1
    tmp27 = tl.load(in_out_ptr0 + (r2 + (384*x3)), rmask & tmp26 & xmask, other=0.0)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp26, tmp27, tmp28)
    tmp30 = tl.where(tmp26, tmp29, tmp6)
    tmp31 = 384.0
    tmp32 = tmp19 / tmp31
    tmp33 = tmp11 * tmp31
    tmp34 = tmp33 - tmp15
    tmp35 = tmp20 * tmp25
    tmp36 = tmp34 - tmp35
    tmp37 = tmp32 * tmp36
    tmp38 = tmp30 + tmp37
    tmp39 = tl.load(in_ptr6 + (r2 + (384*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp2, tmp39, tmp40)
    tmp42 = tl.where(tmp2, tmp41, tmp6)
    tmp43 = tmp38 + tmp42
    tl.store(in_out_ptr0 + (r2 + (384*x3)), tmp38, rmask & xmask)
    tl.store(out_ptr2 + (r2 + (384*x3)), tmp43, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bj/cbjbpfnq4uu5hrfrqbqd7uuvjmwdxbjsi5bq3p34fxb3d5mffsoq.py
# Source Nodes: [l__mod___post_network_1_norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.slice_backward]
# l__mod___post_network_1_norm1 => mul_173, sub_60
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_slice_backward_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_slice_backward_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 122
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp28 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (122*x1)
        tmp1 = tl.full([1, 1], 1576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = (r2 + (122*x1)) % 197
        tmp4 = tl.full([1, 1], 1, tl.int64)
        tmp5 = tmp3 < tmp4
        tmp6 = tmp5 & tmp2
        tmp7 = tl.load(in_ptr0 + (x0 + (384*(((r2 + (122*x1)) // 197) % 8))), rmask & tmp6 & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
        tmp9 = tl.where(tmp6, tmp7, tmp8)
        tmp10 = 0.0
        tmp11 = tl.where(tmp5, tmp9, tmp10)
        tmp12 = tl.load(in_ptr1 + (x0 + (384*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tmp11 + tmp12
        tmp14 = tl.load(in_ptr2 + (x0 + (384*((r2 + (122*x1)) % 1576))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr3 + ((r2 + (122*x1)) % 1576), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tmp14 - tmp15
        tmp17 = tl.load(in_ptr4 + ((r2 + (122*x1)) % 1576), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tmp16 * tmp17
        tmp19 = tmp13 * tmp18
        tmp20 = tl.full(tmp19.shape, 0, tmp19.dtype)
        tmp21 = tl.where(tmp2, tmp19, tmp20)
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
        tmp25 = tl.full(tmp13.shape, 0, tmp13.dtype)
        tmp26 = tl.where(tmp2, tmp13, tmp25)
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(rmask & xmask, tmp29, _tmp28)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp23, xmask)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hy/chytowwqzcqwcguuwp5uai2v7ztcyjxlqadk5lzdkmudpxwlboas.py
# Source Nodes: [l__mod___post_network_0_norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.slice_backward]
# l__mod___post_network_0_norm1 => mul_165, sub_57
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_slice_backward_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_slice_backward_19', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex % 197
    r2 = rindex
    x1 = (xindex // 197)
    x3 = xindex
    tmp8 = tl.load(in_ptr1 + (r2 + (384*x3)), rmask & xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr3 + (r2 + (384*x3)), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (r2 + (384*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = 0.0
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tmp9 = tmp7 + tmp8
    tmp11 = tmp9 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp18 = tmp16 - tmp17
    tmp20 = tmp18 * tmp19
    tmp21 = tmp11 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = tmp0 >= tmp1
    tmp27 = tl.load(in_out_ptr0 + (r2 + (384*x3)), rmask & tmp26 & xmask, other=0.0)
    tmp28 = tl.broadcast_to(x0, [RBLOCK])
    tmp29 = tmp28 < tmp1
    tmp30 = tmp29 & tmp26
    tmp31 = tl.load(in_ptr6 + (r2 + (384*x1)), rmask & tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp30, tmp31, tmp32)
    tmp34 = tl.where(tmp29, tmp33, tmp6)
    tmp35 = tmp27 + tmp34
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp26, tmp35, tmp36)
    tmp38 = tl.where(tmp26, tmp37, tmp6)
    tmp39 = 384.0
    tmp40 = tmp19 / tmp39
    tmp41 = tmp11 * tmp39
    tmp42 = tmp41 - tmp15
    tmp43 = tmp20 * tmp25
    tmp44 = tmp42 - tmp43
    tmp45 = tmp40 * tmp44
    tmp46 = tmp38 + tmp45
    tl.store(in_out_ptr0 + (r2 + (384*x3)), tmp46, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5x/c5x35rza5w6ivxchmanumilng6q5jp2anu7xhgu6do4h7vavvtot.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp5 = tl.load(in_ptr1 + (x0 + (75648*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tl.full([1, 1], 0, tl.int64)
    tmp7 = tl.full([1, 1], 1, tl.int64)
    tmp8 = tmp6 < tmp7
    tmp9 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & tmp8 & xmask, other=0.0)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = 0.0
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tmp5 + tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rb/crbgx6n3caypdk73niafgk6z6lziflzxudh6zgjzd3mrdv42wjtc.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 75264)
    x3 = xindex % 75264
    x1 = (xindex // 384) % 196
    x0 = xindex % 384
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (384 + x3 + (75648*x2)), None)
    tmp1 = 1 + x1
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = tl.load(in_ptr1 + (x0 + (384*x2)), tmp3, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.full(tmp4.shape, 0.0, tmp4.dtype)
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = 0.0
    tmp8 = tl.where(tmp3, tmp6, tmp7)
    tmp9 = tmp0 + tmp8
    tl.store(out_ptr0 + (x4), tmp9, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/p5/cp5ljfwdok4wvxj3v3ozjqz4hlujkmk4dqkwmdyct33o2klcob4c.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
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
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*r2) + (46464*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nv/cnvkm34o6sdffv5ubyojhktibpzd35gwtvhogxment5be542kgt3.py
# Source Nodes: [x_208], Original ATen: [aten.gelu, aten.gelu_backward]
# x_208 => add_160, erf_17, mul_163
triton_poi_fused_gelu_gelu_backward_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_23', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1806336
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


# kernel path: /tmp/torchinductor_youkaichao/63/c63sms36vzmkjga6m4k3tp4zh3lmfdyug4xjutrem52rnlzt4xeg.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_24', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 14976
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 1152)
    x0 = xindex % 1152
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (1152*r2) + (139392*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y5/cy5yt6hwd56mc4hydnnwo3xozrrclobycxg7rf5y2eljoge5nbvs.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_25', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1152*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xh/cxhgx2hmbvcoj6bk3t4abppkpd5c6bsxlkdce4vca53ah5iiyyun.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 196
    x3 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (384 + r1 + (384*x2) + (75648*x3)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 1 + x2
    tmp15 = tl.full([1], 1, tl.int64)
    tmp16 = tmp14 < tmp15
    tmp17 = tl.load(in_ptr4 + (r1 + (384*x3)), rmask & tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp16, tmp17, tmp18)
    tmp20 = 0.0
    tmp21 = tl.where(tmp16, tmp19, tmp20)
    tmp22 = tmp13 + tmp21
    tmp24 = 384.0
    tmp25 = tmp2 * tmp24
    tmp26 = tmp25 - tmp6
    tmp27 = tmp7 * tmp12
    tmp28 = tmp26 - tmp27
    tmp29 = tmp23 * tmp28
    tmp30 = tmp22 + tmp29
    tl.store(out_ptr2 + (r1 + (384*x0)), tmp30, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xh/cxhffo2sdbxtwvwoezxsholdyjo7nucay6dt4qspvutiz2oxnrgr.py
# Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]

triton_red_fused_native_layer_norm_backward_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 384)
    x0 = xindex % 384
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/kw/ckwcta4o5e4dq6fqnegjlviia4hn22zkaokrhljahsx3zm237mzx.py
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
    xnumel = 4992
    rnumel = 121
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
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*r2) + (46464*x1)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/az/cazrvb7cnmg6venq2rpmhg7v4lyksx4ax2j5qlabbntoy5lgvxpw.py
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
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 196
    x2 = (xindex // 6272) % 12
    x3 = (xindex // 75264)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*x2) + (384*x1) + (75264*x3)), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yn/cynnboazk2bnebw462usay4gqtzseeyg52hvvzapd7sc4ae3dymm.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]

triton_per_fused__softmax_backward_data_mul_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_mul_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 18816
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
    tmp9 = 0.1767766952966369
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr1 + (r1 + (196*x0)), tmp10, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/47/c475nqi7gvkos4z5eo4vo77klb7qemee3mddrb7wyacu6eatyp35.py
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
    size_hints=[65536, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 56448
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y6 = (yindex // 2352)
    x4 = xindex
    y7 = yindex
    y0 = yindex % 196
    y8 = (yindex // 196)
    y1 = (yindex // 196) % 12
    y2 = (yindex // 2352) % 8
    y3 = (yindex // 18816)
    tmp0 = y6
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + (32*y7)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 16, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-602112) + y0 + (196*x4) + (6272*y8)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1, 1], 24, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-1204224) + x4 + (32*y7)), tmp15 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tl.where(tmp11, tmp14, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tl.store(out_ptr0 + (x4 + (32*y1) + (384*y3) + (1152*y0) + (225792*y2)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mz/cmzc6mg3k4q3ijrhmqgmvjmtr4mnuztouf2ewn6nxxnqwhfqou2m.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_32', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (384*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 384.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(in_out_ptr0 + (r1 + (384*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dj/cdjiu2za6llg4ni66hbzrhqk7shxihnq6yzmbnxwfejpbebnbspu.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_per_fused_sum_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 75264
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (75264*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bt/cbt4udtglpq3jrpb5ucc5uqtkdhby52jyjii6h32uylcox65act6.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_34', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4992
    rnumel = 121
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
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (384*((r2 + (121*x1)) % 1568))), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ap/capc5ce46nojqe5vstog26sn2wews565falo5hej2h3nnamte6ar.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]

triton_poi_fused__unsafe_view_clone_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + ((784*x1) + (150528*(y0 // 784)) + (y0 % 784)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + (192*y0)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jr/cjr4mur7bftjgvfei63mn6rebi4emqfzkxlexqrrqvig2u6kdwj7.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cn/ccnobu7z4tdgkpazadubpabiij67x3o5m5zng27bhsugsoapeo3y.py
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
    xnumel = 192
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (192*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6g/c6gsxrs2bofy7zmnzmujdfvxz7qqthkmusi2qwp5nhafpvrripbi.py
# Source Nodes: [x_45], Original ATen: [aten.gelu, aten.gelu_backward]
# x_45 => add_61, erf_3, mul_51
triton_poi_fused_gelu_gelu_backward_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_38', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3612672
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


# kernel path: /tmp/torchinductor_youkaichao/7y/c7yoafik65vbw7y753s3bctyv2vnftla7c6tnbfk7znx3uxvxjel.py
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_39', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 28224
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 576
    x1 = (xindex // 576)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (576*r2) + (73728*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mm/cmmndxibgmjw674ruwuqyh7z2w4ly2vmcmjm6hdl3k7ay6d7x6aw.py
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
    xnumel = 576
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (576*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z4/cz4twmpwpuskfknwtxo3pjatxlqqniw6ak7cotpkrgy5zgbyquqx.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]

triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_41', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 784
    x3 = (xindex // 784)
    tmp0 = tl.load(in_ptr0 + (r1 + (192*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (192*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr3 + (x2 + (784*r1) + (150528*x3)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp15 = 192.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(out_ptr2 + (r1 + (192*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (192*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sw/cswk3eob4t2zmlxdtleyyam6gwsomls457eooiorikyfxteyd32z.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_backward_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 192
    x1 = (xindex // 192)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/4i/c4ipdsivgdqt3ohv5ax64ra7xmg6lr5zjm4pbqkawaxwbnadkxnu.py
# Source Nodes: [], Original ATen: [aten.sum]

triton_red_fused_sum_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jk/cjk7eh2kqteoo6oidi34w7m3dw6utfhlndm4p5x2d6wiengdlxvz.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2709504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 32) % 9
    x2 = (xindex // 288) % 196
    x0 = xindex % 32
    x3 = (xindex // 56448) % 6
    x4 = (xindex // 338688)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + ((14*(x1 // 3)) + (x2 // 14)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + ((14*(x1 % 3)) + (x2 % 14)), None, eviction_policy='evict_last')
    tmp1 = tmp0 + 30
    tmp2 = tmp0 < 0
    tmp3 = tl.where(tmp2, tmp1, tmp0)
    tl.device_assert((0 <= tmp3) & (tmp3 < 30), "index out of bounds: 0 <= tmp3 < 30")
    tmp5 = tmp4 + 30
    tmp6 = tmp4 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp4)
    tl.device_assert((0 <= tmp7) & (tmp7 < 30), "index out of bounds: 0 <= tmp7 < 30")
    tmp8 = (-1) + tmp3
    tmp9 = tl.full([1], 0, tl.int64)
    tmp10 = tmp8 >= tmp9
    tmp11 = tl.full([1], 28, tl.int64)
    tmp12 = tmp8 < tmp11
    tmp13 = (-1) + tmp7
    tmp14 = tmp13 >= tmp9
    tmp15 = tmp13 < tmp11
    tmp16 = tmp10 & tmp12
    tmp17 = tmp16 & tmp14
    tmp18 = tmp17 & tmp15
    tmp19 = tl.load(in_ptr2 + ((-5568) + x0 + (32*x3) + (192*tmp7) + (5376*tmp3) + (150528*x4)), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tl.store(out_ptr0 + (x6), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/r5/cr5zzicbmuvi7sz6x4zdmidb56uhbegdspiubziaun26jkvd7hs5.py
# Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.clone]

triton_per_fused__softmax_backward_data_clone_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[131072, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_backward_data_clone_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 84672
    rnumel = 9
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 9
    x3 = (xindex // 9) % 196
    x4 = (xindex // 1764) % 6
    x5 = (xindex // 10584)
    tmp0 = tl.load(in_ptr0 + (r1 + (9*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (9*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tmp1 * tmp6
    tmp8 = tmp2 - tmp7
    tmp9 = 0.1767766952966369
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr1 + (r1 + (9*x2) + (81*x4) + (486*x3) + (95256*x5)), tmp10, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jt/cjt4q6j7vskwrnpgiu54i3beiszfpsvc3girhmedbxepgwxtfhwx.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6318
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 486)
    x0 = xindex % 486
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (486*r2) + (58806*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dx/cdxmfz5i4usjurg2z3cjcj6d56hnnucdhbkmjeeke7myxjmqeciy.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_47', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 486
    rnumel = 13
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (486*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oy/coyvs4fqfiyfmhzig7og23kikyx64heryzaymzpb27vdqs6cpoy4.py
# Source Nodes: [], Original ATen: [aten.col2im]

triton_poi_fused_col2im_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_col2im_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1382400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/h3/ch3fgiqc7qnla7olwjccl6xhmmge67gfdn46lfeg65bypmws5ehg.py
# Source Nodes: [], Original ATen: [aten.clone, aten.col2im]

triton_poi_fused_clone_col2im_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 2048], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_col2im_49', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 1764
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 196
    x3 = (xindex // 196)
    y0 = yindex % 32
    y1 = (yindex // 32)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (32*x3) + (288*x2) + (56448*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x5 + (1764*y4)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ed/cedqnokc2oczf6cr7s2hdakdyzmqvxubqsqazyy577ocb2op332j.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]

triton_poi_fused__unsafe_view_clone_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex
    x1 = xindex
    tmp0 = 1 + ((y0 // 28) % 28)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 30, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 1 + (y0 % 28)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (31 + (30*((y0 // 28) % 28)) + (900*x1) + (172800*(y0 // 784)) + (y0 % 28)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tl.store(out_ptr0 + (x1 + (192*y0)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cx/ccx27vcntvp4xygsstbeatf5m6hlwtayqy5765hr4t5zg2vpapx3.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]

triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_51', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 28
    x1 = (xindex // 28) % 28
    x2 = (xindex // 784)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (192*(tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(14, 1 + (x0 // 2)))))) + (192*(tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(14, 1 + (x0 // 2))))) >= 0, 0, 14))) + (2688*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(14, 1 + (x1 // 2)))))) + (2688*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(14, 1 + (x1 // 2))))) >= 0, 0, 14))) + (37632*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr1 + (r3 + (192*x5)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr2 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.load(in_ptr3 + (r3 + (192*x5)), rmask & xmask, other=0.0)
    tmp25 = tl.load(in_out_ptr0 + (r3 + (192*x5)), rmask & xmask, other=0.0)
    tmp26 = tl.load(in_ptr4 + (x5), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 / 4
    tmp2 = tl.math.max(0, (x1 // 2))
    tmp3 = tl.math.min(14, 1 + (x1 // 2))
    tmp4 = tmp2 < tmp3
    tmp5 = tl.math.max(0, (x0 // 2))
    tmp6 = tl.math.min(14, 1 + (x0 // 2))
    tmp7 = tmp5 < tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp1, tmp9)
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp20 = tmp14 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp27 = 192.0
    tmp28 = tmp14 * tmp27
    tmp29 = tmp28 - tmp18
    tmp30 = tmp19 * tmp24
    tmp31 = tmp29 - tmp30
    tmp32 = tmp26 * tmp31
    tmp33 = tmp25 + tmp32
    tl.store(in_out_ptr0 + (r3 + (192*x5)), tmp33, rmask & xmask)
    tl.store(out_ptr2 + (r3 + (192*x5)), tmp33, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/am/camovokw6w6sgyywir2yq2pb3br3n5vnnu5jsp43ggfganiwbv2y.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_red_fused_add_native_layer_norm_backward_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_52', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 9408
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 192
    x1 = (xindex // 192)
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (192*(tl.math.min(tl.math.max(0, (((r2 + (128*x1)) % 28) // 2)), (-1) + (tl.math.min(14, 1 + (((r2 + (128*x1)) % 28) // 2)))))) + (192*(tl.where((tl.math.min(tl.math.max(0, (((r2 + (128*x1)) % 28) // 2)), (-1) + (tl.math.min(14, 1 + (((r2 + (128*x1)) % 28) // 2))))) >= 0, 0, 14))) + (2688*(tl.math.min(tl.math.max(0, ((((r2 + (128*x1)) // 28) % 28) // 2)), (-1) + (tl.math.min(14, 1 + ((((r2 + (128*x1)) // 28) % 28) // 2)))))) + (2688*(tl.where((tl.math.min(tl.math.max(0, ((((r2 + (128*x1)) // 28) % 28) // 2)), (-1) + (tl.math.min(14, 1 + ((((r2 + (128*x1)) // 28) % 28) // 2))))) >= 0, 0, 14))) + (37632*((r2 + (128*x1)) // 784))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tl.load(in_ptr1 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr2 + (x0 + (192*r2) + (24576*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0 / 4
        tmp2 = tl.math.max(0, ((((r2 + (128*x1)) // 28) % 28) // 2))
        tmp3 = tl.math.min(14, 1 + ((((r2 + (128*x1)) // 28) % 28) // 2))
        tmp4 = tmp2 < tmp3
        tmp5 = tl.math.max(0, (((r2 + (128*x1)) % 28) // 2))
        tmp6 = tl.math.min(14, 1 + (((r2 + (128*x1)) % 28) // 2))
        tmp7 = tmp5 < tmp6
        tmp8 = tmp4 & tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp8, tmp1, tmp9)
        tmp12 = tmp10 + tmp11
        tmp14 = tmp12 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
        tmp18 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp16, xmask)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp19, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sb/csbact7susm37et3fg4soi4spvjohlyg7lk5kr64u3ke6w3j76f2.py
# Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]

triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_53', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (192*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (r1 + (192*x0)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (192*x0)), rmask & xmask, other=0.0)
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp8 = tmp2 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp15 = 192.0
    tmp16 = tmp2 * tmp15
    tmp17 = tmp16 - tmp6
    tmp18 = tmp7 * tmp12
    tmp19 = tmp17 - tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp13 + tmp20
    tl.store(in_out_ptr0 + (r1 + (192*x0)), tmp21, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (192*x0)), tmp21, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ww/cwwhhfnsves2npb4v6pfddchs3jn6gqbsq2st4pozfq5eg3igohc.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_per_fused_add_native_layer_norm_backward_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_54', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x0 = xindex % 28
    x1 = (xindex // 28) % 28
    x2 = (xindex // 784)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (r3 + (192*(tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(14, 1 + (x0 // 2)))))) + (192*(tl.where((tl.math.min(tl.math.max(0, (x0 // 2)), (-1) + (tl.math.min(14, 1 + (x0 // 2))))) >= 0, 0, 14))) + (2688*(tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(14, 1 + (x1 // 2)))))) + (2688*(tl.where((tl.math.min(tl.math.max(0, (x1 // 2)), (-1) + (tl.math.min(14, 1 + (x1 // 2))))) >= 0, 0, 14))) + (37632*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr1 + (r3 + (192*x5)), rmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr2 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.load(in_ptr3 + (r3 + (192*x5)), rmask & xmask, other=0.0)
    tmp25 = tl.load(in_out_ptr0 + (r3 + (192*x5)), rmask & xmask, other=0.0)
    tmp26 = tl.load(in_ptr4 + (x5), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 / 4
    tmp2 = tl.math.max(0, (x1 // 2))
    tmp3 = tl.math.min(14, 1 + (x1 // 2))
    tmp4 = tmp2 < tmp3
    tmp5 = tl.math.max(0, (x0 // 2))
    tmp6 = tl.math.min(14, 1 + (x0 // 2))
    tmp7 = tmp5 < tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp8, tmp1, tmp9)
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp20 = tmp14 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp27 = 192.0
    tmp28 = tmp14 * tmp27
    tmp29 = tmp28 - tmp18
    tmp30 = tmp19 * tmp24
    tmp31 = tmp29 - tmp30
    tmp32 = tmp26 * tmp31
    tmp33 = tmp25 + tmp32
    tl.store(in_out_ptr0 + (r3 + (192*x5)), tmp33, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2g/c2guw7j6yibpdkszgxbfip3q3twzyxkgdl2aitkoxdh3v2mueob2.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.permute]

triton_poi_fused_add_native_layer_norm_backward_permute_55 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_backward_permute_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (192*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (784*y3)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4j/c4jf4ew43dicue6k3cdn4b6ymzgfxxf325trehhecv6ikjh4m4cz.py
# Source Nodes: [], Original ATen: [aten.convolution_backward]

triton_red_fused_convolution_backward_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_backward_56', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (150528*r2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4v/c4vuybqv6agnwn23ie2okyozvtwbyz645f76arzxbgmjvtxw3iz5.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_red_fused_native_batch_norm_backward_threshold_backward_57 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_batch_norm_backward_threshold_backward_57', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 832
    rnumel = 7720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7720*x0)
        tmp1 = tl.full([1, 1], 100352, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((12544*x1) + (802816*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 0.0
        tmp5 = tmp3 <= tmp4
        tmp6 = tl.load(in_ptr1 + ((12544*x1) + (802816*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.where(tmp5, tmp4, tmp6)
        tmp8 = tl.full(tmp7.shape, 0, tmp7.dtype)
        tmp9 = tl.where(tmp2, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
        tmp13 = tl.load(in_ptr2 + ((12544*x1) + (802816*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr3 + (tl.broadcast_to(x1, [XBLOCK, RBLOCK])), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/45/c45vulhvy2tiee2z4oe47yiwjziuv6gjucrixf3554gciai3udim.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_58 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_58', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
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


# kernel path: /tmp/torchinductor_youkaichao/xz/cxzhv4v2rjtesplqknibttvypydt4xreqotri4ygyz4hcu5jopke.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]

triton_per_fused_native_batch_norm_backward_threshold_backward_59 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_batch_norm_backward_threshold_backward_59', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
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


# kernel path: /tmp/torchinductor_youkaichao/l2/cl2ov6kdndls52blt46sgoafl4k2qyqbqsvorlj5khetri553cd5.py
# Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]

triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_60 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_60', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 64
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp3 = tl.load(in_out_ptr0 + (x3), None)
    tmp5 = tl.load(in_ptr1 + (x3), None)
    tmp6 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 <= tmp1
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp7 = tmp5 - tmp6
    tmp9 = 9.964923469387754e-06
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 * tmp13
    tmp15 = tmp4 - tmp14
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 - tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_3, primals_4, primals_6, primals_7, primals_9, primals_10, primals_12, primals_14, primals_21, primals_27, primals_34, primals_40, primals_47, primals_53, primals_60, primals_66, primals_68, primals_73, primals_79, primals_84, primals_90, primals_95, primals_101, primals_106, primals_112, primals_117, primals_123, primals_128, primals_134, primals_139, primals_145, primals_150, primals_156, primals_161, primals_167, primals_172, primals_178, primals_183, primals_189, primals_194, primals_200, primals_205, primals_211, primals_216, primals_222, primals_228, primals_234, primals_240, primals_246, primals_261, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, mul_21, view, add_17, unsqueeze_17, permute_5, view_4, full_default, view_12, mul_24, view_14, addmm_1, view_16, mul_29, view_18, permute_19, view_22, view_30, mul_32, view_32, addmm_4, view_34, mul_37, view_36, permute_33, view_40, view_48, mul_40, view_50, addmm_7, view_52, mul_45, view_54, permute_47, view_58, view_66, mul_48, view_68, addmm_10, view_70, permute_57, mul_53, view_72, view_82, mul_56, view_84, addmm_13, view_86, mul_61, view_88, view_98, mul_64, view_100, addmm_16, view_102, mul_69, view_104, view_114, mul_72, view_116, addmm_19, view_118, mul_77, view_120, view_130, mul_80, view_132, addmm_22, view_134, mul_85, view_136, view_146, mul_88, view_148, addmm_25, view_150, mul_93, view_152, view_162, mul_96, view_164, addmm_28, view_166, mul_101, view_168, view_178, mul_104, view_180, addmm_31, view_182, mul_109, view_184, view_194, mul_112, view_196, addmm_34, view_198, mul_117, view_200, view_210, mul_120, view_212, addmm_37, view_214, mul_125, view_216, view_226, mul_128, view_228, addmm_40, view_230, mul_133, view_232, view_242, mul_136, view_244, addmm_43, view_246, mul_141, view_248, view_258, mul_144, view_260, addmm_46, view_262, mul_149, view_264, view_274, mul_152, view_276, addmm_49, view_278, mul_157, view_280, view_290, mul_160, view_292, addmm_52, view_294, cat, getitem_121, rsqrt_39, view_297, view_300, view_310, mul_168, view_312, addmm_55, view_314, cat_1, getitem_127, rsqrt_41, view_316, view_319, view_329, mul_176, view_331, addmm_58, view_333, cat_2, getitem_133, rsqrt_43, select, view_335, unsqueeze_61, permute_177, permute_179, permute_183, permute_187, div_21, permute_191, permute_196, permute_197, alias_23, permute_198, permute_199, permute_203, permute_208, permute_210, permute_214, div_23, permute_218, permute_223, permute_224, alias_24, permute_225, permute_226, permute_230, permute_235, permute_237, permute_241, div_25, permute_245, permute_250, permute_251, alias_25, permute_252, permute_253, permute_258, div_26, permute_260, permute_264, div_27, permute_268, permute_273, permute_274, alias_26, permute_275, permute_276, permute_281, div_28, permute_283, permute_287, div_29, permute_291, permute_296, permute_297, alias_27, permute_298, permute_299, permute_304, div_30, permute_306, permute_310, div_31, permute_314, permute_319, permute_320, alias_28, permute_321, permute_322, permute_327, div_32, permute_329, permute_333, div_33, permute_337, permute_342, permute_343, alias_29, permute_344, permute_345, permute_350, div_34, permute_352, permute_356, div_35, permute_360, permute_365, permute_366, alias_30, permute_367, permute_368, permute_373, div_36, permute_375, permute_379, div_37, permute_383, permute_388, permute_389, alias_31, permute_390, permute_391, permute_396, div_38, permute_398, permute_402, div_39, permute_406, permute_411, permute_412, alias_32, permute_413, permute_414, permute_419, div_40, permute_421, permute_425, div_41, permute_429, permute_434, permute_435, alias_33, permute_436, permute_437, permute_442, div_42, permute_444, permute_448, div_43, permute_452, permute_457, permute_458, alias_34, permute_459, permute_460, permute_465, div_44, permute_467, permute_471, div_45, permute_475, permute_480, permute_481, alias_35, permute_482, permute_483, permute_488, div_46, permute_490, permute_494, div_47, permute_498, permute_503, permute_504, alias_36, permute_505, permute_506, permute_511, div_48, permute_513, permute_517, div_49, permute_521, permute_526, permute_527, alias_37, permute_528, permute_529, permute_534, div_50, permute_536, permute_540, div_51, permute_544, permute_549, permute_550, alias_38, permute_551, permute_552, permute_557, div_52, permute_561, permute_565, div_53, permute_571, permute_576, permute_577, alias_39, permute_579, permute_590, div_54, permute_592, permute_596, div_55, permute_602, permute_607, permute_608, alias_40, permute_610, permute_621, div_56, permute_623, permute_627, div_57, permute_633, permute_638, permute_639, alias_41, permute_641, permute_652, div_58, permute_654, permute_658, div_59, permute_664, permute_669, permute_670, alias_42, permute_672, permute_683, div_60, unsqueeze_112, unsqueeze_124, unsqueeze_136, tangents_1 = args
    args.clear()
    assert_size_stride(primals_3, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_6, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_9, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_12, (192, 64, 4, 4), (1024, 16, 4, 1))
    assert_size_stride(primals_14, (192, ), (1, ))
    assert_size_stride(primals_21, (192, ), (1, ))
    assert_size_stride(primals_27, (192, ), (1, ))
    assert_size_stride(primals_34, (192, ), (1, ))
    assert_size_stride(primals_40, (192, ), (1, ))
    assert_size_stride(primals_47, (192, ), (1, ))
    assert_size_stride(primals_53, (192, ), (1, ))
    assert_size_stride(primals_60, (192, ), (1, ))
    assert_size_stride(primals_66, (384, 192, 2, 2), (768, 4, 2, 1))
    assert_size_stride(primals_68, (384, ), (1, ))
    assert_size_stride(primals_73, (384, ), (1, ))
    assert_size_stride(primals_79, (384, ), (1, ))
    assert_size_stride(primals_84, (384, ), (1, ))
    assert_size_stride(primals_90, (384, ), (1, ))
    assert_size_stride(primals_95, (384, ), (1, ))
    assert_size_stride(primals_101, (384, ), (1, ))
    assert_size_stride(primals_106, (384, ), (1, ))
    assert_size_stride(primals_112, (384, ), (1, ))
    assert_size_stride(primals_117, (384, ), (1, ))
    assert_size_stride(primals_123, (384, ), (1, ))
    assert_size_stride(primals_128, (384, ), (1, ))
    assert_size_stride(primals_134, (384, ), (1, ))
    assert_size_stride(primals_139, (384, ), (1, ))
    assert_size_stride(primals_145, (384, ), (1, ))
    assert_size_stride(primals_150, (384, ), (1, ))
    assert_size_stride(primals_156, (384, ), (1, ))
    assert_size_stride(primals_161, (384, ), (1, ))
    assert_size_stride(primals_167, (384, ), (1, ))
    assert_size_stride(primals_172, (384, ), (1, ))
    assert_size_stride(primals_178, (384, ), (1, ))
    assert_size_stride(primals_183, (384, ), (1, ))
    assert_size_stride(primals_189, (384, ), (1, ))
    assert_size_stride(primals_194, (384, ), (1, ))
    assert_size_stride(primals_200, (384, ), (1, ))
    assert_size_stride(primals_205, (384, ), (1, ))
    assert_size_stride(primals_211, (384, ), (1, ))
    assert_size_stride(primals_216, (384, ), (1, ))
    assert_size_stride(primals_222, (384, ), (1, ))
    assert_size_stride(primals_228, (384, ), (1, ))
    assert_size_stride(primals_234, (384, ), (1, ))
    assert_size_stride(primals_240, (384, ), (1, ))
    assert_size_stride(primals_246, (384, ), (1, ))
    assert_size_stride(primals_261, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(convolution, (8, 64, 112, 112), (802816, 12544, 112, 1))
    assert_size_stride(squeeze_1, (64, ), (1, ))
    assert_size_stride(relu, (8, 64, 112, 112), (802816, 12544, 112, 1))
    assert_size_stride(convolution_1, (8, 64, 112, 112), (802816, 12544, 112, 1))
    assert_size_stride(squeeze_4, (64, ), (1, ))
    assert_size_stride(relu_1, (8, 64, 112, 112), (802816, 12544, 112, 1))
    assert_size_stride(convolution_2, (8, 64, 112, 112), (802816, 12544, 112, 1))
    assert_size_stride(squeeze_7, (64, ), (1, ))
    assert_size_stride(relu_2, (8, 64, 112, 112), (802816, 12544, 112, 1))
    assert_size_stride(mul_21, (8, 28, 28, 192), (150528, 5376, 192, 1))
    assert_size_stride(view, (6272, 192), (192, 1))
    assert_size_stride(add_17, (3, 14), (14, 1))
    assert_size_stride(unsqueeze_17, (3, 14, 1, 1), (14, 1, 1, 1))
    assert_size_stride(permute_5, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(view_4, (1568, 192), (192, 1))
    assert_size_stride(full_default, (8, 192, 30, 30), (172800, 900, 30, 1))
    assert_size_stride(view_12, (6272, 192), (192, 1))
    assert_size_stride(mul_24, (8, 28, 28, 192), (150528, 5376, 192, 1))
    assert_size_stride(view_14, (6272, 192), (192, 1))
    assert_size_stride(addmm_1, (6272, 576), (576, 1))
    assert_size_stride(view_16, (6272, 576), (576, 1))
    assert_size_stride(mul_29, (8, 28, 28, 192), (150528, 5376, 192, 1))
    assert_size_stride(view_18, (6272, 192), (192, 1))
    assert_size_stride(permute_19, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(view_22, (1568, 192), (192, 1))
    assert_size_stride(view_30, (6272, 192), (192, 1))
    assert_size_stride(mul_32, (8, 28, 28, 192), (150528, 5376, 192, 1))
    assert_size_stride(view_32, (6272, 192), (192, 1))
    assert_size_stride(addmm_4, (6272, 576), (576, 1))
    assert_size_stride(view_34, (6272, 576), (576, 1))
    assert_size_stride(mul_37, (8, 28, 28, 192), (150528, 5376, 192, 1))
    assert_size_stride(view_36, (6272, 192), (192, 1))
    assert_size_stride(permute_33, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(view_40, (1568, 192), (192, 1))
    assert_size_stride(view_48, (6272, 192), (192, 1))
    assert_size_stride(mul_40, (8, 28, 28, 192), (150528, 5376, 192, 1))
    assert_size_stride(view_50, (6272, 192), (192, 1))
    assert_size_stride(addmm_7, (6272, 576), (576, 1))
    assert_size_stride(view_52, (6272, 576), (576, 1))
    assert_size_stride(mul_45, (8, 28, 28, 192), (150528, 5376, 192, 1))
    assert_size_stride(view_54, (6272, 192), (192, 1))
    assert_size_stride(permute_47, (8, 192, 28, 28), (150528, 1, 5376, 192))
    assert_size_stride(view_58, (1568, 192), (192, 1))
    assert_size_stride(view_66, (6272, 192), (192, 1))
    assert_size_stride(mul_48, (8, 28, 28, 192), (150528, 5376, 192, 1))
    assert_size_stride(view_68, (6272, 192), (192, 1))
    assert_size_stride(addmm_10, (6272, 576), (576, 1))
    assert_size_stride(view_70, (6272, 576), (576, 1))
    assert_size_stride(permute_57, (8, 192, 28, 28), (150528, 784, 28, 1))
    assert_size_stride(mul_53, (8, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(view_72, (1568, 384), (384, 1))
    assert_size_stride(view_82, (1568, 384), (384, 1))
    assert_size_stride(mul_56, (8, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(view_84, (1568, 384), (384, 1))
    assert_size_stride(addmm_13, (1568, 1152), (1152, 1))
    assert_size_stride(view_86, (1568, 1152), (1152, 1))
    assert_size_stride(mul_61, (8, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(view_88, (1568, 384), (384, 1))
    assert_size_stride(view_98, (1568, 384), (384, 1))
    assert_size_stride(mul_64, (8, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(view_100, (1568, 384), (384, 1))
    assert_size_stride(addmm_16, (1568, 1152), (1152, 1))
    assert_size_stride(view_102, (1568, 1152), (1152, 1))
    assert_size_stride(mul_69, (8, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(view_104, (1568, 384), (384, 1))
    assert_size_stride(view_114, (1568, 384), (384, 1))
    assert_size_stride(mul_72, (8, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(view_116, (1568, 384), (384, 1))
    assert_size_stride(addmm_19, (1568, 1152), (1152, 1))
    assert_size_stride(view_118, (1568, 1152), (1152, 1))
    assert_size_stride(mul_77, (8, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(view_120, (1568, 384), (384, 1))
    assert_size_stride(view_130, (1568, 384), (384, 1))
    assert_size_stride(mul_80, (8, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(view_132, (1568, 384), (384, 1))
    assert_size_stride(addmm_22, (1568, 1152), (1152, 1))
    assert_size_stride(view_134, (1568, 1152), (1152, 1))
    assert_size_stride(mul_85, (8, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(view_136, (1568, 384), (384, 1))
    assert_size_stride(view_146, (1568, 384), (384, 1))
    assert_size_stride(mul_88, (8, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(view_148, (1568, 384), (384, 1))
    assert_size_stride(addmm_25, (1568, 1152), (1152, 1))
    assert_size_stride(view_150, (1568, 1152), (1152, 1))
    assert_size_stride(mul_93, (8, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(view_152, (1568, 384), (384, 1))
    assert_size_stride(view_162, (1568, 384), (384, 1))
    assert_size_stride(mul_96, (8, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(view_164, (1568, 384), (384, 1))
    assert_size_stride(addmm_28, (1568, 1152), (1152, 1))
    assert_size_stride(view_166, (1568, 1152), (1152, 1))
    assert_size_stride(mul_101, (8, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(view_168, (1568, 384), (384, 1))
    assert_size_stride(view_178, (1568, 384), (384, 1))
    assert_size_stride(mul_104, (8, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(view_180, (1568, 384), (384, 1))
    assert_size_stride(addmm_31, (1568, 1152), (1152, 1))
    assert_size_stride(view_182, (1568, 1152), (1152, 1))
    assert_size_stride(mul_109, (8, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(view_184, (1568, 384), (384, 1))
    assert_size_stride(view_194, (1568, 384), (384, 1))
    assert_size_stride(mul_112, (8, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(view_196, (1568, 384), (384, 1))
    assert_size_stride(addmm_34, (1568, 1152), (1152, 1))
    assert_size_stride(view_198, (1568, 1152), (1152, 1))
    assert_size_stride(mul_117, (8, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(view_200, (1568, 384), (384, 1))
    assert_size_stride(view_210, (1568, 384), (384, 1))
    assert_size_stride(mul_120, (8, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(view_212, (1568, 384), (384, 1))
    assert_size_stride(addmm_37, (1568, 1152), (1152, 1))
    assert_size_stride(view_214, (1568, 1152), (1152, 1))
    assert_size_stride(mul_125, (8, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(view_216, (1568, 384), (384, 1))
    assert_size_stride(view_226, (1568, 384), (384, 1))
    assert_size_stride(mul_128, (8, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(view_228, (1568, 384), (384, 1))
    assert_size_stride(addmm_40, (1568, 1152), (1152, 1))
    assert_size_stride(view_230, (1568, 1152), (1152, 1))
    assert_size_stride(mul_133, (8, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(view_232, (1568, 384), (384, 1))
    assert_size_stride(view_242, (1568, 384), (384, 1))
    assert_size_stride(mul_136, (8, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(view_244, (1568, 384), (384, 1))
    assert_size_stride(addmm_43, (1568, 1152), (1152, 1))
    assert_size_stride(view_246, (1568, 1152), (1152, 1))
    assert_size_stride(mul_141, (8, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(view_248, (1568, 384), (384, 1))
    assert_size_stride(view_258, (1568, 384), (384, 1))
    assert_size_stride(mul_144, (8, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(view_260, (1568, 384), (384, 1))
    assert_size_stride(addmm_46, (1568, 1152), (1152, 1))
    assert_size_stride(view_262, (1568, 1152), (1152, 1))
    assert_size_stride(mul_149, (8, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(view_264, (1568, 384), (384, 1))
    assert_size_stride(view_274, (1568, 384), (384, 1))
    assert_size_stride(mul_152, (8, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(view_276, (1568, 384), (384, 1))
    assert_size_stride(addmm_49, (1568, 1152), (1152, 1))
    assert_size_stride(view_278, (1568, 1152), (1152, 1))
    assert_size_stride(mul_157, (8, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(view_280, (1568, 384), (384, 1))
    assert_size_stride(view_290, (1568, 384), (384, 1))
    assert_size_stride(mul_160, (8, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(view_292, (1568, 384), (384, 1))
    assert_size_stride(addmm_52, (1568, 1152), (1152, 1))
    assert_size_stride(view_294, (1568, 1152), (1152, 1))
    assert_size_stride(cat, (8, 197, 384), (75648, 384, 1))
    assert_size_stride(getitem_121, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_39, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_297, (1576, 384), (384, 1))
    assert_size_stride(view_300, (8, 384), (75648, 1))
    assert_size_stride(view_310, (8, 384), (384, 1))
    assert_size_stride(mul_168, (8, 1, 384), (384, 384, 1))
    assert_size_stride(view_312, (8, 384), (384, 1))
    assert_size_stride(addmm_55, (8, 1152), (1152, 1))
    assert_size_stride(view_314, (8, 1152), (1152, 1))
    assert_size_stride(cat_1, (8, 197, 384), (75648, 384, 1))
    assert_size_stride(getitem_127, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_41, (8, 197, 1), (197, 1, 1))
    assert_size_stride(view_316, (1576, 384), (384, 1))
    assert_size_stride(view_319, (8, 384), (75648, 1))
    assert_size_stride(view_329, (8, 384), (384, 1))
    assert_size_stride(mul_176, (8, 1, 384), (384, 384, 1))
    assert_size_stride(view_331, (8, 384), (384, 1))
    assert_size_stride(addmm_58, (8, 1152), (1152, 1))
    assert_size_stride(view_333, (8, 1152), (1152, 1))
    assert_size_stride(cat_2, (8, 197, 384), (75648, 384, 1))
    assert_size_stride(getitem_133, (8, 197, 1), (197, 1, 1))
    assert_size_stride(rsqrt_43, (8, 197, 1), (197, 1, 1))
    assert_size_stride(select, (8, 384), (75648, 1))
    assert_size_stride(view_335, (1568, 384), (384, 1))
    assert_size_stride(unsqueeze_61, (8, 1, 1000), (1000, 1000, 1))
    assert_size_stride(permute_177, (1000, 384), (384, 1))
    assert_size_stride(permute_179, (1000, 384), (384, 1))
    assert_size_stride(permute_183, (384, 1152), (1152, 1))
    assert_size_stride(permute_187, (1152, 384), (384, 1))
    assert_size_stride(div_21, (8, 1, 1), (1, 1, 1))
    assert_size_stride(permute_191, (384, 384), (384, 1))
    assert_size_stride(permute_196, (96, 197, 1), (197, 1, 0))
    assert_size_stride(permute_197, (96, 32, 197), (6304, 1, 32))
    assert_size_stride(alias_23, (8, 12, 1, 197), (2364, 197, 197, 1))
    assert_size_stride(permute_198, (96, 32, 1), (32, 1, 0))
    assert_size_stride(permute_199, (96, 197, 32), (6304, 1, 197))
    assert_size_stride(permute_203, (384, 384), (384, 1))
    assert_size_stride(permute_208, (768, 384), (384, 1))
    assert_size_stride(permute_210, (384, 1152), (1152, 1))
    assert_size_stride(permute_214, (1152, 384), (384, 1))
    assert_size_stride(div_23, (8, 1, 1), (1, 1, 1))
    assert_size_stride(permute_218, (384, 384), (384, 1))
    assert_size_stride(permute_223, (96, 197, 1), (197, 1, 0))
    assert_size_stride(permute_224, (96, 32, 197), (6304, 1, 32))
    assert_size_stride(alias_24, (8, 12, 1, 197), (2364, 197, 197, 1))
    assert_size_stride(permute_225, (96, 32, 1), (32, 1, 0))
    assert_size_stride(permute_226, (96, 197, 32), (6304, 1, 197))
    assert_size_stride(permute_230, (384, 384), (384, 1))
    assert_size_stride(permute_235, (768, 384), (384, 1))
    assert_size_stride(permute_237, (384, 1152), (1152, 1))
    assert_size_stride(permute_241, (1152, 384), (384, 1))
    assert_size_stride(div_25, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_245, (384, 384), (384, 1))
    assert_size_stride(permute_250, (96, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_251, (96, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_25, (8, 12, 196, 196), (460992, 38416, 196, 1))
    assert_size_stride(permute_252, (96, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_253, (96, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_258, (1152, 384), (384, 1))
    assert_size_stride(div_26, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_260, (384, 1152), (1152, 1))
    assert_size_stride(permute_264, (1152, 384), (384, 1))
    assert_size_stride(div_27, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_268, (384, 384), (384, 1))
    assert_size_stride(permute_273, (96, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_274, (96, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_26, (8, 12, 196, 196), (460992, 38416, 196, 1))
    assert_size_stride(permute_275, (96, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_276, (96, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_281, (1152, 384), (384, 1))
    assert_size_stride(div_28, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_283, (384, 1152), (1152, 1))
    assert_size_stride(permute_287, (1152, 384), (384, 1))
    assert_size_stride(div_29, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_291, (384, 384), (384, 1))
    assert_size_stride(permute_296, (96, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_297, (96, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_27, (8, 12, 196, 196), (460992, 38416, 196, 1))
    assert_size_stride(permute_298, (96, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_299, (96, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_304, (1152, 384), (384, 1))
    assert_size_stride(div_30, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_306, (384, 1152), (1152, 1))
    assert_size_stride(permute_310, (1152, 384), (384, 1))
    assert_size_stride(div_31, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_314, (384, 384), (384, 1))
    assert_size_stride(permute_319, (96, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_320, (96, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_28, (8, 12, 196, 196), (460992, 38416, 196, 1))
    assert_size_stride(permute_321, (96, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_322, (96, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_327, (1152, 384), (384, 1))
    assert_size_stride(div_32, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_329, (384, 1152), (1152, 1))
    assert_size_stride(permute_333, (1152, 384), (384, 1))
    assert_size_stride(div_33, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_337, (384, 384), (384, 1))
    assert_size_stride(permute_342, (96, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_343, (96, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_29, (8, 12, 196, 196), (460992, 38416, 196, 1))
    assert_size_stride(permute_344, (96, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_345, (96, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_350, (1152, 384), (384, 1))
    assert_size_stride(div_34, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_352, (384, 1152), (1152, 1))
    assert_size_stride(permute_356, (1152, 384), (384, 1))
    assert_size_stride(div_35, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_360, (384, 384), (384, 1))
    assert_size_stride(permute_365, (96, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_366, (96, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_30, (8, 12, 196, 196), (460992, 38416, 196, 1))
    assert_size_stride(permute_367, (96, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_368, (96, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_373, (1152, 384), (384, 1))
    assert_size_stride(div_36, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_375, (384, 1152), (1152, 1))
    assert_size_stride(permute_379, (1152, 384), (384, 1))
    assert_size_stride(div_37, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_383, (384, 384), (384, 1))
    assert_size_stride(permute_388, (96, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_389, (96, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_31, (8, 12, 196, 196), (460992, 38416, 196, 1))
    assert_size_stride(permute_390, (96, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_391, (96, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_396, (1152, 384), (384, 1))
    assert_size_stride(div_38, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_398, (384, 1152), (1152, 1))
    assert_size_stride(permute_402, (1152, 384), (384, 1))
    assert_size_stride(div_39, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_406, (384, 384), (384, 1))
    assert_size_stride(permute_411, (96, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_412, (96, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_32, (8, 12, 196, 196), (460992, 38416, 196, 1))
    assert_size_stride(permute_413, (96, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_414, (96, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_419, (1152, 384), (384, 1))
    assert_size_stride(div_40, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_421, (384, 1152), (1152, 1))
    assert_size_stride(permute_425, (1152, 384), (384, 1))
    assert_size_stride(div_41, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_429, (384, 384), (384, 1))
    assert_size_stride(permute_434, (96, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_435, (96, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_33, (8, 12, 196, 196), (460992, 38416, 196, 1))
    assert_size_stride(permute_436, (96, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_437, (96, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_442, (1152, 384), (384, 1))
    assert_size_stride(div_42, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_444, (384, 1152), (1152, 1))
    assert_size_stride(permute_448, (1152, 384), (384, 1))
    assert_size_stride(div_43, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_452, (384, 384), (384, 1))
    assert_size_stride(permute_457, (96, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_458, (96, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_34, (8, 12, 196, 196), (460992, 38416, 196, 1))
    assert_size_stride(permute_459, (96, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_460, (96, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_465, (1152, 384), (384, 1))
    assert_size_stride(div_44, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_467, (384, 1152), (1152, 1))
    assert_size_stride(permute_471, (1152, 384), (384, 1))
    assert_size_stride(div_45, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_475, (384, 384), (384, 1))
    assert_size_stride(permute_480, (96, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_481, (96, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_35, (8, 12, 196, 196), (460992, 38416, 196, 1))
    assert_size_stride(permute_482, (96, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_483, (96, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_488, (1152, 384), (384, 1))
    assert_size_stride(div_46, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_490, (384, 1152), (1152, 1))
    assert_size_stride(permute_494, (1152, 384), (384, 1))
    assert_size_stride(div_47, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_498, (384, 384), (384, 1))
    assert_size_stride(permute_503, (96, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_504, (96, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_36, (8, 12, 196, 196), (460992, 38416, 196, 1))
    assert_size_stride(permute_505, (96, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_506, (96, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_511, (1152, 384), (384, 1))
    assert_size_stride(div_48, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_513, (384, 1152), (1152, 1))
    assert_size_stride(permute_517, (1152, 384), (384, 1))
    assert_size_stride(div_49, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_521, (384, 384), (384, 1))
    assert_size_stride(permute_526, (96, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_527, (96, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_37, (8, 12, 196, 196), (460992, 38416, 196, 1))
    assert_size_stride(permute_528, (96, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_529, (96, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_534, (1152, 384), (384, 1))
    assert_size_stride(div_50, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_536, (384, 1152), (1152, 1))
    assert_size_stride(permute_540, (1152, 384), (384, 1))
    assert_size_stride(div_51, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_544, (384, 384), (384, 1))
    assert_size_stride(permute_549, (96, 196, 196), (38416, 1, 196))
    assert_size_stride(permute_550, (96, 32, 196), (6272, 1, 32))
    assert_size_stride(alias_38, (8, 12, 196, 196), (460992, 38416, 196, 1))
    assert_size_stride(permute_551, (96, 32, 196), (6272, 1, 32))
    assert_size_stride(permute_552, (96, 196, 32), (6272, 1, 196))
    assert_size_stride(permute_557, (1152, 384), (384, 1))
    assert_size_stride(div_52, (8, 14, 14, 1), (196, 14, 1, 1))
    assert_size_stride(permute_561, (192, 576), (576, 1))
    assert_size_stride(permute_565, (576, 192), (192, 1))
    assert_size_stride(div_53, (8, 28, 28, 1), (784, 28, 1, 1))
    assert_size_stride(permute_571, (192, 192), (192, 1))
    assert_size_stride(permute_576, (9408, 9, 9), (81, 1, 9))
    assert_size_stride(permute_577, (9408, 32, 9), (288, 1, 32))
    assert_size_stride(alias_39, (8, 6, 196, 9, 9), (95256, 15876, 81, 9, 1))
    assert_size_stride(permute_579, (486, 192), (192, 1))
    assert_size_stride(permute_590, (192, 192), (192, 1))
    assert_size_stride(div_54, (8, 28, 28, 1), (784, 28, 1, 1))
    assert_size_stride(permute_592, (192, 576), (576, 1))
    assert_size_stride(permute_596, (576, 192), (192, 1))
    assert_size_stride(div_55, (8, 28, 28, 1), (784, 28, 1, 1))
    assert_size_stride(permute_602, (192, 192), (192, 1))
    assert_size_stride(permute_607, (9408, 9, 9), (81, 1, 9))
    assert_size_stride(permute_608, (9408, 32, 9), (288, 1, 32))
    assert_size_stride(alias_40, (8, 6, 196, 9, 9), (95256, 15876, 81, 9, 1))
    assert_size_stride(permute_610, (486, 192), (192, 1))
    assert_size_stride(permute_621, (192, 192), (192, 1))
    assert_size_stride(div_56, (8, 28, 28, 1), (784, 28, 1, 1))
    assert_size_stride(permute_623, (192, 576), (576, 1))
    assert_size_stride(permute_627, (576, 192), (192, 1))
    assert_size_stride(div_57, (8, 28, 28, 1), (784, 28, 1, 1))
    assert_size_stride(permute_633, (192, 192), (192, 1))
    assert_size_stride(permute_638, (9408, 9, 9), (81, 1, 9))
    assert_size_stride(permute_639, (9408, 32, 9), (288, 1, 32))
    assert_size_stride(alias_41, (8, 6, 196, 9, 9), (95256, 15876, 81, 9, 1))
    assert_size_stride(permute_641, (486, 192), (192, 1))
    assert_size_stride(permute_652, (192, 192), (192, 1))
    assert_size_stride(div_58, (8, 28, 28, 1), (784, 28, 1, 1))
    assert_size_stride(permute_654, (192, 576), (576, 1))
    assert_size_stride(permute_658, (576, 192), (192, 1))
    assert_size_stride(div_59, (8, 28, 28, 1), (784, 28, 1, 1))
    assert_size_stride(permute_664, (192, 192), (192, 1))
    assert_size_stride(permute_669, (9408, 9, 9), (81, 1, 9))
    assert_size_stride(permute_670, (9408, 32, 9), (288, 1, 32))
    assert_size_stride(alias_42, (8, 6, 196, 9, 9), (95256, 15876, 81, 9, 1))
    assert_size_stride(permute_672, (486, 192), (192, 1))
    assert_size_stride(permute_683, (192, 192), (192, 1))
    assert_size_stride(div_60, (8, 28, 28, 1), (784, 28, 1, 1))
    assert_size_stride(unsqueeze_112, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_124, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(unsqueeze_136, (1, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 196, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.scatter, aten.zeros]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_scatter_zeros_0.run(buf0, 1568000, grid=grid(1568000), stream=stream0)
        buf1 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mul]
        triton_poi_fused_mul_1.run(tangents_1, buf1, 8000, grid=grid(8000), stream=stream0)
        aten.scatter_(buf0,1,unsqueeze_61,reinterpret_tensor(buf1, (8, 1, 1000), (1000, 0, 1), 0))
        del buf1
        del unsqueeze_61
        buf4 = empty_strided((1, 1, 1000, 13), (13000, 13000, 1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_2.run(buf0, buf4, 13000, 121, grid=grid(13000), stream=stream0)
        buf5 = empty((1, 1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_3.run(buf4, buf5, 1000, 13, grid=grid(1000), stream=stream0)
        del buf4
        buf6 = empty((1000, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (1000, 1568), (1, 1000), 0), view_335, out=buf6)
        del view_335
        buf7 = empty((1568, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (1568, 1000), (1000, 1), 0), permute_177, out=buf7)
        del buf0
        del permute_177
        buf8 = empty((8, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, permute_179, out=buf8)
        del permute_179
        buf9 = empty((1000, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), select, out=buf9)
        del select
        buf10 = empty((1, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_4.run(tangents_1, buf10, 1000, 8, grid=grid(1000), stream=stream0)
        del tangents_1
        buf13 = empty((8, 197, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_234], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_5.run(buf7, buf8, primals_246, cat_2, getitem_133, rsqrt_43, buf13, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_246
        buf14 = empty_strided((384, 13), (1, 384), device='cuda', dtype=torch.float32)
        buf16 = empty_strided((384, 13), (1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_234], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_6.run(buf7, buf8, cat_2, getitem_133, rsqrt_43, buf14, buf16, 4992, 122, grid=grid(4992), stream=stream0)
        del cat_2
        del getitem_133
        del rsqrt_43
        buf15 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_234], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf14, buf15, 384, 13, grid=grid(384), stream=stream0)
        buf17 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf16, buf17, 384, 13, grid=grid(384), stream=stream0)
        buf18 = empty((8, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (8, 384), (75648, 1), 0), permute_183, out=buf18)
        del permute_183
        buf19 = empty((384, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (384, 8), (1, 75648), 0), view_333, out=buf19)
        del view_333
        buf20 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf13, buf20, 384, 8, grid=grid(384), stream=stream0)
        buf21 = reinterpret_tensor(buf18, (8, 1, 1152), (1152, 1152, 1), 0); del buf18  # reuse
        # Source Nodes: [x_226], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf21, addmm_58, 9216, grid=grid(9216), stream=stream0)
        del addmm_58
        buf22 = buf8; del buf8  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf21, (8, 1152), (1152, 1), 0), permute_187, out=buf22)
        del permute_187
        buf23 = empty((1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf21, (1152, 8), (1, 1152), 0), view_331, out=buf23)
        del view_331
        buf24 = empty((1, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf21, buf24, 1152, 8, grid=grid(1152), stream=stream0)
        buf29 = empty((8, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_11.run(buf22, primals_240, mul_176, buf13, div_21, buf29, 8, 384, grid=grid(8), stream=stream0)
        del div_21
        del primals_240
        buf27 = empty((384, ), device='cuda', dtype=torch.float32)
        buf28 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_12.run(buf22, mul_176, buf27, buf28, 384, 8, grid=grid(384), stream=stream0)
        del mul_176
        buf30 = buf22; del buf22  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf29, (8, 384), (384, 1), 0), permute_191, out=buf30)
        del permute_191
        buf31 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf29, (384, 8), (1, 384), 0), view_329, out=buf31)
        del view_329
        buf32 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_13.run(buf29, buf32, 384, 8, grid=grid(384), stream=stream0)
        buf33 = empty((96, 197, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_196, reinterpret_tensor(buf30, (96, 1, 32), (32, 32, 1), 0), out=buf33)
        del permute_196
        buf34 = empty((96, 1, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf30, (96, 1, 32), (32, 32, 1), 0), permute_197, out=buf34)
        del permute_197
        buf36 = empty((8, 12, 1, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_14.run(buf34, alias_23, buf36, 96, 197, grid=grid(96), stream=stream0)
        del alias_23
        buf37 = empty((96, 32, 197), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_198, reinterpret_tensor(buf36, (96, 1, 197), (197, 0, 1), 0), out=buf37)
        del permute_198
        buf38 = reinterpret_tensor(buf30, (96, 1, 32), (32, 32, 1), 0); del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf36, (96, 1, 197), (197, 0, 1), 0), permute_199, out=buf38)
        del permute_199
        buf39 = reinterpret_tensor(buf38, (8, 384), (384, 1), 0); del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf39, 3072, grid=grid(3072), stream=stream0)
        buf40 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (384, 8), (1, 384), 0), view_319, out=buf40)
        del view_319
        buf41 = empty((8, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf39, permute_203, out=buf41)
        del permute_203
        buf42 = empty((8, 197, 2, 12, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf37, buf33, buf42, 37824, 32, grid=grid(37824, 32), stream=stream0)
        buf43 = empty((768, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf42, (768, 1576), (1, 768), 0), view_316, out=buf43)
        del view_316
        buf44 = reinterpret_tensor(buf37, (1576, 384), (384, 1), 0); del buf37  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf42, (1576, 768), (768, 1), 0), permute_208, out=buf44)
        del permute_208
        buf51 = buf13; del buf13  # reuse
        buf52 = reinterpret_tensor(buf33, (8, 197, 384), (75648, 384, 1), 0); del buf33  # reuse
        # Source Nodes: [l__mod___post_network_1_norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.slice_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_slice_backward_17.run(buf51, buf41, buf44, primals_234, cat_1, getitem_127, rsqrt_41, buf29, buf52, 1576, 384, grid=grid(1576), stream=stream0)
        del primals_234
        buf47 = buf16; del buf16  # reuse
        buf49 = buf14; del buf14  # reuse
        # Source Nodes: [l__mod___post_network_1_norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.slice_backward]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_slice_backward_18.run(buf41, buf44, cat_1, getitem_127, rsqrt_41, buf47, buf49, 4992, 122, grid=grid(4992), stream=stream0)
        del cat_1
        del getitem_127
        del rsqrt_41
        buf48 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___post_network_1_norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.slice_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf47, buf48, 384, 13, grid=grid(384), stream=stream0)
        buf50 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.slice_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf49, buf50, 384, 13, grid=grid(384), stream=stream0)
        buf53 = reinterpret_tensor(buf21, (8, 1152), (1152, 1), 0); del buf21  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf52, (8, 384), (75648, 1), 0), permute_210, out=buf53)
        del permute_210
        buf54 = empty((384, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf52, (384, 8), (1, 75648), 0), view_314, out=buf54)
        del view_314
        buf55 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_8.run(buf52, buf55, 384, 8, grid=grid(384), stream=stream0)
        buf56 = reinterpret_tensor(buf53, (8, 1, 1152), (1152, 1152, 1), 0); del buf53  # reuse
        # Source Nodes: [x_219], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_9.run(buf56, addmm_55, 9216, grid=grid(9216), stream=stream0)
        del addmm_55
        buf57 = buf41; del buf41  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf56, (8, 1152), (1152, 1), 0), permute_214, out=buf57)
        del permute_214
        buf58 = empty((1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf56, (1152, 8), (1, 1152), 0), view_312, out=buf58)
        del view_312
        buf59 = empty((1, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_10.run(buf56, buf59, 1152, 8, grid=grid(1152), stream=stream0)
        del buf56
        buf64 = reinterpret_tensor(buf39, (8, 1, 384), (384, 384, 1), 0); del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_11.run(buf57, primals_228, mul_168, buf52, div_23, buf64, 8, 384, grid=grid(8), stream=stream0)
        del div_23
        del primals_228
        buf62 = empty((384, ), device='cuda', dtype=torch.float32)
        buf63 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_native_layer_norm_backward_12.run(buf57, mul_168, buf62, buf63, 384, 8, grid=grid(384), stream=stream0)
        del mul_168
        buf65 = buf57; del buf57  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (8, 384), (384, 1), 0), permute_218, out=buf65)
        del permute_218
        buf66 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf64, (384, 8), (1, 384), 0), view_310, out=buf66)
        del view_310
        buf69 = reinterpret_tensor(buf36, (96, 1, 197), (197, 197, 1), 0); del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf65, (96, 1, 32), (32, 32, 1), 0), permute_224, out=buf69)
        del permute_224
        buf71 = reinterpret_tensor(buf34, (8, 12, 1, 197), (2364, 197, 197, 1), 0); del buf34  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data]
        triton_per_fused__softmax_backward_data_14.run(buf69, alias_24, buf71, 96, 197, grid=grid(96), stream=stream0)
        del alias_24
        del buf69
        buf73 = empty((96, 1, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf71, (96, 1, 197), (197, 0, 1), 0), permute_226, out=buf73)
        del permute_226
        buf74 = reinterpret_tensor(buf73, (8, 384), (384, 1), 0); del buf73  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_15.run(buf74, 3072, grid=grid(3072), stream=stream0)
        buf76 = empty((8, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf74, permute_230, out=buf76)
        del permute_230
        buf68 = reinterpret_tensor(buf52, (96, 197, 32), (6304, 32, 1), 0); del buf52  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_223, reinterpret_tensor(buf65, (96, 1, 32), (32, 32, 1), 0), out=buf68)
        del buf65
        del permute_223
        buf72 = reinterpret_tensor(buf44, (96, 32, 197), (6304, 197, 1), 0); del buf44  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_225, reinterpret_tensor(buf71, (96, 1, 197), (197, 0, 1), 0), out=buf72)
        del buf71
        del permute_225
        buf77 = buf42; del buf42  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf72, buf68, buf77, 37824, 32, grid=grid(37824, 32), stream=stream0)
        del buf68
        buf79 = reinterpret_tensor(buf72, (1576, 384), (384, 1), 0); del buf72  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf77, (1576, 768), (768, 1), 0), permute_235, out=buf79)
        del permute_235
        buf86 = buf51; del buf51  # reuse
        # Source Nodes: [l__mod___post_network_0_norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.slice_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_slice_backward_19.run(buf86, buf76, buf79, primals_222, cat, getitem_121, rsqrt_39, buf29, 1576, 384, grid=grid(1576), stream=stream0)
        del buf29
        del primals_222
        buf67 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf87 = empty((1, 1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_20.run(buf64, buf86, buf67, buf87, 384, 8, grid=grid(384), stream=stream0)
        buf75 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf74, (384, 8), (1, 384), 0), view_300, out=buf75)
        del buf74
        del view_300
        buf78 = empty((768, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf77, (768, 1576), (1, 768), 0), view_297, out=buf78)
        del buf77
        del view_297
        buf82 = buf49; del buf49  # reuse
        buf84 = buf47; del buf47  # reuse
        # Source Nodes: [l__mod___post_network_0_norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.slice_backward]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_slice_backward_18.run(buf76, buf79, cat, getitem_121, rsqrt_39, buf82, buf84, 4992, 122, grid=grid(4992), stream=stream0)
        del buf76
        del buf79
        del cat
        del getitem_121
        del rsqrt_39
        buf83 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___post_network_0_norm1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.slice_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf82, buf83, 384, 13, grid=grid(384), stream=stream0)
        buf85 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.slice_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf84, buf85, 384, 13, grid=grid(384), stream=stream0)
        buf88 = reinterpret_tensor(buf7, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf7  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf86, buf64, buf88, 602112, grid=grid(602112), stream=stream0)
        buf89 = empty((1568, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf88, (1568, 384), (384, 1), 0), permute_237, out=buf89)
        del permute_237
        buf90 = empty((384, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf88, (384, 1568), (1, 384), 0), view_294, out=buf90)
        del view_294
        buf91 = reinterpret_tensor(buf84, (1, 384, 13), (4992, 1, 384), 0); del buf84  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_22.run(buf88, buf91, 4992, 121, grid=grid(4992), stream=stream0)
        buf92 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf91, buf92, 384, 13, grid=grid(384), stream=stream0)
        buf93 = reinterpret_tensor(buf89, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf89  # reuse
        # Source Nodes: [x_208], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_23.run(buf93, addmm_52, 1806336, grid=grid(1806336), stream=stream0)
        del addmm_52
        buf94 = reinterpret_tensor(buf88, (1568, 384), (384, 1), 0); del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf93, (1568, 1152), (1152, 1), 0), permute_241, out=buf94)
        del permute_241
        buf95 = empty((1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf93, (1152, 1568), (1, 1152), 0), view_292, out=buf95)
        del view_292
        buf96 = empty_strided((1, 1152, 13), (14976, 1, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf93, buf96, 14976, 121, grid=grid(14976), stream=stream0)
        buf97 = empty((1, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_25.run(buf96, buf97, 1152, 13, grid=grid(1152), stream=stream0)
        buf104 = empty((8, 14, 14, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_26.run(buf94, primals_216, mul_160, buf86, buf64, div_25, buf104, 1568, 384, grid=grid(1568), stream=stream0)
        del buf64
        del buf86
        del div_25
        del primals_216
        buf100 = reinterpret_tensor(buf91, (384, 13), (1, 384), 0); del buf91  # reuse
        buf102 = buf82; del buf82  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_27.run(buf94, mul_160, buf100, buf102, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_160
        buf101 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf100, buf101, 384, 13, grid=grid(384), stream=stream0)
        buf103 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf102, buf103, 384, 13, grid=grid(384), stream=stream0)
        buf105 = buf94; del buf94  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf104, (1568, 384), (384, 1), 0), permute_245, out=buf105)
        del permute_245
        buf106 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf104, (384, 1568), (1, 384), 0), view_290, out=buf106)
        del view_290
        buf107 = reinterpret_tensor(buf102, (1, 384, 13), (4992, 1, 384), 0); del buf102  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf104, buf107, 4992, 121, grid=grid(4992), stream=stream0)
        buf108 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf107, buf108, 384, 13, grid=grid(384), stream=stream0)
        buf109 = empty((8, 12, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf105, buf109, 602112, grid=grid(602112), stream=stream0)
        buf110 = reinterpret_tensor(buf105, (96, 196, 32), (6272, 32, 1), 0); del buf105  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_250, reinterpret_tensor(buf109, (96, 196, 32), (6272, 32, 1), 0), out=buf110)
        del permute_250
        buf111 = empty((96, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf109, (96, 196, 32), (6272, 32, 1), 0), permute_251, out=buf111)
        del permute_251
        buf113 = empty((8, 12, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_30.run(buf111, alias_25, buf113, 18816, 196, grid=grid(18816), stream=stream0)
        del alias_25
        buf114 = reinterpret_tensor(buf109, (96, 32, 196), (6272, 196, 1), 0); del buf109  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_252, reinterpret_tensor(buf113, (96, 196, 196), (38416, 196, 1), 0), out=buf114)
        del permute_252
        buf115 = empty((96, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf113, (96, 196, 196), (38416, 196, 1), 0), permute_253, out=buf115)
        del permute_253
        buf116 = reinterpret_tensor(buf93, (8, 196, 3, 12, 32), (225792, 1152, 384, 32, 1), 0); del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf115, buf114, buf110, buf116, 56448, 32, grid=grid(56448, 32), stream=stream0)
        buf117 = empty((1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (1152, 1568), (1, 1152), 0), view_280, out=buf117)
        del view_280
        buf118 = reinterpret_tensor(buf115, (1568, 384), (384, 1), 0); del buf115  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (1568, 1152), (1152, 1), 0), permute_258, out=buf118)
        del permute_258
        buf125 = buf104; del buf104  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf125, buf118, primals_211, mul_157, div_26, 1568, 384, grid=grid(1568), stream=stream0)
        del div_26
        del primals_211
        buf121 = reinterpret_tensor(buf107, (384, 13), (1, 384), 0); del buf107  # reuse
        buf123 = buf100; del buf100  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_27.run(buf118, mul_157, buf121, buf123, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_157
        buf122 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf121, buf122, 384, 13, grid=grid(384), stream=stream0)
        buf124 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf123, buf124, 384, 13, grid=grid(384), stream=stream0)
        buf126 = reinterpret_tensor(buf116, (1568, 1152), (1152, 1), 0); del buf116  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (1568, 384), (384, 1), 0), permute_260, out=buf126)
        del permute_260
        buf127 = empty((384, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf125, (384, 1568), (1, 384), 0), view_278, out=buf127)
        del view_278
        buf128 = reinterpret_tensor(buf123, (1, 384, 13), (4992, 1, 384), 0); del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf125, buf128, 4992, 121, grid=grid(4992), stream=stream0)
        buf129 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf128, buf129, 384, 13, grid=grid(384), stream=stream0)
        buf130 = reinterpret_tensor(buf126, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf126  # reuse
        # Source Nodes: [x_197], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_23.run(buf130, addmm_49, 1806336, grid=grid(1806336), stream=stream0)
        del addmm_49
        buf131 = buf118; del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf130, (1568, 1152), (1152, 1), 0), permute_264, out=buf131)
        del permute_264
        buf132 = empty((1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf130, (1152, 1568), (1, 1152), 0), view_276, out=buf132)
        del view_276
        buf133 = buf96; del buf96  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf130, buf133, 14976, 121, grid=grid(14976), stream=stream0)
        buf134 = empty((1, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_25.run(buf133, buf134, 1152, 13, grid=grid(1152), stream=stream0)
        buf141 = buf125; del buf125  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf141, buf131, primals_205, mul_152, div_27, 1568, 384, grid=grid(1568), stream=stream0)
        del div_27
        del primals_205
        buf137 = reinterpret_tensor(buf128, (384, 13), (1, 384), 0); del buf128  # reuse
        buf139 = buf121; del buf121  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_27.run(buf131, mul_152, buf137, buf139, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_152
        buf138 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf137, buf138, 384, 13, grid=grid(384), stream=stream0)
        buf140 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf139, buf140, 384, 13, grid=grid(384), stream=stream0)
        buf142 = buf131; del buf131  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf141, (1568, 384), (384, 1), 0), permute_268, out=buf142)
        del permute_268
        buf143 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf141, (384, 1568), (1, 384), 0), view_274, out=buf143)
        del view_274
        buf144 = reinterpret_tensor(buf139, (1, 384, 13), (4992, 1, 384), 0); del buf139  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf141, buf144, 4992, 121, grid=grid(4992), stream=stream0)
        buf145 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf144, buf145, 384, 13, grid=grid(384), stream=stream0)
        buf146 = reinterpret_tensor(buf114, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf114  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf142, buf146, 602112, grid=grid(602112), stream=stream0)
        buf147 = reinterpret_tensor(buf142, (96, 196, 32), (6272, 32, 1), 0); del buf142  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_273, reinterpret_tensor(buf146, (96, 196, 32), (6272, 32, 1), 0), out=buf147)
        del permute_273
        buf148 = reinterpret_tensor(buf113, (96, 196, 196), (38416, 196, 1), 0); del buf113  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf146, (96, 196, 32), (6272, 32, 1), 0), permute_274, out=buf148)
        del permute_274
        buf150 = reinterpret_tensor(buf111, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf111  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_30.run(buf148, alias_26, buf150, 18816, 196, grid=grid(18816), stream=stream0)
        del alias_26
        buf151 = reinterpret_tensor(buf146, (96, 32, 196), (6272, 196, 1), 0); del buf146  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_275, reinterpret_tensor(buf150, (96, 196, 196), (38416, 196, 1), 0), out=buf151)
        del permute_275
        buf152 = buf110; del buf110  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf150, (96, 196, 196), (38416, 196, 1), 0), permute_276, out=buf152)
        del permute_276
        buf153 = reinterpret_tensor(buf130, (8, 196, 3, 12, 32), (225792, 1152, 384, 32, 1), 0); del buf130  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf152, buf151, buf147, buf153, 56448, 32, grid=grid(56448, 32), stream=stream0)
        buf154 = empty((1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf153, (1152, 1568), (1, 1152), 0), view_264, out=buf154)
        del view_264
        buf155 = reinterpret_tensor(buf152, (1568, 384), (384, 1), 0); del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf153, (1568, 1152), (1152, 1), 0), permute_281, out=buf155)
        del permute_281
        buf162 = buf141; del buf141  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf162, buf155, primals_200, mul_149, div_28, 1568, 384, grid=grid(1568), stream=stream0)
        del div_28
        del primals_200
        buf158 = reinterpret_tensor(buf144, (384, 13), (1, 384), 0); del buf144  # reuse
        buf160 = buf137; del buf137  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_27.run(buf155, mul_149, buf158, buf160, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_149
        buf159 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf158, buf159, 384, 13, grid=grid(384), stream=stream0)
        buf161 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf160, buf161, 384, 13, grid=grid(384), stream=stream0)
        buf163 = reinterpret_tensor(buf153, (1568, 1152), (1152, 1), 0); del buf153  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf162, (1568, 384), (384, 1), 0), permute_283, out=buf163)
        del permute_283
        buf164 = empty((384, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf162, (384, 1568), (1, 384), 0), view_262, out=buf164)
        del view_262
        buf165 = reinterpret_tensor(buf160, (1, 384, 13), (4992, 1, 384), 0); del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf162, buf165, 4992, 121, grid=grid(4992), stream=stream0)
        buf166 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf165, buf166, 384, 13, grid=grid(384), stream=stream0)
        buf167 = reinterpret_tensor(buf163, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf163  # reuse
        # Source Nodes: [x_185], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_23.run(buf167, addmm_46, 1806336, grid=grid(1806336), stream=stream0)
        del addmm_46
        buf168 = buf155; del buf155  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf167, (1568, 1152), (1152, 1), 0), permute_287, out=buf168)
        del permute_287
        buf169 = empty((1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf167, (1152, 1568), (1, 1152), 0), view_260, out=buf169)
        del view_260
        buf170 = buf133; del buf133  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf167, buf170, 14976, 121, grid=grid(14976), stream=stream0)
        buf171 = empty((1, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_25.run(buf170, buf171, 1152, 13, grid=grid(1152), stream=stream0)
        buf178 = buf162; del buf162  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf178, buf168, primals_194, mul_144, div_29, 1568, 384, grid=grid(1568), stream=stream0)
        del div_29
        del primals_194
        buf174 = reinterpret_tensor(buf165, (384, 13), (1, 384), 0); del buf165  # reuse
        buf176 = buf158; del buf158  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_27.run(buf168, mul_144, buf174, buf176, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_144
        buf175 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf174, buf175, 384, 13, grid=grid(384), stream=stream0)
        buf177 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf176, buf177, 384, 13, grid=grid(384), stream=stream0)
        buf179 = buf168; del buf168  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf178, (1568, 384), (384, 1), 0), permute_291, out=buf179)
        del permute_291
        buf180 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf178, (384, 1568), (1, 384), 0), view_258, out=buf180)
        del view_258
        buf181 = reinterpret_tensor(buf176, (1, 384, 13), (4992, 1, 384), 0); del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf178, buf181, 4992, 121, grid=grid(4992), stream=stream0)
        buf182 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf181, buf182, 384, 13, grid=grid(384), stream=stream0)
        buf183 = reinterpret_tensor(buf151, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf151  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf179, buf183, 602112, grid=grid(602112), stream=stream0)
        buf184 = reinterpret_tensor(buf179, (96, 196, 32), (6272, 32, 1), 0); del buf179  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_296, reinterpret_tensor(buf183, (96, 196, 32), (6272, 32, 1), 0), out=buf184)
        del permute_296
        buf185 = reinterpret_tensor(buf150, (96, 196, 196), (38416, 196, 1), 0); del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf183, (96, 196, 32), (6272, 32, 1), 0), permute_297, out=buf185)
        del permute_297
        buf187 = reinterpret_tensor(buf148, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf148  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_30.run(buf185, alias_27, buf187, 18816, 196, grid=grid(18816), stream=stream0)
        del alias_27
        buf188 = reinterpret_tensor(buf183, (96, 32, 196), (6272, 196, 1), 0); del buf183  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_298, reinterpret_tensor(buf187, (96, 196, 196), (38416, 196, 1), 0), out=buf188)
        del permute_298
        buf189 = buf147; del buf147  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf187, (96, 196, 196), (38416, 196, 1), 0), permute_299, out=buf189)
        del permute_299
        buf190 = reinterpret_tensor(buf167, (8, 196, 3, 12, 32), (225792, 1152, 384, 32, 1), 0); del buf167  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf189, buf188, buf184, buf190, 56448, 32, grid=grid(56448, 32), stream=stream0)
        buf191 = empty((1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf190, (1152, 1568), (1, 1152), 0), view_248, out=buf191)
        del view_248
        buf192 = reinterpret_tensor(buf189, (1568, 384), (384, 1), 0); del buf189  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf190, (1568, 1152), (1152, 1), 0), permute_304, out=buf192)
        del permute_304
        buf199 = buf178; del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf199, buf192, primals_189, mul_141, div_30, 1568, 384, grid=grid(1568), stream=stream0)
        del div_30
        del primals_189
        buf195 = reinterpret_tensor(buf181, (384, 13), (1, 384), 0); del buf181  # reuse
        buf197 = buf174; del buf174  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_27.run(buf192, mul_141, buf195, buf197, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_141
        buf196 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf195, buf196, 384, 13, grid=grid(384), stream=stream0)
        buf198 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf197, buf198, 384, 13, grid=grid(384), stream=stream0)
        buf200 = reinterpret_tensor(buf190, (1568, 1152), (1152, 1), 0); del buf190  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf199, (1568, 384), (384, 1), 0), permute_306, out=buf200)
        del permute_306
        buf201 = empty((384, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf199, (384, 1568), (1, 384), 0), view_246, out=buf201)
        del view_246
        buf202 = reinterpret_tensor(buf197, (1, 384, 13), (4992, 1, 384), 0); del buf197  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf199, buf202, 4992, 121, grid=grid(4992), stream=stream0)
        buf203 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf202, buf203, 384, 13, grid=grid(384), stream=stream0)
        buf204 = reinterpret_tensor(buf200, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf200  # reuse
        # Source Nodes: [x_174], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_23.run(buf204, addmm_43, 1806336, grid=grid(1806336), stream=stream0)
        del addmm_43
        buf205 = buf192; del buf192  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf204, (1568, 1152), (1152, 1), 0), permute_310, out=buf205)
        del permute_310
        buf206 = empty((1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf204, (1152, 1568), (1, 1152), 0), view_244, out=buf206)
        del view_244
        buf207 = buf170; del buf170  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf204, buf207, 14976, 121, grid=grid(14976), stream=stream0)
        buf208 = empty((1, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_25.run(buf207, buf208, 1152, 13, grid=grid(1152), stream=stream0)
        buf215 = buf199; del buf199  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf215, buf205, primals_183, mul_136, div_31, 1568, 384, grid=grid(1568), stream=stream0)
        del div_31
        del primals_183
        buf211 = reinterpret_tensor(buf202, (384, 13), (1, 384), 0); del buf202  # reuse
        buf213 = buf195; del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_27.run(buf205, mul_136, buf211, buf213, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_136
        buf212 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf211, buf212, 384, 13, grid=grid(384), stream=stream0)
        buf214 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf213, buf214, 384, 13, grid=grid(384), stream=stream0)
        buf216 = buf205; del buf205  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf215, (1568, 384), (384, 1), 0), permute_314, out=buf216)
        del permute_314
        buf217 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf215, (384, 1568), (1, 384), 0), view_242, out=buf217)
        del view_242
        buf218 = reinterpret_tensor(buf213, (1, 384, 13), (4992, 1, 384), 0); del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf215, buf218, 4992, 121, grid=grid(4992), stream=stream0)
        buf219 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf218, buf219, 384, 13, grid=grid(384), stream=stream0)
        buf220 = reinterpret_tensor(buf188, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf188  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf216, buf220, 602112, grid=grid(602112), stream=stream0)
        buf221 = reinterpret_tensor(buf216, (96, 196, 32), (6272, 32, 1), 0); del buf216  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_319, reinterpret_tensor(buf220, (96, 196, 32), (6272, 32, 1), 0), out=buf221)
        del permute_319
        buf222 = reinterpret_tensor(buf187, (96, 196, 196), (38416, 196, 1), 0); del buf187  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf220, (96, 196, 32), (6272, 32, 1), 0), permute_320, out=buf222)
        del permute_320
        buf224 = reinterpret_tensor(buf185, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf185  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_30.run(buf222, alias_28, buf224, 18816, 196, grid=grid(18816), stream=stream0)
        del alias_28
        buf225 = reinterpret_tensor(buf220, (96, 32, 196), (6272, 196, 1), 0); del buf220  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_321, reinterpret_tensor(buf224, (96, 196, 196), (38416, 196, 1), 0), out=buf225)
        del permute_321
        buf226 = buf184; del buf184  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf224, (96, 196, 196), (38416, 196, 1), 0), permute_322, out=buf226)
        del permute_322
        buf227 = reinterpret_tensor(buf204, (8, 196, 3, 12, 32), (225792, 1152, 384, 32, 1), 0); del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf226, buf225, buf221, buf227, 56448, 32, grid=grid(56448, 32), stream=stream0)
        buf228 = empty((1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf227, (1152, 1568), (1, 1152), 0), view_232, out=buf228)
        del view_232
        buf229 = reinterpret_tensor(buf226, (1568, 384), (384, 1), 0); del buf226  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf227, (1568, 1152), (1152, 1), 0), permute_327, out=buf229)
        del permute_327
        buf236 = buf215; del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf236, buf229, primals_178, mul_133, div_32, 1568, 384, grid=grid(1568), stream=stream0)
        del div_32
        del primals_178
        buf232 = reinterpret_tensor(buf218, (384, 13), (1, 384), 0); del buf218  # reuse
        buf234 = buf211; del buf211  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_27.run(buf229, mul_133, buf232, buf234, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_133
        buf233 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf232, buf233, 384, 13, grid=grid(384), stream=stream0)
        buf235 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf234, buf235, 384, 13, grid=grid(384), stream=stream0)
        buf237 = reinterpret_tensor(buf227, (1568, 1152), (1152, 1), 0); del buf227  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf236, (1568, 384), (384, 1), 0), permute_329, out=buf237)
        del permute_329
        buf238 = empty((384, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf236, (384, 1568), (1, 384), 0), view_230, out=buf238)
        del view_230
        buf239 = reinterpret_tensor(buf234, (1, 384, 13), (4992, 1, 384), 0); del buf234  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf236, buf239, 4992, 121, grid=grid(4992), stream=stream0)
        buf240 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf239, buf240, 384, 13, grid=grid(384), stream=stream0)
        buf241 = reinterpret_tensor(buf237, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf237  # reuse
        # Source Nodes: [x_163], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_23.run(buf241, addmm_40, 1806336, grid=grid(1806336), stream=stream0)
        del addmm_40
        buf242 = buf229; del buf229  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf241, (1568, 1152), (1152, 1), 0), permute_333, out=buf242)
        del permute_333
        buf243 = empty((1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf241, (1152, 1568), (1, 1152), 0), view_228, out=buf243)
        del view_228
        buf244 = buf207; del buf207  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf241, buf244, 14976, 121, grid=grid(14976), stream=stream0)
        buf245 = empty((1, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_25.run(buf244, buf245, 1152, 13, grid=grid(1152), stream=stream0)
        buf252 = buf236; del buf236  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf252, buf242, primals_172, mul_128, div_33, 1568, 384, grid=grid(1568), stream=stream0)
        del div_33
        del primals_172
        buf248 = reinterpret_tensor(buf239, (384, 13), (1, 384), 0); del buf239  # reuse
        buf250 = buf232; del buf232  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_27.run(buf242, mul_128, buf248, buf250, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_128
        buf249 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf248, buf249, 384, 13, grid=grid(384), stream=stream0)
        buf251 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf250, buf251, 384, 13, grid=grid(384), stream=stream0)
        buf253 = buf242; del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf252, (1568, 384), (384, 1), 0), permute_337, out=buf253)
        del permute_337
        buf254 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf252, (384, 1568), (1, 384), 0), view_226, out=buf254)
        del view_226
        buf255 = reinterpret_tensor(buf250, (1, 384, 13), (4992, 1, 384), 0); del buf250  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf252, buf255, 4992, 121, grid=grid(4992), stream=stream0)
        buf256 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf255, buf256, 384, 13, grid=grid(384), stream=stream0)
        buf257 = reinterpret_tensor(buf225, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf225  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf253, buf257, 602112, grid=grid(602112), stream=stream0)
        buf258 = reinterpret_tensor(buf253, (96, 196, 32), (6272, 32, 1), 0); del buf253  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_342, reinterpret_tensor(buf257, (96, 196, 32), (6272, 32, 1), 0), out=buf258)
        del permute_342
        buf259 = reinterpret_tensor(buf224, (96, 196, 196), (38416, 196, 1), 0); del buf224  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf257, (96, 196, 32), (6272, 32, 1), 0), permute_343, out=buf259)
        del permute_343
        buf261 = reinterpret_tensor(buf222, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf222  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_30.run(buf259, alias_29, buf261, 18816, 196, grid=grid(18816), stream=stream0)
        del alias_29
        buf262 = reinterpret_tensor(buf257, (96, 32, 196), (6272, 196, 1), 0); del buf257  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_344, reinterpret_tensor(buf261, (96, 196, 196), (38416, 196, 1), 0), out=buf262)
        del permute_344
        buf263 = buf221; del buf221  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf261, (96, 196, 196), (38416, 196, 1), 0), permute_345, out=buf263)
        del permute_345
        buf264 = reinterpret_tensor(buf241, (8, 196, 3, 12, 32), (225792, 1152, 384, 32, 1), 0); del buf241  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf263, buf262, buf258, buf264, 56448, 32, grid=grid(56448, 32), stream=stream0)
        buf265 = empty((1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf264, (1152, 1568), (1, 1152), 0), view_216, out=buf265)
        del view_216
        buf266 = reinterpret_tensor(buf263, (1568, 384), (384, 1), 0); del buf263  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf264, (1568, 1152), (1152, 1), 0), permute_350, out=buf266)
        del permute_350
        buf273 = buf252; del buf252  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf273, buf266, primals_167, mul_125, div_34, 1568, 384, grid=grid(1568), stream=stream0)
        del div_34
        del primals_167
        buf269 = reinterpret_tensor(buf255, (384, 13), (1, 384), 0); del buf255  # reuse
        buf271 = buf248; del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_27.run(buf266, mul_125, buf269, buf271, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_125
        buf270 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf269, buf270, 384, 13, grid=grid(384), stream=stream0)
        buf272 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf271, buf272, 384, 13, grid=grid(384), stream=stream0)
        buf274 = reinterpret_tensor(buf264, (1568, 1152), (1152, 1), 0); del buf264  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf273, (1568, 384), (384, 1), 0), permute_352, out=buf274)
        del permute_352
        buf275 = empty((384, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf273, (384, 1568), (1, 384), 0), view_214, out=buf275)
        del view_214
        buf276 = reinterpret_tensor(buf271, (1, 384, 13), (4992, 1, 384), 0); del buf271  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf273, buf276, 4992, 121, grid=grid(4992), stream=stream0)
        buf277 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf276, buf277, 384, 13, grid=grid(384), stream=stream0)
        buf278 = reinterpret_tensor(buf274, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf274  # reuse
        # Source Nodes: [x_152], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_23.run(buf278, addmm_37, 1806336, grid=grid(1806336), stream=stream0)
        del addmm_37
        buf279 = buf266; del buf266  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf278, (1568, 1152), (1152, 1), 0), permute_356, out=buf279)
        del permute_356
        buf280 = empty((1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf278, (1152, 1568), (1, 1152), 0), view_212, out=buf280)
        del view_212
        buf281 = buf244; del buf244  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf278, buf281, 14976, 121, grid=grid(14976), stream=stream0)
        buf282 = empty((1, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_25.run(buf281, buf282, 1152, 13, grid=grid(1152), stream=stream0)
        buf289 = buf273; del buf273  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf289, buf279, primals_161, mul_120, div_35, 1568, 384, grid=grid(1568), stream=stream0)
        del div_35
        del primals_161
        buf285 = reinterpret_tensor(buf276, (384, 13), (1, 384), 0); del buf276  # reuse
        buf287 = buf269; del buf269  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_27.run(buf279, mul_120, buf285, buf287, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_120
        buf286 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf285, buf286, 384, 13, grid=grid(384), stream=stream0)
        buf288 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf287, buf288, 384, 13, grid=grid(384), stream=stream0)
        buf290 = buf279; del buf279  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (1568, 384), (384, 1), 0), permute_360, out=buf290)
        del permute_360
        buf291 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (384, 1568), (1, 384), 0), view_210, out=buf291)
        del view_210
        buf292 = reinterpret_tensor(buf287, (1, 384, 13), (4992, 1, 384), 0); del buf287  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf289, buf292, 4992, 121, grid=grid(4992), stream=stream0)
        buf293 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf292, buf293, 384, 13, grid=grid(384), stream=stream0)
        buf294 = reinterpret_tensor(buf262, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf290, buf294, 602112, grid=grid(602112), stream=stream0)
        buf295 = reinterpret_tensor(buf290, (96, 196, 32), (6272, 32, 1), 0); del buf290  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_365, reinterpret_tensor(buf294, (96, 196, 32), (6272, 32, 1), 0), out=buf295)
        del permute_365
        buf296 = reinterpret_tensor(buf261, (96, 196, 196), (38416, 196, 1), 0); del buf261  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf294, (96, 196, 32), (6272, 32, 1), 0), permute_366, out=buf296)
        del permute_366
        buf298 = reinterpret_tensor(buf259, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf259  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_30.run(buf296, alias_30, buf298, 18816, 196, grid=grid(18816), stream=stream0)
        del alias_30
        buf299 = reinterpret_tensor(buf294, (96, 32, 196), (6272, 196, 1), 0); del buf294  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_367, reinterpret_tensor(buf298, (96, 196, 196), (38416, 196, 1), 0), out=buf299)
        del permute_367
        buf300 = buf258; del buf258  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf298, (96, 196, 196), (38416, 196, 1), 0), permute_368, out=buf300)
        del permute_368
        buf301 = reinterpret_tensor(buf278, (8, 196, 3, 12, 32), (225792, 1152, 384, 32, 1), 0); del buf278  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf300, buf299, buf295, buf301, 56448, 32, grid=grid(56448, 32), stream=stream0)
        buf302 = empty((1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf301, (1152, 1568), (1, 1152), 0), view_200, out=buf302)
        del view_200
        buf303 = reinterpret_tensor(buf300, (1568, 384), (384, 1), 0); del buf300  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf301, (1568, 1152), (1152, 1), 0), permute_373, out=buf303)
        del permute_373
        buf310 = buf289; del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf310, buf303, primals_156, mul_117, div_36, 1568, 384, grid=grid(1568), stream=stream0)
        del div_36
        del primals_156
        buf306 = reinterpret_tensor(buf292, (384, 13), (1, 384), 0); del buf292  # reuse
        buf308 = buf285; del buf285  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_27.run(buf303, mul_117, buf306, buf308, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_117
        buf307 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf306, buf307, 384, 13, grid=grid(384), stream=stream0)
        buf309 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf308, buf309, 384, 13, grid=grid(384), stream=stream0)
        buf311 = reinterpret_tensor(buf301, (1568, 1152), (1152, 1), 0); del buf301  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf310, (1568, 384), (384, 1), 0), permute_375, out=buf311)
        del permute_375
        buf312 = empty((384, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf310, (384, 1568), (1, 384), 0), view_198, out=buf312)
        del view_198
        buf313 = reinterpret_tensor(buf308, (1, 384, 13), (4992, 1, 384), 0); del buf308  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf310, buf313, 4992, 121, grid=grid(4992), stream=stream0)
        buf314 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf313, buf314, 384, 13, grid=grid(384), stream=stream0)
        buf315 = reinterpret_tensor(buf311, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf311  # reuse
        # Source Nodes: [x_141], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_23.run(buf315, addmm_34, 1806336, grid=grid(1806336), stream=stream0)
        del addmm_34
        buf316 = buf303; del buf303  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf315, (1568, 1152), (1152, 1), 0), permute_379, out=buf316)
        del permute_379
        buf317 = empty((1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf315, (1152, 1568), (1, 1152), 0), view_196, out=buf317)
        del view_196
        buf318 = buf281; del buf281  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf315, buf318, 14976, 121, grid=grid(14976), stream=stream0)
        buf319 = empty((1, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_25.run(buf318, buf319, 1152, 13, grid=grid(1152), stream=stream0)
        buf326 = buf310; del buf310  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf326, buf316, primals_150, mul_112, div_37, 1568, 384, grid=grid(1568), stream=stream0)
        del div_37
        del primals_150
        buf322 = reinterpret_tensor(buf313, (384, 13), (1, 384), 0); del buf313  # reuse
        buf324 = buf306; del buf306  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_27.run(buf316, mul_112, buf322, buf324, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_112
        buf323 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf322, buf323, 384, 13, grid=grid(384), stream=stream0)
        buf325 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf324, buf325, 384, 13, grid=grid(384), stream=stream0)
        buf327 = buf316; del buf316  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf326, (1568, 384), (384, 1), 0), permute_383, out=buf327)
        del permute_383
        buf328 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf326, (384, 1568), (1, 384), 0), view_194, out=buf328)
        del view_194
        buf329 = reinterpret_tensor(buf324, (1, 384, 13), (4992, 1, 384), 0); del buf324  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf326, buf329, 4992, 121, grid=grid(4992), stream=stream0)
        buf330 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf329, buf330, 384, 13, grid=grid(384), stream=stream0)
        buf331 = reinterpret_tensor(buf299, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf299  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf327, buf331, 602112, grid=grid(602112), stream=stream0)
        buf332 = reinterpret_tensor(buf327, (96, 196, 32), (6272, 32, 1), 0); del buf327  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_388, reinterpret_tensor(buf331, (96, 196, 32), (6272, 32, 1), 0), out=buf332)
        del permute_388
        buf333 = reinterpret_tensor(buf298, (96, 196, 196), (38416, 196, 1), 0); del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf331, (96, 196, 32), (6272, 32, 1), 0), permute_389, out=buf333)
        del permute_389
        buf335 = reinterpret_tensor(buf296, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf296  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_30.run(buf333, alias_31, buf335, 18816, 196, grid=grid(18816), stream=stream0)
        del alias_31
        buf336 = reinterpret_tensor(buf331, (96, 32, 196), (6272, 196, 1), 0); del buf331  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_390, reinterpret_tensor(buf335, (96, 196, 196), (38416, 196, 1), 0), out=buf336)
        del permute_390
        buf337 = buf295; del buf295  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf335, (96, 196, 196), (38416, 196, 1), 0), permute_391, out=buf337)
        del permute_391
        buf338 = reinterpret_tensor(buf315, (8, 196, 3, 12, 32), (225792, 1152, 384, 32, 1), 0); del buf315  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf337, buf336, buf332, buf338, 56448, 32, grid=grid(56448, 32), stream=stream0)
        buf339 = empty((1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf338, (1152, 1568), (1, 1152), 0), view_184, out=buf339)
        del view_184
        buf340 = reinterpret_tensor(buf337, (1568, 384), (384, 1), 0); del buf337  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf338, (1568, 1152), (1152, 1), 0), permute_396, out=buf340)
        del permute_396
        buf347 = buf326; del buf326  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf347, buf340, primals_145, mul_109, div_38, 1568, 384, grid=grid(1568), stream=stream0)
        del div_38
        del primals_145
        buf343 = reinterpret_tensor(buf329, (384, 13), (1, 384), 0); del buf329  # reuse
        buf345 = buf322; del buf322  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_27.run(buf340, mul_109, buf343, buf345, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_109
        buf344 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf343, buf344, 384, 13, grid=grid(384), stream=stream0)
        buf346 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf345, buf346, 384, 13, grid=grid(384), stream=stream0)
        buf348 = reinterpret_tensor(buf338, (1568, 1152), (1152, 1), 0); del buf338  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf347, (1568, 384), (384, 1), 0), permute_398, out=buf348)
        del permute_398
        buf349 = empty((384, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf347, (384, 1568), (1, 384), 0), view_182, out=buf349)
        del view_182
        buf350 = reinterpret_tensor(buf345, (1, 384, 13), (4992, 1, 384), 0); del buf345  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf347, buf350, 4992, 121, grid=grid(4992), stream=stream0)
        buf351 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf350, buf351, 384, 13, grid=grid(384), stream=stream0)
        buf352 = reinterpret_tensor(buf348, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf348  # reuse
        # Source Nodes: [x_130], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_23.run(buf352, addmm_31, 1806336, grid=grid(1806336), stream=stream0)
        del addmm_31
        buf353 = buf340; del buf340  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf352, (1568, 1152), (1152, 1), 0), permute_402, out=buf353)
        del permute_402
        buf354 = empty((1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf352, (1152, 1568), (1, 1152), 0), view_180, out=buf354)
        del view_180
        buf355 = buf318; del buf318  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf352, buf355, 14976, 121, grid=grid(14976), stream=stream0)
        buf356 = empty((1, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_25.run(buf355, buf356, 1152, 13, grid=grid(1152), stream=stream0)
        buf363 = buf347; del buf347  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf363, buf353, primals_139, mul_104, div_39, 1568, 384, grid=grid(1568), stream=stream0)
        del div_39
        del primals_139
        buf359 = reinterpret_tensor(buf350, (384, 13), (1, 384), 0); del buf350  # reuse
        buf361 = buf343; del buf343  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_27.run(buf353, mul_104, buf359, buf361, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_104
        buf360 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf359, buf360, 384, 13, grid=grid(384), stream=stream0)
        buf362 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf361, buf362, 384, 13, grid=grid(384), stream=stream0)
        buf364 = buf353; del buf353  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf363, (1568, 384), (384, 1), 0), permute_406, out=buf364)
        del permute_406
        buf365 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf363, (384, 1568), (1, 384), 0), view_178, out=buf365)
        del view_178
        buf366 = reinterpret_tensor(buf361, (1, 384, 13), (4992, 1, 384), 0); del buf361  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf363, buf366, 4992, 121, grid=grid(4992), stream=stream0)
        buf367 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf366, buf367, 384, 13, grid=grid(384), stream=stream0)
        buf368 = reinterpret_tensor(buf336, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf336  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf364, buf368, 602112, grid=grid(602112), stream=stream0)
        buf369 = reinterpret_tensor(buf364, (96, 196, 32), (6272, 32, 1), 0); del buf364  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_411, reinterpret_tensor(buf368, (96, 196, 32), (6272, 32, 1), 0), out=buf369)
        del permute_411
        buf370 = reinterpret_tensor(buf335, (96, 196, 196), (38416, 196, 1), 0); del buf335  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf368, (96, 196, 32), (6272, 32, 1), 0), permute_412, out=buf370)
        del permute_412
        buf372 = reinterpret_tensor(buf333, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf333  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_30.run(buf370, alias_32, buf372, 18816, 196, grid=grid(18816), stream=stream0)
        del alias_32
        buf373 = reinterpret_tensor(buf368, (96, 32, 196), (6272, 196, 1), 0); del buf368  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_413, reinterpret_tensor(buf372, (96, 196, 196), (38416, 196, 1), 0), out=buf373)
        del permute_413
        buf374 = buf332; del buf332  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf372, (96, 196, 196), (38416, 196, 1), 0), permute_414, out=buf374)
        del permute_414
        buf375 = reinterpret_tensor(buf352, (8, 196, 3, 12, 32), (225792, 1152, 384, 32, 1), 0); del buf352  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf374, buf373, buf369, buf375, 56448, 32, grid=grid(56448, 32), stream=stream0)
        buf376 = empty((1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf375, (1152, 1568), (1, 1152), 0), view_168, out=buf376)
        del view_168
        buf377 = reinterpret_tensor(buf374, (1568, 384), (384, 1), 0); del buf374  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf375, (1568, 1152), (1152, 1), 0), permute_419, out=buf377)
        del permute_419
        buf384 = buf363; del buf363  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf384, buf377, primals_134, mul_101, div_40, 1568, 384, grid=grid(1568), stream=stream0)
        del div_40
        del primals_134
        buf380 = reinterpret_tensor(buf366, (384, 13), (1, 384), 0); del buf366  # reuse
        buf382 = buf359; del buf359  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_27.run(buf377, mul_101, buf380, buf382, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_101
        buf381 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf380, buf381, 384, 13, grid=grid(384), stream=stream0)
        buf383 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf382, buf383, 384, 13, grid=grid(384), stream=stream0)
        buf385 = reinterpret_tensor(buf375, (1568, 1152), (1152, 1), 0); del buf375  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf384, (1568, 384), (384, 1), 0), permute_421, out=buf385)
        del permute_421
        buf386 = empty((384, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf384, (384, 1568), (1, 384), 0), view_166, out=buf386)
        del view_166
        buf387 = reinterpret_tensor(buf382, (1, 384, 13), (4992, 1, 384), 0); del buf382  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf384, buf387, 4992, 121, grid=grid(4992), stream=stream0)
        buf388 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf387, buf388, 384, 13, grid=grid(384), stream=stream0)
        buf389 = reinterpret_tensor(buf385, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf385  # reuse
        # Source Nodes: [x_119], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_23.run(buf389, addmm_28, 1806336, grid=grid(1806336), stream=stream0)
        del addmm_28
        buf390 = buf377; del buf377  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf389, (1568, 1152), (1152, 1), 0), permute_425, out=buf390)
        del permute_425
        buf391 = empty((1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf389, (1152, 1568), (1, 1152), 0), view_164, out=buf391)
        del view_164
        buf392 = buf355; del buf355  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf389, buf392, 14976, 121, grid=grid(14976), stream=stream0)
        buf393 = empty((1, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_25.run(buf392, buf393, 1152, 13, grid=grid(1152), stream=stream0)
        buf400 = buf384; del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf400, buf390, primals_128, mul_96, div_41, 1568, 384, grid=grid(1568), stream=stream0)
        del div_41
        del primals_128
        buf396 = reinterpret_tensor(buf387, (384, 13), (1, 384), 0); del buf387  # reuse
        buf398 = buf380; del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_27.run(buf390, mul_96, buf396, buf398, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_96
        buf397 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf396, buf397, 384, 13, grid=grid(384), stream=stream0)
        buf399 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf398, buf399, 384, 13, grid=grid(384), stream=stream0)
        buf401 = buf390; del buf390  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf400, (1568, 384), (384, 1), 0), permute_429, out=buf401)
        del permute_429
        buf402 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf400, (384, 1568), (1, 384), 0), view_162, out=buf402)
        del view_162
        buf403 = reinterpret_tensor(buf398, (1, 384, 13), (4992, 1, 384), 0); del buf398  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf400, buf403, 4992, 121, grid=grid(4992), stream=stream0)
        buf404 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf403, buf404, 384, 13, grid=grid(384), stream=stream0)
        buf405 = reinterpret_tensor(buf373, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf373  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf401, buf405, 602112, grid=grid(602112), stream=stream0)
        buf406 = reinterpret_tensor(buf401, (96, 196, 32), (6272, 32, 1), 0); del buf401  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_434, reinterpret_tensor(buf405, (96, 196, 32), (6272, 32, 1), 0), out=buf406)
        del permute_434
        buf407 = reinterpret_tensor(buf372, (96, 196, 196), (38416, 196, 1), 0); del buf372  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf405, (96, 196, 32), (6272, 32, 1), 0), permute_435, out=buf407)
        del permute_435
        buf409 = reinterpret_tensor(buf370, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf370  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_30.run(buf407, alias_33, buf409, 18816, 196, grid=grid(18816), stream=stream0)
        del alias_33
        buf410 = reinterpret_tensor(buf405, (96, 32, 196), (6272, 196, 1), 0); del buf405  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_436, reinterpret_tensor(buf409, (96, 196, 196), (38416, 196, 1), 0), out=buf410)
        del permute_436
        buf411 = buf369; del buf369  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf409, (96, 196, 196), (38416, 196, 1), 0), permute_437, out=buf411)
        del permute_437
        buf412 = reinterpret_tensor(buf389, (8, 196, 3, 12, 32), (225792, 1152, 384, 32, 1), 0); del buf389  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf411, buf410, buf406, buf412, 56448, 32, grid=grid(56448, 32), stream=stream0)
        buf413 = empty((1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf412, (1152, 1568), (1, 1152), 0), view_152, out=buf413)
        del view_152
        buf414 = reinterpret_tensor(buf411, (1568, 384), (384, 1), 0); del buf411  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf412, (1568, 1152), (1152, 1), 0), permute_442, out=buf414)
        del permute_442
        buf421 = buf400; del buf400  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf421, buf414, primals_123, mul_93, div_42, 1568, 384, grid=grid(1568), stream=stream0)
        del div_42
        del primals_123
        buf417 = reinterpret_tensor(buf403, (384, 13), (1, 384), 0); del buf403  # reuse
        buf419 = buf396; del buf396  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_27.run(buf414, mul_93, buf417, buf419, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_93
        buf418 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf417, buf418, 384, 13, grid=grid(384), stream=stream0)
        buf420 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf419, buf420, 384, 13, grid=grid(384), stream=stream0)
        buf422 = reinterpret_tensor(buf412, (1568, 1152), (1152, 1), 0); del buf412  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf421, (1568, 384), (384, 1), 0), permute_444, out=buf422)
        del permute_444
        buf423 = empty((384, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf421, (384, 1568), (1, 384), 0), view_150, out=buf423)
        del view_150
        buf424 = reinterpret_tensor(buf419, (1, 384, 13), (4992, 1, 384), 0); del buf419  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf421, buf424, 4992, 121, grid=grid(4992), stream=stream0)
        buf425 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf424, buf425, 384, 13, grid=grid(384), stream=stream0)
        buf426 = reinterpret_tensor(buf422, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf422  # reuse
        # Source Nodes: [x_108], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_23.run(buf426, addmm_25, 1806336, grid=grid(1806336), stream=stream0)
        del addmm_25
        buf427 = buf414; del buf414  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf426, (1568, 1152), (1152, 1), 0), permute_448, out=buf427)
        del permute_448
        buf428 = empty((1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf426, (1152, 1568), (1, 1152), 0), view_148, out=buf428)
        del view_148
        buf429 = buf392; del buf392  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf426, buf429, 14976, 121, grid=grid(14976), stream=stream0)
        buf430 = empty((1, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_25.run(buf429, buf430, 1152, 13, grid=grid(1152), stream=stream0)
        buf437 = buf421; del buf421  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf437, buf427, primals_117, mul_88, div_43, 1568, 384, grid=grid(1568), stream=stream0)
        del div_43
        del primals_117
        buf433 = reinterpret_tensor(buf424, (384, 13), (1, 384), 0); del buf424  # reuse
        buf435 = buf417; del buf417  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_27.run(buf427, mul_88, buf433, buf435, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_88
        buf434 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf433, buf434, 384, 13, grid=grid(384), stream=stream0)
        buf436 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf435, buf436, 384, 13, grid=grid(384), stream=stream0)
        buf438 = buf427; del buf427  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf437, (1568, 384), (384, 1), 0), permute_452, out=buf438)
        del permute_452
        buf439 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf437, (384, 1568), (1, 384), 0), view_146, out=buf439)
        del view_146
        buf440 = reinterpret_tensor(buf435, (1, 384, 13), (4992, 1, 384), 0); del buf435  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf437, buf440, 4992, 121, grid=grid(4992), stream=stream0)
        buf441 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf440, buf441, 384, 13, grid=grid(384), stream=stream0)
        buf442 = reinterpret_tensor(buf410, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf410  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf438, buf442, 602112, grid=grid(602112), stream=stream0)
        buf443 = reinterpret_tensor(buf438, (96, 196, 32), (6272, 32, 1), 0); del buf438  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_457, reinterpret_tensor(buf442, (96, 196, 32), (6272, 32, 1), 0), out=buf443)
        del permute_457
        buf444 = reinterpret_tensor(buf409, (96, 196, 196), (38416, 196, 1), 0); del buf409  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf442, (96, 196, 32), (6272, 32, 1), 0), permute_458, out=buf444)
        del permute_458
        buf446 = reinterpret_tensor(buf407, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf407  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_30.run(buf444, alias_34, buf446, 18816, 196, grid=grid(18816), stream=stream0)
        del alias_34
        buf447 = reinterpret_tensor(buf442, (96, 32, 196), (6272, 196, 1), 0); del buf442  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_459, reinterpret_tensor(buf446, (96, 196, 196), (38416, 196, 1), 0), out=buf447)
        del permute_459
        buf448 = buf406; del buf406  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf446, (96, 196, 196), (38416, 196, 1), 0), permute_460, out=buf448)
        del permute_460
        buf449 = reinterpret_tensor(buf426, (8, 196, 3, 12, 32), (225792, 1152, 384, 32, 1), 0); del buf426  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf448, buf447, buf443, buf449, 56448, 32, grid=grid(56448, 32), stream=stream0)
        buf450 = empty((1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf449, (1152, 1568), (1, 1152), 0), view_136, out=buf450)
        del view_136
        buf451 = reinterpret_tensor(buf448, (1568, 384), (384, 1), 0); del buf448  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf449, (1568, 1152), (1152, 1), 0), permute_465, out=buf451)
        del permute_465
        buf458 = buf437; del buf437  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf458, buf451, primals_112, mul_85, div_44, 1568, 384, grid=grid(1568), stream=stream0)
        del div_44
        del primals_112
        buf454 = reinterpret_tensor(buf440, (384, 13), (1, 384), 0); del buf440  # reuse
        buf456 = buf433; del buf433  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_27.run(buf451, mul_85, buf454, buf456, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_85
        buf455 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf454, buf455, 384, 13, grid=grid(384), stream=stream0)
        buf457 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf456, buf457, 384, 13, grid=grid(384), stream=stream0)
        buf459 = reinterpret_tensor(buf449, (1568, 1152), (1152, 1), 0); del buf449  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf458, (1568, 384), (384, 1), 0), permute_467, out=buf459)
        del permute_467
        buf460 = empty((384, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf458, (384, 1568), (1, 384), 0), view_134, out=buf460)
        del view_134
        buf461 = reinterpret_tensor(buf456, (1, 384, 13), (4992, 1, 384), 0); del buf456  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf458, buf461, 4992, 121, grid=grid(4992), stream=stream0)
        buf462 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf461, buf462, 384, 13, grid=grid(384), stream=stream0)
        buf463 = reinterpret_tensor(buf459, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf459  # reuse
        # Source Nodes: [x_96], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_23.run(buf463, addmm_22, 1806336, grid=grid(1806336), stream=stream0)
        del addmm_22
        buf464 = buf451; del buf451  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf463, (1568, 1152), (1152, 1), 0), permute_471, out=buf464)
        del permute_471
        buf465 = empty((1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf463, (1152, 1568), (1, 1152), 0), view_132, out=buf465)
        del view_132
        buf466 = buf429; del buf429  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf463, buf466, 14976, 121, grid=grid(14976), stream=stream0)
        buf467 = empty((1, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_25.run(buf466, buf467, 1152, 13, grid=grid(1152), stream=stream0)
        buf474 = buf458; del buf458  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf474, buf464, primals_106, mul_80, div_45, 1568, 384, grid=grid(1568), stream=stream0)
        del div_45
        del primals_106
        buf470 = reinterpret_tensor(buf461, (384, 13), (1, 384), 0); del buf461  # reuse
        buf472 = buf454; del buf454  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_27.run(buf464, mul_80, buf470, buf472, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_80
        buf471 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf470, buf471, 384, 13, grid=grid(384), stream=stream0)
        buf473 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf472, buf473, 384, 13, grid=grid(384), stream=stream0)
        buf475 = buf464; del buf464  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf474, (1568, 384), (384, 1), 0), permute_475, out=buf475)
        del permute_475
        buf476 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf474, (384, 1568), (1, 384), 0), view_130, out=buf476)
        del view_130
        buf477 = reinterpret_tensor(buf472, (1, 384, 13), (4992, 1, 384), 0); del buf472  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf474, buf477, 4992, 121, grid=grid(4992), stream=stream0)
        buf478 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf477, buf478, 384, 13, grid=grid(384), stream=stream0)
        buf479 = reinterpret_tensor(buf447, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf447  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf475, buf479, 602112, grid=grid(602112), stream=stream0)
        buf480 = reinterpret_tensor(buf475, (96, 196, 32), (6272, 32, 1), 0); del buf475  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_480, reinterpret_tensor(buf479, (96, 196, 32), (6272, 32, 1), 0), out=buf480)
        del permute_480
        buf481 = reinterpret_tensor(buf446, (96, 196, 196), (38416, 196, 1), 0); del buf446  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf479, (96, 196, 32), (6272, 32, 1), 0), permute_481, out=buf481)
        del permute_481
        buf483 = reinterpret_tensor(buf444, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf444  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_30.run(buf481, alias_35, buf483, 18816, 196, grid=grid(18816), stream=stream0)
        del alias_35
        buf484 = reinterpret_tensor(buf479, (96, 32, 196), (6272, 196, 1), 0); del buf479  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_482, reinterpret_tensor(buf483, (96, 196, 196), (38416, 196, 1), 0), out=buf484)
        del permute_482
        buf485 = buf443; del buf443  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf483, (96, 196, 196), (38416, 196, 1), 0), permute_483, out=buf485)
        del permute_483
        buf486 = reinterpret_tensor(buf463, (8, 196, 3, 12, 32), (225792, 1152, 384, 32, 1), 0); del buf463  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf485, buf484, buf480, buf486, 56448, 32, grid=grid(56448, 32), stream=stream0)
        buf487 = empty((1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf486, (1152, 1568), (1, 1152), 0), view_120, out=buf487)
        del view_120
        buf488 = reinterpret_tensor(buf485, (1568, 384), (384, 1), 0); del buf485  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf486, (1568, 1152), (1152, 1), 0), permute_488, out=buf488)
        del permute_488
        buf495 = buf474; del buf474  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf495, buf488, primals_101, mul_77, div_46, 1568, 384, grid=grid(1568), stream=stream0)
        del div_46
        del primals_101
        buf491 = reinterpret_tensor(buf477, (384, 13), (1, 384), 0); del buf477  # reuse
        buf493 = buf470; del buf470  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_27.run(buf488, mul_77, buf491, buf493, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_77
        buf492 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf491, buf492, 384, 13, grid=grid(384), stream=stream0)
        buf494 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf493, buf494, 384, 13, grid=grid(384), stream=stream0)
        buf496 = reinterpret_tensor(buf486, (1568, 1152), (1152, 1), 0); del buf486  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf495, (1568, 384), (384, 1), 0), permute_490, out=buf496)
        del permute_490
        buf497 = empty((384, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf495, (384, 1568), (1, 384), 0), view_118, out=buf497)
        del view_118
        buf498 = reinterpret_tensor(buf493, (1, 384, 13), (4992, 1, 384), 0); del buf493  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf495, buf498, 4992, 121, grid=grid(4992), stream=stream0)
        buf499 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf498, buf499, 384, 13, grid=grid(384), stream=stream0)
        buf500 = reinterpret_tensor(buf496, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf496  # reuse
        # Source Nodes: [x_85], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_23.run(buf500, addmm_19, 1806336, grid=grid(1806336), stream=stream0)
        del addmm_19
        buf501 = buf488; del buf488  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf500, (1568, 1152), (1152, 1), 0), permute_494, out=buf501)
        del permute_494
        buf502 = empty((1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf500, (1152, 1568), (1, 1152), 0), view_116, out=buf502)
        del view_116
        buf503 = buf466; del buf466  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf500, buf503, 14976, 121, grid=grid(14976), stream=stream0)
        buf504 = empty((1, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_25.run(buf503, buf504, 1152, 13, grid=grid(1152), stream=stream0)
        buf511 = buf495; del buf495  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf511, buf501, primals_95, mul_72, div_47, 1568, 384, grid=grid(1568), stream=stream0)
        del div_47
        del primals_95
        buf507 = reinterpret_tensor(buf498, (384, 13), (1, 384), 0); del buf498  # reuse
        buf509 = buf491; del buf491  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_27.run(buf501, mul_72, buf507, buf509, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_72
        buf508 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf507, buf508, 384, 13, grid=grid(384), stream=stream0)
        buf510 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf509, buf510, 384, 13, grid=grid(384), stream=stream0)
        buf512 = buf501; del buf501  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf511, (1568, 384), (384, 1), 0), permute_498, out=buf512)
        del permute_498
        buf513 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf511, (384, 1568), (1, 384), 0), view_114, out=buf513)
        del view_114
        buf514 = reinterpret_tensor(buf509, (1, 384, 13), (4992, 1, 384), 0); del buf509  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf511, buf514, 4992, 121, grid=grid(4992), stream=stream0)
        buf515 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf514, buf515, 384, 13, grid=grid(384), stream=stream0)
        buf516 = reinterpret_tensor(buf484, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf484  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf512, buf516, 602112, grid=grid(602112), stream=stream0)
        buf517 = reinterpret_tensor(buf512, (96, 196, 32), (6272, 32, 1), 0); del buf512  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_503, reinterpret_tensor(buf516, (96, 196, 32), (6272, 32, 1), 0), out=buf517)
        del permute_503
        buf518 = reinterpret_tensor(buf483, (96, 196, 196), (38416, 196, 1), 0); del buf483  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf516, (96, 196, 32), (6272, 32, 1), 0), permute_504, out=buf518)
        del permute_504
        buf520 = reinterpret_tensor(buf481, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf481  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_30.run(buf518, alias_36, buf520, 18816, 196, grid=grid(18816), stream=stream0)
        del alias_36
        buf521 = reinterpret_tensor(buf516, (96, 32, 196), (6272, 196, 1), 0); del buf516  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_505, reinterpret_tensor(buf520, (96, 196, 196), (38416, 196, 1), 0), out=buf521)
        del permute_505
        buf522 = buf480; del buf480  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf520, (96, 196, 196), (38416, 196, 1), 0), permute_506, out=buf522)
        del permute_506
        buf523 = reinterpret_tensor(buf500, (8, 196, 3, 12, 32), (225792, 1152, 384, 32, 1), 0); del buf500  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf522, buf521, buf517, buf523, 56448, 32, grid=grid(56448, 32), stream=stream0)
        buf524 = empty((1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf523, (1152, 1568), (1, 1152), 0), view_104, out=buf524)
        del view_104
        buf525 = reinterpret_tensor(buf522, (1568, 384), (384, 1), 0); del buf522  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf523, (1568, 1152), (1152, 1), 0), permute_511, out=buf525)
        del permute_511
        buf532 = buf511; del buf511  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf532, buf525, primals_90, mul_69, div_48, 1568, 384, grid=grid(1568), stream=stream0)
        del div_48
        del primals_90
        buf528 = reinterpret_tensor(buf514, (384, 13), (1, 384), 0); del buf514  # reuse
        buf530 = buf507; del buf507  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_27.run(buf525, mul_69, buf528, buf530, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_69
        buf529 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf528, buf529, 384, 13, grid=grid(384), stream=stream0)
        buf531 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf530, buf531, 384, 13, grid=grid(384), stream=stream0)
        buf533 = reinterpret_tensor(buf523, (1568, 1152), (1152, 1), 0); del buf523  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf532, (1568, 384), (384, 1), 0), permute_513, out=buf533)
        del permute_513
        buf534 = empty((384, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf532, (384, 1568), (1, 384), 0), view_102, out=buf534)
        del view_102
        buf535 = reinterpret_tensor(buf530, (1, 384, 13), (4992, 1, 384), 0); del buf530  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf532, buf535, 4992, 121, grid=grid(4992), stream=stream0)
        buf536 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf535, buf536, 384, 13, grid=grid(384), stream=stream0)
        buf537 = reinterpret_tensor(buf533, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf533  # reuse
        # Source Nodes: [x_74], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_23.run(buf537, addmm_16, 1806336, grid=grid(1806336), stream=stream0)
        del addmm_16
        buf538 = buf525; del buf525  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf537, (1568, 1152), (1152, 1), 0), permute_517, out=buf538)
        del permute_517
        buf539 = empty((1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf537, (1152, 1568), (1, 1152), 0), view_100, out=buf539)
        del view_100
        buf540 = buf503; del buf503  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf537, buf540, 14976, 121, grid=grid(14976), stream=stream0)
        buf541 = empty((1, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_25.run(buf540, buf541, 1152, 13, grid=grid(1152), stream=stream0)
        buf548 = buf532; del buf532  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf548, buf538, primals_84, mul_64, div_49, 1568, 384, grid=grid(1568), stream=stream0)
        del div_49
        del primals_84
        buf544 = reinterpret_tensor(buf535, (384, 13), (1, 384), 0); del buf535  # reuse
        buf546 = buf528; del buf528  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_27.run(buf538, mul_64, buf544, buf546, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_64
        buf545 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf544, buf545, 384, 13, grid=grid(384), stream=stream0)
        buf547 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf546, buf547, 384, 13, grid=grid(384), stream=stream0)
        buf549 = buf538; del buf538  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf548, (1568, 384), (384, 1), 0), permute_521, out=buf549)
        del permute_521
        buf550 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf548, (384, 1568), (1, 384), 0), view_98, out=buf550)
        del view_98
        buf551 = reinterpret_tensor(buf546, (1, 384, 13), (4992, 1, 384), 0); del buf546  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf548, buf551, 4992, 121, grid=grid(4992), stream=stream0)
        buf552 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf551, buf552, 384, 13, grid=grid(384), stream=stream0)
        buf553 = reinterpret_tensor(buf521, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf521  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf549, buf553, 602112, grid=grid(602112), stream=stream0)
        buf554 = reinterpret_tensor(buf549, (96, 196, 32), (6272, 32, 1), 0); del buf549  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_526, reinterpret_tensor(buf553, (96, 196, 32), (6272, 32, 1), 0), out=buf554)
        del permute_526
        buf555 = reinterpret_tensor(buf520, (96, 196, 196), (38416, 196, 1), 0); del buf520  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf553, (96, 196, 32), (6272, 32, 1), 0), permute_527, out=buf555)
        del permute_527
        buf557 = reinterpret_tensor(buf518, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf518  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_30.run(buf555, alias_37, buf557, 18816, 196, grid=grid(18816), stream=stream0)
        del alias_37
        buf558 = reinterpret_tensor(buf553, (96, 32, 196), (6272, 196, 1), 0); del buf553  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_528, reinterpret_tensor(buf557, (96, 196, 196), (38416, 196, 1), 0), out=buf558)
        del permute_528
        buf559 = buf517; del buf517  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf557, (96, 196, 196), (38416, 196, 1), 0), permute_529, out=buf559)
        del permute_529
        buf560 = reinterpret_tensor(buf537, (8, 196, 3, 12, 32), (225792, 1152, 384, 32, 1), 0); del buf537  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf559, buf558, buf554, buf560, 56448, 32, grid=grid(56448, 32), stream=stream0)
        buf561 = empty((1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf560, (1152, 1568), (1, 1152), 0), view_88, out=buf561)
        del view_88
        buf562 = reinterpret_tensor(buf559, (1568, 384), (384, 1), 0); del buf559  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf560, (1568, 1152), (1152, 1), 0), permute_534, out=buf562)
        del permute_534
        buf569 = buf548; del buf548  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf569, buf562, primals_79, mul_61, div_50, 1568, 384, grid=grid(1568), stream=stream0)
        del div_50
        del primals_79
        buf565 = reinterpret_tensor(buf551, (384, 13), (1, 384), 0); del buf551  # reuse
        buf567 = buf544; del buf544  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_27.run(buf562, mul_61, buf565, buf567, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_61
        buf566 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf565, buf566, 384, 13, grid=grid(384), stream=stream0)
        buf568 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf567, buf568, 384, 13, grid=grid(384), stream=stream0)
        buf570 = reinterpret_tensor(buf560, (1568, 1152), (1152, 1), 0); del buf560  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf569, (1568, 384), (384, 1), 0), permute_536, out=buf570)
        del permute_536
        buf571 = empty((384, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf569, (384, 1568), (1, 384), 0), view_86, out=buf571)
        del view_86
        buf572 = reinterpret_tensor(buf567, (1, 384, 13), (4992, 1, 384), 0); del buf567  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf569, buf572, 4992, 121, grid=grid(4992), stream=stream0)
        buf573 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf572, buf573, 384, 13, grid=grid(384), stream=stream0)
        buf574 = reinterpret_tensor(buf570, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf570  # reuse
        # Source Nodes: [x_63], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_23.run(buf574, addmm_13, 1806336, grid=grid(1806336), stream=stream0)
        del addmm_13
        buf575 = buf562; del buf562  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf574, (1568, 1152), (1152, 1), 0), permute_540, out=buf575)
        del permute_540
        buf576 = empty((1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf574, (1152, 1568), (1, 1152), 0), view_84, out=buf576)
        del view_84
        buf577 = buf540; del buf540  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_24.run(buf574, buf577, 14976, 121, grid=grid(14976), stream=stream0)
        buf578 = empty((1, 1152), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_25.run(buf577, buf578, 1152, 13, grid=grid(1152), stream=stream0)
        del buf577
        buf585 = buf569; del buf569  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf585, buf575, primals_73, mul_56, div_51, 1568, 384, grid=grid(1568), stream=stream0)
        del div_51
        del primals_73
        buf581 = reinterpret_tensor(buf572, (384, 13), (1, 384), 0); del buf572  # reuse
        buf583 = buf565; del buf565  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_27.run(buf575, mul_56, buf581, buf583, 4992, 121, grid=grid(4992), stream=stream0)
        del mul_56
        buf582 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf581, buf582, 384, 13, grid=grid(384), stream=stream0)
        buf584 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf583, buf584, 384, 13, grid=grid(384), stream=stream0)
        buf586 = buf575; del buf575  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf585, (1568, 384), (384, 1), 0), permute_544, out=buf586)
        del permute_544
        buf587 = empty((384, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf585, (384, 1568), (1, 384), 0), view_82, out=buf587)
        del view_82
        buf588 = reinterpret_tensor(buf583, (1, 384, 13), (4992, 1, 384), 0); del buf583  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_28.run(buf585, buf588, 4992, 121, grid=grid(4992), stream=stream0)
        buf589 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf588, buf589, 384, 13, grid=grid(384), stream=stream0)
        buf590 = reinterpret_tensor(buf558, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf558  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf586, buf590, 602112, grid=grid(602112), stream=stream0)
        buf591 = reinterpret_tensor(buf586, (96, 196, 32), (6272, 32, 1), 0); del buf586  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_549, reinterpret_tensor(buf590, (96, 196, 32), (6272, 32, 1), 0), out=buf591)
        del permute_549
        buf592 = reinterpret_tensor(buf557, (96, 196, 196), (38416, 196, 1), 0); del buf557  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf590, (96, 196, 32), (6272, 32, 1), 0), permute_550, out=buf592)
        del permute_550
        buf594 = reinterpret_tensor(buf555, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf555  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.mul]
        triton_per_fused__softmax_backward_data_mul_30.run(buf592, alias_38, buf594, 18816, 196, grid=grid(18816), stream=stream0)
        del alias_38
        del buf592
        buf595 = reinterpret_tensor(buf590, (96, 32, 196), (6272, 196, 1), 0); del buf590  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_551, reinterpret_tensor(buf594, (96, 196, 196), (38416, 196, 1), 0), out=buf595)
        del permute_551
        buf596 = buf554; del buf554  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf594, (96, 196, 196), (38416, 196, 1), 0), permute_552, out=buf596)
        del buf594
        del permute_552
        buf597 = reinterpret_tensor(buf574, (8, 196, 3, 12, 32), (225792, 1152, 384, 32, 1), 0); del buf574  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf596, buf595, buf591, buf597, 56448, 32, grid=grid(56448, 32), stream=stream0)
        del buf591
        del buf595
        buf598 = empty((1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf597, (1152, 1568), (1, 1152), 0), view_72, out=buf598)
        del view_72
        buf599 = reinterpret_tensor(buf596, (1568, 384), (384, 1), 0); del buf596  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf597, (1568, 1152), (1152, 1), 0), permute_557, out=buf599)
        del buf597
        del permute_557
        buf606 = buf585; del buf585  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_32.run(buf606, buf599, primals_68, mul_53, div_52, 1568, 384, grid=grid(1568), stream=stream0)
        del div_52
        del primals_68
        buf602 = reinterpret_tensor(buf588, (384, 13), (1, 384), 0); del buf588  # reuse
        buf604 = buf581; del buf581  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_27.run(buf599, mul_53, buf602, buf604, 4992, 121, grid=grid(4992), stream=stream0)
        del buf599
        del mul_53
        buf603 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf602, buf603, 384, 13, grid=grid(384), stream=stream0)
        del buf602
        buf605 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf604, buf605, 384, 13, grid=grid(384), stream=stream0)
        buf607 = empty((1, 14, 14, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_33.run(buf606, buf607, 75264, 8, grid=grid(75264), stream=stream0)
        buf608 = buf604; del buf604  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_34.run(buf606, buf608, 4992, 121, grid=grid(4992), stream=stream0)
        buf609 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_7.run(buf608, buf609, 384, 13, grid=grid(384), stream=stream0)
        del buf608
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf610 = aten.convolution_backward(reinterpret_tensor(buf606, (8, 384, 14, 14), (75264, 1, 5376, 384), 0), permute_57, primals_66, [384], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf606
        del permute_57
        del primals_66
        buf611 = buf610[0]
        buf612 = buf610[1]
        del buf610
        buf613 = empty((6272, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_35.run(buf611, buf613, 6272, 192, grid=grid(6272, 192), stream=stream0)
        buf614 = empty((6272, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf613, permute_561, out=buf614)
        del permute_561
        buf615 = empty((192, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf613, (192, 6272), (1, 192), 0), view_70, out=buf615)
        del view_70
        buf616 = empty_strided((1, 192, 49), (9408, 1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf613, buf616, 9408, 128, grid=grid(9408), stream=stream0)
        buf617 = empty((1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_37.run(buf616, buf617, 192, 49, grid=grid(192), stream=stream0)
        buf618 = reinterpret_tensor(buf614, (8, 28, 28, 576), (451584, 16128, 576, 1), 0); del buf614  # reuse
        # Source Nodes: [x_45], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_38.run(buf618, addmm_10, 3612672, grid=grid(3612672), stream=stream0)
        del addmm_10
        buf619 = buf613; del buf613  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf618, (6272, 576), (576, 1), 0), permute_565, out=buf619)
        del permute_565
        buf620 = empty((576, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf618, (576, 6272), (1, 576), 0), view_68, out=buf620)
        del view_68
        buf621 = empty_strided((1, 576, 49), (28224, 1, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf618, buf621, 28224, 128, grid=grid(28224), stream=stream0)
        buf622 = empty((1, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf621, buf622, 576, 49, grid=grid(576), stream=stream0)
        buf629 = empty((8, 28, 28, 192), device='cuda', dtype=torch.float32)
        buf632 = empty((6272, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_41.run(buf619, primals_60, mul_48, buf611, div_53, buf629, buf632, 6272, 192, grid=grid(6272), stream=stream0)
        del buf611
        del div_53
        del primals_60
        buf625 = reinterpret_tensor(buf616, (192, 49), (1, 192), 0); del buf616  # reuse
        buf627 = empty_strided((192, 49), (1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_42.run(buf619, mul_48, buf625, buf627, 9408, 128, grid=grid(9408), stream=stream0)
        del mul_48
        buf626 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_37.run(buf625, buf626, 192, 49, grid=grid(192), stream=stream0)
        buf628 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_37.run(buf627, buf628, 192, 49, grid=grid(192), stream=stream0)
        buf630 = reinterpret_tensor(buf627, (1, 1, 1, 192, 49), (9408, 9408, 9408, 1, 192), 0); del buf627  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_43.run(buf629, buf630, 9408, 128, grid=grid(9408), stream=stream0)
        buf631 = empty((1, 1, 1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_37.run(buf630, buf631, 192, 49, grid=grid(192), stream=stream0)
        buf633 = empty((192, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf632, (192, 6272), (1, 192), 0), view_66, out=buf633)
        del view_66
        buf634 = buf619; del buf619  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf632, permute_571, out=buf634)
        del permute_571
        buf635 = empty((8, 6, 196, 9, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(unsqueeze_17, add_17, buf634, buf635, 2709504, grid=grid(2709504), stream=stream0)
        buf636 = empty((9408, 9, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_576, reinterpret_tensor(buf635, (9408, 9, 32), (288, 32, 1), 0), out=buf636)
        del permute_576
        buf637 = empty((9408, 9, 9), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf635, (9408, 9, 32), (288, 32, 1), 0), permute_577, out=buf637)
        del permute_577
        buf639 = empty((8, 196, 6, 9, 9), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.clone]
        triton_per_fused__softmax_backward_data_clone_45.run(buf637, alias_39, buf639, 84672, 9, grid=grid(84672), stream=stream0)
        del alias_39
        buf640 = empty((1568, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf639, (1568, 486), (486, 1), 0), permute_579, out=buf640)
        del permute_579
        buf641 = empty((486, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf639, (486, 1568), (1, 486), 0), view_58, out=buf641)
        del view_58
        buf642 = empty_strided((1, 486, 13), (6318, 1, 486), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_46.run(buf639, buf642, 6318, 121, grid=grid(6318), stream=stream0)
        buf643 = empty((1, 486), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_47.run(buf642, buf643, 486, 13, grid=grid(486), stream=stream0)
        buf644 = empty((8, 192, 30, 30), device='cuda', dtype=torch.float32)
        buf690 = empty((8, 192, 30, 30), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_48.run(full_default, buf644, buf690, 1382400, grid=grid(1382400), stream=stream0)
        buf645 = reinterpret_tensor(buf635, (8, 6, 32, 9, 196), (338688, 56448, 1764, 196, 1), 0); del buf635  # reuse
        buf646 = reinterpret_tensor(buf645, (8, 192, 3, 14, 3, 14), (338688, 1764, 588, 14, 196, 1), 0); del buf645  # reuse
        # Source Nodes: [], Original ATen: [aten.clone, aten.col2im]
        triton_poi_fused_clone_col2im_49.run(buf646, buf636, 1536, 1764, grid=grid(1536, 1764), stream=stream0)
        aten.index_put_(buf644, [None, None, unsqueeze_17, add_17], buf646, True)
        buf649 = buf634; del buf634  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_50.run(buf644, buf649, 6272, 192, grid=grid(6272, 192), stream=stream0)
        buf650 = empty((192, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf649, (192, 6272), (1, 192), 0), view_54, out=buf650)
        del view_54
        buf651 = buf632; del buf632  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf649, permute_590, out=buf651)
        del permute_590
        buf658 = buf629; del buf629  # reuse
        buf659 = buf649; del buf649  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_51.run(buf658, buf640, buf651, primals_53, mul_45, div_54, buf659, 6272, 192, grid=grid(6272), stream=stream0)
        del div_54
        del primals_53
        buf654 = reinterpret_tensor(buf630, (192, 49), (1, 192), 0); del buf630  # reuse
        buf656 = buf625; del buf625  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_52.run(buf640, buf651, mul_45, buf654, buf656, 9408, 128, grid=grid(9408), stream=stream0)
        del mul_45
        buf655 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_sum_37.run(buf654, buf655, 192, 49, grid=grid(192), stream=stream0)
        buf657 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_sum_37.run(buf656, buf657, 192, 49, grid=grid(192), stream=stream0)
        buf660 = reinterpret_tensor(buf618, (6272, 576), (576, 1), 0); del buf618  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf659, permute_592, out=buf660)
        del permute_592
        buf661 = empty((192, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf659, (192, 6272), (1, 192), 0), view_52, out=buf661)
        del view_52
        buf662 = reinterpret_tensor(buf656, (1, 192, 49), (9408, 1, 192), 0); del buf656  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf659, buf662, 9408, 128, grid=grid(9408), stream=stream0)
        buf663 = empty((1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_37.run(buf662, buf663, 192, 49, grid=grid(192), stream=stream0)
        buf664 = reinterpret_tensor(buf660, (8, 28, 28, 576), (451584, 16128, 576, 1), 0); del buf660  # reuse
        # Source Nodes: [x_33], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_38.run(buf664, addmm_7, 3612672, grid=grid(3612672), stream=stream0)
        del addmm_7
        buf665 = buf659; del buf659  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf664, (6272, 576), (576, 1), 0), permute_596, out=buf665)
        del permute_596
        buf666 = empty((576, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf664, (576, 6272), (1, 576), 0), view_50, out=buf666)
        del view_50
        buf667 = buf621; del buf621  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf664, buf667, 28224, 128, grid=grid(28224), stream=stream0)
        buf668 = empty((1, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf667, buf668, 576, 49, grid=grid(576), stream=stream0)
        buf675 = buf658; del buf658  # reuse
        buf678 = buf651; del buf651  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_53.run(buf675, buf665, primals_47, mul_40, div_55, buf678, 6272, 192, grid=grid(6272), stream=stream0)
        del div_55
        del primals_47
        buf671 = reinterpret_tensor(buf662, (192, 49), (1, 192), 0); del buf662  # reuse
        buf673 = buf654; del buf654  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_42.run(buf665, mul_40, buf671, buf673, 9408, 128, grid=grid(9408), stream=stream0)
        del mul_40
        buf672 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_37.run(buf671, buf672, 192, 49, grid=grid(192), stream=stream0)
        buf674 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_37.run(buf673, buf674, 192, 49, grid=grid(192), stream=stream0)
        buf676 = reinterpret_tensor(buf673, (1, 1, 1, 192, 49), (9408, 9408, 9408, 1, 192), 0); del buf673  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_43.run(buf675, buf676, 9408, 128, grid=grid(9408), stream=stream0)
        buf677 = empty((1, 1, 1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_37.run(buf676, buf677, 192, 49, grid=grid(192), stream=stream0)
        buf679 = empty((192, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf678, (192, 6272), (1, 192), 0), view_48, out=buf679)
        del view_48
        buf680 = buf665; del buf665  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf678, permute_602, out=buf680)
        del permute_602
        buf681 = reinterpret_tensor(buf646, (8, 6, 196, 9, 32), (338688, 56448, 288, 32, 1), 0); del buf646  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(unsqueeze_17, add_17, buf680, buf681, 2709504, grid=grid(2709504), stream=stream0)
        buf682 = buf636; del buf636  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_607, reinterpret_tensor(buf681, (9408, 9, 32), (288, 32, 1), 0), out=buf682)
        del permute_607
        buf683 = reinterpret_tensor(buf639, (9408, 9, 9), (81, 9, 1), 0); del buf639  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf681, (9408, 9, 32), (288, 32, 1), 0), permute_608, out=buf683)
        del permute_608
        buf685 = reinterpret_tensor(buf637, (8, 196, 6, 9, 9), (95256, 486, 81, 9, 1), 0); del buf637  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.clone]
        triton_per_fused__softmax_backward_data_clone_45.run(buf683, alias_40, buf685, 84672, 9, grid=grid(84672), stream=stream0)
        del alias_40
        buf686 = buf640; del buf640  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf685, (1568, 486), (486, 1), 0), permute_610, out=buf686)
        del permute_610
        buf687 = empty((486, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf685, (486, 1568), (1, 486), 0), view_40, out=buf687)
        del view_40
        buf688 = buf642; del buf642  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_46.run(buf685, buf688, 6318, 121, grid=grid(6318), stream=stream0)
        buf689 = empty((1, 486), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_47.run(buf688, buf689, 486, 13, grid=grid(486), stream=stream0)
        buf691 = reinterpret_tensor(buf681, (8, 6, 32, 9, 196), (338688, 56448, 1764, 196, 1), 0); del buf681  # reuse
        buf692 = reinterpret_tensor(buf691, (8, 192, 3, 14, 3, 14), (338688, 1764, 588, 14, 196, 1), 0); del buf691  # reuse
        # Source Nodes: [], Original ATen: [aten.clone, aten.col2im]
        triton_poi_fused_clone_col2im_49.run(buf692, buf682, 1536, 1764, grid=grid(1536, 1764), stream=stream0)
        aten.index_put_(buf690, [None, None, unsqueeze_17, add_17], buf692, True)
        buf695 = buf680; del buf680  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_50.run(buf690, buf695, 6272, 192, grid=grid(6272, 192), stream=stream0)
        buf696 = empty((192, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf695, (192, 6272), (1, 192), 0), view_36, out=buf696)
        del view_36
        buf697 = buf678; del buf678  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf695, permute_621, out=buf697)
        del permute_621
        buf704 = buf675; del buf675  # reuse
        buf705 = buf695; del buf695  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_51.run(buf704, buf686, buf697, primals_40, mul_37, div_56, buf705, 6272, 192, grid=grid(6272), stream=stream0)
        del div_56
        del primals_40
        buf700 = reinterpret_tensor(buf676, (192, 49), (1, 192), 0); del buf676  # reuse
        buf702 = buf671; del buf671  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_52.run(buf686, buf697, mul_37, buf700, buf702, 9408, 128, grid=grid(9408), stream=stream0)
        del mul_37
        buf701 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_sum_37.run(buf700, buf701, 192, 49, grid=grid(192), stream=stream0)
        buf703 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_sum_37.run(buf702, buf703, 192, 49, grid=grid(192), stream=stream0)
        buf706 = reinterpret_tensor(buf664, (6272, 576), (576, 1), 0); del buf664  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf705, permute_623, out=buf706)
        del permute_623
        buf707 = empty((192, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf705, (192, 6272), (1, 192), 0), view_34, out=buf707)
        del view_34
        buf708 = reinterpret_tensor(buf702, (1, 192, 49), (9408, 1, 192), 0); del buf702  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf705, buf708, 9408, 128, grid=grid(9408), stream=stream0)
        buf709 = empty((1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_37.run(buf708, buf709, 192, 49, grid=grid(192), stream=stream0)
        buf710 = reinterpret_tensor(buf706, (8, 28, 28, 576), (451584, 16128, 576, 1), 0); del buf706  # reuse
        # Source Nodes: [x_21], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_38.run(buf710, addmm_4, 3612672, grid=grid(3612672), stream=stream0)
        del addmm_4
        buf711 = buf705; del buf705  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf710, (6272, 576), (576, 1), 0), permute_627, out=buf711)
        del permute_627
        buf712 = empty((576, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf710, (576, 6272), (1, 576), 0), view_32, out=buf712)
        del view_32
        buf713 = buf667; del buf667  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf710, buf713, 28224, 128, grid=grid(28224), stream=stream0)
        buf714 = empty((1, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf713, buf714, 576, 49, grid=grid(576), stream=stream0)
        buf721 = buf704; del buf704  # reuse
        buf724 = buf697; del buf697  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_53.run(buf721, buf711, primals_34, mul_32, div_57, buf724, 6272, 192, grid=grid(6272), stream=stream0)
        del div_57
        del primals_34
        buf717 = reinterpret_tensor(buf708, (192, 49), (1, 192), 0); del buf708  # reuse
        buf719 = buf700; del buf700  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_42.run(buf711, mul_32, buf717, buf719, 9408, 128, grid=grid(9408), stream=stream0)
        del mul_32
        buf718 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_37.run(buf717, buf718, 192, 49, grid=grid(192), stream=stream0)
        buf720 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_37.run(buf719, buf720, 192, 49, grid=grid(192), stream=stream0)
        buf722 = reinterpret_tensor(buf719, (1, 1, 1, 192, 49), (9408, 9408, 9408, 1, 192), 0); del buf719  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_43.run(buf721, buf722, 9408, 128, grid=grid(9408), stream=stream0)
        buf723 = empty((1, 1, 1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_37.run(buf722, buf723, 192, 49, grid=grid(192), stream=stream0)
        buf725 = empty((192, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf724, (192, 6272), (1, 192), 0), view_30, out=buf725)
        del view_30
        buf726 = buf711; del buf711  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf724, permute_633, out=buf726)
        del permute_633
        buf727 = reinterpret_tensor(buf692, (8, 6, 196, 9, 32), (338688, 56448, 288, 32, 1), 0); del buf692  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(unsqueeze_17, add_17, buf726, buf727, 2709504, grid=grid(2709504), stream=stream0)
        buf728 = buf682; del buf682  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_638, reinterpret_tensor(buf727, (9408, 9, 32), (288, 32, 1), 0), out=buf728)
        del permute_638
        buf729 = reinterpret_tensor(buf685, (9408, 9, 9), (81, 9, 1), 0); del buf685  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf727, (9408, 9, 32), (288, 32, 1), 0), permute_639, out=buf729)
        del permute_639
        buf731 = reinterpret_tensor(buf683, (8, 196, 6, 9, 9), (95256, 486, 81, 9, 1), 0); del buf683  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.clone]
        triton_per_fused__softmax_backward_data_clone_45.run(buf729, alias_41, buf731, 84672, 9, grid=grid(84672), stream=stream0)
        del alias_41
        buf732 = buf686; del buf686  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf731, (1568, 486), (486, 1), 0), permute_641, out=buf732)
        del permute_641
        buf733 = empty((486, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf731, (486, 1568), (1, 486), 0), view_22, out=buf733)
        del view_22
        buf734 = buf688; del buf688  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_46.run(buf731, buf734, 6318, 121, grid=grid(6318), stream=stream0)
        buf735 = empty((1, 486), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_47.run(buf734, buf735, 486, 13, grid=grid(486), stream=stream0)
        buf736 = buf690; del buf690  # reuse
        buf782 = buf644; del buf644  # reuse
        # Source Nodes: [], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_48.run(full_default, buf736, buf782, 1382400, grid=grid(1382400), stream=stream0)
        del full_default
        buf737 = reinterpret_tensor(buf727, (8, 6, 32, 9, 196), (338688, 56448, 1764, 196, 1), 0); del buf727  # reuse
        buf738 = reinterpret_tensor(buf737, (8, 192, 3, 14, 3, 14), (338688, 1764, 588, 14, 196, 1), 0); del buf737  # reuse
        # Source Nodes: [], Original ATen: [aten.clone, aten.col2im]
        triton_poi_fused_clone_col2im_49.run(buf738, buf728, 1536, 1764, grid=grid(1536, 1764), stream=stream0)
        aten.index_put_(buf736, [None, None, unsqueeze_17, add_17], buf738, True)
        buf741 = buf726; del buf726  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_50.run(buf736, buf741, 6272, 192, grid=grid(6272, 192), stream=stream0)
        del buf736
        buf742 = empty((192, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf741, (192, 6272), (1, 192), 0), view_18, out=buf742)
        del view_18
        buf743 = buf724; del buf724  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf741, permute_652, out=buf743)
        del permute_652
        buf750 = buf721; del buf721  # reuse
        buf751 = buf741; del buf741  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_51.run(buf750, buf732, buf743, primals_27, mul_29, div_58, buf751, 6272, 192, grid=grid(6272), stream=stream0)
        del div_58
        del primals_27
        buf746 = reinterpret_tensor(buf722, (192, 49), (1, 192), 0); del buf722  # reuse
        buf748 = buf717; del buf717  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_52.run(buf732, buf743, mul_29, buf746, buf748, 9408, 128, grid=grid(9408), stream=stream0)
        del mul_29
        buf747 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_sum_37.run(buf746, buf747, 192, 49, grid=grid(192), stream=stream0)
        buf749 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_sum_37.run(buf748, buf749, 192, 49, grid=grid(192), stream=stream0)
        buf752 = reinterpret_tensor(buf710, (6272, 576), (576, 1), 0); del buf710  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf751, permute_654, out=buf752)
        del permute_654
        buf753 = empty((192, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf751, (192, 6272), (1, 192), 0), view_16, out=buf753)
        del view_16
        buf754 = reinterpret_tensor(buf748, (1, 192, 49), (9408, 1, 192), 0); del buf748  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_36.run(buf751, buf754, 9408, 128, grid=grid(9408), stream=stream0)
        buf755 = empty((1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_37.run(buf754, buf755, 192, 49, grid=grid(192), stream=stream0)
        buf756 = reinterpret_tensor(buf752, (8, 28, 28, 576), (451584, 16128, 576, 1), 0); del buf752  # reuse
        # Source Nodes: [x_9], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_38.run(buf756, addmm_1, 3612672, grid=grid(3612672), stream=stream0)
        del addmm_1
        buf757 = buf751; del buf751  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf756, (6272, 576), (576, 1), 0), permute_658, out=buf757)
        del permute_658
        buf758 = empty((576, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf756, (576, 6272), (1, 576), 0), view_14, out=buf758)
        del view_14
        buf759 = buf713; del buf713  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_39.run(buf756, buf759, 28224, 128, grid=grid(28224), stream=stream0)
        del buf756
        buf760 = empty((1, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_40.run(buf759, buf760, 576, 49, grid=grid(576), stream=stream0)
        del buf759
        buf767 = buf750; del buf750  # reuse
        buf770 = buf743; del buf743  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.add, aten.clone, aten.native_layer_norm_backward]
        triton_per_fused__unsafe_view_add_clone_native_layer_norm_backward_53.run(buf767, buf757, primals_21, mul_24, div_59, buf770, 6272, 192, grid=grid(6272), stream=stream0)
        del div_59
        del primals_21
        buf763 = reinterpret_tensor(buf754, (192, 49), (1, 192), 0); del buf754  # reuse
        buf765 = buf746; del buf746  # reuse
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_red_fused_native_layer_norm_backward_42.run(buf757, mul_24, buf763, buf765, 9408, 128, grid=grid(9408), stream=stream0)
        del mul_24
        buf764 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_37.run(buf763, buf764, 192, 49, grid=grid(192), stream=stream0)
        buf766 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_layer_norm_backward]
        triton_per_fused_sum_37.run(buf765, buf766, 192, 49, grid=grid(192), stream=stream0)
        buf768 = reinterpret_tensor(buf765, (1, 1, 1, 192, 49), (9408, 9408, 9408, 1, 192), 0); del buf765  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_43.run(buf767, buf768, 9408, 128, grid=grid(9408), stream=stream0)
        buf769 = empty((1, 1, 1, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_37.run(buf768, buf769, 192, 49, grid=grid(192), stream=stream0)
        buf771 = empty((192, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf770, (192, 6272), (1, 192), 0), view_12, out=buf771)
        del view_12
        buf772 = buf757; del buf757  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf770, permute_664, out=buf772)
        del permute_664
        buf773 = reinterpret_tensor(buf738, (8, 6, 196, 9, 32), (338688, 56448, 288, 32, 1), 0); del buf738  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_44.run(unsqueeze_17, add_17, buf772, buf773, 2709504, grid=grid(2709504), stream=stream0)
        buf774 = buf728; del buf728  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_669, reinterpret_tensor(buf773, (9408, 9, 32), (288, 32, 1), 0), out=buf774)
        del permute_669
        buf775 = reinterpret_tensor(buf731, (9408, 9, 9), (81, 9, 1), 0); del buf731  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf773, (9408, 9, 32), (288, 32, 1), 0), permute_670, out=buf775)
        del permute_670
        buf777 = reinterpret_tensor(buf729, (8, 196, 6, 9, 9), (95256, 486, 81, 9, 1), 0); del buf729  # reuse
        # Source Nodes: [], Original ATen: [aten._softmax_backward_data, aten.clone]
        triton_per_fused__softmax_backward_data_clone_45.run(buf775, alias_42, buf777, 84672, 9, grid=grid(84672), stream=stream0)
        del alias_42
        del buf775
        buf778 = buf732; del buf732  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf777, (1568, 486), (486, 1), 0), permute_672, out=buf778)
        del permute_672
        buf779 = empty((486, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf777, (486, 1568), (1, 486), 0), view_4, out=buf779)
        del view_4
        buf780 = buf734; del buf734  # reuse
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_red_fused_sum_46.run(buf777, buf780, 6318, 121, grid=grid(6318), stream=stream0)
        del buf777
        buf781 = empty((1, 486), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.sum]
        triton_per_fused_sum_47.run(buf780, buf781, 486, 13, grid=grid(486), stream=stream0)
        del buf780
        buf783 = reinterpret_tensor(buf773, (8, 6, 32, 9, 196), (338688, 56448, 1764, 196, 1), 0); del buf773  # reuse
        buf784 = reinterpret_tensor(buf783, (8, 192, 3, 14, 3, 14), (338688, 1764, 588, 14, 196, 1), 0); del buf783  # reuse
        # Source Nodes: [], Original ATen: [aten.clone, aten.col2im]
        triton_poi_fused_clone_col2im_49.run(buf784, buf774, 1536, 1764, grid=grid(1536, 1764), stream=stream0)
        del buf774
        aten.index_put_(buf782, [None, None, unsqueeze_17, add_17], buf784, True)
        del add_17
        del buf784
        del unsqueeze_17
        buf787 = buf772; del buf772  # reuse
        # Source Nodes: [], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_50.run(buf782, buf787, 6272, 192, grid=grid(6272, 192), stream=stream0)
        del buf782
        buf788 = empty((192, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf787, (192, 6272), (1, 192), 0), view, out=buf788)
        del view
        buf789 = buf770; del buf770  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf787, permute_683, out=buf789)
        del buf787
        del permute_683
        buf796 = buf767; del buf767  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_backward_54.run(buf796, buf778, buf789, primals_14, mul_21, div_60, 6272, 192, grid=grid(6272), stream=stream0)
        del div_60
        del primals_14
        buf792 = reinterpret_tensor(buf768, (192, 49), (1, 192), 0); del buf768  # reuse
        buf794 = buf763; del buf763  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_backward_52.run(buf778, buf789, mul_21, buf792, buf794, 9408, 128, grid=grid(9408), stream=stream0)
        del buf778
        del mul_21
        buf793 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_sum_37.run(buf792, buf793, 192, 49, grid=grid(192), stream=stream0)
        del buf792
        buf795 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]
        triton_per_fused_sum_37.run(buf794, buf795, 192, 49, grid=grid(192), stream=stream0)
        del buf794
        buf797 = reinterpret_tensor(buf789, (8, 192, 28, 28), (150528, 784, 28, 1), 0); del buf789  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.permute]
        triton_poi_fused_add_native_layer_norm_backward_permute_55.run(buf796, buf797, 1536, 784, grid=grid(1536, 784), stream=stream0)
        del buf796
        buf798 = empty((192, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        triton_red_fused_convolution_backward_56.run(buf797, buf798, 192, 6272, grid=grid(192), stream=stream0)
        # Source Nodes: [], Original ATen: [aten.convolution_backward]
        buf799 = aten.convolution_backward(buf797, relu_2, primals_12, [192], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf797
        del primals_12
        buf800 = buf799[0]
        buf801 = buf799[1]
        del buf799
        buf802 = empty((64, 13), device='cuda', dtype=torch.float32)
        buf804 = empty((64, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_57.run(relu_2, buf800, convolution_2, unsqueeze_112, buf802, buf804, 832, 7720, grid=grid(832), stream=stream0)
        buf803 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_58.run(buf802, buf803, 64, 13, grid=grid(64), stream=stream0)
        buf805 = empty((64, ), device='cuda', dtype=torch.float32)
        buf806 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_59.run(buf804, squeeze_7, buf805, buf806, 64, 13, grid=grid(64), stream=stream0)
        buf807 = buf800; del buf800  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_60.run(buf807, relu_2, convolution_2, unsqueeze_112, buf805, squeeze_7, buf803, primals_10, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_2
        del primals_10
        del relu_2
        del squeeze_7
        del unsqueeze_112
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf808 = aten.convolution_backward(buf807, relu_1, primals_9, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf807
        del primals_9
        buf809 = buf808[0]
        buf810 = buf808[1]
        del buf808
        buf811 = buf804; del buf804  # reuse
        buf813 = buf802; del buf802  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_57.run(relu_1, buf809, convolution_1, unsqueeze_124, buf811, buf813, 832, 7720, grid=grid(832), stream=stream0)
        buf812 = buf805; del buf805  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_58.run(buf811, buf812, 64, 13, grid=grid(64), stream=stream0)
        buf814 = empty((64, ), device='cuda', dtype=torch.float32)
        buf815 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_59.run(buf813, squeeze_4, buf814, buf815, 64, 13, grid=grid(64), stream=stream0)
        buf816 = buf809; del buf809  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_60.run(buf816, relu_1, convolution_1, unsqueeze_124, buf814, squeeze_4, buf812, primals_7, 6422528, grid=grid(6422528), stream=stream0)
        del convolution_1
        del primals_7
        del relu_1
        del squeeze_4
        del unsqueeze_124
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf817 = aten.convolution_backward(buf816, relu, primals_6, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        del buf816
        del primals_6
        buf818 = buf817[0]
        buf819 = buf817[1]
        del buf817
        buf820 = buf813; del buf813  # reuse
        buf822 = buf811; del buf811  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_red_fused_native_batch_norm_backward_threshold_backward_57.run(relu, buf818, convolution, unsqueeze_136, buf820, buf822, 832, 7720, grid=grid(832), stream=stream0)
        buf821 = buf814; del buf814  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_58.run(buf820, buf821, 64, 13, grid=grid(64), stream=stream0)
        del buf820
        buf823 = empty((64, ), device='cuda', dtype=torch.float32)
        buf824 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward, aten.threshold_backward]
        triton_per_fused_native_batch_norm_backward_threshold_backward_59.run(buf822, squeeze_1, buf823, buf824, 64, 13, grid=grid(64), stream=stream0)
        del buf822
        buf825 = buf818; del buf818  # reuse
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        triton_poi_fused_convolution_backward_native_batch_norm_backward_threshold_backward_60.run(buf825, relu, convolution, unsqueeze_136, buf823, squeeze_1, buf821, primals_4, 6422528, grid=grid(6422528), stream=stream0)
        del buf823
        del convolution
        del primals_4
        del relu
        del squeeze_1
        del unsqueeze_136
        # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
        buf826 = aten.convolution_backward(buf825, primals_261, primals_3, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False])
        del buf825
        del primals_261
        del primals_3
        buf827 = buf826[1]
        return (buf607, buf87, buf827, buf824, buf821, buf819, buf815, buf812, buf810, buf806, buf803, buf801, buf798, buf793, buf795, reinterpret_tensor(buf788, (192, 192), (192, 1), 0), reinterpret_tensor(buf779, (486, 192), (192, 1), 0), reinterpret_tensor(buf781, (486, ), (1, ), 0), reinterpret_tensor(buf771, (192, 192), (192, 1), 0), reinterpret_tensor(buf769, (192, ), (1, ), 0), buf764, buf766, reinterpret_tensor(buf758, (576, 192), (192, 1), 0), reinterpret_tensor(buf760, (576, ), (1, ), 0), reinterpret_tensor(buf753, (192, 576), (576, 1), 0), reinterpret_tensor(buf755, (192, ), (1, ), 0), buf747, buf749, reinterpret_tensor(buf742, (192, 192), (192, 1), 0), reinterpret_tensor(buf733, (486, 192), (192, 1), 0), reinterpret_tensor(buf735, (486, ), (1, ), 0), reinterpret_tensor(buf725, (192, 192), (192, 1), 0), reinterpret_tensor(buf723, (192, ), (1, ), 0), buf718, buf720, reinterpret_tensor(buf712, (576, 192), (192, 1), 0), reinterpret_tensor(buf714, (576, ), (1, ), 0), reinterpret_tensor(buf707, (192, 576), (576, 1), 0), reinterpret_tensor(buf709, (192, ), (1, ), 0), buf701, buf703, reinterpret_tensor(buf696, (192, 192), (192, 1), 0), reinterpret_tensor(buf687, (486, 192), (192, 1), 0), reinterpret_tensor(buf689, (486, ), (1, ), 0), reinterpret_tensor(buf679, (192, 192), (192, 1), 0), reinterpret_tensor(buf677, (192, ), (1, ), 0), buf672, buf674, reinterpret_tensor(buf666, (576, 192), (192, 1), 0), reinterpret_tensor(buf668, (576, ), (1, ), 0), reinterpret_tensor(buf661, (192, 576), (576, 1), 0), reinterpret_tensor(buf663, (192, ), (1, ), 0), buf655, buf657, reinterpret_tensor(buf650, (192, 192), (192, 1), 0), reinterpret_tensor(buf641, (486, 192), (192, 1), 0), reinterpret_tensor(buf643, (486, ), (1, ), 0), reinterpret_tensor(buf633, (192, 192), (192, 1), 0), reinterpret_tensor(buf631, (192, ), (1, ), 0), buf626, buf628, reinterpret_tensor(buf620, (576, 192), (192, 1), 0), reinterpret_tensor(buf622, (576, ), (1, ), 0), reinterpret_tensor(buf615, (192, 576), (576, 1), 0), reinterpret_tensor(buf617, (192, ), (1, ), 0), buf612, buf609, buf603, buf605, reinterpret_tensor(buf598, (1152, 384), (384, 1), 0), reinterpret_tensor(buf587, (384, 384), (384, 1), 0), reinterpret_tensor(buf589, (384, ), (1, ), 0), buf582, buf584, reinterpret_tensor(buf576, (1152, 384), (384, 1), 0), reinterpret_tensor(buf578, (1152, ), (1, ), 0), reinterpret_tensor(buf571, (384, 1152), (1152, 1), 0), reinterpret_tensor(buf573, (384, ), (1, ), 0), buf566, buf568, reinterpret_tensor(buf561, (1152, 384), (384, 1), 0), reinterpret_tensor(buf550, (384, 384), (384, 1), 0), reinterpret_tensor(buf552, (384, ), (1, ), 0), buf545, buf547, reinterpret_tensor(buf539, (1152, 384), (384, 1), 0), reinterpret_tensor(buf541, (1152, ), (1, ), 0), reinterpret_tensor(buf534, (384, 1152), (1152, 1), 0), reinterpret_tensor(buf536, (384, ), (1, ), 0), buf529, buf531, reinterpret_tensor(buf524, (1152, 384), (384, 1), 0), reinterpret_tensor(buf513, (384, 384), (384, 1), 0), reinterpret_tensor(buf515, (384, ), (1, ), 0), buf508, buf510, reinterpret_tensor(buf502, (1152, 384), (384, 1), 0), reinterpret_tensor(buf504, (1152, ), (1, ), 0), reinterpret_tensor(buf497, (384, 1152), (1152, 1), 0), reinterpret_tensor(buf499, (384, ), (1, ), 0), buf492, buf494, reinterpret_tensor(buf487, (1152, 384), (384, 1), 0), reinterpret_tensor(buf476, (384, 384), (384, 1), 0), reinterpret_tensor(buf478, (384, ), (1, ), 0), buf471, buf473, reinterpret_tensor(buf465, (1152, 384), (384, 1), 0), reinterpret_tensor(buf467, (1152, ), (1, ), 0), reinterpret_tensor(buf460, (384, 1152), (1152, 1), 0), reinterpret_tensor(buf462, (384, ), (1, ), 0), buf455, buf457, reinterpret_tensor(buf450, (1152, 384), (384, 1), 0), reinterpret_tensor(buf439, (384, 384), (384, 1), 0), reinterpret_tensor(buf441, (384, ), (1, ), 0), buf434, buf436, reinterpret_tensor(buf428, (1152, 384), (384, 1), 0), reinterpret_tensor(buf430, (1152, ), (1, ), 0), reinterpret_tensor(buf423, (384, 1152), (1152, 1), 0), reinterpret_tensor(buf425, (384, ), (1, ), 0), buf418, buf420, reinterpret_tensor(buf413, (1152, 384), (384, 1), 0), reinterpret_tensor(buf402, (384, 384), (384, 1), 0), reinterpret_tensor(buf404, (384, ), (1, ), 0), buf397, buf399, reinterpret_tensor(buf391, (1152, 384), (384, 1), 0), reinterpret_tensor(buf393, (1152, ), (1, ), 0), reinterpret_tensor(buf386, (384, 1152), (1152, 1), 0), reinterpret_tensor(buf388, (384, ), (1, ), 0), buf381, buf383, reinterpret_tensor(buf376, (1152, 384), (384, 1), 0), reinterpret_tensor(buf365, (384, 384), (384, 1), 0), reinterpret_tensor(buf367, (384, ), (1, ), 0), buf360, buf362, reinterpret_tensor(buf354, (1152, 384), (384, 1), 0), reinterpret_tensor(buf356, (1152, ), (1, ), 0), reinterpret_tensor(buf349, (384, 1152), (1152, 1), 0), reinterpret_tensor(buf351, (384, ), (1, ), 0), buf344, buf346, reinterpret_tensor(buf339, (1152, 384), (384, 1), 0), reinterpret_tensor(buf328, (384, 384), (384, 1), 0), reinterpret_tensor(buf330, (384, ), (1, ), 0), buf323, buf325, reinterpret_tensor(buf317, (1152, 384), (384, 1), 0), reinterpret_tensor(buf319, (1152, ), (1, ), 0), reinterpret_tensor(buf312, (384, 1152), (1152, 1), 0), reinterpret_tensor(buf314, (384, ), (1, ), 0), buf307, buf309, reinterpret_tensor(buf302, (1152, 384), (384, 1), 0), reinterpret_tensor(buf291, (384, 384), (384, 1), 0), reinterpret_tensor(buf293, (384, ), (1, ), 0), buf286, buf288, reinterpret_tensor(buf280, (1152, 384), (384, 1), 0), reinterpret_tensor(buf282, (1152, ), (1, ), 0), reinterpret_tensor(buf275, (384, 1152), (1152, 1), 0), reinterpret_tensor(buf277, (384, ), (1, ), 0), buf270, buf272, reinterpret_tensor(buf265, (1152, 384), (384, 1), 0), reinterpret_tensor(buf254, (384, 384), (384, 1), 0), reinterpret_tensor(buf256, (384, ), (1, ), 0), buf249, buf251, reinterpret_tensor(buf243, (1152, 384), (384, 1), 0), reinterpret_tensor(buf245, (1152, ), (1, ), 0), reinterpret_tensor(buf238, (384, 1152), (1152, 1), 0), reinterpret_tensor(buf240, (384, ), (1, ), 0), buf233, buf235, reinterpret_tensor(buf228, (1152, 384), (384, 1), 0), reinterpret_tensor(buf217, (384, 384), (384, 1), 0), reinterpret_tensor(buf219, (384, ), (1, ), 0), buf212, buf214, reinterpret_tensor(buf206, (1152, 384), (384, 1), 0), reinterpret_tensor(buf208, (1152, ), (1, ), 0), reinterpret_tensor(buf201, (384, 1152), (1152, 1), 0), reinterpret_tensor(buf203, (384, ), (1, ), 0), buf196, buf198, reinterpret_tensor(buf191, (1152, 384), (384, 1), 0), reinterpret_tensor(buf180, (384, 384), (384, 1), 0), reinterpret_tensor(buf182, (384, ), (1, ), 0), buf175, buf177, reinterpret_tensor(buf169, (1152, 384), (384, 1), 0), reinterpret_tensor(buf171, (1152, ), (1, ), 0), reinterpret_tensor(buf164, (384, 1152), (1152, 1), 0), reinterpret_tensor(buf166, (384, ), (1, ), 0), buf159, buf161, reinterpret_tensor(buf154, (1152, 384), (384, 1), 0), reinterpret_tensor(buf143, (384, 384), (384, 1), 0), reinterpret_tensor(buf145, (384, ), (1, ), 0), buf138, buf140, reinterpret_tensor(buf132, (1152, 384), (384, 1), 0), reinterpret_tensor(buf134, (1152, ), (1, ), 0), reinterpret_tensor(buf127, (384, 1152), (1152, 1), 0), reinterpret_tensor(buf129, (384, ), (1, ), 0), buf122, buf124, reinterpret_tensor(buf117, (1152, 384), (384, 1), 0), reinterpret_tensor(buf106, (384, 384), (384, 1), 0), reinterpret_tensor(buf108, (384, ), (1, ), 0), buf101, buf103, reinterpret_tensor(buf95, (1152, 384), (384, 1), 0), reinterpret_tensor(buf97, (1152, ), (1, ), 0), reinterpret_tensor(buf90, (384, 1152), (1152, 1), 0), reinterpret_tensor(buf92, (384, ), (1, ), 0), buf83, buf85, reinterpret_tensor(buf78, (768, 384), (384, 1), 0), reinterpret_tensor(buf75, (384, 384), (384, 1), 0), reinterpret_tensor(buf66, (384, 384), (384, 1), 0), reinterpret_tensor(buf67, (384, ), (1, ), 0), buf62, buf63, reinterpret_tensor(buf58, (1152, 384), (384, 1), 0), reinterpret_tensor(buf59, (1152, ), (1, ), 0), reinterpret_tensor(buf54, (384, 1152), (1152, 1), 0), reinterpret_tensor(buf55, (384, ), (1, ), 0), buf48, buf50, reinterpret_tensor(buf43, (768, 384), (384, 1), 0), reinterpret_tensor(buf40, (384, 384), (384, 1), 0), reinterpret_tensor(buf31, (384, 384), (384, 1), 0), reinterpret_tensor(buf32, (384, ), (1, ), 0), buf27, buf28, reinterpret_tensor(buf23, (1152, 384), (384, 1), 0), reinterpret_tensor(buf24, (1152, ), (1, ), 0), reinterpret_tensor(buf19, (384, 1152), (1152, 1), 0), reinterpret_tensor(buf20, (384, ), (1, ), 0), buf15, buf17, reinterpret_tensor(buf9, (1000, 384), (384, 1), 0), reinterpret_tensor(buf10, (1000, ), (1, ), 0), reinterpret_tensor(buf6, (1000, 384), (384, 1), 0), reinterpret_tensor(buf5, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_3 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((192, 64, 4, 4), (1024, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((384, 192, 2, 2), (768, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    convolution = rand_strided((8, 64, 112, 112), (802816, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    squeeze_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu = rand_strided((8, 64, 112, 112), (802816, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    convolution_1 = rand_strided((8, 64, 112, 112), (802816, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    squeeze_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_1 = rand_strided((8, 64, 112, 112), (802816, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    convolution_2 = rand_strided((8, 64, 112, 112), (802816, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    squeeze_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    relu_2 = rand_strided((8, 64, 112, 112), (802816, 12544, 112, 1), device='cuda:0', dtype=torch.float32)
    mul_21 = rand_strided((8, 28, 28, 192), (150528, 5376, 192, 1), device='cuda:0', dtype=torch.float32)
    view = rand_strided((6272, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    add_17 = rand_strided((3, 14), (14, 1), device='cuda:0', dtype=torch.int64)
    unsqueeze_17 = rand_strided((3, 14, 1, 1), (14, 1, 1, 1), device='cuda:0', dtype=torch.int64)
    permute_5 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    view_4 = rand_strided((1568, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    full_default = rand_strided((8, 192, 30, 30), (172800, 900, 30, 1), device='cuda:0', dtype=torch.float32)
    view_12 = rand_strided((6272, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_24 = rand_strided((8, 28, 28, 192), (150528, 5376, 192, 1), device='cuda:0', dtype=torch.float32)
    view_14 = rand_strided((6272, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    addmm_1 = rand_strided((6272, 576), (576, 1), device='cuda:0', dtype=torch.float32)
    view_16 = rand_strided((6272, 576), (576, 1), device='cuda:0', dtype=torch.float32)
    mul_29 = rand_strided((8, 28, 28, 192), (150528, 5376, 192, 1), device='cuda:0', dtype=torch.float32)
    view_18 = rand_strided((6272, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_19 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    view_22 = rand_strided((1568, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    view_30 = rand_strided((6272, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_32 = rand_strided((8, 28, 28, 192), (150528, 5376, 192, 1), device='cuda:0', dtype=torch.float32)
    view_32 = rand_strided((6272, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    addmm_4 = rand_strided((6272, 576), (576, 1), device='cuda:0', dtype=torch.float32)
    view_34 = rand_strided((6272, 576), (576, 1), device='cuda:0', dtype=torch.float32)
    mul_37 = rand_strided((8, 28, 28, 192), (150528, 5376, 192, 1), device='cuda:0', dtype=torch.float32)
    view_36 = rand_strided((6272, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_33 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    view_40 = rand_strided((1568, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    view_48 = rand_strided((6272, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_40 = rand_strided((8, 28, 28, 192), (150528, 5376, 192, 1), device='cuda:0', dtype=torch.float32)
    view_50 = rand_strided((6272, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    addmm_7 = rand_strided((6272, 576), (576, 1), device='cuda:0', dtype=torch.float32)
    view_52 = rand_strided((6272, 576), (576, 1), device='cuda:0', dtype=torch.float32)
    mul_45 = rand_strided((8, 28, 28, 192), (150528, 5376, 192, 1), device='cuda:0', dtype=torch.float32)
    view_54 = rand_strided((6272, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_47 = rand_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    view_58 = rand_strided((1568, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    view_66 = rand_strided((6272, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    mul_48 = rand_strided((8, 28, 28, 192), (150528, 5376, 192, 1), device='cuda:0', dtype=torch.float32)
    view_68 = rand_strided((6272, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    addmm_10 = rand_strided((6272, 576), (576, 1), device='cuda:0', dtype=torch.float32)
    view_70 = rand_strided((6272, 576), (576, 1), device='cuda:0', dtype=torch.float32)
    permute_57 = rand_strided((8, 192, 28, 28), (150528, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    mul_53 = rand_strided((8, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    view_72 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_82 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_56 = rand_strided((8, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    view_84 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    addmm_13 = rand_strided((1568, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    view_86 = rand_strided((1568, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    mul_61 = rand_strided((8, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    view_88 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_98 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_64 = rand_strided((8, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    view_100 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    addmm_16 = rand_strided((1568, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    view_102 = rand_strided((1568, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    mul_69 = rand_strided((8, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    view_104 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_114 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_72 = rand_strided((8, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    view_116 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    addmm_19 = rand_strided((1568, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    view_118 = rand_strided((1568, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    mul_77 = rand_strided((8, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    view_120 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_130 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_80 = rand_strided((8, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    view_132 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    addmm_22 = rand_strided((1568, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    view_134 = rand_strided((1568, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    mul_85 = rand_strided((8, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    view_136 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_146 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_88 = rand_strided((8, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    view_148 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    addmm_25 = rand_strided((1568, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    view_150 = rand_strided((1568, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    mul_93 = rand_strided((8, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    view_152 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_162 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_96 = rand_strided((8, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    view_164 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    addmm_28 = rand_strided((1568, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    view_166 = rand_strided((1568, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    mul_101 = rand_strided((8, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    view_168 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_178 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_104 = rand_strided((8, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    view_180 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    addmm_31 = rand_strided((1568, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    view_182 = rand_strided((1568, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    mul_109 = rand_strided((8, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    view_184 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_194 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_112 = rand_strided((8, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    view_196 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    addmm_34 = rand_strided((1568, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    view_198 = rand_strided((1568, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    mul_117 = rand_strided((8, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    view_200 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_210 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_120 = rand_strided((8, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    view_212 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    addmm_37 = rand_strided((1568, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    view_214 = rand_strided((1568, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    mul_125 = rand_strided((8, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    view_216 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_226 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_128 = rand_strided((8, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    view_228 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    addmm_40 = rand_strided((1568, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    view_230 = rand_strided((1568, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    mul_133 = rand_strided((8, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    view_232 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_242 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_136 = rand_strided((8, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    view_244 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    addmm_43 = rand_strided((1568, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    view_246 = rand_strided((1568, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    mul_141 = rand_strided((8, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    view_248 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_258 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_144 = rand_strided((8, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    view_260 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    addmm_46 = rand_strided((1568, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    view_262 = rand_strided((1568, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    mul_149 = rand_strided((8, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    view_264 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_274 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_152 = rand_strided((8, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    view_276 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    addmm_49 = rand_strided((1568, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    view_278 = rand_strided((1568, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    mul_157 = rand_strided((8, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    view_280 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_290 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_160 = rand_strided((8, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    view_292 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    addmm_52 = rand_strided((1568, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    view_294 = rand_strided((1568, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    cat = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_121 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_39 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_297 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_300 = rand_strided((8, 384), (75648, 1), device='cuda:0', dtype=torch.float32)
    view_310 = rand_strided((8, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_168 = rand_strided((8, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    view_312 = rand_strided((8, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    addmm_55 = rand_strided((8, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    view_314 = rand_strided((8, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    cat_1 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_127 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_41 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    view_316 = rand_strided((1576, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_319 = rand_strided((8, 384), (75648, 1), device='cuda:0', dtype=torch.float32)
    view_329 = rand_strided((8, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    mul_176 = rand_strided((8, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    view_331 = rand_strided((8, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    addmm_58 = rand_strided((8, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    view_333 = rand_strided((8, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    cat_2 = rand_strided((8, 197, 384), (75648, 384, 1), device='cuda:0', dtype=torch.float32)
    getitem_133 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_43 = rand_strided((8, 197, 1), (197, 1, 1), device='cuda:0', dtype=torch.float32)
    select = rand_strided((8, 384), (75648, 1), device='cuda:0', dtype=torch.float32)
    view_335 = rand_strided((1568, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_61 = rand_strided((8, 1, 1000), (1000, 1000, 1), device='cuda:0', dtype=torch.int64)
    permute_177 = rand_strided((1000, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_179 = rand_strided((1000, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_183 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    permute_187 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_21 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_191 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_196 = rand_strided((96, 197, 1), (197, 1, 0), device='cuda:0', dtype=torch.float32)
    permute_197 = rand_strided((96, 32, 197), (6304, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_23 = rand_strided((8, 12, 1, 197), (2364, 197, 197, 1), device='cuda:0', dtype=torch.float32)
    permute_198 = rand_strided((96, 32, 1), (32, 1, 0), device='cuda:0', dtype=torch.float32)
    permute_199 = rand_strided((96, 197, 32), (6304, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_203 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_208 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_210 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    permute_214 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_23 = rand_strided((8, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_218 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_223 = rand_strided((96, 197, 1), (197, 1, 0), device='cuda:0', dtype=torch.float32)
    permute_224 = rand_strided((96, 32, 197), (6304, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_24 = rand_strided((8, 12, 1, 197), (2364, 197, 197, 1), device='cuda:0', dtype=torch.float32)
    permute_225 = rand_strided((96, 32, 1), (32, 1, 0), device='cuda:0', dtype=torch.float32)
    permute_226 = rand_strided((96, 197, 32), (6304, 1, 197), device='cuda:0', dtype=torch.float32)
    permute_230 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_235 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_237 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    permute_241 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_25 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_245 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_250 = rand_strided((96, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_251 = rand_strided((96, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_25 = rand_strided((8, 12, 196, 196), (460992, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_252 = rand_strided((96, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_253 = rand_strided((96, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_258 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_26 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_260 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    permute_264 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_27 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_268 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_273 = rand_strided((96, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_274 = rand_strided((96, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_26 = rand_strided((8, 12, 196, 196), (460992, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_275 = rand_strided((96, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_276 = rand_strided((96, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_281 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_28 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_283 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    permute_287 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_29 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_291 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_296 = rand_strided((96, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_297 = rand_strided((96, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_27 = rand_strided((8, 12, 196, 196), (460992, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_298 = rand_strided((96, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_299 = rand_strided((96, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_304 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_30 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_306 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    permute_310 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_31 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_314 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_319 = rand_strided((96, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_320 = rand_strided((96, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_28 = rand_strided((8, 12, 196, 196), (460992, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_321 = rand_strided((96, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_322 = rand_strided((96, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_327 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_32 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_329 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    permute_333 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_33 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_337 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_342 = rand_strided((96, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_343 = rand_strided((96, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_29 = rand_strided((8, 12, 196, 196), (460992, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_344 = rand_strided((96, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_345 = rand_strided((96, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_350 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_34 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_352 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    permute_356 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_35 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_360 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_365 = rand_strided((96, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_366 = rand_strided((96, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_30 = rand_strided((8, 12, 196, 196), (460992, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_367 = rand_strided((96, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_368 = rand_strided((96, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_373 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_36 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_375 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    permute_379 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_37 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_383 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_388 = rand_strided((96, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_389 = rand_strided((96, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_31 = rand_strided((8, 12, 196, 196), (460992, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_390 = rand_strided((96, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_391 = rand_strided((96, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_396 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_38 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_398 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    permute_402 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_39 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_406 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_411 = rand_strided((96, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_412 = rand_strided((96, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_32 = rand_strided((8, 12, 196, 196), (460992, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_413 = rand_strided((96, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_414 = rand_strided((96, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_419 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_40 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_421 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    permute_425 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_41 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_429 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_434 = rand_strided((96, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_435 = rand_strided((96, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_33 = rand_strided((8, 12, 196, 196), (460992, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_436 = rand_strided((96, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_437 = rand_strided((96, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_442 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_42 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_444 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    permute_448 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_43 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_452 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_457 = rand_strided((96, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_458 = rand_strided((96, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_34 = rand_strided((8, 12, 196, 196), (460992, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_459 = rand_strided((96, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_460 = rand_strided((96, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_465 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_44 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_467 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    permute_471 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_45 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_475 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_480 = rand_strided((96, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_481 = rand_strided((96, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_35 = rand_strided((8, 12, 196, 196), (460992, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_482 = rand_strided((96, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_483 = rand_strided((96, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_488 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_46 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_490 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    permute_494 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_47 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_498 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_503 = rand_strided((96, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_504 = rand_strided((96, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_36 = rand_strided((8, 12, 196, 196), (460992, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_505 = rand_strided((96, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_506 = rand_strided((96, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_511 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_48 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_513 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    permute_517 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_49 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_521 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_526 = rand_strided((96, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_527 = rand_strided((96, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_37 = rand_strided((8, 12, 196, 196), (460992, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_528 = rand_strided((96, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_529 = rand_strided((96, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_534 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_50 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_536 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    permute_540 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_51 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_544 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    permute_549 = rand_strided((96, 196, 196), (38416, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_550 = rand_strided((96, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_38 = rand_strided((8, 12, 196, 196), (460992, 38416, 196, 1), device='cuda:0', dtype=torch.float32)
    permute_551 = rand_strided((96, 32, 196), (6272, 1, 32), device='cuda:0', dtype=torch.float32)
    permute_552 = rand_strided((96, 196, 32), (6272, 1, 196), device='cuda:0', dtype=torch.float32)
    permute_557 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    div_52 = rand_strided((8, 14, 14, 1), (196, 14, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_561 = rand_strided((192, 576), (576, 1), device='cuda:0', dtype=torch.float32)
    permute_565 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    div_53 = rand_strided((8, 28, 28, 1), (784, 28, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_571 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_576 = rand_strided((9408, 9, 9), (81, 1, 9), device='cuda:0', dtype=torch.float32)
    permute_577 = rand_strided((9408, 32, 9), (288, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_39 = rand_strided((8, 6, 196, 9, 9), (95256, 15876, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    permute_579 = rand_strided((486, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_590 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    div_54 = rand_strided((8, 28, 28, 1), (784, 28, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_592 = rand_strided((192, 576), (576, 1), device='cuda:0', dtype=torch.float32)
    permute_596 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    div_55 = rand_strided((8, 28, 28, 1), (784, 28, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_602 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_607 = rand_strided((9408, 9, 9), (81, 1, 9), device='cuda:0', dtype=torch.float32)
    permute_608 = rand_strided((9408, 32, 9), (288, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_40 = rand_strided((8, 6, 196, 9, 9), (95256, 15876, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    permute_610 = rand_strided((486, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_621 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    div_56 = rand_strided((8, 28, 28, 1), (784, 28, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_623 = rand_strided((192, 576), (576, 1), device='cuda:0', dtype=torch.float32)
    permute_627 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    div_57 = rand_strided((8, 28, 28, 1), (784, 28, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_633 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_638 = rand_strided((9408, 9, 9), (81, 1, 9), device='cuda:0', dtype=torch.float32)
    permute_639 = rand_strided((9408, 32, 9), (288, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_41 = rand_strided((8, 6, 196, 9, 9), (95256, 15876, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    permute_641 = rand_strided((486, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_652 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    div_58 = rand_strided((8, 28, 28, 1), (784, 28, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_654 = rand_strided((192, 576), (576, 1), device='cuda:0', dtype=torch.float32)
    permute_658 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    div_59 = rand_strided((8, 28, 28, 1), (784, 28, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_664 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_669 = rand_strided((9408, 9, 9), (81, 1, 9), device='cuda:0', dtype=torch.float32)
    permute_670 = rand_strided((9408, 32, 9), (288, 1, 32), device='cuda:0', dtype=torch.float32)
    alias_42 = rand_strided((8, 6, 196, 9, 9), (95256, 15876, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    permute_672 = rand_strided((486, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    permute_683 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    div_60 = rand_strided((8, 28, 28, 1), (784, 28, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_112 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_124 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_136 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_3, primals_4, primals_6, primals_7, primals_9, primals_10, primals_12, primals_14, primals_21, primals_27, primals_34, primals_40, primals_47, primals_53, primals_60, primals_66, primals_68, primals_73, primals_79, primals_84, primals_90, primals_95, primals_101, primals_106, primals_112, primals_117, primals_123, primals_128, primals_134, primals_139, primals_145, primals_150, primals_156, primals_161, primals_167, primals_172, primals_178, primals_183, primals_189, primals_194, primals_200, primals_205, primals_211, primals_216, primals_222, primals_228, primals_234, primals_240, primals_246, primals_261, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, mul_21, view, add_17, unsqueeze_17, permute_5, view_4, full_default, view_12, mul_24, view_14, addmm_1, view_16, mul_29, view_18, permute_19, view_22, view_30, mul_32, view_32, addmm_4, view_34, mul_37, view_36, permute_33, view_40, view_48, mul_40, view_50, addmm_7, view_52, mul_45, view_54, permute_47, view_58, view_66, mul_48, view_68, addmm_10, view_70, permute_57, mul_53, view_72, view_82, mul_56, view_84, addmm_13, view_86, mul_61, view_88, view_98, mul_64, view_100, addmm_16, view_102, mul_69, view_104, view_114, mul_72, view_116, addmm_19, view_118, mul_77, view_120, view_130, mul_80, view_132, addmm_22, view_134, mul_85, view_136, view_146, mul_88, view_148, addmm_25, view_150, mul_93, view_152, view_162, mul_96, view_164, addmm_28, view_166, mul_101, view_168, view_178, mul_104, view_180, addmm_31, view_182, mul_109, view_184, view_194, mul_112, view_196, addmm_34, view_198, mul_117, view_200, view_210, mul_120, view_212, addmm_37, view_214, mul_125, view_216, view_226, mul_128, view_228, addmm_40, view_230, mul_133, view_232, view_242, mul_136, view_244, addmm_43, view_246, mul_141, view_248, view_258, mul_144, view_260, addmm_46, view_262, mul_149, view_264, view_274, mul_152, view_276, addmm_49, view_278, mul_157, view_280, view_290, mul_160, view_292, addmm_52, view_294, cat, getitem_121, rsqrt_39, view_297, view_300, view_310, mul_168, view_312, addmm_55, view_314, cat_1, getitem_127, rsqrt_41, view_316, view_319, view_329, mul_176, view_331, addmm_58, view_333, cat_2, getitem_133, rsqrt_43, select, view_335, unsqueeze_61, permute_177, permute_179, permute_183, permute_187, div_21, permute_191, permute_196, permute_197, alias_23, permute_198, permute_199, permute_203, permute_208, permute_210, permute_214, div_23, permute_218, permute_223, permute_224, alias_24, permute_225, permute_226, permute_230, permute_235, permute_237, permute_241, div_25, permute_245, permute_250, permute_251, alias_25, permute_252, permute_253, permute_258, div_26, permute_260, permute_264, div_27, permute_268, permute_273, permute_274, alias_26, permute_275, permute_276, permute_281, div_28, permute_283, permute_287, div_29, permute_291, permute_296, permute_297, alias_27, permute_298, permute_299, permute_304, div_30, permute_306, permute_310, div_31, permute_314, permute_319, permute_320, alias_28, permute_321, permute_322, permute_327, div_32, permute_329, permute_333, div_33, permute_337, permute_342, permute_343, alias_29, permute_344, permute_345, permute_350, div_34, permute_352, permute_356, div_35, permute_360, permute_365, permute_366, alias_30, permute_367, permute_368, permute_373, div_36, permute_375, permute_379, div_37, permute_383, permute_388, permute_389, alias_31, permute_390, permute_391, permute_396, div_38, permute_398, permute_402, div_39, permute_406, permute_411, permute_412, alias_32, permute_413, permute_414, permute_419, div_40, permute_421, permute_425, div_41, permute_429, permute_434, permute_435, alias_33, permute_436, permute_437, permute_442, div_42, permute_444, permute_448, div_43, permute_452, permute_457, permute_458, alias_34, permute_459, permute_460, permute_465, div_44, permute_467, permute_471, div_45, permute_475, permute_480, permute_481, alias_35, permute_482, permute_483, permute_488, div_46, permute_490, permute_494, div_47, permute_498, permute_503, permute_504, alias_36, permute_505, permute_506, permute_511, div_48, permute_513, permute_517, div_49, permute_521, permute_526, permute_527, alias_37, permute_528, permute_529, permute_534, div_50, permute_536, permute_540, div_51, permute_544, permute_549, permute_550, alias_38, permute_551, permute_552, permute_557, div_52, permute_561, permute_565, div_53, permute_571, permute_576, permute_577, alias_39, permute_579, permute_590, div_54, permute_592, permute_596, div_55, permute_602, permute_607, permute_608, alias_40, permute_610, permute_621, div_56, permute_623, permute_627, div_57, permute_633, permute_638, permute_639, alias_41, permute_641, permute_652, div_58, permute_654, permute_658, div_59, permute_664, permute_669, permute_670, alias_42, permute_672, permute_683, div_60, unsqueeze_112, unsqueeze_124, unsqueeze_136, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('volo_d1_224', benchmark_compiled_module)
